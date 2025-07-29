"""
Performance Alert System - Wave 5.0 Implementation.

Provides comprehensive performance alerting capabilities including:
- Real-time alert generation and management
- Multi-channel alert delivery (email, webhook, logs)
- Alert escalation and notification policies
- Alert suppression and rate limiting
- Alert correlation and deduplication
- Integration with performance monitoring system
"""

import asyncio
import json
import logging
import smtplib
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp

from src.services.performance_monitor import (
    AlertSeverity,
    AlertStatus,
    PerformanceAlert,
    PerformanceMetricType,
    get_performance_monitor,
)


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    LOG = "log"
    SLACK = "slack"
    CUSTOM = "custom"


class EscalationPolicy(Enum):
    """Alert escalation policies."""

    IMMEDIATE = "immediate"  # Alert immediately
    DELAYED = "delayed"  # Alert after delay
    BATCHED = "batched"  # Batch alerts together
    SUPPRESSED = "suppressed"  # Suppress duplicate alerts


@dataclass
class AlertChannel_Config:
    """Configuration for an alert delivery channel."""

    channel_type: AlertChannel
    enabled: bool = True

    # Email configuration
    smtp_server: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    email_recipients: list[str] = field(default_factory=list)

    # Webhook configuration
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = field(default_factory=dict)
    webhook_timeout: float = 10.0

    # Slack configuration
    slack_webhook_url: str | None = None
    slack_channel: str | None = None

    # Custom handler
    custom_handler: Callable | None = None

    # Channel-specific settings
    rate_limit_per_hour: int = 100
    batch_size: int = 10
    batch_timeout_seconds: float = 300.0  # 5 minutes


@dataclass
class AlertRule:
    """Defines rules for alert generation and delivery."""

    rule_id: str
    name: str
    description: str

    # Trigger conditions
    component_pattern: str = "*"  # Component name pattern (* for all)
    metric_types: list[PerformanceMetricType] = field(default_factory=list)
    severities: list[AlertSeverity] = field(default_factory=list)

    # Delivery configuration
    channels: list[AlertChannel] = field(default_factory=list)
    escalation_policy: EscalationPolicy = EscalationPolicy.IMMEDIATE
    escalation_delay_seconds: float = 300.0  # 5 minutes

    # Rate limiting and suppression
    rate_limit_per_hour: int = 10
    suppression_duration_seconds: float = 1800.0  # 30 minutes
    duplicate_threshold_seconds: float = 60.0  # 1 minute

    # Rule state
    enabled: bool = True
    last_triggered: float | None = None
    trigger_count: int = 0
    suppressed_until: float | None = None

    def can_trigger(self, current_time: float) -> bool:
        """Check if this rule can be triggered."""
        if not self.enabled:
            return False

        # Check suppression
        if self.suppressed_until and current_time < self.suppressed_until:
            return False

        # Check rate limiting
        if self.last_triggered:
            time_since_last = current_time - self.last_triggered
            if time_since_last < (3600 / self.rate_limit_per_hour):  # Convert rate to seconds
                return False

        return True

    def matches_alert(self, alert: PerformanceAlert) -> bool:
        """Check if this rule matches an alert."""
        # Check component pattern
        if self.component_pattern != "*" and self.component_pattern != alert.component:
            return False

        # Check metric types
        if self.metric_types and alert.metric_name not in [mt.value for mt in self.metric_types]:
            return False

        # Check severities
        if self.severities and alert.severity not in self.severities:
            return False

        return True


@dataclass
class AlertDelivery:
    """Represents an alert delivery attempt."""

    delivery_id: str
    alert_id: str
    rule_id: str
    channel: AlertChannel
    timestamp: float
    success: bool
    error_message: str | None = None
    delivery_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertBatch:
    """Represents a batch of alerts for delivery."""

    batch_id: str
    alerts: list[PerformanceAlert]
    channel: AlertChannel
    created_at: float
    scheduled_delivery: float
    delivered: bool = False

    @property
    def is_ready(self) -> bool:
        """Check if batch is ready for delivery."""
        current_time = time.time()
        return current_time >= self.scheduled_delivery


class PerformanceAlertSystem:
    """
    Comprehensive performance alert system.

    Manages alert generation, delivery, escalation, and notification across
    multiple channels with intelligent rate limiting and deduplication.
    """

    def __init__(self):
        """Initialize the performance alert system."""
        self.logger = logging.getLogger(__name__)

        # Performance monitoring integration
        self.performance_monitor = get_performance_monitor()

        # Alert system state
        self._alert_rules: list[AlertRule] = []
        self._alert_channels: dict[AlertChannel, AlertChannel_Config] = {}
        self._delivery_history: deque = deque(maxlen=10000)
        self._alert_batches: dict[str, AlertBatch] = {}

        # Alert processing
        self._alert_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: asyncio.Task | None = None
        self._delivery_task: asyncio.Task | None = None
        self._is_running = False

        # Alert correlation and deduplication
        self._recent_alerts: dict[str, PerformanceAlert] = {}
        self._suppressed_alerts: dict[str, float] = {}  # alert_key -> suppression_end_time

        # Statistics
        self._alert_stats = {
            "total_alerts_received": 0,
            "total_alerts_delivered": 0,
            "total_alerts_suppressed": 0,
            "delivery_failures": 0,
            "duplicate_alerts": 0,
        }

        # Initialize default rules and channels
        self._initialize_default_configuration()

        self.logger.info("PerformanceAlertSystem initialized")

    def _initialize_default_configuration(self):
        """Initialize default alert rules and channels."""
        # Default log channel
        log_channel = AlertChannel_Config(channel_type=AlertChannel.LOG, enabled=True)
        self._alert_channels[AlertChannel.LOG] = log_channel

        # Default alert rules
        critical_rule = AlertRule(
            rule_id="critical_alerts",
            name="Critical Performance Alerts",
            description="Immediate notification for critical performance issues",
            severities=[AlertSeverity.CRITICAL],
            channels=[AlertChannel.LOG],
            escalation_policy=EscalationPolicy.IMMEDIATE,
            rate_limit_per_hour=20,
        )

        error_rule = AlertRule(
            rule_id="error_alerts",
            name="Error Performance Alerts",
            description="Notification for error-level performance issues",
            severities=[AlertSeverity.ERROR],
            channels=[AlertChannel.LOG],
            escalation_policy=EscalationPolicy.DELAYED,
            escalation_delay_seconds=60.0,  # 1 minute delay
            rate_limit_per_hour=10,
        )

        warning_rule = AlertRule(
            rule_id="warning_alerts",
            name="Warning Performance Alerts",
            description="Batched notification for warning-level issues",
            severities=[AlertSeverity.WARNING],
            channels=[AlertChannel.LOG],
            escalation_policy=EscalationPolicy.BATCHED,
            rate_limit_per_hour=5,
        )

        self._alert_rules.extend([critical_rule, error_rule, warning_rule])

    async def start_alert_system(self):
        """Start the performance alert system."""
        if self._is_running:
            return {"status": "already_running", "message": "Alert system already running"}

        try:
            self._is_running = True

            # Start alert processing tasks
            self._processing_task = asyncio.create_task(self._alert_processing_loop())
            self._delivery_task = asyncio.create_task(self._alert_delivery_loop())

            # Subscribe to performance monitor alerts
            if hasattr(self.performance_monitor, "alert_callback"):
                original_callback = self.performance_monitor.alert_callback
                self.performance_monitor.alert_callback = self._create_combined_callback(original_callback)
            else:
                self.performance_monitor.alert_callback = self._handle_performance_alert

            self.logger.info("Performance alert system started")
            return {"status": "started", "message": "Alert system started successfully"}

        except Exception as e:
            self.logger.error(f"Error starting alert system: {e}")
            self._is_running = False
            return {"status": "error", "message": f"Failed to start alert system: {e}"}

    async def stop_alert_system(self):
        """Stop the performance alert system."""
        self._is_running = False

        # Cancel processing tasks
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass

        # Process remaining alerts in queue
        while not self._alert_queue.empty():
            try:
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                await self._process_alert(alert)
            except asyncio.TimeoutError:
                break

        self.logger.info("Performance alert system stopped")
        return {"status": "stopped", "message": "Alert system stopped successfully"}

    def _create_combined_callback(self, original_callback: Callable | None) -> Callable:
        """Create a combined callback that calls both original and alert system callbacks."""

        async def combined_callback(alert: PerformanceAlert):
            # Call original callback first
            if original_callback:
                try:
                    if asyncio.iscoroutinefunction(original_callback):
                        await original_callback(alert)
                    else:
                        original_callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in original alert callback: {e}")

            # Call our alert handler
            await self._handle_performance_alert(alert)

        return combined_callback

    async def _handle_performance_alert(self, alert: PerformanceAlert):
        """Handle alerts from the performance monitor."""
        try:
            await self._alert_queue.put(alert)
            self._alert_stats["total_alerts_received"] += 1

        except Exception as e:
            self.logger.error(f"Error handling performance alert: {e}")

    async def _alert_processing_loop(self):
        """Main alert processing loop."""
        self.logger.info("Alert processing loop started")

        while self._is_running:
            try:
                # Get alert from queue with timeout
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                await self._process_alert(alert)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(1.0)

        self.logger.info("Alert processing loop stopped")

    async def _alert_delivery_loop(self):
        """Alert delivery loop for batched alerts."""
        self.logger.info("Alert delivery loop started")

        while self._is_running:
            try:
                # Check for ready batches
                ready_batches = [batch for batch in self._alert_batches.values() if batch.is_ready and not batch.delivered]

                # Deliver ready batches
                for batch in ready_batches:
                    await self._deliver_alert_batch(batch)
                    batch.delivered = True

                # Clean up old batches
                current_time = time.time()
                old_batches = [
                    batch_id
                    for batch_id, batch in self._alert_batches.items()
                    if batch.delivered and (current_time - batch.created_at) > 3600  # 1 hour old
                ]

                for batch_id in old_batches:
                    del self._alert_batches[batch_id]

                # Wait before next check
                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert delivery loop: {e}")
                await asyncio.sleep(30.0)

        self.logger.info("Alert delivery loop stopped")

    async def _process_alert(self, alert: PerformanceAlert):
        """Process an individual alert through the alert system."""
        try:
            current_time = time.time()

            # Check for duplicate alerts
            alert_key = f"{alert.component}_{alert.metric_name}_{alert.alert_type}"

            if self._is_duplicate_alert(alert_key, alert, current_time):
                self._alert_stats["duplicate_alerts"] += 1
                return

            # Store recent alert
            self._recent_alerts[alert_key] = alert

            # Find matching rules
            matching_rules = [rule for rule in self._alert_rules if rule.matches_alert(alert) and rule.can_trigger(current_time)]

            if not matching_rules:
                return

            # Process alert for each matching rule
            for rule in matching_rules:
                await self._process_alert_for_rule(alert, rule, current_time)

        except Exception as e:
            self.logger.error(f"Error processing alert: {e}")

    def _is_duplicate_alert(self, alert_key: str, alert: PerformanceAlert, current_time: float) -> bool:
        """Check if this is a duplicate alert that should be suppressed."""
        if alert_key in self._recent_alerts:
            recent_alert = self._recent_alerts[alert_key]
            time_diff = current_time - recent_alert.timestamp

            # Check if this is within the duplicate threshold
            if time_diff < 60.0:  # 1 minute duplicate threshold
                return True

        return False

    async def _process_alert_for_rule(self, alert: PerformanceAlert, rule: AlertRule, current_time: float):
        """Process an alert for a specific rule."""
        try:
            # Update rule state
            rule.last_triggered = current_time
            rule.trigger_count += 1

            # Handle different escalation policies
            if rule.escalation_policy == EscalationPolicy.IMMEDIATE:
                await self._deliver_alert_immediately(alert, rule)

            elif rule.escalation_policy == EscalationPolicy.DELAYED:
                await self._schedule_delayed_alert(alert, rule)

            elif rule.escalation_policy == EscalationPolicy.BATCHED:
                await self._add_alert_to_batch(alert, rule)

            elif rule.escalation_policy == EscalationPolicy.SUPPRESSED:
                self._suppress_alert(alert, rule, current_time)

        except Exception as e:
            self.logger.error(f"Error processing alert for rule {rule.rule_id}: {e}")

    async def _deliver_alert_immediately(self, alert: PerformanceAlert, rule: AlertRule):
        """Deliver an alert immediately to all configured channels."""
        try:
            for channel_type in rule.channels:
                if channel_type in self._alert_channels:
                    channel_config = self._alert_channels[channel_type]
                    if channel_config.enabled:
                        await self._deliver_alert_to_channel(alert, rule, channel_config)

        except Exception as e:
            self.logger.error(f"Error delivering immediate alert: {e}")

    async def _schedule_delayed_alert(self, alert: PerformanceAlert, rule: AlertRule):
        """Schedule an alert for delayed delivery."""
        try:
            # For delayed alerts, we create a single-alert batch with delay
            batch_id = f"delayed_{rule.rule_id}_{int(time.time())}"
            delivery_time = time.time() + rule.escalation_delay_seconds

            for channel_type in rule.channels:
                batch = AlertBatch(
                    batch_id=f"{batch_id}_{channel_type.value}",
                    alerts=[alert],
                    channel=channel_type,
                    created_at=time.time(),
                    scheduled_delivery=delivery_time,
                )
                self._alert_batches[batch.batch_id] = batch

        except Exception as e:
            self.logger.error(f"Error scheduling delayed alert: {e}")

    async def _add_alert_to_batch(self, alert: PerformanceAlert, rule: AlertRule):
        """Add an alert to batches for batched delivery."""
        try:
            current_time = time.time()

            for channel_type in rule.channels:
                # Find existing batch or create new one
                batch_key = f"batch_{rule.rule_id}_{channel_type.value}"
                existing_batch = None

                for batch in self._alert_batches.values():
                    if (
                        batch.batch_id.startswith(batch_key)
                        and not batch.delivered
                        and len(batch.alerts) < self._alert_channels.get(channel_type, AlertChannel_Config(channel_type)).batch_size
                    ):
                        existing_batch = batch
                        break

                if existing_batch:
                    # Add to existing batch
                    existing_batch.alerts.append(alert)
                else:
                    # Create new batch
                    channel_config = self._alert_channels.get(channel_type, AlertChannel_Config(channel_type))
                    batch_id = f"{batch_key}_{int(current_time)}"

                    batch = AlertBatch(
                        batch_id=batch_id,
                        alerts=[alert],
                        channel=channel_type,
                        created_at=current_time,
                        scheduled_delivery=current_time + channel_config.batch_timeout_seconds,
                    )
                    self._alert_batches[batch_id] = batch

        except Exception as e:
            self.logger.error(f"Error adding alert to batch: {e}")

    def _suppress_alert(self, alert: PerformanceAlert, rule: AlertRule, current_time: float):
        """Suppress an alert based on rule configuration."""
        try:
            alert_key = f"{alert.component}_{alert.metric_name}"
            suppression_end = current_time + rule.suppression_duration_seconds
            self._suppressed_alerts[alert_key] = suppression_end

            rule.suppressed_until = suppression_end
            self._alert_stats["total_alerts_suppressed"] += 1

            self.logger.info(f"Alert suppressed: {alert_key} until {datetime.fromtimestamp(suppression_end)}")

        except Exception as e:
            self.logger.error(f"Error suppressing alert: {e}")

    async def _deliver_alert_batch(self, batch: AlertBatch):
        """Deliver a batch of alerts."""
        try:
            if batch.channel in self._alert_channels:
                channel_config = self._alert_channels[batch.channel]
                if channel_config.enabled:
                    await self._deliver_batch_to_channel(batch, channel_config)

        except Exception as e:
            self.logger.error(f"Error delivering alert batch {batch.batch_id}: {e}")

    async def _deliver_alert_to_channel(self, alert: PerformanceAlert, rule: AlertRule, channel_config: AlertChannel_Config):
        """Deliver a single alert to a specific channel."""
        delivery_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            success = False
            error_message = None

            if channel_config.channel_type == AlertChannel.EMAIL:
                success, error_message = await self._deliver_email_alert(alert, channel_config)

            elif channel_config.channel_type == AlertChannel.WEBHOOK:
                success, error_message = await self._deliver_webhook_alert(alert, channel_config)

            elif channel_config.channel_type == AlertChannel.SLACK:
                success, error_message = await self._deliver_slack_alert(alert, channel_config)

            elif channel_config.channel_type == AlertChannel.LOG:
                success, error_message = await self._deliver_log_alert(alert, channel_config)

            elif channel_config.channel_type == AlertChannel.CUSTOM:
                success, error_message = await self._deliver_custom_alert(alert, channel_config)

            # Record delivery
            delivery_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            delivery = AlertDelivery(
                delivery_id=delivery_id,
                alert_id=alert.alert_id,
                rule_id=rule.rule_id,
                channel=channel_config.channel_type,
                timestamp=time.time(),
                success=success,
                error_message=error_message,
                delivery_time_ms=delivery_time,
            )

            self._delivery_history.append(delivery)

            if success:
                self._alert_stats["total_alerts_delivered"] += 1
            else:
                self._alert_stats["delivery_failures"] += 1

        except Exception as e:
            self.logger.error(f"Error delivering alert to {channel_config.channel_type.value}: {e}")
            self._alert_stats["delivery_failures"] += 1

    async def _deliver_batch_to_channel(self, batch: AlertBatch, channel_config: AlertChannel_Config):
        """Deliver a batch of alerts to a specific channel."""
        try:
            if channel_config.channel_type == AlertChannel.EMAIL:
                await self._deliver_email_batch(batch, channel_config)
            elif channel_config.channel_type == AlertChannel.WEBHOOK:
                await self._deliver_webhook_batch(batch, channel_config)
            elif channel_config.channel_type == AlertChannel.SLACK:
                await self._deliver_slack_batch(batch, channel_config)
            elif channel_config.channel_type == AlertChannel.LOG:
                await self._deliver_log_batch(batch, channel_config)
            elif channel_config.channel_type == AlertChannel.CUSTOM:
                await self._deliver_custom_batch(batch, channel_config)

        except Exception as e:
            self.logger.error(f"Error delivering batch to {channel_config.channel_type.value}: {e}")

    async def _deliver_email_alert(self, alert: PerformanceAlert, config: AlertChannel_Config) -> tuple[bool, str | None]:
        """Deliver alert via email."""
        try:
            if not config.email_recipients:
                return False, "No email recipients configured"

            # Create email message
            msg = MIMEMultipart()
            msg["From"] = config.smtp_username or "performance-alerts@system.local"
            msg["To"] = ", ".join(config.email_recipients)
            msg["Subject"] = f"Performance Alert: {alert.severity.value.upper()} - {alert.component}"

            # Email body
            body = f"""
Performance Alert Notification

Severity: {alert.severity.value.upper()}
Component: {alert.component}
Metric: {alert.metric_name}
Message: {alert.message}

Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
Timestamp: {datetime.fromtimestamp(alert.timestamp)}

Alert ID: {alert.alert_id}

This is an automated message from the Performance Monitoring System.
            """.strip()

            msg.attach(MIMEText(body, "plain"))

            # Send email
            if config.smtp_server and config.smtp_username and config.smtp_password:
                server = smtplib.SMTP(config.smtp_server, config.smtp_port)
                server.starttls()
                server.login(config.smtp_username, config.smtp_password)
                text = msg.as_string()
                server.sendmail(config.smtp_username, config.email_recipients, text)
                server.quit()

                return True, None
            else:
                return False, "SMTP configuration incomplete"

        except Exception as e:
            return False, str(e)

    async def _deliver_webhook_alert(self, alert: PerformanceAlert, config: AlertChannel_Config) -> tuple[bool, str | None]:
        """Deliver alert via webhook."""
        try:
            if not config.webhook_url:
                return False, "No webhook URL configured"

            # Prepare webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity.value,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp,
                "status": alert.status.value,
                "metadata": alert.metadata,
            }

            # Send webhook
            timeout = aiohttp.ClientTimeout(total=config.webhook_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config.webhook_url, json=payload, headers=config.webhook_headers) as response:
                    if response.status == 200:
                        return True, None
                    else:
                        return False, f"HTTP {response.status}: {await response.text()}"

        except Exception as e:
            return False, str(e)

    async def _deliver_slack_alert(self, alert: PerformanceAlert, config: AlertChannel_Config) -> tuple[bool, str | None]:
        """Deliver alert via Slack webhook."""
        try:
            if not config.slack_webhook_url:
                return False, "No Slack webhook URL configured"

            # Determine color based on severity
            color_map = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.ERROR: "warning",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "good",
            }

            # Create Slack message
            payload = {
                "channel": config.slack_channel,
                "username": "Performance Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "title": f"Performance Alert: {alert.severity.value.upper()}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Component", "value": alert.component, "short": True},
                            {"title": "Metric", "value": alert.metric_name, "short": True},
                            {"title": "Current Value", "value": str(alert.current_value), "short": True},
                            {"title": "Threshold", "value": str(alert.threshold_value), "short": True},
                        ],
                        "footer": "Performance Monitoring System",
                        "ts": int(alert.timestamp),
                    }
                ],
            }

            # Send to Slack
            timeout = aiohttp.ClientTimeout(total=config.webhook_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        return True, None
                    else:
                        return False, f"HTTP {response.status}: {await response.text()}"

        except Exception as e:
            return False, str(e)

    async def _deliver_log_alert(self, alert: PerformanceAlert, config: AlertChannel_Config) -> tuple[bool, str | None]:
        """Deliver alert via logging."""
        try:
            log_message = (
                f"PERFORMANCE ALERT [{alert.severity.value.upper()}] "
                f"{alert.component}: {alert.message} "
                f"(Current: {alert.current_value}, Threshold: {alert.threshold_value})"
            )

            if alert.severity == AlertSeverity.CRITICAL:
                self.logger.critical(log_message)
            elif alert.severity == AlertSeverity.ERROR:
                self.logger.error(log_message)
            elif alert.severity == AlertSeverity.WARNING:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)

            return True, None

        except Exception as e:
            return False, str(e)

    async def _deliver_custom_alert(self, alert: PerformanceAlert, config: AlertChannel_Config) -> tuple[bool, str | None]:
        """Deliver alert via custom handler."""
        try:
            if not config.custom_handler:
                return False, "No custom handler configured"

            if asyncio.iscoroutinefunction(config.custom_handler):
                result = await config.custom_handler(alert, config)
            else:
                result = config.custom_handler(alert, config)

            if isinstance(result, tuple):
                return result
            else:
                return bool(result), None

        except Exception as e:
            return False, str(e)

    async def _deliver_email_batch(self, batch: AlertBatch, config: AlertChannel_Config):
        """Deliver batch of alerts via email."""
        # Implementation similar to single email but with multiple alerts
        pass

    async def _deliver_webhook_batch(self, batch: AlertBatch, config: AlertChannel_Config):
        """Deliver batch of alerts via webhook."""
        # Implementation for batch webhook delivery
        pass

    async def _deliver_slack_batch(self, batch: AlertBatch, config: AlertChannel_Config):
        """Deliver batch of alerts via Slack."""
        # Implementation for batch Slack delivery
        pass

    async def _deliver_log_batch(self, batch: AlertBatch, config: AlertChannel_Config):
        """Deliver batch of alerts via logging."""
        try:
            alert_summaries = [f"{alert.component}: {alert.message}" for alert in batch.alerts]

            log_message = f"PERFORMANCE ALERT BATCH ({len(batch.alerts)} alerts): {'; '.join(alert_summaries)}"
            self.logger.warning(log_message)

        except Exception as e:
            self.logger.error(f"Error delivering log batch: {e}")

    async def _deliver_custom_batch(self, batch: AlertBatch, config: AlertChannel_Config):
        """Deliver batch of alerts via custom handler."""
        # Implementation for custom batch delivery
        pass

    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self._alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.rule_id}")

    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        for i, rule in enumerate(self._alert_rules):
            if rule.rule_id == rule_id:
                del self._alert_rules[i]
                self.logger.info(f"Removed alert rule: {rule_id}")
                return True
        return False

    def configure_channel(self, channel_config: AlertChannel_Config):
        """Configure or update an alert channel."""
        self._alert_channels[channel_config.channel_type] = channel_config
        self.logger.info(f"Configured alert channel: {channel_config.channel_type.value}")

    def get_alert_status(self) -> dict[str, Any]:
        """Get the current status of the alert system."""
        return {
            "running": self._is_running,
            "total_rules": len(self._alert_rules),
            "active_rules": len([r for r in self._alert_rules if r.enabled]),
            "configured_channels": list(self._alert_channels.keys()),
            "pending_batches": len(self._alert_batches),
            "queue_size": self._alert_queue.qsize(),
            "statistics": self._alert_stats.copy(),
            "recent_deliveries": len([d for d in self._delivery_history if time.time() - d.timestamp <= 3600]),  # Last hour
        }

    def get_delivery_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get alert delivery history."""
        cutoff_time = time.time() - (hours * 3600)

        return [
            {
                "delivery_id": delivery.delivery_id,
                "alert_id": delivery.alert_id,
                "rule_id": delivery.rule_id,
                "channel": delivery.channel.value,
                "timestamp": delivery.timestamp,
                "success": delivery.success,
                "delivery_time_ms": delivery.delivery_time_ms,
                "error_message": delivery.error_message,
            }
            for delivery in self._delivery_history
            if delivery.timestamp > cutoff_time
        ]

    async def test_channel(self, channel_type: AlertChannel) -> dict[str, Any]:
        """Test an alert channel configuration."""
        try:
            if channel_type not in self._alert_channels:
                return {"success": False, "message": "Channel not configured"}

            config = self._alert_channels[channel_type]

            # Create test alert
            test_alert = PerformanceAlert(
                alert_id="test_alert",
                alert_type="test",
                severity=AlertSeverity.INFO,
                component="test_component",
                metric_name="test_metric",
                current_value=100.0,
                threshold_value=90.0,
                message="This is a test alert from the Performance Alert System",
                timestamp=time.time(),
            )

            # Test delivery
            if channel_type == AlertChannel.EMAIL:
                success, error = await self._deliver_email_alert(test_alert, config)
            elif channel_type == AlertChannel.WEBHOOK:
                success, error = await self._deliver_webhook_alert(test_alert, config)
            elif channel_type == AlertChannel.SLACK:
                success, error = await self._deliver_slack_alert(test_alert, config)
            elif channel_type == AlertChannel.LOG:
                success, error = await self._deliver_log_alert(test_alert, config)
            elif channel_type == AlertChannel.CUSTOM:
                success, error = await self._deliver_custom_alert(test_alert, config)
            else:
                success, error = False, f"Unsupported channel type: {channel_type}"

            return {
                "success": success,
                "message": "Test alert delivered successfully" if success else f"Test failed: {error}",
                "channel": channel_type.value,
                "error": error,
            }

        except Exception as e:
            return {"success": False, "message": f"Test failed with exception: {e}", "channel": channel_type.value}

    async def shutdown(self):
        """Shutdown the alert system."""
        self.logger.info("Shutting down PerformanceAlertSystem")
        await self.stop_alert_system()

        # Clear state
        self._alert_rules.clear()
        self._alert_channels.clear()
        self._delivery_history.clear()
        self._alert_batches.clear()
        self._recent_alerts.clear()
        self._suppressed_alerts.clear()

        self.logger.info("PerformanceAlertSystem shutdown complete")


# Global alert system instance
_alert_system: PerformanceAlertSystem | None = None


def get_alert_system() -> PerformanceAlertSystem:
    """Get the global alert system instance."""
    global _alert_system
    if _alert_system is None:
        _alert_system = PerformanceAlertSystem()
    return _alert_system
