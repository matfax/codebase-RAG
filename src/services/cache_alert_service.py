"""
Cache alert service for monitoring cache performance and triggering notifications.

This service implements configurable alert thresholds and notification mechanisms
for cache performance issues, capacity planning warnings, and operational alerts.
"""

import asyncio
import json
import logging
import smtplib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional, Union

from ..utils.performance_monitor import CachePerformanceMonitor, get_cache_performance_monitor


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of cache alerts."""

    HIGH_ERROR_RATE = "high_error_rate"
    SLOW_RESPONSE_TIME = "slow_response_time"
    LOW_HIT_RATE = "low_hit_rate"
    HIGH_MEMORY_USAGE = "high_memory_usage"
    MEMORY_PRESSURE = "memory_pressure"
    CAPACITY_WARNING = "capacity_warning"
    CONNECTION_FAILURE = "connection_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CACHE_UNAVAILABLE = "cache_unavailable"
    EVICTION_RATE_HIGH = "eviction_rate_high"
    CUSTOM = "custom"


@dataclass
class AlertThreshold:
    """Defines an alert threshold configuration."""

    alert_type: AlertType
    severity: AlertSeverity
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    metric_name: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    description: str = ""
    action_required: str = ""
    escalation_enabled: bool = False
    escalation_delay_seconds: int = 1800  # 30 minutes
    escalation_severity: AlertSeverity = AlertSeverity.HIGH

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.alert_type.value} alert"
        if not self.action_required:
            self.action_required = f"Investigate {self.alert_type.value} issue"


@dataclass
class Alert:
    """Represents a cache alert event."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    cache_name: str
    metric_name: str
    current_value: float
    threshold_value: float
    threshold_config: AlertThreshold
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_timestamp: float | None = None
    escalated: bool = False
    escalated_timestamp: float | None = None
    acknowledgment_timestamp: float | None = None
    acknowledged_by: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get the age of the alert in seconds."""
        return time.time() - self.timestamp

    @property
    def age_formatted(self) -> str:
        """Get the age of the alert formatted as a string."""
        age = self.age_seconds
        if age < 60:
            return f"{age:.0f}s"
        elif age < 3600:
            return f"{age // 60:.0f}m {age % 60:.0f}s"
        else:
            hours = age // 3600
            minutes = (age % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "cache_name": self.cache_name,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp,
            "escalated": self.escalated,
            "escalated_timestamp": self.escalated_timestamp,
            "acknowledgment_timestamp": self.acknowledgment_timestamp,
            "acknowledged_by": self.acknowledged_by,
            "age_seconds": self.age_seconds,
            "age_formatted": self.age_formatted,
            "additional_data": self.additional_data,
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """Send a notification for the given alert."""
        pass


class LogNotificationChannel(NotificationChannel):
    """Notification channel that logs alerts."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    async def send_notification(self, alert: Alert) -> bool:
        """Send notification by logging the alert."""
        try:
            log_level = {
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
            }.get(alert.severity, logging.WARNING)

            message = (
                f"Cache Alert [{alert.severity.value.upper()}] - {alert.alert_type.value}: "
                f"{alert.cache_name} - {alert.metric_name}={alert.current_value} "
                f"(threshold: {alert.threshold_value})"
            )

            self.logger.log(log_level, message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send log notification: {e}")
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
        use_tls: bool = True,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    async def send_notification(self, alert: Alert) -> bool:
        """Send notification via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"Cache Alert [{alert.severity.value.upper()}] - {alert.cache_name}"

            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, "html"))

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            return True
        except Exception:
            return False

    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body for the alert."""
        severity_color = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545",
        }.get(alert.severity, "#6c757d")

        return f"""
        <html>
        <body>
            <h2 style="color: {severity_color};">Cache Alert - {alert.alert_type.value.replace('_', ' ').title()}</h2>
            <p><strong>Cache:</strong> {alert.cache_name}</p>
            <p><strong>Severity:</strong> <span style="color: {severity_color};">{alert.severity.value.upper()}</span></p>
            <p><strong>Metric:</strong> {alert.metric_name}</p>
            <p><strong>Current Value:</strong> {alert.current_value}</p>
            <p><strong>Threshold:</strong> {alert.threshold_value}</p>
            <p><strong>Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}</p>
            <p><strong>Age:</strong> {alert.age_formatted}</p>
            <p><strong>Description:</strong> {alert.threshold_config.description}</p>
            <p><strong>Action Required:</strong> {alert.threshold_config.action_required}</p>

            <hr>
            <p><em>This alert was generated by the Cache Alert Service</em></p>
        </body>
        </html>
        """


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""

    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send_notification(self, alert: Alert) -> bool:
        """Send notification via webhook."""
        try:
            import aiohttp

            payload = {
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "cache_name": alert.cache_name,
                "message": f"{alert.alert_type.value}: {alert.cache_name} - {alert.metric_name}={alert.current_value}",
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp,
                "additional_data": alert.additional_data,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, json=payload, headers=self.headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
        except Exception:
            return False


class CacheAlertService:
    """Main cache alert service that monitors cache performance and triggers notifications."""

    def __init__(
        self,
        cache_monitor: CachePerformanceMonitor | None = None,
        logger: logging.Logger | None = None,
    ):
        self.cache_monitor = cache_monitor or get_cache_performance_monitor()
        self.logger = logger or logging.getLogger(__name__)

        # Alert configuration
        self.thresholds: dict[str, AlertThreshold] = {}
        self.notification_channels: list[NotificationChannel] = []

        # Alert state
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.max_history_size = 1000

        # Monitoring state
        self.monitoring_enabled = True
        self.monitoring_interval = 30.0  # seconds
        self.monitoring_task: asyncio.Task | None = None

        # Cooldown tracking
        self.alert_cooldowns: dict[str, float] = {}

        # Statistics
        self.stats = {
            "total_alerts_generated": 0,
            "total_alerts_resolved": 0,
            "total_alerts_escalated": 0,
            "total_notifications_sent": 0,
            "total_notifications_failed": 0,
            "monitoring_start_time": time.time(),
        }

        # Initialize default thresholds
        self._initialize_default_thresholds()

    def _initialize_default_thresholds(self) -> None:
        """Initialize default alert thresholds."""
        default_thresholds = [
            AlertThreshold(
                alert_type=AlertType.HIGH_ERROR_RATE,
                severity=AlertSeverity.HIGH,
                threshold_value=0.05,  # 5%
                comparison_operator="gt",
                metric_name="error_rate",
                description="Cache error rate is above acceptable threshold",
                action_required="Investigate cache connection and error patterns",
            ),
            AlertThreshold(
                alert_type=AlertType.SLOW_RESPONSE_TIME,
                severity=AlertSeverity.MEDIUM,
                threshold_value=1000,  # 1 second in ms
                comparison_operator="gt",
                metric_name="average_response_time_ms",
                description="Cache response time is slower than expected",
                action_required="Check cache performance and network latency",
            ),
            AlertThreshold(
                alert_type=AlertType.LOW_HIT_RATE,
                severity=AlertSeverity.MEDIUM,
                threshold_value=0.7,  # 70%
                comparison_operator="lt",
                metric_name="hit_rate",
                description="Cache hit rate is below optimal threshold",
                action_required="Review cache configuration and data patterns",
            ),
            AlertThreshold(
                alert_type=AlertType.HIGH_MEMORY_USAGE,
                severity=AlertSeverity.HIGH,
                threshold_value=500,  # 500MB
                comparison_operator="gt",
                metric_name="current_size_mb",
                description="Cache memory usage is high",
                action_required="Consider cache eviction policies or size limits",
            ),
            AlertThreshold(
                alert_type=AlertType.EVICTION_RATE_HIGH,
                severity=AlertSeverity.MEDIUM,
                threshold_value=100,  # 100 evictions per hour
                comparison_operator="gt",
                metric_name="eviction_rate_per_hour",
                description="Cache eviction rate is high",
                action_required="Review cache size limits and usage patterns",
            ),
        ]

        for threshold in default_thresholds:
            self.add_threshold(threshold)

    def add_threshold(self, threshold: AlertThreshold) -> None:
        """Add or update an alert threshold."""
        key = f"{threshold.alert_type.value}_{threshold.metric_name}"
        self.thresholds[key] = threshold
        self.logger.info(f"Added alert threshold: {key}")

    def remove_threshold(self, alert_type: AlertType, metric_name: str) -> None:
        """Remove an alert threshold."""
        key = f"{alert_type.value}_{metric_name}"
        if key in self.thresholds:
            del self.thresholds[key]
            self.logger.info(f"Removed alert threshold: {key}")

    def get_threshold(self, alert_type: AlertType, metric_name: str) -> AlertThreshold | None:
        """Get an alert threshold."""
        key = f"{alert_type.value}_{metric_name}"
        return self.thresholds.get(key)

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.notification_channels.append(channel)
        self.logger.info(f"Added notification channel: {type(channel).__name__}")

    def remove_notification_channel(self, channel: NotificationChannel) -> None:
        """Remove a notification channel."""
        if channel in self.notification_channels:
            self.notification_channels.remove(channel)
            self.logger.info(f"Removed notification channel: {type(channel).__name__}")

    def start_monitoring(self) -> None:
        """Start the alert monitoring system."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_enabled = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started cache alert monitoring")

    def stop_monitoring(self) -> None:
        """Stop the alert monitoring system."""
        self.monitoring_enabled = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
        self.logger.info("Stopped cache alert monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._check_all_alerts()
                await self._check_escalations()
                await self._cleanup_resolved_alerts()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    async def _check_all_alerts(self) -> None:
        """Check all configured alert thresholds."""
        cache_metrics = self.cache_monitor.get_all_cache_metrics()

        for cache_name, metrics in cache_metrics.items():
            await self._check_cache_alerts(cache_name, metrics)

    async def _check_cache_alerts(self, cache_name: str, metrics: dict[str, Any]) -> None:
        """Check alerts for a specific cache."""
        for threshold_key, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue

            # Check if we're in cooldown period
            cooldown_key = f"{cache_name}_{threshold_key}"
            if cooldown_key in self.alert_cooldowns:
                if time.time() - self.alert_cooldowns[cooldown_key] < threshold.cooldown_seconds:
                    continue

            # Get the metric value
            metric_value = self._extract_metric_value(metrics, threshold.metric_name)
            if metric_value is None:
                continue

            # Check threshold
            if self._check_threshold(metric_value, threshold):
                await self._trigger_alert(cache_name, threshold, metric_value, metrics)
            else:
                # Check if we should resolve an existing alert
                alert_id = f"{cache_name}_{threshold.alert_type.value}_{threshold.metric_name}"
                if alert_id in self.active_alerts:
                    await self._resolve_alert(alert_id, metric_value)

    def _extract_metric_value(self, metrics: dict[str, Any], metric_name: str) -> float | None:
        """Extract metric value from cache metrics."""
        try:
            # Handle nested metric paths
            parts = metric_name.split(".")
            value = metrics

            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None

            # Handle special computed metrics
            if metric_name == "current_size_mb":
                return value / 1024 / 1024 if isinstance(value, int | float) else None
            elif metric_name == "eviction_rate_per_hour":
                # Calculate eviction rate from eviction count and uptime
                eviction_count = metrics.get("memory", {}).get("eviction_count", 0)
                uptime = metrics.get("timestamps", {}).get("uptime_seconds", 0)
                if uptime > 0:
                    return (eviction_count / uptime) * 3600  # per hour
                return 0

            return float(value) if isinstance(value, int | float) else None
        except (KeyError, TypeError, ValueError):
            return None

    def _check_threshold(self, value: float, threshold: AlertThreshold) -> bool:
        """Check if a value exceeds the threshold."""
        if threshold.comparison_operator == "gt":
            return value > threshold.threshold_value
        elif threshold.comparison_operator == "lt":
            return value < threshold.threshold_value
        elif threshold.comparison_operator == "eq":
            return abs(value - threshold.threshold_value) < 0.0001
        elif threshold.comparison_operator == "gte":
            return value >= threshold.threshold_value
        elif threshold.comparison_operator == "lte":
            return value <= threshold.threshold_value
        else:
            return False

    async def _trigger_alert(
        self,
        cache_name: str,
        threshold: AlertThreshold,
        metric_value: float,
        metrics: dict[str, Any],
    ) -> None:
        """Trigger a new alert."""
        alert_id = f"{cache_name}_{threshold.alert_type.value}_{threshold.metric_name}"

        # Check if alert already exists
        if alert_id in self.active_alerts:
            return

        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=threshold.alert_type,
            severity=threshold.severity,
            cache_name=cache_name,
            metric_name=threshold.metric_name,
            current_value=metric_value,
            threshold_value=threshold.threshold_value,
            threshold_config=threshold,
            additional_data={"full_metrics": metrics},
        )

        # Add to active alerts
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.stats["total_alerts_generated"] += 1

        # Set cooldown
        cooldown_key = f"{cache_name}_{threshold.alert_type.value}_{threshold.metric_name}"
        self.alert_cooldowns[cooldown_key] = time.time()

        # Send notifications
        await self._send_notifications(alert)

        self.logger.warning(f"Alert triggered: {alert_id}")

    async def _resolve_alert(self, alert_id: str, current_value: float) -> None:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_timestamp = time.time()

        # Remove from active alerts
        del self.active_alerts[alert_id]
        self.stats["total_alerts_resolved"] += 1

        # Create resolution notification
        resolution_alert = Alert(
            alert_id=f"{alert_id}_resolved",
            alert_type=alert.alert_type,
            severity=AlertSeverity.LOW,
            cache_name=alert.cache_name,
            metric_name=alert.metric_name,
            current_value=current_value,
            threshold_value=alert.threshold_value,
            threshold_config=alert.threshold_config,
            additional_data={"resolution": True, "original_alert_id": alert_id},
        )

        await self._send_notifications(resolution_alert)

        self.logger.info(f"Alert resolved: {alert_id}")

    async def _check_escalations(self) -> None:
        """Check for alerts that need escalation."""
        for alert in self.active_alerts.values():
            if (
                alert.threshold_config.escalation_enabled
                and not alert.escalated
                and alert.age_seconds > alert.threshold_config.escalation_delay_seconds
            ):
                await self._escalate_alert(alert)

    async def _escalate_alert(self, alert: Alert) -> None:
        """Escalate an alert to higher severity."""
        alert.escalated = True
        alert.escalated_timestamp = time.time()
        original_severity = alert.severity
        alert.severity = alert.threshold_config.escalation_severity

        self.stats["total_alerts_escalated"] += 1

        # Create escalation notification
        escalation_alert = Alert(
            alert_id=f"{alert.alert_id}_escalated",
            alert_type=alert.alert_type,
            severity=alert.severity,
            cache_name=alert.cache_name,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            threshold_config=alert.threshold_config,
            additional_data={
                "escalation": True,
                "original_severity": original_severity.value,
                "escalated_after_seconds": alert.threshold_config.escalation_delay_seconds,
            },
        )

        await self._send_notifications(escalation_alert)

        self.logger.error(f"Alert escalated: {alert.alert_id}")

    async def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for channel in self.notification_channels:
            try:
                success = await channel.send_notification(alert)
                if success:
                    self.stats["total_notifications_sent"] += 1
                else:
                    self.stats["total_notifications_failed"] += 1
            except Exception as e:
                self.logger.error(f"Failed to send notification via {type(channel).__name__}: {e}")
                self.stats["total_notifications_failed"] += 1

    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up old resolved alerts from history."""
        if len(self.alert_history) > self.max_history_size:
            # Keep only the most recent alerts
            self.alert_history = self.alert_history[-self.max_history_size :]

        # Clean up old cooldowns
        current_time = time.time()
        expired_cooldowns = [key for key, timestamp in self.alert_cooldowns.items() if current_time - timestamp > 3600]  # 1 hour

        for key in expired_cooldowns:
            del self.alert_cooldowns[key]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledgment_timestamp = time.time()
            alert.acknowledged_by = acknowledged_by
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:]

    def get_alert_statistics(self) -> dict[str, Any]:
        """Get alert statistics."""
        active_count = len(self.active_alerts)
        severity_counts = {severity.value: 0 for severity in AlertSeverity}

        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        return {
            "active_alerts": active_count,
            "total_alerts_generated": self.stats["total_alerts_generated"],
            "total_alerts_resolved": self.stats["total_alerts_resolved"],
            "total_alerts_escalated": self.stats["total_alerts_escalated"],
            "total_notifications_sent": self.stats["total_notifications_sent"],
            "total_notifications_failed": self.stats["total_notifications_failed"],
            "monitoring_enabled": self.monitoring_enabled,
            "monitoring_uptime_seconds": time.time() - self.stats["monitoring_start_time"],
            "active_alerts_by_severity": severity_counts,
            "configured_thresholds": len(self.thresholds),
            "notification_channels": len(self.notification_channels),
        }

    def export_configuration(self) -> dict[str, Any]:
        """Export alert configuration."""
        return {
            "thresholds": [
                {
                    "alert_type": threshold.alert_type.value,
                    "severity": threshold.severity.value,
                    "threshold_value": threshold.threshold_value,
                    "comparison_operator": threshold.comparison_operator,
                    "metric_name": threshold.metric_name,
                    "enabled": threshold.enabled,
                    "cooldown_seconds": threshold.cooldown_seconds,
                    "description": threshold.description,
                    "action_required": threshold.action_required,
                    "escalation_enabled": threshold.escalation_enabled,
                    "escalation_delay_seconds": threshold.escalation_delay_seconds,
                    "escalation_severity": threshold.escalation_severity.value,
                }
                for threshold in self.thresholds.values()
            ],
            "monitoring_interval": self.monitoring_interval,
            "max_history_size": self.max_history_size,
        }

    def import_configuration(self, config: dict[str, Any]) -> None:
        """Import alert configuration."""
        # Clear existing thresholds
        self.thresholds.clear()

        # Import thresholds
        for threshold_data in config.get("thresholds", []):
            threshold = AlertThreshold(
                alert_type=AlertType(threshold_data["alert_type"]),
                severity=AlertSeverity(threshold_data["severity"]),
                threshold_value=threshold_data["threshold_value"],
                comparison_operator=threshold_data["comparison_operator"],
                metric_name=threshold_data["metric_name"],
                enabled=threshold_data.get("enabled", True),
                cooldown_seconds=threshold_data.get("cooldown_seconds", 300),
                description=threshold_data.get("description", ""),
                action_required=threshold_data.get("action_required", ""),
                escalation_enabled=threshold_data.get("escalation_enabled", False),
                escalation_delay_seconds=threshold_data.get("escalation_delay_seconds", 1800),
                escalation_severity=AlertSeverity(threshold_data.get("escalation_severity", "high")),
            )
            self.add_threshold(threshold)

        # Import other settings
        self.monitoring_interval = config.get("monitoring_interval", 30.0)
        self.max_history_size = config.get("max_history_size", 1000)

        self.logger.info("Imported alert configuration")


# Global alert service instance
_alert_service: CacheAlertService | None = None


def get_cache_alert_service() -> CacheAlertService:
    """Get the global cache alert service instance."""
    global _alert_service
    if _alert_service is None:
        _alert_service = CacheAlertService()
    return _alert_service


def initialize_alert_service(
    cache_monitor: CachePerformanceMonitor | None = None,
    logger: logging.Logger | None = None,
) -> CacheAlertService:
    """Initialize the global cache alert service."""
    global _alert_service
    _alert_service = CacheAlertService(cache_monitor, logger)
    return _alert_service
