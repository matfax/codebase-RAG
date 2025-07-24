"""
Cache alert management tool for MCP server.

This module provides tools to manage cache alerts, thresholds, and notifications.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional, Union

from src.services.cache_alert_service import (
    AlertSeverity,
    AlertThreshold,
    AlertType,
    CacheAlertService,
    EmailNotificationChannel,
    LogNotificationChannel,
    WebhookNotificationChannel,
    get_cache_alert_service,
)

logger = logging.getLogger(__name__)


async def cache_alert_status() -> dict[str, Any]:
    """
    Get current cache alert service status and statistics.

    Returns:
        Dict containing alert service status, active alerts, and statistics.
    """
    try:
        alert_service = get_cache_alert_service()

        # Get alert statistics
        stats = alert_service.get_alert_statistics()

        # Get active alerts
        active_alerts = alert_service.get_active_alerts()

        # Get alert history
        alert_history = alert_service.get_alert_history(limit=10)

        # Get threshold configuration
        thresholds = alert_service.export_configuration()

        return {
            "status": "ok",
            "message": "Cache alert service status retrieved successfully",
            "alert_service": {
                "monitoring_enabled": alert_service.monitoring_enabled,
                "monitoring_interval": alert_service.monitoring_interval,
                "notification_channels": len(alert_service.notification_channels),
                "configured_thresholds": len(alert_service.thresholds),
            },
            "statistics": stats,
            "active_alerts": {
                "count": len(active_alerts),
                "alerts": [alert.to_dict() for alert in active_alerts],
            },
            "recent_history": {
                "count": len(alert_history),
                "alerts": [alert.to_dict() for alert in alert_history],
            },
            "thresholds": thresholds,
            "timestamp": stats.get("monitoring_uptime_seconds", 0),
        }

    except Exception as e:
        logger.error(f"Failed to get cache alert status: {e}")
        return {
            "status": "error",
            "message": f"Failed to get cache alert status: {str(e)}",
            "error": str(e),
        }


async def configure_alert_threshold(
    alert_type: str,
    metric_name: str,
    threshold_value: float,
    comparison_operator: str = "gt",
    severity: str = "medium",
    enabled: bool = True,
    cooldown_seconds: int = 300,
    description: str = "",
    action_required: str = "",
    escalation_enabled: bool = False,
    escalation_delay_seconds: int = 1800,
    escalation_severity: str = "high",
) -> dict[str, Any]:
    """
    Configure or update an alert threshold.

    Args:
        alert_type: Type of alert (e.g., "high_error_rate", "low_hit_rate")
        metric_name: Name of the metric to monitor
        threshold_value: Threshold value to trigger the alert
        comparison_operator: Comparison operator ("gt", "lt", "eq", "gte", "lte")
        severity: Alert severity ("low", "medium", "high", "critical")
        enabled: Whether the threshold is enabled
        cooldown_seconds: Cooldown period between alerts
        description: Human-readable description of the alert
        action_required: Action required when alert is triggered
        escalation_enabled: Whether to escalate unresolved alerts
        escalation_delay_seconds: Delay before escalating
        escalation_severity: Severity to escalate to

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        # Validate alert type
        try:
            alert_type_enum = AlertType(alert_type)
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid alert type: {alert_type}",
                "valid_types": [t.value for t in AlertType],
            }

        # Validate severity
        try:
            severity_enum = AlertSeverity(severity)
            escalation_severity_enum = AlertSeverity(escalation_severity)
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid severity: {severity} or {escalation_severity}",
                "valid_severities": [s.value for s in AlertSeverity],
            }

        # Validate comparison operator
        valid_operators = ["gt", "lt", "eq", "gte", "lte"]
        if comparison_operator not in valid_operators:
            return {
                "status": "error",
                "message": f"Invalid comparison operator: {comparison_operator}",
                "valid_operators": valid_operators,
            }

        # Create threshold
        threshold = AlertThreshold(
            alert_type=alert_type_enum,
            severity=severity_enum,
            threshold_value=threshold_value,
            comparison_operator=comparison_operator,
            metric_name=metric_name,
            enabled=enabled,
            cooldown_seconds=cooldown_seconds,
            description=description or f"{alert_type} alert for {metric_name}",
            action_required=action_required or f"Investigate {alert_type} issue",
            escalation_enabled=escalation_enabled,
            escalation_delay_seconds=escalation_delay_seconds,
            escalation_severity=escalation_severity_enum,
        )

        # Add threshold to service
        alert_service.add_threshold(threshold)

        return {
            "status": "ok",
            "message": f"Alert threshold configured successfully for {alert_type}:{metric_name}",
            "threshold": {
                "alert_type": alert_type,
                "metric_name": metric_name,
                "threshold_value": threshold_value,
                "comparison_operator": comparison_operator,
                "severity": severity,
                "enabled": enabled,
                "cooldown_seconds": cooldown_seconds,
                "escalation_enabled": escalation_enabled,
            },
        }

    except Exception as e:
        logger.error(f"Failed to configure alert threshold: {e}")
        return {
            "status": "error",
            "message": f"Failed to configure alert threshold: {str(e)}",
            "error": str(e),
        }


async def remove_alert_threshold(alert_type: str, metric_name: str) -> dict[str, Any]:
    """
    Remove an alert threshold.

    Args:
        alert_type: Type of alert to remove
        metric_name: Metric name for the alert

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        # Validate alert type
        try:
            alert_type_enum = AlertType(alert_type)
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid alert type: {alert_type}",
                "valid_types": [t.value for t in AlertType],
            }

        # Check if threshold exists
        existing_threshold = alert_service.get_threshold(alert_type_enum, metric_name)
        if not existing_threshold:
            return {
                "status": "error",
                "message": f"No threshold found for {alert_type}:{metric_name}",
            }

        # Remove threshold
        alert_service.remove_threshold(alert_type_enum, metric_name)

        return {
            "status": "ok",
            "message": f"Alert threshold removed successfully for {alert_type}:{metric_name}",
        }

    except Exception as e:
        logger.error(f"Failed to remove alert threshold: {e}")
        return {
            "status": "error",
            "message": f"Failed to remove alert threshold: {str(e)}",
            "error": str(e),
        }


async def add_notification_channel(channel_type: str, **channel_config) -> dict[str, Any]:
    """
    Add a notification channel to the alert service.

    Args:
        channel_type: Type of notification channel ("log", "email", "webhook")
        **channel_config: Configuration parameters for the channel

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        if channel_type == "log":
            # Log notification channel
            channel = LogNotificationChannel(logger=logger if channel_config.get("use_service_logger") else None)

        elif channel_type == "email":
            # Email notification channel
            required_fields = ["smtp_server", "smtp_port", "username", "password", "from_email", "to_emails"]
            for field in required_fields:
                if field not in channel_config:
                    return {
                        "status": "error",
                        "message": f"Missing required field for email channel: {field}",
                        "required_fields": required_fields,
                    }

            channel = EmailNotificationChannel(
                smtp_server=channel_config["smtp_server"],
                smtp_port=channel_config["smtp_port"],
                username=channel_config["username"],
                password=channel_config["password"],
                from_email=channel_config["from_email"],
                to_emails=channel_config["to_emails"],
                use_tls=channel_config.get("use_tls", True),
            )

        elif channel_type == "webhook":
            # Webhook notification channel
            if "webhook_url" not in channel_config:
                return {
                    "status": "error",
                    "message": "Missing required field for webhook channel: webhook_url",
                    "required_fields": ["webhook_url"],
                }

            channel = WebhookNotificationChannel(
                webhook_url=channel_config["webhook_url"],
                headers=channel_config.get("headers", {}),
            )

        else:
            return {
                "status": "error",
                "message": f"Unknown notification channel type: {channel_type}",
                "valid_types": ["log", "email", "webhook"],
            }

        # Add channel to service
        alert_service.add_notification_channel(channel)

        return {
            "status": "ok",
            "message": f"Notification channel added successfully: {channel_type}",
            "channel_type": channel_type,
            "total_channels": len(alert_service.notification_channels),
        }

    except Exception as e:
        logger.error(f"Failed to add notification channel: {e}")
        return {
            "status": "error",
            "message": f"Failed to add notification channel: {str(e)}",
            "error": str(e),
        }


async def start_alert_monitoring() -> dict[str, Any]:
    """
    Start the cache alert monitoring system.

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        if alert_service.monitoring_enabled:
            return {
                "status": "ok",
                "message": "Cache alert monitoring is already running",
                "monitoring_enabled": True,
            }

        # Start monitoring
        alert_service.start_monitoring()

        # Give it a moment to start
        await asyncio.sleep(0.1)

        return {
            "status": "ok",
            "message": "Cache alert monitoring started successfully",
            "monitoring_enabled": alert_service.monitoring_enabled,
            "monitoring_interval": alert_service.monitoring_interval,
            "configured_thresholds": len(alert_service.thresholds),
            "notification_channels": len(alert_service.notification_channels),
        }

    except Exception as e:
        logger.error(f"Failed to start alert monitoring: {e}")
        return {
            "status": "error",
            "message": f"Failed to start alert monitoring: {str(e)}",
            "error": str(e),
        }


async def stop_alert_monitoring() -> dict[str, Any]:
    """
    Stop the cache alert monitoring system.

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        if not alert_service.monitoring_enabled:
            return {
                "status": "ok",
                "message": "Cache alert monitoring is not running",
                "monitoring_enabled": False,
            }

        # Stop monitoring
        alert_service.stop_monitoring()

        return {
            "status": "ok",
            "message": "Cache alert monitoring stopped successfully",
            "monitoring_enabled": alert_service.monitoring_enabled,
        }

    except Exception as e:
        logger.error(f"Failed to stop alert monitoring: {e}")
        return {
            "status": "error",
            "message": f"Failed to stop alert monitoring: {str(e)}",
            "error": str(e),
        }


async def acknowledge_alert(alert_id: str, acknowledged_by: str) -> dict[str, Any]:
    """
    Acknowledge an active alert.

    Args:
        alert_id: ID of the alert to acknowledge
        acknowledged_by: Name/ID of the person acknowledging the alert

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        success = alert_service.acknowledge_alert(alert_id, acknowledged_by)

        if success:
            return {
                "status": "ok",
                "message": f"Alert acknowledged successfully: {alert_id}",
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by,
            }
        else:
            return {
                "status": "error",
                "message": f"Alert not found or already acknowledged: {alert_id}",
                "alert_id": alert_id,
            }

    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        return {
            "status": "error",
            "message": f"Failed to acknowledge alert: {str(e)}",
            "error": str(e),
        }


async def get_alert_history(limit: int = 50) -> dict[str, Any]:
    """
    Get alert history.

    Args:
        limit: Maximum number of historical alerts to retrieve

    Returns:
        Dict containing alert history.
    """
    try:
        alert_service = get_cache_alert_service()

        history = alert_service.get_alert_history(limit=limit)

        return {
            "status": "ok",
            "message": f"Retrieved {len(history)} historical alerts",
            "history": [alert.to_dict() for alert in history],
            "total_alerts": len(history),
            "limit": limit,
        }

    except Exception as e:
        logger.error(f"Failed to get alert history: {e}")
        return {
            "status": "error",
            "message": f"Failed to get alert history: {str(e)}",
            "error": str(e),
        }


async def export_alert_configuration() -> dict[str, Any]:
    """
    Export the complete alert configuration.

    Returns:
        Dict containing the complete alert configuration.
    """
    try:
        alert_service = get_cache_alert_service()

        config = alert_service.export_configuration()

        return {
            "status": "ok",
            "message": "Alert configuration exported successfully",
            "configuration": config,
            "export_timestamp": alert_service.stats.get("monitoring_start_time", 0),
        }

    except Exception as e:
        logger.error(f"Failed to export alert configuration: {e}")
        return {
            "status": "error",
            "message": f"Failed to export alert configuration: {str(e)}",
            "error": str(e),
        }


async def import_alert_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    """
    Import alert configuration.

    Args:
        configuration: Alert configuration to import

    Returns:
        Dict containing operation result.
    """
    try:
        alert_service = get_cache_alert_service()

        # Validate configuration structure
        if "thresholds" not in configuration:
            return {
                "status": "error",
                "message": "Invalid configuration: missing 'thresholds' field",
            }

        # Import configuration
        alert_service.import_configuration(configuration)

        return {
            "status": "ok",
            "message": "Alert configuration imported successfully",
            "imported_thresholds": len(configuration.get("thresholds", [])),
            "monitoring_interval": configuration.get("monitoring_interval", 30.0),
        }

    except Exception as e:
        logger.error(f"Failed to import alert configuration: {e}")
        return {
            "status": "error",
            "message": f"Failed to import alert configuration: {str(e)}",
            "error": str(e),
        }


async def test_notification_channels() -> dict[str, Any]:
    """
    Test all configured notification channels.

    Returns:
        Dict containing test results.
    """
    try:
        alert_service = get_cache_alert_service()

        if not alert_service.notification_channels:
            return {
                "status": "warning",
                "message": "No notification channels configured",
                "channels_tested": 0,
            }

        # Create a test alert
        from src.services.cache_alert_service import Alert, AlertSeverity, AlertThreshold, AlertType

        test_threshold = AlertThreshold(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.LOW,
            threshold_value=0.0,
            comparison_operator="gt",
            metric_name="test_metric",
            description="Test alert for notification channel validation",
            action_required="This is a test alert - no action required",
        )

        test_alert = Alert(
            alert_id="test_alert_" + str(int(time.time())),
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.LOW,
            cache_name="test_cache",
            metric_name="test_metric",
            current_value=1.0,
            threshold_value=0.0,
            threshold_config=test_threshold,
            additional_data={"test": True},
        )

        # Test each channel
        results = []
        for i, channel in enumerate(alert_service.notification_channels):
            try:
                success = await channel.send_notification(test_alert)
                results.append(
                    {
                        "channel_index": i,
                        "channel_type": type(channel).__name__,
                        "success": success,
                        "error": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "channel_index": i,
                        "channel_type": type(channel).__name__,
                        "success": False,
                        "error": str(e),
                    }
                )

        successful_tests = sum(1 for r in results if r["success"])

        return {
            "status": "ok" if successful_tests == len(results) else "warning",
            "message": f"Tested {len(results)} notification channels, {successful_tests} successful",
            "channels_tested": len(results),
            "successful_tests": successful_tests,
            "failed_tests": len(results) - successful_tests,
            "test_results": results,
        }

    except Exception as e:
        logger.error(f"Failed to test notification channels: {e}")
        return {
            "status": "error",
            "message": f"Failed to test notification channels: {str(e)}",
            "error": str(e),
        }


# Example usage and initialization
async def initialize_default_alert_system() -> dict[str, Any]:
    """
    Initialize the alert system with default configuration.

    Returns:
        Dict containing initialization result.
    """
    try:
        # Add default log notification channel
        await add_notification_channel("log", use_service_logger=True)

        # Configure default thresholds (these are already set up in the service)
        default_thresholds = [
            {
                "alert_type": "high_error_rate",
                "metric_name": "error_rate",
                "threshold_value": 0.05,
                "comparison_operator": "gt",
                "severity": "high",
                "description": "Cache error rate exceeds 5%",
            },
            {
                "alert_type": "low_hit_rate",
                "metric_name": "hit_rate",
                "threshold_value": 0.7,
                "comparison_operator": "lt",
                "severity": "medium",
                "description": "Cache hit rate below 70%",
            },
            {
                "alert_type": "high_memory_usage",
                "metric_name": "current_size_mb",
                "threshold_value": 500,
                "comparison_operator": "gt",
                "severity": "high",
                "description": "Cache memory usage exceeds 500MB",
            },
        ]

        # Start monitoring
        await start_alert_monitoring()

        return {
            "status": "ok",
            "message": "Default alert system initialized successfully",
            "default_thresholds": len(default_thresholds),
            "monitoring_started": True,
        }

    except Exception as e:
        logger.error(f"Failed to initialize default alert system: {e}")
        return {
            "status": "error",
            "message": f"Failed to initialize default alert system: {str(e)}",
            "error": str(e),
        }
