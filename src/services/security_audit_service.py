"""
Security audit service for cache access monitoring and compliance.

This service provides comprehensive security auditing for cache operations,
including access logging, compliance reporting, and security analytics.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from ..utils.secure_cache_utils import SecurityAuditLog, SecurityContext, SecurityLevel

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    CACHE_ACCESS = "cache_access"
    CACHE_WRITE = "cache_write"
    CACHE_DELETE = "cache_delete"
    CACHE_CLEAR = "cache_clear"
    ACCESS_DENIED = "access_denied"
    ENCRYPTION_ERROR = "encryption_error"
    KEY_ROTATION = "key_rotation"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Comprehensive audit event structure."""

    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: float
    project_id: str
    user_id: str | None = None
    session_id: str | None = None
    operation: str = ""
    resource: str = ""
    client_ip: str | None = None
    user_agent: str | None = None
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        if self.security_level:
            data["security_level"] = self.security_level.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Create from dictionary."""
        event_type = AuditEventType(data["event_type"])
        severity = AuditSeverity(data["severity"])
        security_level = SecurityLevel(data["security_level"]) if data.get("security_level") else None

        return cls(
            event_type=event_type,
            severity=severity,
            timestamp=data["timestamp"],
            project_id=data["project_id"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            operation=data.get("operation", ""),
            resource=data.get("resource", ""),
            client_ip=data.get("client_ip"),
            user_agent=data.get("user_agent"),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            security_level=security_level,
        )


@dataclass
class AuditStatistics:
    """Audit statistics for reporting."""

    total_events: int = 0
    events_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events_by_severity: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events_by_project: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events_by_user: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_operations: int = 0
    security_violations: int = 0
    access_denials: int = 0
    time_period_start: float = field(default_factory=time.time)
    time_period_end: float = field(default_factory=time.time)

    def add_event(self, event: AuditEvent) -> None:
        """Add an event to statistics."""
        self.total_events += 1
        self.events_by_type[event.event_type.value] += 1
        self.events_by_severity[event.severity.value] += 1
        self.events_by_project[event.project_id] += 1

        if event.user_id:
            self.events_by_user[event.user_id] += 1

        if not event.success:
            self.failed_operations += 1

        if event.event_type == AuditEventType.ACCESS_DENIED:
            self.access_denials += 1

        if event.event_type == AuditEventType.SECURITY_VIOLATION:
            self.security_violations += 1

        self.time_period_end = max(self.time_period_end, event.timestamp)
        self.time_period_start = min(self.time_period_start, event.timestamp)


class SecurityAuditService:
    """
    Comprehensive security audit service for cache operations.

    Provides detailed audit logging, compliance reporting, security analytics,
    and real-time monitoring of cache access patterns.
    """

    def __init__(
        self,
        audit_log_path: str | None = None,
        max_events_in_memory: int = 10000,
        enable_file_logging: bool = True,
        enable_real_time_alerts: bool = True,
        retention_days: int = 90,
    ):
        """
        Initialize security audit service.

        Args:
            audit_log_path: Path to store audit log files
            max_events_in_memory: Maximum events to keep in memory
            enable_file_logging: Whether to log to files
            enable_real_time_alerts: Whether to enable real-time alerts
            retention_days: Days to retain audit logs
        """
        self.audit_log_path = audit_log_path or os.getenv("AUDIT_LOG_PATH", "./audit_logs")
        self.max_events_in_memory = max_events_in_memory
        self.enable_file_logging = enable_file_logging
        self.enable_real_time_alerts = enable_real_time_alerts
        self.retention_days = retention_days

        # In-memory event storage
        self.events: list[AuditEvent] = []
        self.statistics = AuditStatistics()
        self._lock = threading.RLock()

        # Alert thresholds
        self.alert_thresholds = {
            "failed_operations_per_minute": 10,
            "access_denials_per_minute": 5,
            "security_violations_per_hour": 3,
            "suspicious_patterns_threshold": 0.8,
        }

        # Tracking for pattern detection
        self._recent_events = defaultdict(list)  # Track recent events by user/project
        self._alert_cooldowns = {}  # Prevent alert spam

        # Ensure log directory exists
        if self.enable_file_logging:
            Path(self.audit_log_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Security audit service initialized with log path: {self.audit_log_path}")

    def log_cache_access(
        self,
        operation: str,
        resource: str,
        security_context: SecurityContext,
        success: bool = True,
        error_message: str | None = None,
        client_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a cache access event.

        Args:
            operation: Operation performed (get, set, delete, etc.)
            resource: Resource accessed (cache key)
            security_context: Security context of the operation
            success: Whether operation succeeded
            error_message: Error message if failed
            client_info: Client information (IP, user agent, etc.)
        """
        severity = AuditSeverity.LOW
        event_type = AuditEventType.CACHE_ACCESS

        # Determine event type and severity
        if operation in ["get", "exists"]:
            event_type = AuditEventType.CACHE_ACCESS
        elif operation in ["set", "update"]:
            event_type = AuditEventType.CACHE_WRITE
        elif operation in ["delete", "remove"]:
            event_type = AuditEventType.CACHE_DELETE
        elif operation == "clear":
            event_type = AuditEventType.CACHE_CLEAR
            severity = AuditSeverity.MEDIUM

        if not success:
            severity = AuditSeverity.MEDIUM
            if error_message and ("permission" in error_message.lower() or "access" in error_message.lower()):
                event_type = AuditEventType.ACCESS_DENIED
                severity = AuditSeverity.HIGH

        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            project_id=security_context.project_id,
            user_id=security_context.user_id,
            session_id=security_context.session_id,
            operation=operation,
            resource=resource,
            success=success,
            error_message=error_message,
            security_level=security_context.security_level,
            client_ip=client_info.get("ip") if client_info else None,
            user_agent=client_info.get("user_agent") if client_info else None,
            metadata=client_info or {},
        )

        self._record_event(event)

    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        security_context: SecurityContext,
        severity: AuditSeverity = AuditSeverity.HIGH,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a security violation event.

        Args:
            violation_type: Type of violation
            description: Description of the violation
            security_context: Security context
            severity: Severity level
            metadata: Additional metadata
        """
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=severity,
            timestamp=time.time(),
            project_id=security_context.project_id,
            user_id=security_context.user_id,
            session_id=security_context.session_id,
            operation=violation_type,
            resource=description,
            success=False,
            security_level=security_context.security_level,
            metadata=metadata or {},
        )

        self._record_event(event)

    def log_encryption_error(
        self, operation: str, error_details: str, security_context: SecurityContext, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Log an encryption-related error.

        Args:
            operation: Operation that failed
            error_details: Error details
            security_context: Security context
            metadata: Additional metadata
        """
        event = AuditEvent(
            event_type=AuditEventType.ENCRYPTION_ERROR,
            severity=AuditSeverity.HIGH,
            timestamp=time.time(),
            project_id=security_context.project_id,
            user_id=security_context.user_id,
            session_id=security_context.session_id,
            operation=operation,
            resource=error_details,
            success=False,
            error_message=error_details,
            security_level=security_context.security_level,
            metadata=metadata or {},
        )

        self._record_event(event)

    def log_key_rotation(self, old_key_id: str, new_key_id: str, project_id: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Log a key rotation event.

        Args:
            old_key_id: Previous key ID
            new_key_id: New key ID
            project_id: Project ID
            metadata: Additional metadata
        """
        event = AuditEvent(
            event_type=AuditEventType.KEY_ROTATION,
            severity=AuditSeverity.MEDIUM,
            timestamp=time.time(),
            project_id=project_id,
            operation="key_rotation",
            resource=f"old:{old_key_id} -> new:{new_key_id}",
            success=True,
            metadata=metadata or {},
        )

        self._record_event(event)

    def _record_event(self, event: AuditEvent) -> None:
        """
        Record an audit event.

        Args:
            event: Audit event to record
        """
        with self._lock:
            # Add to in-memory storage
            self.events.append(event)
            self.statistics.add_event(event)

            # Maintain memory limit
            if len(self.events) > self.max_events_in_memory:
                # Remove oldest events
                self.events = self.events[-self.max_events_in_memory // 2 :]

            # File logging
            if self.enable_file_logging:
                self._write_to_file(event)

            # Real-time analysis
            if self.enable_real_time_alerts:
                self._analyze_for_alerts(event)

            # Pattern detection
            self._update_pattern_tracking(event)

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to audit log file."""
        try:
            # Create daily log files
            date_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d")
            log_file = Path(self.audit_log_path) / f"audit_{date_str}.jsonl"

            with open(log_file, "a") as f:
                json.dump(event.to_dict(), f)
                f.write("\n")

        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")

    def _analyze_for_alerts(self, event: AuditEvent) -> None:
        """Analyze event for real-time alerts."""
        now = time.time()

        # Check for high failure rates
        if not event.success:
            recent_failures = [
                e for e in self.events[-100:] if not e.success and (now - e.timestamp) < 60  # Last 100 events  # Last minute
            ]

            if len(recent_failures) >= self.alert_thresholds["failed_operations_per_minute"]:
                self._trigger_alert(
                    "high_failure_rate",
                    f"High failure rate detected: {len(recent_failures)} failures in last minute",
                    AuditSeverity.HIGH,
                    {"failure_count": len(recent_failures), "time_window": 60},
                )

        # Check for access denial patterns
        if event.event_type == AuditEventType.ACCESS_DENIED:
            recent_denials = [e for e in self.events[-50:] if e.event_type == AuditEventType.ACCESS_DENIED and (now - e.timestamp) < 60]

            if len(recent_denials) >= self.alert_thresholds["access_denials_per_minute"]:
                self._trigger_alert(
                    "high_access_denial_rate",
                    f"High access denial rate: {len(recent_denials)} denials in last minute",
                    AuditSeverity.HIGH,
                    {"denial_count": len(recent_denials), "time_window": 60},
                )

        # Check for security violations
        if event.event_type == AuditEventType.SECURITY_VIOLATION:
            recent_violations = [
                e for e in self.events[-100:] if e.event_type == AuditEventType.SECURITY_VIOLATION and (now - e.timestamp) < 3600
            ]

            if len(recent_violations) >= self.alert_thresholds["security_violations_per_hour"]:
                self._trigger_alert(
                    "multiple_security_violations",
                    f"Multiple security violations: {len(recent_violations)} in last hour",
                    AuditSeverity.CRITICAL,
                    {"violation_count": len(recent_violations), "time_window": 3600},
                )

    def _update_pattern_tracking(self, event: AuditEvent) -> None:
        """Update pattern tracking for anomaly detection."""
        # Track events by user and project
        tracking_key = f"{event.project_id}:{event.user_id or 'anonymous'}"

        self._recent_events[tracking_key].append(event)

        # Keep only recent events (last hour)
        cutoff_time = time.time() - 3600
        self._recent_events[tracking_key] = [e for e in self._recent_events[tracking_key] if e.timestamp > cutoff_time]

        # Analyze patterns
        user_events = self._recent_events[tracking_key]
        if len(user_events) > 20:  # Enough data for analysis
            self._analyze_user_patterns(tracking_key, user_events)

    def _analyze_user_patterns(self, tracking_key: str, events: list[AuditEvent]) -> None:
        """Analyze user behavior patterns for anomalies."""
        # Calculate failure rate
        failures = sum(1 for e in events if not e.success)
        failure_rate = failures / len(events)

        # Check for unusual activity patterns
        if failure_rate > self.alert_thresholds["suspicious_patterns_threshold"]:
            self._trigger_alert(
                "suspicious_user_pattern",
                f"Suspicious activity pattern for {tracking_key}: {failure_rate:.2%} failure rate",
                AuditSeverity.MEDIUM,
                {"tracking_key": tracking_key, "failure_rate": failure_rate, "event_count": len(events), "time_window": 3600},
            )

    def _trigger_alert(self, alert_type: str, message: str, severity: AuditSeverity, metadata: dict[str, Any]) -> None:
        """Trigger a security alert."""
        # Implement alert cooldown to prevent spam
        cooldown_key = f"{alert_type}:{metadata.get('tracking_key', 'global')}"
        now = time.time()

        if cooldown_key in self._alert_cooldowns:
            if now - self._alert_cooldowns[cooldown_key] < 300:  # 5 minute cooldown
                return

        self._alert_cooldowns[cooldown_key] = now

        # Log the alert
        logger.warning(f"SECURITY ALERT [{severity.value.upper()}]: {alert_type} - {message}")

        # Create system audit event for the alert
        alert_event = AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            severity=severity,
            timestamp=now,
            project_id=metadata.get("project_id", "system"),
            operation="security_alert",
            resource=alert_type,
            success=True,
            metadata={"alert_type": alert_type, "message": message, **metadata},
        )

        # Record the alert event (without triggering more alerts)
        with self._lock:
            self.events.append(alert_event)
            self.statistics.add_event(alert_event)

    def get_events(
        self,
        project_id: str | None = None,
        user_id: str | None = None,
        event_type: AuditEventType | None = None,
        severity: AuditSeverity | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
        include_successful: bool = True,
        include_failed: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get audit events with filtering.

        Args:
            project_id: Filter by project ID
            user_id: Filter by user ID
            event_type: Filter by event type
            severity: Filter by severity
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of events
            include_successful: Include successful operations
            include_failed: Include failed operations

        Returns:
            List[Dict[str, Any]]: Filtered audit events
        """
        with self._lock:
            filtered_events = []

            for event in reversed(self.events):  # Most recent first
                # Apply filters
                if project_id and event.project_id != project_id:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if severity and event.severity != severity:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if not include_successful and event.success:
                    continue
                if not include_failed and not event.success:
                    continue

                filtered_events.append(event.to_dict())

                if len(filtered_events) >= limit:
                    break

            return filtered_events

    def get_statistics(self, project_id: str | None = None, time_window_hours: int = 24) -> dict[str, Any]:
        """
        Get audit statistics.

        Args:
            project_id: Filter by project ID
            time_window_hours: Time window in hours

        Returns:
            Dict[str, Any]: Statistics
        """
        with self._lock:
            cutoff_time = time.time() - (time_window_hours * 3600)

            # Filter events
            filtered_events = [
                event for event in self.events if event.timestamp > cutoff_time and (not project_id or event.project_id == project_id)
            ]

            # Calculate statistics
            stats = AuditStatistics()
            for event in filtered_events:
                stats.add_event(event)

            return {
                "total_events": stats.total_events,
                "events_by_type": dict(stats.events_by_type),
                "events_by_severity": dict(stats.events_by_severity),
                "events_by_project": dict(stats.events_by_project),
                "events_by_user": dict(stats.events_by_user),
                "failed_operations": stats.failed_operations,
                "security_violations": stats.security_violations,
                "access_denials": stats.access_denials,
                "success_rate": (stats.total_events - stats.failed_operations) / stats.total_events if stats.total_events > 0 else 1.0,
                "time_window_hours": time_window_hours,
                "time_period_start": stats.time_period_start,
                "time_period_end": stats.time_period_end,
            }

    def generate_compliance_report(self, project_id: str, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """
        Generate a compliance report for audit purposes.

        Args:
            project_id: Project ID to report on
            start_date: Start date for report
            end_date: End date for report

        Returns:
            Dict[str, Any]: Compliance report
        """
        start_time = start_date.timestamp()
        end_time = end_date.timestamp()

        events = self.get_events(
            project_id=project_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,  # High limit for comprehensive report
        )

        # Analyze events for compliance metrics
        total_events = len(events)
        failed_events = sum(1 for e in events if not e["success"])
        access_denials = sum(1 for e in events if e["event_type"] == AuditEventType.ACCESS_DENIED.value)
        security_violations = sum(1 for e in events if e["event_type"] == AuditEventType.SECURITY_VIOLATION.value)

        # Group by security levels
        security_level_stats = defaultdict(int)
        for event in events:
            if event.get("security_level"):
                security_level_stats[event["security_level"]] += 1

        # User activity analysis
        user_activity = defaultdict(int)
        for event in events:
            if event.get("user_id"):
                user_activity[event["user_id"]] += 1

        return {
            "report_metadata": {
                "project_id": project_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "generated_at": datetime.now().isoformat(),
                "total_events": total_events,
            },
            "security_metrics": {
                "total_events": total_events,
                "failed_events": failed_events,
                "success_rate": (total_events - failed_events) / total_events if total_events > 0 else 1.0,
                "access_denials": access_denials,
                "security_violations": security_violations,
                "security_level_breakdown": dict(security_level_stats),
            },
            "user_activity": dict(user_activity),
            "compliance_status": {
                "audit_coverage": "complete" if total_events > 0 else "no_activity",
                "security_incidents": security_violations + access_denials,
                "risk_level": self._calculate_risk_level(failed_events, total_events, security_violations),
            },
            "recommendations": self._generate_recommendations(events),
        }

    def _calculate_risk_level(self, failed_events: int, total_events: int, security_violations: int) -> str:
        """Calculate risk level based on metrics."""
        if security_violations > 0:
            return "high"

        if total_events == 0:
            return "unknown"

        failure_rate = failed_events / total_events

        if failure_rate > 0.1:  # More than 10% failure rate
            return "medium"
        elif failure_rate > 0.05:  # More than 5% failure rate
            return "low"
        else:
            return "minimal"

    def _generate_recommendations(self, events: list[dict[str, Any]]) -> list[str]:
        """Generate security recommendations based on audit data."""
        recommendations = []

        if not events:
            recommendations.append("Enable audit logging and monitor cache access patterns")
            return recommendations

        # Analyze failure patterns
        failed_events = [e for e in events if not e["success"]]
        failure_rate = len(failed_events) / len(events)

        if failure_rate > 0.1:
            recommendations.append("High failure rate detected - review error patterns and system stability")

        # Check for security issues
        access_denials = sum(1 for e in events if e["event_type"] == AuditEventType.ACCESS_DENIED.value)
        if access_denials > 0:
            recommendations.append("Access denials detected - review user permissions and access patterns")

        security_violations = sum(1 for e in events if e["event_type"] == AuditEventType.SECURITY_VIOLATION.value)
        if security_violations > 0:
            recommendations.append("Security violations detected - immediate review and remediation required")

        # Check encryption usage
        encrypted_events = sum(1 for e in events if e.get("security_level") in ["sensitive", "confidential"])
        if encrypted_events == 0 and len(events) > 0:
            recommendations.append("Consider using encryption for sensitive cache data")

        return recommendations

    def cleanup_old_logs(self) -> int:
        """Clean up old audit logs based on retention policy."""
        if not self.enable_file_logging:
            return 0

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        cleaned_count = 0
        log_dir = Path(self.audit_log_path)

        try:
            for log_file in log_dir.glob("audit_*.jsonl"):
                # Extract date from filename
                try:
                    date_str = log_file.stem.replace("audit_", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")

                    if file_date < cutoff_date:
                        log_file.unlink()
                        cleaned_count += 1

                except (ValueError, OSError) as e:
                    logger.warning(f"Failed to process log file {log_file}: {e}")

            logger.info(f"Cleaned up {cleaned_count} old audit log files")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")
            return 0


# Global security audit service instance
_security_audit_service: SecurityAuditService | None = None


def get_security_audit_service() -> SecurityAuditService:
    """Get the global security audit service instance."""
    global _security_audit_service
    if _security_audit_service is None:
        _security_audit_service = SecurityAuditService()
    return _security_audit_service


# Convenience functions for common audit operations
def audit_cache_access(
    operation: str,
    resource: str,
    security_context: SecurityContext,
    success: bool = True,
    error_message: str | None = None,
    client_info: dict[str, Any] | None = None,
) -> None:
    """Log a cache access audit event."""
    service = get_security_audit_service()
    service.log_cache_access(operation, resource, security_context, success, error_message, client_info)


def audit_security_violation(
    violation_type: str,
    description: str,
    security_context: SecurityContext,
    severity: AuditSeverity = AuditSeverity.HIGH,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a security violation audit event."""
    service = get_security_audit_service()
    service.log_security_violation(violation_type, description, security_context, severity, metadata)


def audit_encryption_error(
    operation: str, error_details: str, security_context: SecurityContext, metadata: dict[str, Any] | None = None
) -> None:
    """Log an encryption error audit event."""
    service = get_security_audit_service()
    service.log_encryption_error(operation, error_details, security_context, metadata)


def audit_key_rotation(old_key_id: str, new_key_id: str, project_id: str, metadata: dict[str, Any] | None = None) -> None:
    """Log a key rotation audit event."""
    service = get_security_audit_service()
    service.log_key_rotation(old_key_id, new_key_id, project_id, metadata)


# Export public interface
__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "AuditEvent",
    "AuditStatistics",
    "SecurityAuditService",
    "get_security_audit_service",
    "audit_cache_access",
    "audit_security_violation",
    "audit_encryption_error",
    "audit_key_rotation",
]
