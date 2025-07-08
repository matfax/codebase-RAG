"""
Unit tests for security audit service.
"""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ..utils.secure_cache_utils import SecurityContext, SecurityLevel
from .security_audit_service import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditStatistics,
    SecurityAuditService,
    audit_cache_access,
    audit_encryption_error,
    audit_key_rotation,
    audit_security_violation,
    get_security_audit_service,
)


class TestAuditEvent:
    """Test AuditEvent class."""

    def test_audit_event_creation(self):
        """Test audit event creation."""
        event = AuditEvent(
            event_type=AuditEventType.CACHE_ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=time.time(),
            project_id="test_project",
            user_id="test_user",
            operation="get",
            resource="test_key",
            success=True,
        )

        assert event.event_type == AuditEventType.CACHE_ACCESS
        assert event.severity == AuditSeverity.LOW
        assert event.project_id == "test_project"
        assert event.user_id == "test_user"
        assert event.operation == "get"
        assert event.resource == "test_key"
        assert event.success is True

    def test_audit_event_to_dict(self):
        """Test audit event serialization."""
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.HIGH,
            timestamp=1234567890.0,
            project_id="test_project",
            user_id="test_user",
            operation="unauthorized_access",
            resource="sensitive_data",
            success=False,
            error_message="Access denied",
            security_level=SecurityLevel.SENSITIVE,
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "security_violation"
        assert event_dict["severity"] == "high"
        assert event_dict["timestamp"] == 1234567890.0
        assert event_dict["project_id"] == "test_project"
        assert event_dict["user_id"] == "test_user"
        assert event_dict["operation"] == "unauthorized_access"
        assert event_dict["resource"] == "sensitive_data"
        assert event_dict["success"] is False
        assert event_dict["error_message"] == "Access denied"
        assert event_dict["security_level"] == "sensitive"

    def test_audit_event_from_dict(self):
        """Test audit event deserialization."""
        event_dict = {
            "event_type": "cache_access",
            "severity": "low",
            "timestamp": 1234567890.0,
            "project_id": "test_project",
            "user_id": "test_user",
            "operation": "get",
            "resource": "test_key",
            "success": True,
            "security_level": "internal",
        }

        event = AuditEvent.from_dict(event_dict)

        assert event.event_type == AuditEventType.CACHE_ACCESS
        assert event.severity == AuditSeverity.LOW
        assert event.timestamp == 1234567890.0
        assert event.project_id == "test_project"
        assert event.user_id == "test_user"
        assert event.operation == "get"
        assert event.resource == "test_key"
        assert event.success is True
        assert event.security_level == SecurityLevel.INTERNAL


class TestAuditStatistics:
    """Test AuditStatistics class."""

    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = AuditStatistics()

        assert stats.total_events == 0
        assert stats.failed_operations == 0
        assert stats.security_violations == 0
        assert stats.access_denials == 0
        assert len(stats.events_by_type) == 0
        assert len(stats.events_by_severity) == 0

    def test_add_event_to_statistics(self):
        """Test adding events to statistics."""
        stats = AuditStatistics()

        # Add successful event
        event1 = AuditEvent(
            event_type=AuditEventType.CACHE_ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=time.time(),
            project_id="test_project",
            user_id="user1",
            success=True,
        )
        stats.add_event(event1)

        assert stats.total_events == 1
        assert stats.failed_operations == 0
        assert stats.events_by_type["cache_access"] == 1
        assert stats.events_by_severity["low"] == 1
        assert stats.events_by_project["test_project"] == 1
        assert stats.events_by_user["user1"] == 1

        # Add failed event
        event2 = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            severity=AuditSeverity.HIGH,
            timestamp=time.time(),
            project_id="test_project",
            user_id="user2",
            success=False,
        )
        stats.add_event(event2)

        assert stats.total_events == 2
        assert stats.failed_operations == 1
        assert stats.access_denials == 1
        assert stats.events_by_type["access_denied"] == 1
        assert stats.events_by_severity["high"] == 1
        assert stats.events_by_user["user2"] == 1

        # Add security violation
        event3 = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            timestamp=time.time(),
            project_id="test_project",
            success=False,
        )
        stats.add_event(event3)

        assert stats.total_events == 3
        assert stats.failed_operations == 2
        assert stats.security_violations == 1
        assert stats.events_by_type["security_violation"] == 1
        assert stats.events_by_severity["critical"] == 1


class TestSecurityAuditService:
    """Test SecurityAuditService class."""

    def test_service_initialization(self):
        """Test service initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SecurityAuditService(
                audit_log_path=temp_dir, max_events_in_memory=1000, enable_file_logging=True, enable_real_time_alerts=True
            )

            assert service.audit_log_path == temp_dir
            assert service.max_events_in_memory == 1000
            assert service.enable_file_logging is True
            assert service.enable_real_time_alerts is True
            assert len(service.events) == 0

    def test_log_cache_access(self):
        """Test logging cache access events."""
        service = SecurityAuditService(enable_file_logging=False)

        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.INTERNAL)

        # Log successful access
        service.log_cache_access("get", "test_key", context, success=True)

        assert len(service.events) == 1
        event = service.events[0]
        assert event.event_type == AuditEventType.CACHE_ACCESS
        assert event.severity == AuditSeverity.LOW
        assert event.operation == "get"
        assert event.resource == "test_key"
        assert event.success is True
        assert event.project_id == "test_project"
        assert event.user_id == "test_user"

        # Log failed access
        service.log_cache_access("get", "sensitive_key", context, success=False, error_message="Access denied")

        assert len(service.events) == 2
        event = service.events[1]
        assert event.event_type == AuditEventType.ACCESS_DENIED
        assert event.severity == AuditSeverity.HIGH
        assert event.success is False
        assert event.error_message == "Access denied"

    def test_log_security_violation(self):
        """Test logging security violations."""
        service = SecurityAuditService(enable_file_logging=False)

        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        service.log_security_violation(
            "unauthorized_cross_project_access",
            "User attempted to access data from different project",
            context,
            severity=AuditSeverity.CRITICAL,
            metadata={"attempted_project": "other_project"},
        )

        assert len(service.events) == 1
        event = service.events[0]
        assert event.event_type == AuditEventType.SECURITY_VIOLATION
        assert event.severity == AuditSeverity.CRITICAL
        assert event.operation == "unauthorized_cross_project_access"
        assert event.success is False
        assert event.metadata["attempted_project"] == "other_project"

    def test_log_encryption_error(self):
        """Test logging encryption errors."""
        service = SecurityAuditService(enable_file_logging=False)

        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.CONFIDENTIAL)

        service.log_encryption_error("decrypt", "Failed to decrypt cache entry: invalid key", context, metadata={"key_id": "test_key_123"})

        assert len(service.events) == 1
        event = service.events[0]
        assert event.event_type == AuditEventType.ENCRYPTION_ERROR
        assert event.severity == AuditSeverity.HIGH
        assert event.operation == "decrypt"
        assert event.success is False
        assert event.error_message == "Failed to decrypt cache entry: invalid key"
        assert event.metadata["key_id"] == "test_key_123"

    def test_log_key_rotation(self):
        """Test logging key rotation events."""
        service = SecurityAuditService(enable_file_logging=False)

        service.log_key_rotation("old_key_123", "new_key_456", "test_project", metadata={"rotation_reason": "scheduled"})

        assert len(service.events) == 1
        event = service.events[0]
        assert event.event_type == AuditEventType.KEY_ROTATION
        assert event.severity == AuditSeverity.MEDIUM
        assert event.operation == "key_rotation"
        assert event.resource == "old:old_key_123 -> new:new_key_456"
        assert event.success is True
        assert event.metadata["rotation_reason"] == "scheduled"

    def test_file_logging(self):
        """Test file logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = SecurityAuditService(audit_log_path=temp_dir, enable_file_logging=True)

            context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

            # Log an event
            service.log_cache_access("get", "test_key", context)

            # Check that log file was created
            log_files = list(Path(temp_dir).glob("audit_*.jsonl"))
            assert len(log_files) == 1

            # Read and verify log content
            with open(log_files[0]) as f:
                log_line = f.readline().strip()
                event_data = json.loads(log_line)

                assert event_data["event_type"] == "cache_access"
                assert event_data["operation"] == "get"
                assert event_data["resource"] == "test_key"
                assert event_data["project_id"] == "test_project"

    def test_event_filtering(self):
        """Test event filtering functionality."""
        service = SecurityAuditService(enable_file_logging=False)

        # Create different contexts
        context1 = SecurityContext(project_id="project1", user_id="user1", security_level=SecurityLevel.PUBLIC)
        context2 = SecurityContext(project_id="project2", user_id="user2", security_level=SecurityLevel.INTERNAL)

        # Log various events
        service.log_cache_access("get", "key1", context1, success=True)
        service.log_cache_access("set", "key2", context1, success=False)
        service.log_cache_access("get", "key3", context2, success=True)
        service.log_security_violation("test_violation", "Test", context2)

        # Test filtering by project
        project1_events = service.get_events(project_id="project1")
        assert len(project1_events) == 2

        # Test filtering by user
        user1_events = service.get_events(user_id="user1")
        assert len(user1_events) == 2

        # Test filtering by event type
        access_events = service.get_events(event_type=AuditEventType.CACHE_ACCESS)
        assert len(access_events) == 3

        violation_events = service.get_events(event_type=AuditEventType.SECURITY_VIOLATION)
        assert len(violation_events) == 1

        # Test filtering by success status
        failed_events = service.get_events(include_successful=False, include_failed=True)
        assert len(failed_events) == 2  # One failed cache access + one security violation

        successful_events = service.get_events(include_successful=True, include_failed=False)
        assert len(successful_events) == 2

    def test_statistics_generation(self):
        """Test statistics generation."""
        service = SecurityAuditService(enable_file_logging=False)

        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.INTERNAL)

        # Log various events
        service.log_cache_access("get", "key1", context, success=True)
        service.log_cache_access("set", "key2", context, success=False, error_message="Permission denied")
        service.log_security_violation("test_violation", "Test violation", context)

        # Get statistics
        stats = service.get_statistics()

        assert stats["total_events"] == 3
        assert stats["failed_operations"] == 2
        assert stats["security_violations"] == 1
        assert stats["access_denials"] == 1  # Failed access becomes access denial
        assert stats["success_rate"] == 1 / 3  # 1 success out of 3 events
        assert stats["events_by_type"]["cache_access"] == 1
        assert stats["events_by_type"]["access_denied"] == 1
        assert stats["events_by_type"]["security_violation"] == 1

    def test_compliance_report_generation(self):
        """Test compliance report generation."""
        service = SecurityAuditService(enable_file_logging=False)

        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        # Log events
        service.log_cache_access("get", "key1", context, success=True)
        service.log_cache_access("set", "key2", context, success=False)
        service.log_security_violation("unauthorized_access", "Test violation", context)

        # Generate compliance report
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now() + timedelta(days=1)

        report = service.generate_compliance_report("test_project", start_date, end_date)

        assert report["report_metadata"]["project_id"] == "test_project"
        assert report["security_metrics"]["total_events"] == 3
        assert report["security_metrics"]["failed_events"] == 2
        assert report["security_metrics"]["security_violations"] == 1
        assert report["security_metrics"]["success_rate"] == 1 / 3
        assert "sensitive" in report["security_metrics"]["security_level_breakdown"]
        assert "test_user" in report["user_activity"]
        assert report["compliance_status"]["risk_level"] == "high"  # Due to security violations
        assert len(report["recommendations"]) > 0

    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced."""
        service = SecurityAuditService(enable_file_logging=False, max_events_in_memory=5)

        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        # Log more events than the memory limit
        for i in range(10):
            service.log_cache_access("get", f"key_{i}", context)

        # Should have cleaned up to half the limit
        assert len(service.events) <= 5

    @patch("src.services.security_audit_service.logger")
    def test_real_time_alerts(self, mock_logger):
        """Test real-time alert functionality."""
        service = SecurityAuditService(enable_file_logging=False, enable_real_time_alerts=True)

        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.INTERNAL)

        # Trigger high failure rate alert
        for i in range(15):  # More than threshold of 10
            service.log_cache_access("get", f"key_{i}", context, success=False, error_message="Test failure")

        # Should have triggered an alert
        mock_logger.warning.assert_called()
        alert_calls = [call for call in mock_logger.warning.call_args_list if "SECURITY ALERT" in str(call)]
        assert len(alert_calls) > 0

    def test_pattern_analysis(self):
        """Test user pattern analysis."""
        service = SecurityAuditService(enable_file_logging=False, enable_real_time_alerts=True)

        context = SecurityContext(project_id="test_project", user_id="suspicious_user", security_level=SecurityLevel.INTERNAL)

        # Create suspicious pattern - high failure rate
        for i in range(25):  # Enough events for pattern analysis
            success = i < 5  # Only first 5 succeed, rest fail
            service.log_cache_access(
                "get", f"key_{i}", context, success=success, error_message="Permission denied" if not success else None
            )

        # Should have detected suspicious pattern
        tracking_key = f"{context.project_id}:{context.user_id}"
        assert tracking_key in service._recent_events

        # Should have events for this user
        user_events = service._recent_events[tracking_key]
        assert len(user_events) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_audit_cache_access_function(self):
        """Test audit_cache_access convenience function."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.PUBLIC)

        # Should not raise error
        audit_cache_access("get", "test_key", context, success=True)

        # Verify event was logged
        service = get_security_audit_service()
        events = service.get_events(project_id="test_project")
        assert len(events) > 0

    def test_audit_security_violation_function(self):
        """Test audit_security_violation convenience function."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        # Should not raise error
        audit_security_violation("test_violation", "Test security violation", context, severity=AuditSeverity.HIGH)

        # Verify event was logged
        service = get_security_audit_service()
        events = service.get_events(project_id="test_project", event_type=AuditEventType.SECURITY_VIOLATION)
        assert len(events) > 0

    def test_audit_encryption_error_function(self):
        """Test audit_encryption_error convenience function."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.CONFIDENTIAL)

        # Should not raise error
        audit_encryption_error("decrypt", "Test encryption error", context, metadata={"test": "data"})

        # Verify event was logged
        service = get_security_audit_service()
        events = service.get_events(project_id="test_project", event_type=AuditEventType.ENCRYPTION_ERROR)
        assert len(events) > 0

    def test_audit_key_rotation_function(self):
        """Test audit_key_rotation convenience function."""
        # Should not raise error
        audit_key_rotation("old_key", "new_key", "test_project", metadata={"reason": "scheduled"})

        # Verify event was logged
        service = get_security_audit_service()
        events = service.get_events(project_id="test_project", event_type=AuditEventType.KEY_ROTATION)
        assert len(events) > 0


if __name__ == "__main__":
    pytest.main([__file__])
