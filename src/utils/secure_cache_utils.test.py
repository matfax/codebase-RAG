"""
Unit tests for secure cache utilities.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from .secure_cache_utils import (
    CacheSecurityError,
    SecureCacheEntry,
    SecureCacheKeyGenerator,
    SecureCacheManager,
    SecurityAuditLog,
    SecurityContext,
    SecurityLevel,
    create_secure_key,
    deserialize_with_security,
    get_secure_cache_manager,
    log_cache_access,
    serialize_with_security,
    validate_secure_access,
)


class TestSecurityContext:
    """Test SecurityContext class."""

    def test_valid_context(self):
        """Test valid security context creation."""
        context = SecurityContext(
            project_id="test_project", user_id="test_user", session_id="test_session", security_level=SecurityLevel.SENSITIVE
        )

        assert context.project_id == "test_project"
        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert context.security_level == SecurityLevel.SENSITIVE

    def test_missing_project_id(self):
        """Test that missing project ID raises error."""
        with pytest.raises(CacheSecurityError):
            SecurityContext(project_id="")

    def test_sensitive_requires_user_id(self):
        """Test that sensitive level requires user ID."""
        with pytest.raises(CacheSecurityError):
            SecurityContext(project_id="test_project", security_level=SecurityLevel.SENSITIVE)

    def test_confidential_requires_user_id(self):
        """Test that confidential level requires user ID."""
        with pytest.raises(CacheSecurityError):
            SecurityContext(project_id="test_project", security_level=SecurityLevel.CONFIDENTIAL)

    def test_public_internal_no_user_required(self):
        """Test that public/internal levels don't require user ID."""
        # Should not raise error
        SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        SecurityContext(project_id="test_project", security_level=SecurityLevel.INTERNAL)


class TestSecureCacheEntry:
    """Test SecureCacheEntry class."""

    def test_entry_creation(self):
        """Test secure cache entry creation."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        entry = SecureCacheEntry(encrypted_data={"test": "data"}, security_context=context)

        assert entry.encrypted_data == {"test": "data"}
        assert entry.security_context == context
        assert entry.accessed_count == 0

    def test_touch_updates_access(self):
        """Test that touch updates access tracking."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        entry = SecureCacheEntry(encrypted_data={}, security_context=context)

        original_count = entry.accessed_count
        original_time = entry.last_accessed

        time.sleep(0.01)  # Small delay
        entry.touch()

        assert entry.accessed_count == original_count + 1
        assert entry.last_accessed > original_time

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        entry = SecureCacheEntry(encrypted_data={"test": "data"}, security_context=context, metadata={"key": "value"})

        # Convert to dict
        entry_dict = entry.to_dict()

        # Convert back
        restored_entry = SecureCacheEntry.from_dict(entry_dict)

        assert restored_entry.encrypted_data == entry.encrypted_data
        assert restored_entry.security_context.project_id == entry.security_context.project_id
        assert restored_entry.security_context.user_id == entry.security_context.user_id
        assert restored_entry.security_context.security_level == entry.security_context.security_level
        assert restored_entry.metadata == entry.metadata


class TestSecurityAuditLog:
    """Test SecurityAuditLog class."""

    def test_audit_log_creation(self):
        """Test audit log creation."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        log = SecurityAuditLog(operation="get", key="test_key", security_context=context, success=True)

        assert log.operation == "get"
        assert log.key == "test_key"
        assert log.security_context == context
        assert log.success is True
        assert log.error_message is None

    def test_audit_log_to_dict(self):
        """Test audit log serialization."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        log = SecurityAuditLog(operation="set", key="test_key", security_context=context, success=False, error_message="Test error")

        log_dict = log.to_dict()

        assert log_dict["operation"] == "set"
        assert log_dict["key"] == "test_key"
        assert log_dict["project_id"] == "test_project"
        assert log_dict["user_id"] == "test_user"
        assert log_dict["security_level"] == "sensitive"
        assert log_dict["success"] is False
        assert log_dict["error_message"] == "Test error"


class TestSecureCacheKeyGenerator:
    """Test SecureCacheKeyGenerator class."""

    def test_public_key_generation(self):
        """Test key generation for public data."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        key = SecureCacheKeyGenerator.generate_key("test_key", context)

        expected = "public:test_project:test_key"
        assert key == expected

    def test_internal_key_generation(self):
        """Test key generation for internal data."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.INTERNAL)

        key = SecureCacheKeyGenerator.generate_key("test_key", context)

        expected = "internal:test_project:test_key"
        assert key == expected

    def test_sensitive_key_generation(self):
        """Test key generation for sensitive data."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        key = SecureCacheKeyGenerator.generate_key("test_key", context)

        expected = "sensitive:test_project:test_user:test_key"
        assert key == expected

    def test_confidential_key_generation(self):
        """Test key generation for confidential data."""
        context = SecurityContext(
            project_id="test_project", user_id="test_user", session_id="test_session", security_level=SecurityLevel.CONFIDENTIAL
        )

        key = SecureCacheKeyGenerator.generate_key("test_key", context)

        expected = "confidential:test_project:test_user:test_session:test_key"
        assert key == expected

    def test_key_access_validation_public(self):
        """Test key access validation for public data."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        # Same project - should allow
        key = "public:test_project:test_key"
        assert SecureCacheKeyGenerator.validate_key_access(key, context)

        # Different project - should deny
        key = "public:other_project:test_key"
        assert not SecureCacheKeyGenerator.validate_key_access(key, context)

    def test_key_access_validation_sensitive(self):
        """Test key access validation for sensitive data."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        # Same project and user - should allow
        key = "sensitive:test_project:test_user:test_key"
        assert SecureCacheKeyGenerator.validate_key_access(key, context)

        # Same project, different user - should deny
        key = "sensitive:test_project:other_user:test_key"
        assert not SecureCacheKeyGenerator.validate_key_access(key, context)

        # Different project - should deny
        key = "sensitive:other_project:test_user:test_key"
        assert not SecureCacheKeyGenerator.validate_key_access(key, context)

    def test_key_access_validation_confidential(self):
        """Test key access validation for confidential data."""
        context = SecurityContext(
            project_id="test_project", user_id="test_user", session_id="test_session", security_level=SecurityLevel.CONFIDENTIAL
        )

        # Same project, user, and session - should allow
        key = "confidential:test_project:test_user:test_session:test_key"
        assert SecureCacheKeyGenerator.validate_key_access(key, context)

        # Same project and user, different session - should deny
        key = "confidential:test_project:test_user:other_session:test_key"
        assert not SecureCacheKeyGenerator.validate_key_access(key, context)

        # Different user - should deny
        key = "confidential:test_project:other_user:test_session:test_key"
        assert not SecureCacheKeyGenerator.validate_key_access(key, context)

    def test_invalid_key_format(self):
        """Test validation of invalid key formats."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        # Invalid keys should be denied
        assert not SecureCacheKeyGenerator.validate_key_access("invalid_key", context)
        assert not SecureCacheKeyGenerator.validate_key_access("public:test_project", context)  # Too short
        assert not SecureCacheKeyGenerator.validate_key_access("", context)


class TestSecureCacheManager:
    """Test SecureCacheManager class."""

    def test_manager_initialization(self):
        """Test secure cache manager initialization."""
        manager = SecureCacheManager()

        assert manager.enable_audit_logging is True
        assert manager.key_generator is not None
        assert len(manager.audit_logs) == 0

    def test_serialize_public_data(self):
        """Test serialization of public data."""
        manager = SecureCacheManager()
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        data = {"test": "data", "number": 42}

        serialized_data, metadata = manager.serialize_secure_data(data, context)

        assert isinstance(serialized_data, bytes)
        assert metadata["security_level"] == "public"
        assert metadata["encrypted"] is False

    @patch("src.utils.secure_cache_utils.get_default_serializer")
    def test_serialize_encrypted_data(self, mock_serializer):
        """Test serialization of encrypted data."""
        # Mock the serializer
        mock_serializer_instance = MagicMock()
        mock_serializer_instance.serialize_and_encrypt.return_value = {
            "ciphertext": "fake_ciphertext",
            "iv": "fake_iv",
            "key_id": "test_key",
        }
        mock_serializer.return_value = mock_serializer_instance

        manager = SecureCacheManager()
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        data = {"sensitive": "data"}

        serialized_data, metadata = manager.serialize_secure_data(data, context)

        assert isinstance(serialized_data, bytes)
        assert metadata["security_level"] == "sensitive"
        assert metadata["encrypted"] is True
        mock_serializer_instance.serialize_and_encrypt.assert_called_once_with(data, key_id=None)

    @patch("src.utils.secure_cache_utils.get_default_serializer")
    def test_deserialize_encrypted_data(self, mock_serializer):
        """Test deserialization of encrypted data."""
        # Mock the serializer
        mock_serializer_instance = MagicMock()
        mock_serializer_instance.decrypt_and_deserialize.return_value = {"sensitive": "data"}
        mock_serializer.return_value = mock_serializer_instance

        manager = SecureCacheManager()
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        # Create a secure entry
        entry = SecureCacheEntry(encrypted_data={"ciphertext": "fake_ciphertext"}, security_context=context)

        # Serialize the entry to bytes (simulating cache storage)
        from .cache_utils import SerializationFormat, serialize_data

        entry_bytes = serialize_data(entry.to_dict(), SerializationFormat.JSON)

        metadata = {"encrypted": True, "compressed": False, "serialization_format": "json"}

        result = manager.deserialize_secure_data(entry_bytes, metadata, context)

        assert result == {"sensitive": "data"}
        mock_serializer_instance.decrypt_and_deserialize.assert_called_once()

    def test_deserialize_access_denied(self):
        """Test access denial for unauthorized decryption."""
        manager = SecureCacheManager()

        # Create entry with one context
        entry_context = SecurityContext(project_id="test_project", user_id="user1", security_level=SecurityLevel.SENSITIVE)

        # Try to access with different context
        request_context = SecurityContext(project_id="test_project", user_id="user2", security_level=SecurityLevel.SENSITIVE)

        entry = SecureCacheEntry(encrypted_data={"ciphertext": "fake_ciphertext"}, security_context=entry_context)

        from .cache_utils import SerializationFormat, serialize_data

        entry_bytes = serialize_data(entry.to_dict(), SerializationFormat.JSON)

        metadata = {"encrypted": True, "compressed": False, "serialization_format": "json"}

        with pytest.raises(CacheSecurityError, match="Insufficient permissions"):
            manager.deserialize_secure_data(entry_bytes, metadata, request_context)

    def test_generate_secure_key(self):
        """Test secure key generation."""
        manager = SecureCacheManager()
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        key = manager.generate_secure_key("test_key", context)

        expected = "sensitive:test_project:test_user:test_key"
        assert key == expected

    def test_validate_key_access(self):
        """Test key access validation."""
        manager = SecureCacheManager()
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        # Valid key
        valid_key = "sensitive:test_project:test_user:test_key"
        assert manager.validate_key_access(valid_key, context)

        # Invalid key (different user)
        invalid_key = "sensitive:test_project:other_user:test_key"
        assert not manager.validate_key_access(invalid_key, context)

    def test_project_keys_pattern(self):
        """Test project keys pattern generation."""
        manager = SecureCacheManager()

        pattern = manager.get_project_keys_pattern("test_project")
        assert pattern == "*:test_project:*"

    def test_user_keys_pattern(self):
        """Test user keys pattern generation."""
        manager = SecureCacheManager()

        pattern = manager.get_user_keys_pattern("test_project", "test_user")
        assert pattern == "*:test_project:test_user:*"

    def test_audit_logging(self):
        """Test security audit logging."""
        manager = SecureCacheManager(enable_audit_logging=True)
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        # Log an event
        manager._log_security_event("get", "test_key", context, success=True)

        assert len(manager.audit_logs) == 1
        assert manager.audit_logs[0].operation == "get"
        assert manager.audit_logs[0].key == "test_key"
        assert manager.audit_logs[0].success is True

    def test_get_audit_logs_filtering(self):
        """Test audit log filtering."""
        manager = SecureCacheManager(enable_audit_logging=True)

        # Create different contexts
        context1 = SecurityContext(project_id="project1", user_id="user1", security_level=SecurityLevel.SENSITIVE)
        context2 = SecurityContext(project_id="project2", user_id="user2", security_level=SecurityLevel.SENSITIVE)

        # Log events
        manager._log_security_event("get", "key1", context1)
        manager._log_security_event("set", "key2", context1)
        manager._log_security_event("get", "key3", context2)

        # Filter by project
        project1_logs = manager.get_audit_logs(project_id="project1")
        assert len(project1_logs) == 2

        # Filter by user
        user1_logs = manager.get_audit_logs(user_id="user1")
        assert len(user1_logs) == 2

        # Filter by operation
        get_logs = manager.get_audit_logs(operation="get")
        assert len(get_logs) == 2

    def test_clear_project_audit_logs(self):
        """Test clearing project audit logs."""
        manager = SecureCacheManager(enable_audit_logging=True)

        context1 = SecurityContext(project_id="project1", security_level=SecurityLevel.PUBLIC)
        context2 = SecurityContext(project_id="project2", security_level=SecurityLevel.PUBLIC)

        # Log events for both projects
        manager._log_security_event("get", "key1", context1)
        manager._log_security_event("get", "key2", context2)

        assert len(manager.audit_logs) == 2

        # Clear logs for project1
        cleared_count = manager.clear_project_audit_logs("project1")

        assert cleared_count == 1
        assert len(manager.audit_logs) == 1
        assert manager.audit_logs[0].security_context.project_id == "project2"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_serialize_with_security(self):
        """Test serialize_with_security convenience function."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        data = {"test": "data"}

        serialized_data, metadata = serialize_with_security(data, context)

        assert isinstance(serialized_data, bytes)
        assert "security_level" in metadata

    def test_create_secure_key(self):
        """Test create_secure_key convenience function."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        key = create_secure_key("test_key", context)

        assert key == "public:test_project:test_key"

    def test_validate_secure_access(self):
        """Test validate_secure_access convenience function."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        valid_key = "public:test_project:test_key"
        invalid_key = "public:other_project:test_key"

        assert validate_secure_access(valid_key, context)
        assert not validate_secure_access(invalid_key, context)

    def test_log_cache_access(self):
        """Test log_cache_access convenience function."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        # Should not raise error
        log_cache_access("get", "test_key", context)

        # Check that log was recorded
        manager = get_secure_cache_manager()
        logs = manager.get_audit_logs(project_id="test_project")
        assert len(logs) > 0


class TestSecurityLevels:
    """Test different security levels."""

    def test_public_level(self):
        """Test public security level behavior."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.PUBLIC)

        manager = SecureCacheManager()
        data = {"public": "data"}

        serialized_data, metadata = manager.serialize_secure_data(data, context)

        # Public data should not be encrypted
        assert metadata["encrypted"] is False
        assert metadata["security_level"] == "public"

    def test_internal_level(self):
        """Test internal security level behavior."""
        context = SecurityContext(project_id="test_project", security_level=SecurityLevel.INTERNAL)

        manager = SecureCacheManager()
        key = manager.generate_secure_key("test_key", context)

        # Should include project isolation
        assert key.startswith("internal:test_project:")

    def test_sensitive_level(self):
        """Test sensitive security level behavior."""
        context = SecurityContext(project_id="test_project", user_id="test_user", security_level=SecurityLevel.SENSITIVE)

        manager = SecureCacheManager()
        key = manager.generate_secure_key("test_key", context)

        # Should include user isolation
        assert key == "sensitive:test_project:test_user:test_key"

    def test_confidential_level(self):
        """Test confidential security level behavior."""
        context = SecurityContext(
            project_id="test_project", user_id="test_user", session_id="test_session", security_level=SecurityLevel.CONFIDENTIAL
        )

        manager = SecureCacheManager()
        key = manager.generate_secure_key("test_key", context)

        # Should include session isolation
        assert key == "confidential:test_project:test_user:test_session:test_key"


if __name__ == "__main__":
    pytest.main([__file__])
