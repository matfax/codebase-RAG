"""
Secure cache utilities with encryption for sensitive data protection.

This module extends the existing cache utilities with encryption capabilities,
project-based isolation, and security auditing for sensitive cache data.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from .cache_utils import (
    CacheUtilsError,
    CompressionFormat,
    SerializationFormat,
    compress_data,
    decompress_data,
    deserialize_data,
    estimate_size,
    serialize_data,
)
from .encryption_utils import EncryptedData, EncryptionError, decrypt_json, encrypt_json, get_default_encryption, get_default_serializer

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for cache data."""

    PUBLIC = "public"  # No encryption needed
    INTERNAL = "internal"  # Basic encryption
    SENSITIVE = "sensitive"  # Strong encryption with project isolation
    CONFIDENTIAL = "confidential"  # Strong encryption with user isolation


class CacheSecurityError(CacheUtilsError):
    """Exception raised for cache security violations."""

    pass


@dataclass
class SecurityContext:
    """Security context for cache operations."""

    project_id: str
    user_id: str | None = None
    session_id: str | None = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    encryption_key_id: str | None = None

    def __post_init__(self):
        """Validate security context."""
        if not self.project_id:
            raise CacheSecurityError("Project ID is required for security context")

        # Require user_id for sensitive and confidential data
        if self.security_level in [SecurityLevel.SENSITIVE, SecurityLevel.CONFIDENTIAL]:
            if not self.user_id:
                raise CacheSecurityError(f"User ID is required for {self.security_level.value} security level")


@dataclass
class SecureCacheEntry:
    """Secure cache entry with encryption metadata."""

    encrypted_data: dict[str, Any]
    security_context: SecurityContext
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update access tracking."""
        self.accessed_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "encrypted_data": self.encrypted_data,
            "security_context": {
                "project_id": self.security_context.project_id,
                "user_id": self.security_context.user_id,
                "session_id": self.security_context.session_id,
                "security_level": self.security_context.security_level.value,
                "encryption_key_id": self.security_context.encryption_key_id,
            },
            "metadata": self.metadata,
            "created_at": self.created_at,
            "accessed_count": self.accessed_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecureCacheEntry":
        """Create from dictionary."""
        security_context = SecurityContext(
            project_id=data["security_context"]["project_id"],
            user_id=data["security_context"].get("user_id"),
            session_id=data["security_context"].get("session_id"),
            security_level=SecurityLevel(data["security_context"]["security_level"]),
            encryption_key_id=data["security_context"].get("encryption_key_id"),
        )

        return cls(
            encrypted_data=data["encrypted_data"],
            security_context=security_context,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            accessed_count=data.get("accessed_count", 0),
            last_accessed=data.get("last_accessed", time.time()),
        )


@dataclass
class SecurityAuditLog:
    """Security audit log entry."""

    operation: str  # get, set, delete, access_denied
    key: str
    security_context: SecurityContext
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: str | None = None
    client_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "operation": self.operation,
            "key": self.key,
            "project_id": self.security_context.project_id,
            "user_id": self.security_context.user_id,
            "session_id": self.security_context.session_id,
            "security_level": self.security_context.security_level.value,
            "timestamp": self.timestamp,
            "success": self.success,
            "error_message": self.error_message,
            "client_info": self.client_info,
        }


class SecureCacheKeyGenerator:
    """Generates secure, isolated cache keys."""

    @staticmethod
    def generate_key(base_key: str, security_context: SecurityContext) -> str:
        """
        Generate a secure cache key with proper isolation.

        Args:
            base_key: Base cache key
            security_context: Security context for isolation

        Returns:
            str: Secure cache key with isolation
        """
        # Start with security level and project isolation
        key_parts = [security_context.security_level.value, security_context.project_id]

        # Add user isolation for sensitive/confidential data
        if security_context.security_level in [SecurityLevel.SENSITIVE, SecurityLevel.CONFIDENTIAL]:
            if security_context.user_id:
                key_parts.append(security_context.user_id)

        # Add session isolation for confidential data
        if security_context.security_level == SecurityLevel.CONFIDENTIAL:
            if security_context.session_id:
                key_parts.append(security_context.session_id)

        # Add the base key
        key_parts.append(base_key)

        # Join with separators
        return ":".join(key_parts)

    @staticmethod
    def validate_key_access(key: str, security_context: SecurityContext) -> bool:
        """
        Validate that a security context can access a given key.

        Args:
            key: Cache key to validate
            security_context: Security context requesting access

        Returns:
            bool: True if access is allowed
        """
        try:
            # Parse key components
            key_parts = key.split(":")
            if len(key_parts) < 3:  # security_level:project_id:base_key minimum
                return False

            key_security_level = SecurityLevel(key_parts[0])
            key_project_id = key_parts[1]

            # Check project isolation
            if key_project_id != security_context.project_id:
                return False

            # Check security level requirements
            if key_security_level in [SecurityLevel.SENSITIVE, SecurityLevel.CONFIDENTIAL]:
                if len(key_parts) < 4:  # Should have user_id
                    return False

                key_user_id = key_parts[2]
                if key_user_id != security_context.user_id:
                    return False

                # Check session isolation for confidential data
                if key_security_level == SecurityLevel.CONFIDENTIAL:
                    if len(key_parts) < 5:  # Should have session_id
                        return False

                    key_session_id = key_parts[3]
                    if key_session_id != security_context.session_id:
                        return False

            return True

        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid key format for access validation: {e}")
            return False


class SecureCacheManager:
    """Manages secure cache operations with encryption and isolation."""

    def __init__(self, enable_audit_logging: bool = True):
        """
        Initialize secure cache manager.

        Args:
            enable_audit_logging: Whether to enable security audit logging
        """
        self.enable_audit_logging = enable_audit_logging
        self.key_generator = SecureCacheKeyGenerator()
        self.audit_logs: list[SecurityAuditLog] = []
        self._encryption = get_default_encryption()
        self._serializer = get_default_serializer()

    def serialize_secure_data(self, data: Any, security_context: SecurityContext, compress: bool = True) -> tuple[bytes, dict[str, Any]]:
        """
        Serialize data with appropriate security measures.

        Args:
            data: Data to serialize
            security_context: Security context
            compress: Whether to compress data

        Returns:
            Tuple[bytes, Dict[str, Any]]: (serialized_data, metadata)
        """
        try:
            start_time = time.time()

            # Determine if encryption is needed
            needs_encryption = security_context.security_level != SecurityLevel.PUBLIC

            if needs_encryption:
                # Create secure cache entry
                entry = SecureCacheEntry(encrypted_data={}, security_context=security_context)  # Will be filled below

                # Encrypt the actual data
                encrypted_data = self._serializer.serialize_and_encrypt(data, key_id=security_context.encryption_key_id)
                entry.encrypted_data = encrypted_data

                # Serialize the secure entry
                serialized_data = serialize_data(entry.to_dict(), SerializationFormat.JSON)

            else:
                # No encryption for public data
                serialized_data = serialize_data(data, SerializationFormat.PICKLE)

            # Apply compression if requested
            if compress:
                compression_format = CompressionFormat.GZIP
                final_data = compress_data(serialized_data, compression_format)
            else:
                compression_format = CompressionFormat.NONE
                final_data = serialized_data

            # Build metadata
            metadata = {
                "security_level": security_context.security_level.value,
                "encrypted": needs_encryption,
                "compressed": compress,
                "compression_format": compression_format.value,
                "serialization_format": SerializationFormat.JSON.value if needs_encryption else SerializationFormat.PICKLE.value,
                "original_size": estimate_size(data),
                "serialized_size": len(serialized_data),
                "final_size": len(final_data),
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
            }

            return final_data, metadata

        except Exception as e:
            logger.error(f"Failed to serialize secure data: {e}")
            raise CacheSecurityError(f"Failed to serialize secure data: {e}")

    def deserialize_secure_data(self, data: bytes, metadata: dict[str, Any], security_context: SecurityContext) -> Any:
        """
        Deserialize data with security validation.

        Args:
            data: Serialized data
            metadata: Serialization metadata
            security_context: Security context for validation

        Returns:
            Any: Deserialized data
        """
        try:
            # Decompress if needed
            if metadata.get("compressed", False):
                compression_format = CompressionFormat(metadata["compression_format"])
                decompressed_data = decompress_data(data, compression_format)
            else:
                decompressed_data = data

            # Check if data is encrypted
            if metadata.get("encrypted", False):
                # Deserialize secure entry
                serialization_format = SerializationFormat(metadata["serialization_format"])
                entry_dict = deserialize_data(decompressed_data, serialization_format)
                entry = SecureCacheEntry.from_dict(entry_dict)

                # Validate security context access
                if not self._validate_context_access(entry.security_context, security_context):
                    self._log_security_event(
                        "access_denied",
                        "unauthorized_access",
                        security_context,
                        success=False,
                        error_message="Insufficient permissions for encrypted data",
                    )
                    raise CacheSecurityError("Insufficient permissions to access encrypted data")

                # Decrypt the actual data
                decrypted_data = self._serializer.decrypt_and_deserialize(entry.encrypted_data)

                # Update access tracking
                entry.touch()

                return decrypted_data

            else:
                # No encryption - deserialize directly
                serialization_format = SerializationFormat(metadata["serialization_format"])
                return deserialize_data(decompressed_data, serialization_format)

        except EncryptionError as e:
            self._log_security_event("get", "decryption_failed", security_context, success=False, error_message=str(e))
            raise CacheSecurityError(f"Failed to decrypt cache data: {e}")
        except Exception as e:
            logger.error(f"Failed to deserialize secure data: {e}")
            raise CacheSecurityError(f"Failed to deserialize secure data: {e}")

    def generate_secure_key(self, base_key: str, security_context: SecurityContext) -> str:
        """
        Generate a secure cache key with proper isolation.

        Args:
            base_key: Base cache key
            security_context: Security context

        Returns:
            str: Secure cache key
        """
        return self.key_generator.generate_key(base_key, security_context)

    def validate_key_access(self, key: str, security_context: SecurityContext) -> bool:
        """
        Validate access to a cache key.

        Args:
            key: Cache key
            security_context: Security context

        Returns:
            bool: True if access is allowed
        """
        access_allowed = self.key_generator.validate_key_access(key, security_context)

        if not access_allowed and self.enable_audit_logging:
            self._log_security_event("access_denied", key, security_context, success=False, error_message="Key access validation failed")

        return access_allowed

    def get_project_keys_pattern(self, project_id: str) -> str:
        """
        Get pattern for finding all keys belonging to a project.

        Args:
            project_id: Project ID

        Returns:
            str: Redis pattern for project keys
        """
        return f"*:{project_id}:*"

    def get_user_keys_pattern(self, project_id: str, user_id: str) -> str:
        """
        Get pattern for finding all keys belonging to a user within a project.

        Args:
            project_id: Project ID
            user_id: User ID

        Returns:
            str: Redis pattern for user keys
        """
        return f"*:{project_id}:{user_id}:*"

    def _validate_context_access(self, entry_context: SecurityContext, request_context: SecurityContext) -> bool:
        """
        Validate that request context can access data with entry context.

        Args:
            entry_context: Security context of cached entry
            request_context: Security context of request

        Returns:
            bool: True if access is allowed
        """
        # Project isolation - must match
        if entry_context.project_id != request_context.project_id:
            return False

        # For sensitive and confidential data, user must match
        if entry_context.security_level in [SecurityLevel.SENSITIVE, SecurityLevel.CONFIDENTIAL]:
            if entry_context.user_id != request_context.user_id:
                return False

        # For confidential data, session must match
        if entry_context.security_level == SecurityLevel.CONFIDENTIAL:
            if entry_context.session_id != request_context.session_id:
                return False

        return True

    def _log_security_event(
        self, operation: str, key: str, security_context: SecurityContext, success: bool = True, error_message: str | None = None
    ) -> None:
        """
        Log a security event for auditing.

        Args:
            operation: Operation performed
            key: Cache key involved
            security_context: Security context
            success: Whether operation succeeded
            error_message: Error message if failed
        """
        if not self.enable_audit_logging:
            return

        audit_log = SecurityAuditLog(
            operation=operation, key=key, security_context=security_context, success=success, error_message=error_message
        )

        self.audit_logs.append(audit_log)

        # Log to system logger as well
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"Security audit: {operation} on {key} for project {security_context.project_id}, "
            f"user {security_context.user_id}, success: {success}",
        )

        # Limit audit log size
        if len(self.audit_logs) > 10000:
            self.audit_logs = self.audit_logs[-5000:]  # Keep last 5000 entries

    def get_audit_logs(
        self,
        project_id: str | None = None,
        user_id: str | None = None,
        operation: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get security audit logs with filtering.

        Args:
            project_id: Filter by project ID
            user_id: Filter by user ID
            operation: Filter by operation type
            since: Filter by timestamp (Unix timestamp)
            limit: Maximum number of entries to return

        Returns:
            List[Dict[str, Any]]: Filtered audit logs
        """
        filtered_logs = []

        for log in reversed(self.audit_logs):  # Most recent first
            # Apply filters
            if project_id and log.security_context.project_id != project_id:
                continue
            if user_id and log.security_context.user_id != user_id:
                continue
            if operation and log.operation != operation:
                continue
            if since and log.timestamp < since:
                continue

            filtered_logs.append(log.to_dict())

            if len(filtered_logs) >= limit:
                break

        return filtered_logs

    def clear_project_audit_logs(self, project_id: str) -> int:
        """
        Clear audit logs for a specific project.

        Args:
            project_id: Project ID to clear logs for

        Returns:
            int: Number of logs cleared
        """
        original_count = len(self.audit_logs)
        self.audit_logs = [log for log in self.audit_logs if log.security_context.project_id != project_id]
        cleared_count = original_count - len(self.audit_logs)

        logger.info(f"Cleared {cleared_count} audit logs for project {project_id}")
        return cleared_count


# Global secure cache manager instance
_secure_cache_manager: SecureCacheManager | None = None


def get_secure_cache_manager() -> SecureCacheManager:
    """Get the global secure cache manager instance."""
    global _secure_cache_manager
    if _secure_cache_manager is None:
        _secure_cache_manager = SecureCacheManager()
    return _secure_cache_manager


# Convenience functions
def serialize_with_security(data: Any, security_context: SecurityContext, compress: bool = True) -> tuple[bytes, dict[str, Any]]:
    """
    Convenience function to serialize data with security.

    Args:
        data: Data to serialize
        security_context: Security context
        compress: Whether to compress

    Returns:
        Tuple[bytes, Dict[str, Any]]: (serialized_data, metadata)
    """
    manager = get_secure_cache_manager()
    return manager.serialize_secure_data(data, security_context, compress)


def deserialize_with_security(data: bytes, metadata: dict[str, Any], security_context: SecurityContext) -> Any:
    """
    Convenience function to deserialize secure data.

    Args:
        data: Serialized data
        metadata: Metadata
        security_context: Security context

    Returns:
        Any: Deserialized data
    """
    manager = get_secure_cache_manager()
    return manager.deserialize_secure_data(data, metadata, security_context)


def create_secure_key(base_key: str, security_context: SecurityContext) -> str:
    """
    Convenience function to create secure cache key.

    Args:
        base_key: Base key
        security_context: Security context

    Returns:
        str: Secure cache key
    """
    manager = get_secure_cache_manager()
    return manager.generate_secure_key(base_key, security_context)


def validate_secure_access(key: str, security_context: SecurityContext) -> bool:
    """
    Convenience function to validate key access.

    Args:
        key: Cache key
        security_context: Security context

    Returns:
        bool: True if access allowed
    """
    manager = get_secure_cache_manager()
    return manager.validate_key_access(key, security_context)


def log_cache_access(
    operation: str, key: str, security_context: SecurityContext, success: bool = True, error_message: str | None = None
) -> None:
    """
    Convenience function to log cache access for security auditing.

    Args:
        operation: Operation type
        key: Cache key
        security_context: Security context
        success: Whether operation succeeded
        error_message: Error message if failed
    """
    manager = get_secure_cache_manager()
    manager._log_security_event(operation, key, security_context, success, error_message)


# Export public interface
__all__ = [
    "SecurityLevel",
    "CacheSecurityError",
    "SecurityContext",
    "SecureCacheEntry",
    "SecurityAuditLog",
    "SecureCacheKeyGenerator",
    "SecureCacheManager",
    "get_secure_cache_manager",
    "serialize_with_security",
    "deserialize_with_security",
    "create_secure_key",
    "validate_secure_access",
    "log_cache_access",
]
