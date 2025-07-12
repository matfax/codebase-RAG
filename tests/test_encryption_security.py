"""
Unit tests for encryption and security - Wave 15.1.3
Tests encryption utilities, security services, and secure cache operations.
"""

import base64
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.services.security_audit_service import SecurityAuditService
from src.utils.encryption_utils import (
    AESEncryption,
    BatchEncryption,
    EncryptedData,
    EncryptionCache,
    EncryptionError,
    EncryptionKey,
    EncryptionMetrics,
    KeyManager,
    SecureSerializer,
    decrypt_data,
    encrypt_data,
    generate_encryption_key,
    rotate_encryption_key,
    validate_encrypted_data,
)
from src.utils.secure_cache_utils import (
    SecureCacheWrapper,
    SecurityContext,
    access_control_check,
    audit_cache_access,
    mask_sensitive_data,
    validate_security_policy,
)


class TestEncryptionCore:
    """Test core encryption functionality."""

    def test_generate_encryption_key(self):
        """Test encryption key generation."""
        key = generate_encryption_key()

        assert key is not None
        assert len(key) >= 32  # At least 256 bits
        assert isinstance(key, bytes)

        # Keys should be unique
        key2 = generate_encryption_key()
        assert key != key2

    def test_encrypt_decrypt_string(self):
        """Test basic string encryption and decryption."""
        key = generate_encryption_key()
        plaintext = "This is sensitive data"

        # Encrypt
        encrypted = encrypt_data(plaintext, key)
        assert encrypted != plaintext
        assert isinstance(encrypted, str)

        # Decrypt
        decrypted = decrypt_data(encrypted, key)
        assert decrypted == plaintext

    def test_encrypt_decrypt_complex_data(self):
        """Test encryption of complex data structures."""
        key = generate_encryption_key()
        data = {
            "user": {
                "id": 123,
                "name": "Test User",
                "email": "test@example.com",
                "credentials": {"api_key": "secret_key_123", "token": "bearer_token_456"},
            },
            "metadata": {"created": datetime.now().isoformat(), "permissions": ["read", "write", "admin"]},
        }

        # Encrypt
        encrypted = encrypt_data(json.dumps(data), key)

        # Decrypt
        decrypted_json = decrypt_data(encrypted, key)
        decrypted_data = json.loads(decrypted_json)

        assert decrypted_data == data

    def test_encryption_with_wrong_key(self):
        """Test decryption with wrong key fails gracefully."""
        key1 = generate_encryption_key()
        key2 = generate_encryption_key()
        plaintext = "Secret message"

        encrypted = encrypt_data(plaintext, key1)

        # Attempting to decrypt with wrong key should fail
        with pytest.raises(EncryptionError):
            decrypt_data(encrypted, key2)

    def test_validate_encrypted_data(self):
        """Test encrypted data validation."""
        key = generate_encryption_key()
        plaintext = "Valid data"

        # Valid encrypted data
        encrypted = encrypt_data(plaintext, key)
        assert validate_encrypted_data(encrypted) is True

        # Invalid data
        assert validate_encrypted_data("not_encrypted_data") is False
        assert validate_encrypted_data("") is False
        assert validate_encrypted_data(None) is False


class TestAESEncryption:
    """Test AES encryption implementation."""

    def test_aes_encryption_initialization(self):
        """Test AES encryption initialization."""
        aes = AESEncryption()

        assert aes.key is not None
        assert len(aes.key) == 32  # 256-bit key
        assert aes.algorithm == "AES-256-CBC"

    def test_aes_encrypt_decrypt(self):
        """Test AES encryption and decryption."""
        aes = AESEncryption()
        plaintext = b"Sensitive information that needs protection"

        # Encrypt
        encrypted_data = aes.encrypt(plaintext)
        assert encrypted_data.ciphertext != plaintext
        assert encrypted_data.iv is not None
        assert encrypted_data.tag is not None

        # Decrypt
        decrypted = aes.decrypt(encrypted_data)
        assert decrypted == plaintext

    def test_aes_with_additional_data(self):
        """Test AES encryption with additional authenticated data."""
        aes = AESEncryption()
        plaintext = b"Secret message"
        aad = b"user_id:123:session:abc"

        # Encrypt with AAD
        encrypted_data = aes.encrypt(plaintext, additional_data=aad)

        # Decrypt with correct AAD
        decrypted = aes.decrypt(encrypted_data, additional_data=aad)
        assert decrypted == plaintext

        # Decrypt with wrong AAD should fail
        with pytest.raises(EncryptionError):
            aes.decrypt(encrypted_data, additional_data=b"wrong_aad")


class TestKeyManager:
    """Test encryption key management."""

    @pytest.fixture
    def key_manager(self):
        """Create a key manager instance."""
        return KeyManager()

    def test_key_generation_and_storage(self, key_manager):
        """Test key generation and storage."""
        key_id = "test_key_001"
        key = key_manager.generate_key(key_id)

        assert key is not None
        assert key_manager.get_key(key_id) == key
        assert key_id in key_manager.list_keys()

    def test_key_rotation(self, key_manager):
        """Test encryption key rotation."""
        key_id = "rotating_key"

        # Generate initial key
        old_key = key_manager.generate_key(key_id)

        # Rotate key
        new_key = key_manager.rotate_key(key_id)

        assert new_key != old_key
        assert key_manager.get_key(key_id) == new_key
        assert key_manager.get_key(key_id, version="previous") == old_key

    def test_key_expiration(self, key_manager):
        """Test key expiration handling."""
        key_id = "expiring_key"

        # Generate key with short expiration
        key = key_manager.generate_key(key_id, expires_in_seconds=1)

        # Key should be valid immediately
        assert key_manager.is_key_valid(key_id) is True

        # Wait for expiration
        time.sleep(2)

        # Key should be expired
        assert key_manager.is_key_valid(key_id) is False

    def test_master_key_derivation(self, key_manager):
        """Test master key derivation for key hierarchy."""
        master_key = key_manager.get_master_key()

        # Derive tenant-specific keys
        tenant1_key = key_manager.derive_key(master_key, "tenant:123")
        tenant2_key = key_manager.derive_key(master_key, "tenant:456")

        assert tenant1_key != tenant2_key
        assert len(tenant1_key) == len(master_key)


class TestBatchEncryption:
    """Test batch encryption operations."""

    def test_batch_encrypt(self):
        """Test batch encryption of multiple items."""
        batch_encryptor = BatchEncryption()

        items = [{"id": 1, "data": "secret1"}, {"id": 2, "data": "secret2"}, {"id": 3, "data": "secret3"}]

        encrypted_items = batch_encryptor.encrypt_batch(items)

        assert len(encrypted_items) == len(items)
        for i, encrypted in enumerate(encrypted_items):
            assert encrypted["id"] == items[i]["id"]
            assert encrypted["data"] != items[i]["data"]
            assert "encryption_metadata" in encrypted

    def test_batch_decrypt(self):
        """Test batch decryption of multiple items."""
        batch_encryptor = BatchEncryption()

        # Encrypt batch
        items = [{"id": i, "data": f"secret{i}"} for i in range(10)]
        encrypted_items = batch_encryptor.encrypt_batch(items)

        # Decrypt batch
        decrypted_items = batch_encryptor.decrypt_batch(encrypted_items)

        assert len(decrypted_items) == len(items)
        for i, decrypted in enumerate(decrypted_items):
            assert decrypted == items[i]

    def test_batch_partial_failure(self):
        """Test batch operations with partial failures."""
        batch_encryptor = BatchEncryption()

        # Mix valid and invalid items
        items = [{"id": 1, "data": "valid"}, {"id": 2, "data": None}, {"id": 3, "data": "valid"}]  # Invalid

        results = batch_encryptor.encrypt_batch(items, fail_fast=False)

        assert results["success_count"] == 2
        assert results["failure_count"] == 1
        assert len(results["failed_items"]) == 1


class TestSecurityAuditService:
    """Test security audit service functionality."""

    @pytest.fixture
    async def audit_service(self):
        """Create a security audit service instance."""
        service = SecurityAuditService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_audit_log_creation(self, audit_service):
        """Test security audit log creation."""
        await audit_service.log_access(
            user_id="user123",
            resource="cache:sensitive:data",
            action="read",
            result="success",
            metadata={"ip": "192.168.1.1", "session": "abc123"},
        )

        # Retrieve audit logs
        logs = await audit_service.get_audit_logs(user_id="user123", start_time=datetime.now() - timedelta(minutes=5))

        assert len(logs) == 1
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["action"] == "read"
        assert logs[0]["result"] == "success"

    @pytest.mark.asyncio
    async def test_suspicious_activity_detection(self, audit_service):
        """Test detection of suspicious access patterns."""
        user_id = "suspicious_user"

        # Simulate rapid access attempts
        for i in range(20):
            await audit_service.log_access(user_id=user_id, resource=f"cache:resource:{i}", action="read", result="success")

        # Check for anomaly detection
        anomalies = await audit_service.detect_anomalies(user_id)

        assert len(anomalies) > 0
        assert any(a["type"] == "rapid_access" for a in anomalies)

    @pytest.mark.asyncio
    async def test_compliance_reporting(self, audit_service):
        """Test compliance report generation."""
        # Log various activities
        activities = [
            {"user": "admin", "action": "delete", "resource": "cache:user:*"},
            {"user": "user1", "action": "read", "resource": "cache:public:data"},
            {"user": "user2", "action": "write", "resource": "cache:private:data"},
        ]

        for activity in activities:
            await audit_service.log_access(
                user_id=activity["user"], resource=activity["resource"], action=activity["action"], result="success"
            )

        # Generate compliance report
        report = await audit_service.generate_compliance_report(start_date=datetime.now() - timedelta(days=1), end_date=datetime.now())

        assert "summary" in report
        assert "high_risk_actions" in report
        assert report["summary"]["total_actions"] == 3


class TestSecureCacheOperations:
    """Test secure cache operations with encryption."""

    @pytest.mark.asyncio
    async def test_secure_cache_wrapper(self):
        """Test SecureCacheWrapper functionality."""
        base_cache = AsyncMock()
        security_context = SecurityContext(user_id="user123", encryption_required=True, audit_enabled=True)

        secure_cache = SecureCacheWrapper(base_cache, security_context)

        # Test secure set
        await secure_cache.set("key123", {"sensitive": "data"})

        # Verify encryption was applied
        base_cache.set.assert_called_once()
        stored_value = base_cache.set.call_args[0][1]
        assert "sensitive" not in str(stored_value)

    @pytest.mark.asyncio
    async def test_access_control(self):
        """Test access control for cache operations."""
        security_context = SecurityContext(user_id="user123", permissions=["read", "write"], tenant_id="tenant456")

        # Test allowed access
        assert access_control_check(security_context, resource="cache:tenant456:data", action="read") is True

        # Test denied access - wrong tenant
        assert access_control_check(security_context, resource="cache:tenant789:data", action="read") is False

        # Test denied access - missing permission
        assert access_control_check(security_context, resource="cache:tenant456:data", action="delete") is False

    def test_mask_sensitive_data(self):
        """Test sensitive data masking."""
        data = {
            "user_id": "12345",
            "email": "test@example.com",
            "password": "secret123",
            "api_key": "sk_live_abcdef123456",
            "credit_card": "4111111111111111",
            "ssn": "123-45-6789",
        }

        masked = mask_sensitive_data(data)

        assert masked["password"] == "****"
        assert masked["api_key"].startswith("sk_live_") and masked["api_key"].endswith("***")
        assert "****" in masked["credit_card"]
        assert "****" in masked["ssn"]
        assert masked["user_id"] == "12345"  # Non-sensitive data unchanged

    @pytest.mark.asyncio
    async def test_encryption_performance_metrics(self):
        """Test encryption performance metrics collection."""
        metrics = EncryptionMetrics()

        # Simulate encryption operations
        for i in range(100):
            start_time = time.time()
            key = generate_encryption_key()
            encrypted = encrypt_data(f"data_{i}", key)
            metrics.record_operation(operation="encrypt", duration=time.time() - start_time, data_size=len(f"data_{i}"))

        stats = metrics.get_statistics()

        assert stats["total_operations"] == 100
        assert stats["average_duration"] > 0
        assert stats["operations_per_second"] > 0
        assert stats["total_bytes_processed"] > 0


class TestEncryptionCache:
    """Test encryption result caching."""

    def test_encryption_cache_hit(self):
        """Test encryption cache for repeated operations."""
        cache = EncryptionCache(max_size=100)
        key = generate_encryption_key()

        # First encryption
        plaintext = "Repeated data"
        encrypted1 = cache.encrypt_with_cache(plaintext, key)

        # Second encryption of same data
        encrypted2 = cache.encrypt_with_cache(plaintext, key)

        # Should return cached result
        assert encrypted1 == encrypted2
        assert cache.hit_rate() > 0

    def test_encryption_cache_eviction(self):
        """Test encryption cache eviction policy."""
        cache = EncryptionCache(max_size=5)
        key = generate_encryption_key()

        # Fill cache beyond capacity
        for i in range(10):
            cache.encrypt_with_cache(f"data_{i}", key)

        # Cache should maintain size limit
        assert cache.size() <= 5

        # Check LRU eviction
        early_items = [f"data_{i}" for i in range(5)]
        for item in early_items:
            assert cache.get(item) is None  # Should be evicted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
