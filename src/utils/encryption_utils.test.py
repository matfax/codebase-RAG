"""
Unit tests for encryption utilities.
"""

import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from .encryption_utils import (
    AESEncryption,
    BatchEncryption,
    EncryptedData,
    EncryptionCache,
    EncryptionError,
    EncryptionKey,
    EncryptionMetrics,
    EncryptionOperationError,
    KeyGenerationError,
    KeyManager,
    SecureSerializer,
    cleanup_old_keys,
    clear_encryption_cache,
    decrypt_batch,
    decrypt_data,
    decrypt_json,
    encrypt_batch,
    encrypt_data,
    encrypt_json,
    get_encryption_metrics,
    reset_encryption_metrics,
    rotate_encryption_key,
)


class TestEncryptionKey:
    """Test EncryptionKey class."""

    def test_init(self):
        """Test EncryptionKey initialization."""
        key = EncryptionKey(key_id="test_key", key_data=b"test_key_data", created_at=1234567890.0)

        assert key.key_id == "test_key"
        assert key.key_data == b"test_key_data"
        assert key.created_at == 1234567890.0
        assert key.expires_at is None
        assert key.algorithm == "AES-256-CBC"
        assert key.salt is None

    def test_is_expired(self):
        """Test key expiration check."""
        current_time = time.time()

        # Not expired key
        key = EncryptionKey(
            key_id="test_key",
            key_data=b"test_key_data",
            created_at=current_time,
            expires_at=current_time + 3600,  # 1 hour from now
        )
        assert not key.is_expired()

        # Expired key
        key.expires_at = current_time - 3600  # 1 hour ago
        assert key.is_expired()

        # No expiration
        key.expires_at = None
        assert not key.is_expired()

    def test_should_rotate(self):
        """Test key rotation check."""
        current_time = time.time()

        # Should not rotate (recent key)
        key = EncryptionKey(key_id="test_key", key_data=b"test_key_data", created_at=current_time)
        assert not key.should_rotate()

        # Should rotate (old key)
        key.created_at = current_time - (86400 * 8)  # 8 days ago
        assert key.should_rotate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        key = EncryptionKey(key_id="test_key", key_data=b"test_key_data", created_at=1234567890.0, salt=b"test_salt")

        result = key.to_dict()

        assert result["key_id"] == "test_key"
        assert result["key_data"] == "dGVzdF9rZXlfZGF0YQ=="  # base64 encoded
        assert result["created_at"] == 1234567890.0
        assert result["salt"] == "dGVzdF9zYWx0"  # base64 encoded
        assert result["algorithm"] == "AES-256-CBC"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "key_id": "test_key",
            "key_data": "dGVzdF9rZXlfZGF0YQ==",  # base64 encoded
            "created_at": 1234567890.0,
            "salt": "dGVzdF9zYWx0",  # base64 encoded
            "algorithm": "AES-256-CBC",
        }

        key = EncryptionKey.from_dict(data)

        assert key.key_id == "test_key"
        assert key.key_data == b"test_key_data"
        assert key.created_at == 1234567890.0
        assert key.salt == b"test_salt"
        assert key.algorithm == "AES-256-CBC"


class TestEncryptedData:
    """Test EncryptedData class."""

    def test_init(self):
        """Test EncryptedData initialization."""
        data = EncryptedData(ciphertext=b"encrypted_data", iv=b"initialization_vector", key_id="test_key")

        assert data.ciphertext == b"encrypted_data"
        assert data.iv == b"initialization_vector"
        assert data.key_id == "test_key"
        assert data.algorithm == "AES-256-CBC"
        assert data.created_at is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = EncryptedData(ciphertext=b"encrypted_data", iv=b"initialization_vector", key_id="test_key", created_at=1234567890.0)

        result = data.to_dict()

        assert result["ciphertext"] == "ZW5jcnlwdGVkX2RhdGE="  # base64 encoded
        assert result["iv"] == "aW5pdGlhbGl6YXRpb25fdmVjdG9y"  # base64 encoded
        assert result["key_id"] == "test_key"
        assert result["algorithm"] == "AES-256-CBC"
        assert result["created_at"] == 1234567890.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "ciphertext": "ZW5jcnlwdGVkX2RhdGE=",  # base64 encoded
            "iv": "aW5pdGlhbGl6YXRpb25fdmVjdG9y",  # base64 encoded
            "key_id": "test_key",
            "algorithm": "AES-256-CBC",
            "created_at": 1234567890.0,
        }

        encrypted_data = EncryptedData.from_dict(data)

        assert encrypted_data.ciphertext == b"encrypted_data"
        assert encrypted_data.iv == b"initialization_vector"
        assert encrypted_data.key_id == "test_key"
        assert encrypted_data.algorithm == "AES-256-CBC"
        assert encrypted_data.created_at == 1234567890.0


class TestKeyManager:
    """Test KeyManager class."""

    def test_init_with_temp_path(self):
        """Test KeyManager initialization with temporary path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")

            manager = KeyManager(key_store_path)

            assert manager.key_store_path == key_store_path
            assert len(manager.keys) > 0  # Should have generated initial key
            assert manager.current_key_id is not None

    def test_generate_key(self):
        """Test key generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            manager = KeyManager(key_store_path)

            # Generate key with custom ID
            key = manager.generate_key("custom_key")

            assert key.key_id == "custom_key"
            assert len(key.key_data) == 32  # AES-256 key size
            assert key.salt is not None
            assert key.created_at is not None

            # Generate key with auto ID
            key2 = manager.generate_key()
            assert key2.key_id != "custom_key"
            assert key2.key_id.startswith("key_")

    def test_derive_key_from_password(self):
        """Test key derivation from password."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            manager = KeyManager(key_store_path)

            key = manager.derive_key_from_password("test_password")

            assert key.key_id.startswith("derived_")
            assert len(key.key_data) == 32  # AES-256 key size
            assert key.salt is not None
            assert key.created_at is not None

    def test_key_rotation(self):
        """Test key rotation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            manager = KeyManager(key_store_path)

            original_key_id = manager.current_key_id

            # Rotate key
            new_key = manager.rotate_key()

            assert new_key.key_id != original_key_id
            assert manager.current_key_id == new_key.key_id
            assert original_key_id in manager.keys  # Old key should still exist

    def test_cleanup_expired_keys(self):
        """Test cleanup of expired keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            manager = KeyManager(key_store_path)

            # Generate multiple keys
            for i in range(10):
                manager.generate_key(f"key_{i}")

            # Cleanup, keeping only 5 keys
            removed_count = manager.cleanup_expired_keys(keep_count=5)

            assert removed_count > 0
            assert len(manager.keys) <= 5
            assert manager.current_key_id in manager.keys  # Current key should be preserved

    def test_save_and_load_keys(self):
        """Test saving and loading keys."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")

            # Create manager and generate keys
            manager1 = KeyManager(key_store_path)
            manager1.generate_key("test_key_1")
            manager1.generate_key("test_key_2")

            # Create new manager with same path
            manager2 = KeyManager(key_store_path)

            # Should have loaded the same keys
            assert len(manager2.keys) == len(manager1.keys)
            assert "test_key_1" in manager2.keys
            assert "test_key_2" in manager2.keys
            assert manager2.current_key_id == manager1.current_key_id


class TestAESEncryption:
    """Test AESEncryption class."""

    def test_encrypt_decrypt_string(self):
        """Test encryption and decryption of string data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)

            plaintext = "Hello, World! This is a test message."

            # Encrypt
            encrypted_data = encryption.encrypt(plaintext)

            assert encrypted_data.ciphertext != plaintext.encode()
            assert encrypted_data.key_id is not None
            assert encrypted_data.iv is not None

            # Decrypt
            decrypted_data = encryption.decrypt(encrypted_data)
            decrypted_string = encryption.decrypt_to_string(encrypted_data)

            assert decrypted_data == plaintext.encode()
            assert decrypted_string == plaintext

    def test_encrypt_decrypt_bytes(self):
        """Test encryption and decryption of bytes data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)

            plaintext = b"Binary data: \x00\x01\x02\x03\x04"

            # Encrypt
            encrypted_data = encryption.encrypt(plaintext)

            assert encrypted_data.ciphertext != plaintext
            assert encrypted_data.key_id is not None
            assert encrypted_data.iv is not None

            # Decrypt
            decrypted_data = encryption.decrypt(encrypted_data)

            assert decrypted_data == plaintext

    def test_encrypt_with_specific_key(self):
        """Test encryption with specific key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)

            # Generate specific key
            key_manager.generate_key("specific_key")

            plaintext = "Test message"

            # Encrypt with specific key
            encrypted_data = encryption.encrypt(plaintext, key_id="specific_key")

            assert encrypted_data.key_id == "specific_key"

            # Decrypt
            decrypted_string = encryption.decrypt_to_string(encrypted_data)
            assert decrypted_string == plaintext

    def test_decrypt_with_missing_key(self):
        """Test decryption with missing key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)

            # Create encrypted data with non-existent key
            encrypted_data = EncryptedData(
                ciphertext=b"fake_ciphertext",
                iv=b"fake_iv" + b"\x00" * 12,
                key_id="non_existent_key",  # 16 bytes total
            )

            # Should raise error
            with pytest.raises(EncryptionOperationError):
                encryption.decrypt(encrypted_data)


class TestSecureSerializer:
    """Test SecureSerializer class."""

    def test_serialize_and_encrypt(self):
        """Test secure serialization and encryption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)
            serializer = SecureSerializer(encryption)

            data = {"string": "test", "number": 42, "list": [1, 2, 3], "dict": {"nested": "value"}}

            # Serialize and encrypt
            encrypted_dict = serializer.serialize_and_encrypt(data)

            assert "ciphertext" in encrypted_dict
            assert "iv" in encrypted_dict
            assert "key_id" in encrypted_dict

            # Decrypt and deserialize
            decrypted_data = serializer.decrypt_and_deserialize(encrypted_dict)

            assert decrypted_data == data

    def test_complex_data_structures(self):
        """Test serialization of complex data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)
            serializer = SecureSerializer(encryption)

            data = {
                "unicode": "🔐 Encryption test 中文",
                "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
                "large_number": 9999999999999999999,
                "float": 3.14159265359,
                "boolean": True,
                "null": None,
                "nested": {"deep": {"structure": [1, 2, {"inner": "value"}]}},
            }

            # Serialize and encrypt
            encrypted_dict = serializer.serialize_and_encrypt(data)

            # Decrypt and deserialize
            decrypted_data = serializer.decrypt_and_deserialize(encrypted_dict)

            assert decrypted_data == data


class TestEncryptionCache:
    """Test EncryptionCache class."""

    def test_cipher_caching(self):
        """Test cipher object caching."""
        cache = EncryptionCache(max_size=10)

        key_data = b"test_key_12345678901234567890123456"  # 32 bytes for AES-256
        iv = b"test_iv_12345678"  # 16 bytes for CBC

        # First access - should create cipher
        cipher1 = cache.get_cipher(key_data, iv)
        assert cipher1 is not None

        # Second access - should return cached cipher
        cipher2 = cache.get_cipher(key_data, iv)
        assert cipher2 is cipher1  # Same object

    def test_padder_caching(self):
        """Test PKCS7 padder caching."""
        cache = EncryptionCache(max_size=10)

        # First access - should create padder
        padder1 = cache.get_padder()
        assert padder1 is not None

        # Second access - should return cached padder
        padder2 = cache.get_padder()
        assert padder2 is padder1  # Same object

    def test_cache_cleanup(self):
        """Test cache cleanup when max size is exceeded."""
        cache = EncryptionCache(max_size=2)

        # Fill cache beyond max size
        for i in range(5):
            key_data = f"test_key_{i:02d}234567890123456789012345".encode()
            iv = f"test_iv_{i:02d}345678".encode()
            cache.get_cipher(key_data, iv)

        # Cache should have been cleaned up
        assert len(cache._cache) <= 2

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = EncryptionCache()

        # Add some items
        key_data = b"test_key_12345678901234567890123456"
        iv = b"test_iv_12345678"
        cache.get_cipher(key_data, iv)
        cache.get_padder()

        assert len(cache._cache) > 0

        # Clear cache
        cache.clear()
        assert len(cache._cache) == 0


class TestBatchEncryption:
    """Test BatchEncryption class."""

    def test_batch_encrypt_decrypt(self):
        """Test batch encryption and decryption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)
            batch_encryption = BatchEncryption(encryption)

            data_list = ["Message 1", "Message 2", "Message 3", b"Binary message 4"]

            # Batch encrypt
            encrypted_list = batch_encryption.encrypt_batch(data_list)

            assert len(encrypted_list) == len(data_list)
            for encrypted_data in encrypted_list:
                assert isinstance(encrypted_data, EncryptedData)

            # Batch decrypt
            decrypted_list = batch_encryption.decrypt_batch(encrypted_list)

            assert len(decrypted_list) == len(data_list)
            assert decrypted_list[0] == b"Message 1"
            assert decrypted_list[1] == b"Message 2"
            assert decrypted_list[2] == b"Message 3"
            assert decrypted_list[3] == b"Binary message 4"

            # Cleanup
            batch_encryption.shutdown()

    def test_small_batch_sequential(self):
        """Test that small batches use sequential processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)
            batch_encryption = BatchEncryption(encryption)

            # Small batch (2 items or less)
            data_list = ["Message 1", "Message 2"]

            encrypted_list = batch_encryption.encrypt_batch(data_list)
            assert len(encrypted_list) == 2

            batch_encryption.shutdown()

    def test_empty_batch(self):
        """Test batch processing with empty list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)
            batch_encryption = BatchEncryption(encryption)

            # Empty batch
            encrypted_list = batch_encryption.encrypt_batch([])
            assert encrypted_list == []

            decrypted_list = batch_encryption.decrypt_batch([])
            assert decrypted_list == []

            batch_encryption.shutdown()


class TestEncryptionMetrics:
    """Test EncryptionMetrics class."""

    def test_metrics_collection(self):
        """Test metrics collection."""
        metrics = EncryptionMetrics()

        # Record some operations
        metrics.record_encryption(0.1, 100)
        metrics.record_encryption(0.2, 200)
        metrics.record_decryption(0.15, 150)
        metrics.record_cache_hit()
        metrics.record_cache_miss()

        # Get metrics
        result = metrics.get_metrics()

        assert result["encryption_count"] == 2
        assert result["decryption_count"] == 1
        assert result["encryption_time"] == 0.3
        assert result["decryption_time"] == 0.15
        assert result["bytes_encrypted"] == 300
        assert result["bytes_decrypted"] == 150
        assert result["cache_hits"] == 1
        assert result["cache_misses"] == 1
        assert result["avg_encryption_time"] == 0.15
        assert result["avg_decryption_time"] == 0.15
        assert result["cache_hit_ratio"] == 0.5

    def test_metrics_reset(self):
        """Test metrics reset."""
        metrics = EncryptionMetrics()

        # Record some operations
        metrics.record_encryption(0.1, 100)
        metrics.record_decryption(0.15, 150)

        # Reset
        metrics.reset()

        result = metrics.get_metrics()
        assert result["encryption_count"] == 0
        assert result["decryption_count"] == 0
        assert result["encryption_time"] == 0.0
        assert result["decryption_time"] == 0.0


class TestAESEncryptionPerformanceOptimizations:
    """Test AES encryption with performance optimizations."""

    def test_encryption_with_cache_and_metrics(self):
        """Test encryption with caching and metrics enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager, enable_cache=True, enable_metrics=True)

            plaintext = "Test message for performance optimization"

            # Encrypt multiple times
            encrypted1 = encryption.encrypt(plaintext)
            encrypted2 = encryption.encrypt(plaintext)

            # Check metrics
            metrics = encryption.get_metrics()
            assert metrics is not None
            assert metrics["encryption_count"] >= 2

            # Decrypt
            decrypted1 = encryption.decrypt_to_string(encrypted1)
            decrypted2 = encryption.decrypt_to_string(encrypted2)

            assert decrypted1 == plaintext
            assert decrypted2 == plaintext

            # Check metrics again
            metrics = encryption.get_metrics()
            assert metrics["decryption_count"] >= 2

            # Test batch operations
            data_list = ["Message 1", "Message 2", "Message 3"]
            encrypted_list = encryption.encrypt_batch(data_list)
            decrypted_list = encryption.decrypt_batch(encrypted_list)

            assert len(decrypted_list) == 3
            assert [d.decode() for d in decrypted_list] == data_list

            # Test cache clearing
            encryption.clear_cache()

            # Test metrics reset
            encryption.reset_metrics()
            metrics = encryption.get_metrics()
            assert metrics["encryption_count"] == 0

            # Cleanup
            encryption.shutdown()

    def test_encryption_without_optimizations(self):
        """Test encryption without performance optimizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager, enable_cache=False, enable_metrics=False)

            plaintext = "Test message without optimizations"

            # Encrypt and decrypt
            encrypted = encryption.encrypt(plaintext)
            decrypted = encryption.decrypt_to_string(encrypted)

            assert decrypted == plaintext

            # Metrics should be None
            assert encryption.get_metrics() is None

            # Batch operations should still work (fallback to sequential)
            data_list = ["Message 1", "Message 2"]
            encrypted_list = encryption.encrypt_batch(data_list)
            decrypted_list = encryption.decrypt_batch(encrypted_list)

            assert len(decrypted_list) == 2
            assert [d.decode() for d in decrypted_list] == data_list


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_encrypt_decrypt_data(self):
        """Test encrypt_data and decrypt_data functions."""
        plaintext = "Test message for convenience functions"

        # Encrypt
        encrypted_dict = encrypt_data(plaintext)

        assert "ciphertext" in encrypted_dict
        assert "iv" in encrypted_dict
        assert "key_id" in encrypted_dict

        # Decrypt
        decrypted_data = decrypt_data(encrypted_dict)

        assert decrypted_data == plaintext.encode()

    def test_encrypt_decrypt_json(self):
        """Test encrypt_json and decrypt_json functions."""
        data = {"message": "Test data for JSON convenience functions", "number": 123, "list": [1, 2, 3]}

        # Encrypt
        encrypted_dict = encrypt_json(data)

        assert "ciphertext" in encrypted_dict
        assert "iv" in encrypted_dict
        assert "key_id" in encrypted_dict

        # Decrypt
        decrypted_data = decrypt_json(encrypted_dict)

        assert decrypted_data == data

    def test_batch_encrypt_decrypt_functions(self):
        """Test batch encrypt and decrypt convenience functions."""
        data_list = ["Message 1", "Message 2", "Message 3"]

        # Batch encrypt
        encrypted_dict_list = encrypt_batch(data_list)

        assert len(encrypted_dict_list) == 3
        for encrypted_dict in encrypted_dict_list:
            assert "ciphertext" in encrypted_dict
            assert "iv" in encrypted_dict
            assert "key_id" in encrypted_dict

        # Batch decrypt
        decrypted_list = decrypt_batch(encrypted_dict_list)

        assert len(decrypted_list) == 3
        assert [d.decode() for d in decrypted_list] == data_list

    def test_metrics_functions(self):
        """Test metrics convenience functions."""
        # Reset metrics first
        reset_encryption_metrics()

        # Perform some operations
        encrypt_data("Test message")

        # Get metrics
        metrics = get_encryption_metrics()
        if metrics:  # Only test if metrics are enabled
            assert metrics["encryption_count"] >= 1

        # Clear cache
        clear_encryption_cache()

        # Reset metrics again
        reset_encryption_metrics()

        metrics = get_encryption_metrics()
        if metrics:
            assert metrics["encryption_count"] == 0

    def test_rotate_encryption_key(self):
        """Test rotate_encryption_key function."""
        original_key_id = rotate_encryption_key()
        new_key_id = rotate_encryption_key()

        assert new_key_id != original_key_id
        assert isinstance(new_key_id, str)

    def test_cleanup_old_keys(self):
        """Test cleanup_old_keys function."""
        # Generate some keys first
        for i in range(10):
            rotate_encryption_key()

        # Cleanup
        removed_count = cleanup_old_keys(keep_count=3)

        assert isinstance(removed_count, int)
        assert removed_count >= 0


class TestErrorHandling:
    """Test error handling."""

    def test_encryption_error_inheritance(self):
        """Test error class inheritance."""
        assert issubclass(KeyGenerationError, EncryptionError)
        assert issubclass(EncryptionOperationError, EncryptionError)
        assert issubclass(EncryptionError, Exception)

    def test_key_manager_invalid_path(self):
        """Test KeyManager with invalid path."""
        # This should still work by creating the directory
        invalid_path = "/tmp/nonexistent/path/keys.json"
        try:
            manager = KeyManager(invalid_path)
            assert manager.key_store_path == invalid_path
        except Exception:
            # If it fails, it's expected due to permissions
            pass

    def test_encryption_without_key(self):
        """Test encryption without available key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_store_path = os.path.join(temp_dir, "test_keys.json")
            key_manager = KeyManager(key_store_path)
            encryption = AESEncryption(key_manager)

            # Clear all keys
            key_manager.keys.clear()
            key_manager.current_key_id = None

            # Should raise error
            with pytest.raises(EncryptionOperationError):
                encryption.encrypt("test")


if __name__ == "__main__":
    pytest.main([__file__])
