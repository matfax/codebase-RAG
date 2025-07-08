"""
Encryption utilities for cache data protection.

This module provides AES-256 encryption functionality for securing cached data,
with support for key generation, rotation, secure serialization, and performance optimizations.
"""

import base64
import json
import logging
import os
import secrets
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.padding import PKCS7

logger = logging.getLogger(__name__)

# Constants for encryption
AES_KEY_SIZE = 32  # 256 bits
AES_BLOCK_SIZE = 16  # 128 bits
SALT_SIZE = 16  # 128 bits
IV_SIZE = 16  # 128 bits
PBKDF2_ITERATIONS = 100000
KEY_ROTATION_INTERVAL = 86400 * 7  # 7 days in seconds


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata."""

    key_id: str
    key_data: bytes
    created_at: float
    expires_at: float | None = None
    algorithm: str = "AES-256-CBC"
    salt: bytes | None = None

    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def should_rotate(self) -> bool:
        """Check if the key should be rotated."""
        return time.time() > (self.created_at + KEY_ROTATION_INTERVAL)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert bytes to base64 for JSON serialization
        if self.key_data:
            data["key_data"] = base64.b64encode(self.key_data).decode("utf-8")
        if self.salt:
            data["salt"] = base64.b64encode(self.salt).decode("utf-8")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncryptionKey":
        """Create from dictionary."""
        if "key_data" in data and isinstance(data["key_data"], str):
            data["key_data"] = base64.b64decode(data["key_data"])
        if "salt" in data and isinstance(data["salt"], str):
            data["salt"] = base64.b64decode(data["salt"])
        return cls(**data)


@dataclass
class EncryptedData:
    """Represents encrypted data with metadata."""

    ciphertext: bytes
    iv: bytes
    key_id: str
    algorithm: str = "AES-256-CBC"
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode("utf-8"),
            "iv": base64.b64encode(self.iv).decode("utf-8"),
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EncryptedData":
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            iv=base64.b64decode(data["iv"]),
            key_id=data["key_id"],
            algorithm=data.get("algorithm", "AES-256-CBC"),
            created_at=data.get("created_at", time.time()),
        )


class EncryptionError(Exception):
    """Base exception for encryption-related errors."""

    pass


class KeyGenerationError(EncryptionError):
    """Exception raised when key generation fails."""

    pass


class EncryptionOperationError(EncryptionError):
    """Exception raised when encryption/decryption operations fail."""

    pass


class KeyManager:
    """Manages encryption keys with rotation and storage."""

    def __init__(self, key_store_path: str | None = None):
        """
        Initialize the key manager.

        Args:
            key_store_path: Path to store encryption keys. If None, uses environment variable.
        """
        self.key_store_path = key_store_path or os.getenv("ENCRYPTION_KEY_STORE_PATH", ".cache_keys")
        self.keys: dict[str, EncryptionKey] = {}
        self.current_key_id: str | None = None
        self._backend = default_backend()

        # Ensure key store directory exists
        Path(self.key_store_path).parent.mkdir(parents=True, exist_ok=True)

        # Load existing keys
        self._load_keys()

        # Generate initial key if none exists
        if not self.keys:
            self._generate_initial_key()

    def _load_keys(self):
        """Load keys from storage."""
        if os.path.exists(self.key_store_path):
            try:
                with open(self.key_store_path) as f:
                    data = json.load(f)

                for key_data in data.get("keys", []):
                    key = EncryptionKey.from_dict(key_data)
                    self.keys[key.key_id] = key

                self.current_key_id = data.get("current_key_id")
                logger.info(f"Loaded {len(self.keys)} encryption keys")

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to load encryption keys: {e}")
                # Create backup of corrupted file
                backup_path = f"{self.key_store_path}.backup.{int(time.time())}"
                os.rename(self.key_store_path, backup_path)
                logger.warning(f"Corrupted key file backed up to {backup_path}")

    def _save_keys(self):
        """Save keys to storage."""
        try:
            data = {"keys": [key.to_dict() for key in self.keys.values()], "current_key_id": self.current_key_id}

            # Write to temporary file first
            temp_path = f"{self.key_store_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            os.rename(temp_path, self.key_store_path)

            # Set secure permissions
            os.chmod(self.key_store_path, 0o600)

        except Exception as e:
            logger.error(f"Failed to save encryption keys: {e}")
            raise KeyGenerationError(f"Failed to save encryption keys: {e}")

    def _generate_initial_key(self):
        """Generate the initial encryption key."""
        key = self.generate_key()
        self.current_key_id = key.key_id
        self._save_keys()
        logger.info(f"Generated initial encryption key: {key.key_id}")

    def generate_key(self, key_id: str | None = None) -> EncryptionKey:
        """
        Generate a new encryption key.

        Args:
            key_id: Optional key ID. If None, generates a unique ID.

        Returns:
            EncryptionKey: The generated key.
        """
        try:
            if key_id is None:
                key_id = f"key_{secrets.token_hex(8)}_{int(time.time())}"

            # Generate random key
            key_data = secrets.token_bytes(AES_KEY_SIZE)

            # Generate salt for key derivation
            salt = secrets.token_bytes(SALT_SIZE)

            key = EncryptionKey(key_id=key_id, key_data=key_data, created_at=time.time(), salt=salt)

            self.keys[key_id] = key
            logger.info(f"Generated new encryption key: {key_id}")

            return key

        except Exception as e:
            logger.error(f"Failed to generate encryption key: {e}")
            raise KeyGenerationError(f"Failed to generate encryption key: {e}")

    def derive_key_from_password(self, password: str, salt: bytes | None = None) -> EncryptionKey:
        """
        Derive an encryption key from a password using PBKDF2.

        Args:
            password: The password to derive from.
            salt: Optional salt. If None, generates a new salt.

        Returns:
            EncryptionKey: The derived key.
        """
        try:
            if salt is None:
                salt = secrets.token_bytes(SALT_SIZE)

            kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=AES_KEY_SIZE, salt=salt, iterations=PBKDF2_ITERATIONS, backend=self._backend)

            key_data = kdf.derive(password.encode("utf-8"))
            key_id = f"derived_{secrets.token_hex(8)}_{int(time.time())}"

            key = EncryptionKey(key_id=key_id, key_data=key_data, created_at=time.time(), salt=salt)

            self.keys[key_id] = key
            logger.info(f"Derived encryption key from password: {key_id}")

            return key

        except Exception as e:
            logger.error(f"Failed to derive encryption key: {e}")
            raise KeyGenerationError(f"Failed to derive encryption key: {e}")

    def get_current_key(self) -> EncryptionKey | None:
        """Get the current encryption key."""
        if self.current_key_id and self.current_key_id in self.keys:
            return self.keys[self.current_key_id]
        return None

    def get_key(self, key_id: str) -> EncryptionKey | None:
        """Get a specific encryption key by ID."""
        return self.keys.get(key_id)

    def rotate_key(self) -> EncryptionKey:
        """
        Rotate the current encryption key.

        Returns:
            EncryptionKey: The new current key.
        """
        try:
            # Generate new key
            new_key = self.generate_key()

            # Update current key
            old_key_id = self.current_key_id
            self.current_key_id = new_key.key_id

            # Save changes
            self._save_keys()

            logger.info(f"Rotated encryption key from {old_key_id} to {new_key.key_id}")

            return new_key

        except Exception as e:
            logger.error(f"Failed to rotate encryption key: {e}")
            raise KeyGenerationError(f"Failed to rotate encryption key: {e}")

    def cleanup_expired_keys(self, keep_count: int = 5) -> int:
        """
        Clean up expired keys, keeping a minimum number for decryption.

        Args:
            keep_count: Minimum number of keys to keep.

        Returns:
            int: Number of keys removed.
        """
        if len(self.keys) <= keep_count:
            return 0

        # Sort keys by creation time (newest first)
        sorted_keys = sorted(self.keys.values(), key=lambda k: k.created_at, reverse=True)

        # Keep current key and most recent keys
        keys_to_keep = set()
        if self.current_key_id:
            keys_to_keep.add(self.current_key_id)

        for key in sorted_keys[:keep_count]:
            keys_to_keep.add(key.key_id)

        # Remove old keys
        keys_to_remove = set(self.keys.keys()) - keys_to_keep
        removed_count = 0

        for key_id in keys_to_remove:
            if key_id != self.current_key_id:  # Never remove current key
                del self.keys[key_id]
                removed_count += 1

        if removed_count > 0:
            self._save_keys()
            logger.info(f"Cleaned up {removed_count} expired encryption keys")

        return removed_count


class EncryptionCache:
    """Thread-safe cache for encryption objects to improve performance."""

    def __init__(self, max_size: int = 100):
        """
        Initialize encryption cache.

        Args:
            max_size: Maximum number of cached objects.
        """
        self.max_size = max_size
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._access_times: dict[str, float] = {}

    def get_cipher(self, key_data: bytes, iv: bytes) -> Cipher:
        """
        Get or create a cipher object with caching.

        Args:
            key_data: The encryption key.
            iv: The initialization vector.

        Returns:
            Cipher: The cipher object.
        """
        cache_key = f"cipher_{hash((key_data, iv))}"

        with self._lock:
            if cache_key in self._cache:
                self._access_times[cache_key] = time.time()
                return self._cache[cache_key]

            # Create new cipher
            cipher = Cipher(algorithms.AES(key_data), modes.CBC(iv), backend=default_backend())

            # Add to cache
            self._cache[cache_key] = cipher
            self._access_times[cache_key] = time.time()

            # Cleanup if necessary
            self._cleanup_if_needed()

            return cipher

    def get_padder(self) -> PKCS7:
        """
        Get or create a PKCS7 padder with caching.

        Returns:
            PKCS7: The padder object.
        """
        cache_key = "padder"

        with self._lock:
            if cache_key in self._cache:
                self._access_times[cache_key] = time.time()
                return self._cache[cache_key]

            # Create new padder
            padder = PKCS7(AES_BLOCK_SIZE * 8)

            # Add to cache
            self._cache[cache_key] = padder
            self._access_times[cache_key] = time.time()

            return padder

    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds max size."""
        if len(self._cache) <= self.max_size:
            return

        # Remove least recently used items
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])

        items_to_remove = len(self._cache) - self.max_size + 10  # Remove extra items

        for cache_key, _ in sorted_items[:items_to_remove]:
            self._cache.pop(cache_key, None)
            self._access_times.pop(cache_key, None)

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class BatchEncryption:
    """Optimized batch encryption operations."""

    def __init__(self, encryption: "AESEncryption", max_workers: int = 4):
        """
        Initialize batch encryption.

        Args:
            encryption: The AES encryption instance.
            max_workers: Maximum number of worker threads.
        """
        self.encryption = encryption
        self.max_workers = max_workers
        self._executor = None
        self._lock = threading.Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def encrypt_batch(self, data_list: list[str | bytes], key_id: str | None = None) -> list[EncryptedData]:
        """
        Encrypt multiple data items in parallel.

        Args:
            data_list: List of data items to encrypt.
            key_id: Optional key ID for encryption.

        Returns:
            List[EncryptedData]: List of encrypted data objects.
        """
        if not data_list:
            return []

        # For small batches, use sequential processing
        if len(data_list) <= 2:
            return [self.encryption.encrypt(data, key_id) for data in data_list]

        # Use parallel processing for larger batches
        executor = self._get_executor()

        futures = [executor.submit(self.encryption.encrypt, data, key_id) for data in data_list]

        return [future.result() for future in futures]

    def decrypt_batch(self, encrypted_data_list: list[EncryptedData]) -> list[bytes]:
        """
        Decrypt multiple data items in parallel.

        Args:
            encrypted_data_list: List of encrypted data objects.

        Returns:
            List[bytes]: List of decrypted data.
        """
        if not encrypted_data_list:
            return []

        # For small batches, use sequential processing
        if len(encrypted_data_list) <= 2:
            return [self.encryption.decrypt(data) for data in encrypted_data_list]

        # Use parallel processing for larger batches
        executor = self._get_executor()

        futures = [executor.submit(self.encryption.decrypt, data) for data in encrypted_data_list]

        return [future.result() for future in futures]

    def shutdown(self):
        """Shutdown the thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class EncryptionMetrics:
    """Performance metrics for encryption operations."""

    def __init__(self):
        """Initialize metrics collection."""
        self.encryption_count = 0
        self.decryption_count = 0
        self.encryption_time = 0.0
        self.decryption_time = 0.0
        self.bytes_encrypted = 0
        self.bytes_decrypted = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.Lock()

    def record_encryption(self, duration: float, data_size: int):
        """
        Record encryption operation metrics.

        Args:
            duration: Time taken for encryption.
            data_size: Size of data encrypted.
        """
        with self._lock:
            self.encryption_count += 1
            self.encryption_time += duration
            self.bytes_encrypted += data_size

    def record_decryption(self, duration: float, data_size: int):
        """
        Record decryption operation metrics.

        Args:
            duration: Time taken for decryption.
            data_size: Size of data decrypted.
        """
        with self._lock:
            self.decryption_count += 1
            self.decryption_time += duration
            self.bytes_decrypted += data_size

    def record_cache_hit(self):
        """Record cache hit."""
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss."""
        with self._lock:
            self.cache_misses += 1

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dict: Metrics data.
        """
        with self._lock:
            return {
                "encryption_count": self.encryption_count,
                "decryption_count": self.decryption_count,
                "encryption_time": self.encryption_time,
                "decryption_time": self.decryption_time,
                "bytes_encrypted": self.bytes_encrypted,
                "bytes_decrypted": self.bytes_decrypted,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "avg_encryption_time": (self.encryption_time / self.encryption_count if self.encryption_count > 0 else 0),
                "avg_decryption_time": (self.decryption_time / self.decryption_count if self.decryption_count > 0 else 0),
                "cache_hit_ratio": (
                    self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                ),
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.encryption_count = 0
            self.decryption_count = 0
            self.encryption_time = 0.0
            self.decryption_time = 0.0
            self.bytes_encrypted = 0
            self.bytes_decrypted = 0
            self.cache_hits = 0
            self.cache_misses = 0


class AESEncryption:
    """AES-256 encryption implementation with performance optimizations."""

    def __init__(self, key_manager: KeyManager | None = None, enable_cache: bool = True, enable_metrics: bool = True):
        """
        Initialize AES encryption.

        Args:
            key_manager: Optional key manager. If None, creates a new one.
            enable_cache: Whether to enable object caching for performance.
            enable_metrics: Whether to collect performance metrics.
        """
        self.key_manager = key_manager or KeyManager()
        self._backend = default_backend()

        # Performance optimizations
        self._cache = EncryptionCache() if enable_cache else None
        self._metrics = EncryptionMetrics() if enable_metrics else None
        self._batch_encryption = BatchEncryption(self) if enable_cache else None

    def encrypt(self, plaintext: str | bytes, key_id: str | None = None) -> EncryptedData:
        """
        Encrypt plaintext using AES-256-CBC with performance optimizations.

        Args:
            plaintext: The data to encrypt.
            key_id: Optional key ID. If None, uses current key.

        Returns:
            EncryptedData: The encrypted data with metadata.
        """
        start_time = time.time() if self._metrics else None

        try:
            # Get encryption key
            if key_id:
                key = self.key_manager.get_key(key_id)
                if not key:
                    raise EncryptionOperationError(f"Key not found: {key_id}")
            else:
                key = self.key_manager.get_current_key()
                if not key:
                    raise EncryptionOperationError("No encryption key available")

            # Convert string to bytes if needed
            if isinstance(plaintext, str):
                plaintext = plaintext.encode("utf-8")

            # Generate random IV
            iv = secrets.token_bytes(IV_SIZE)

            # Create cipher (with caching if enabled)
            if self._cache:
                cipher = self._cache.get_cipher(key.key_data, iv)
                if self._metrics:
                    self._metrics.record_cache_hit()
            else:
                cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv), backend=self._backend)
                if self._metrics:
                    self._metrics.record_cache_miss()

            encryptor = cipher.encryptor()

            # Apply PKCS7 padding (with caching if enabled)
            if self._cache:
                padder_class = self._cache.get_padder()
                padder = padder_class.padder()
            else:
                padder = PKCS7(AES_BLOCK_SIZE * 8).padder()

            padded_data = padder.update(plaintext) + padder.finalize()

            # Encrypt
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            # Record metrics
            if self._metrics and start_time:
                duration = time.time() - start_time
                self._metrics.record_encryption(duration, len(plaintext))

            return EncryptedData(ciphertext=ciphertext, iv=iv, key_id=key.key_id, algorithm="AES-256-CBC")

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionOperationError(f"Encryption failed: {e}")

    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """
        Decrypt encrypted data with performance optimizations.

        Args:
            encrypted_data: The encrypted data to decrypt.

        Returns:
            bytes: The decrypted plaintext.
        """
        start_time = time.time() if self._metrics else None

        try:
            # Get decryption key
            key = self.key_manager.get_key(encrypted_data.key_id)
            if not key:
                raise EncryptionOperationError(f"Decryption key not found: {encrypted_data.key_id}")

            # Create cipher (with caching if enabled)
            if self._cache:
                cipher = self._cache.get_cipher(key.key_data, encrypted_data.iv)
                if self._metrics:
                    self._metrics.record_cache_hit()
            else:
                cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(encrypted_data.iv), backend=self._backend)
                if self._metrics:
                    self._metrics.record_cache_miss()

            decryptor = cipher.decryptor()

            # Decrypt
            padded_plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()

            # Remove PKCS7 padding (with caching if enabled)
            if self._cache:
                padder_class = self._cache.get_padder()
                unpadder = padder_class.unpadder()
            else:
                unpadder = PKCS7(AES_BLOCK_SIZE * 8).unpadder()

            plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

            # Record metrics
            if self._metrics and start_time:
                duration = time.time() - start_time
                self._metrics.record_decryption(duration, len(plaintext))

            return plaintext

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionOperationError(f"Decryption failed: {e}")

    def decrypt_to_string(self, encrypted_data: EncryptedData) -> str:
        """
        Decrypt encrypted data to string.

        Args:
            encrypted_data: The encrypted data to decrypt.

        Returns:
            str: The decrypted plaintext as string.
        """
        plaintext = self.decrypt(encrypted_data)
        return plaintext.decode("utf-8")

    def encrypt_batch(self, data_list: list[str | bytes], key_id: str | None = None) -> list[EncryptedData]:
        """
        Encrypt multiple data items using batch processing.

        Args:
            data_list: List of data items to encrypt.
            key_id: Optional key ID for encryption.

        Returns:
            List[EncryptedData]: List of encrypted data objects.
        """
        if self._batch_encryption:
            return self._batch_encryption.encrypt_batch(data_list, key_id)
        else:
            # Fallback to sequential processing
            return [self.encrypt(data, key_id) for data in data_list]

    def decrypt_batch(self, encrypted_data_list: list[EncryptedData]) -> list[bytes]:
        """
        Decrypt multiple data items using batch processing.

        Args:
            encrypted_data_list: List of encrypted data objects.

        Returns:
            List[bytes]: List of decrypted data.
        """
        if self._batch_encryption:
            return self._batch_encryption.decrypt_batch(encrypted_data_list)
        else:
            # Fallback to sequential processing
            return [self.decrypt(data) for data in encrypted_data_list]

    def get_metrics(self) -> dict[str, Any] | None:
        """
        Get performance metrics.

        Returns:
            Optional[Dict]: Metrics data if enabled, None otherwise.
        """
        return self._metrics.get_metrics() if self._metrics else None

    def reset_metrics(self):
        """Reset performance metrics."""
        if self._metrics:
            self._metrics.reset()

    def clear_cache(self):
        """Clear the encryption cache."""
        if self._cache:
            self._cache.clear()

    def shutdown(self):
        """Shutdown and cleanup resources."""
        if self._batch_encryption:
            self._batch_encryption.shutdown()
        if self._cache:
            self._cache.clear()


class SecureSerializer:
    """Secure serialization with encryption."""

    def __init__(self, encryption: AESEncryption | None = None):
        """
        Initialize secure serializer.

        Args:
            encryption: Optional AES encryption instance.
        """
        self.encryption = encryption or AESEncryption()

    def serialize_and_encrypt(self, data: Any, key_id: str | None = None) -> dict[str, Any]:
        """
        Serialize data and encrypt it.

        Args:
            data: The data to serialize and encrypt.
            key_id: Optional key ID for encryption.

        Returns:
            Dict: Encrypted data dictionary.
        """
        try:
            # Serialize data to JSON
            json_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

            # Encrypt
            encrypted_data = self.encryption.encrypt(json_data, key_id)

            return encrypted_data.to_dict()

        except Exception as e:
            logger.error(f"Secure serialization failed: {e}")
            raise EncryptionOperationError(f"Secure serialization failed: {e}")

    def decrypt_and_deserialize(self, encrypted_dict: dict[str, Any]) -> Any:
        """
        Decrypt and deserialize data.

        Args:
            encrypted_dict: Encrypted data dictionary.

        Returns:
            Any: The deserialized data.
        """
        try:
            # Create encrypted data object
            encrypted_data = EncryptedData.from_dict(encrypted_dict)

            # Decrypt
            json_data = self.encryption.decrypt_to_string(encrypted_data)

            # Deserialize
            return json.loads(json_data)

        except Exception as e:
            logger.error(f"Secure deserialization failed: {e}")
            raise EncryptionOperationError(f"Secure deserialization failed: {e}")


# Global instances for convenience
_default_key_manager = None
_default_encryption = None
_default_serializer = None


def get_default_key_manager() -> KeyManager:
    """Get the default key manager instance."""
    global _default_key_manager
    if _default_key_manager is None:
        _default_key_manager = KeyManager()
    return _default_key_manager


def get_default_encryption() -> AESEncryption:
    """Get the default encryption instance."""
    global _default_encryption
    if _default_encryption is None:
        _default_encryption = AESEncryption(get_default_key_manager())
    return _default_encryption


def get_default_serializer() -> SecureSerializer:
    """Get the default secure serializer instance."""
    global _default_serializer
    if _default_serializer is None:
        _default_serializer = SecureSerializer(get_default_encryption())
    return _default_serializer


# Convenience functions
def encrypt_data(data: str | bytes, key_id: str | None = None) -> dict[str, Any]:
    """
    Encrypt data using the default encryption instance.

    Args:
        data: The data to encrypt.
        key_id: Optional key ID.

    Returns:
        Dict: Encrypted data dictionary.
    """
    encrypted_data = get_default_encryption().encrypt(data, key_id)
    return encrypted_data.to_dict()


def decrypt_data(encrypted_dict: dict[str, Any]) -> bytes:
    """
    Decrypt data using the default encryption instance.

    Args:
        encrypted_dict: Encrypted data dictionary.

    Returns:
        bytes: Decrypted data.
    """
    encrypted_data = EncryptedData.from_dict(encrypted_dict)
    return get_default_encryption().decrypt(encrypted_data)


def encrypt_json(data: Any, key_id: str | None = None) -> dict[str, Any]:
    """
    Serialize and encrypt JSON data.

    Args:
        data: The data to serialize and encrypt.
        key_id: Optional key ID.

    Returns:
        Dict: Encrypted data dictionary.
    """
    return get_default_serializer().serialize_and_encrypt(data, key_id)


def decrypt_json(encrypted_dict: dict[str, Any]) -> Any:
    """
    Decrypt and deserialize JSON data.

    Args:
        encrypted_dict: Encrypted data dictionary.

    Returns:
        Any: The deserialized data.
    """
    return get_default_serializer().decrypt_and_deserialize(encrypted_dict)


def rotate_encryption_key() -> str:
    """
    Rotate the current encryption key.

    Returns:
        str: The new key ID.
    """
    key = get_default_key_manager().rotate_key()
    return key.key_id


def cleanup_old_keys(keep_count: int = 5) -> int:
    """
    Clean up old encryption keys.

    Args:
        keep_count: Number of keys to keep.

    Returns:
        int: Number of keys removed.
    """
    return get_default_key_manager().cleanup_expired_keys(keep_count)


def encrypt_batch(data_list: list[str | bytes], key_id: str | None = None) -> list[dict[str, Any]]:
    """
    Encrypt multiple data items using batch processing.

    Args:
        data_list: List of data items to encrypt.
        key_id: Optional key ID.

    Returns:
        List[Dict]: List of encrypted data dictionaries.
    """
    encrypted_list = get_default_encryption().encrypt_batch(data_list, key_id)
    return [encrypted.to_dict() for encrypted in encrypted_list]


def decrypt_batch(encrypted_dict_list: list[dict[str, Any]]) -> list[bytes]:
    """
    Decrypt multiple data items using batch processing.

    Args:
        encrypted_dict_list: List of encrypted data dictionaries.

    Returns:
        List[bytes]: List of decrypted data.
    """
    encrypted_list = [EncryptedData.from_dict(d) for d in encrypted_dict_list]
    return get_default_encryption().decrypt_batch(encrypted_list)


def get_encryption_metrics() -> dict[str, Any] | None:
    """
    Get encryption performance metrics.

    Returns:
        Optional[Dict]: Metrics data if available.
    """
    return get_default_encryption().get_metrics()


def reset_encryption_metrics():
    """Reset encryption performance metrics."""
    get_default_encryption().reset_metrics()


def clear_encryption_cache():
    """Clear the encryption cache."""
    get_default_encryption().clear_cache()


def shutdown_encryption():
    """Shutdown encryption services and cleanup resources."""
    get_default_encryption().shutdown()
