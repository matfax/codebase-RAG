# Cache Security and Encryption Implementation

## Overview

This document provides comprehensive coverage of the security and encryption features implemented in the Query Caching Layer, including data protection mechanisms, access controls, and security best practices.

## Security Architecture

### Multi-Layer Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer Security                  │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │   Access      │ │   Session    │ │    API Key             │ │
│  │   Controls    │ │   Isolation  │ │    Authentication      │ │
│  └───────────────┘ └──────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Cache Layer Security                         │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │  Data         │ │   Project    │ │    Cache Key           │ │
│  │  Encryption   │ │   Isolation  │ │    Namespacing         │ │
│  └───────────────┘ └──────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Transport Layer Security                       │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │   Redis       │ │   Network    │ │    Certificate         │ │
│  │   AUTH        │ │   TLS/SSL    │ │    Management          │ │
│  └───────────────┘ └──────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                Infrastructure Security                         │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────────────┐ │
│  │   Network     │ │   Container  │ │    Host                │ │
│  │   Isolation   │ │   Security   │ │    Security            │ │
│  └───────────────┘ └──────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Encryption Implementation

### 1. AES-256-CBC Encryption

The cache system implements AES-256-CBC encryption for sensitive data protection:

```python
class AESEncryption:
    """
    AES-256-CBC encryption implementation for cache data.

    Features:
    - AES-256-CBC encryption with PKCS7 padding
    - Secure key generation and management
    - Performance-optimized with caching
    - Thread-safe operation
    """

    def __init__(self, key_manager: KeyManager, enable_cache: bool = True):
        self.key_manager = key_manager
        self.enable_cache = enable_cache
        self._cipher_cache = {}
        self._cache_lock = threading.Lock()

    async def encrypt(self, plaintext: str | bytes, key_id: str = None) -> EncryptedData:
        """
        Encrypt data using AES-256-CBC.

        Args:
            plaintext: Data to encrypt
            key_id: Specific key ID to use (uses current key if None)

        Returns:
            EncryptedData: Encrypted data with metadata
        """
        # Convert string to bytes if necessary
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')

        # Get encryption key
        key = self.key_manager.get_current_key() if key_id is None else self.key_manager.get_key(key_id)
        if not key:
            raise EncryptionOperationError("No encryption key available")

        # Generate random IV
        iv = secrets.token_bytes(IV_SIZE)

        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(iv),
            backend=default_backend()
        )

        # Pad data to block size
        padder = PKCS7(AES_BLOCK_SIZE * 8).padder()
        padded_data = padder.update(plaintext)
        padded_data += padder.finalize()

        # Encrypt
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        return EncryptedData(
            ciphertext=ciphertext,
            iv=iv,
            key_id=key.key_id,
            algorithm="AES-256-CBC"
        )

    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """
        Decrypt data using AES-256-CBC.

        Args:
            encrypted_data: Encrypted data to decrypt

        Returns:
            bytes: Decrypted plaintext data
        """
        # Get decryption key
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise EncryptionOperationError(f"Encryption key {encrypted_data.key_id} not found")

        # Create cipher
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(encrypted_data.iv),
            backend=default_backend()
        )

        # Decrypt
        decryptor = cipher.decryptor()
        padded_plaintext = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()

        # Remove padding
        unpadder = PKCS7(AES_BLOCK_SIZE * 8).unpadder()
        plaintext = unpadder.update(padded_plaintext)
        plaintext += unpadder.finalize()

        return plaintext
```

### 2. Key Management System

#### Key Generation and Storage

```python
class KeyManager:
    """
    Secure key management with rotation and storage.

    Features:
    - Automatic key generation
    - Secure key storage with file permissions
    - Key rotation with versioning
    - Multiple key support for migration
    """

    def generate_key(self, key_id: str = None) -> EncryptionKey:
        """Generate new encryption key."""
        if key_id is None:
            key_id = f"key_{int(time.time())}_{secrets.token_hex(8)}"

        # Generate random key
        key_data = secrets.token_bytes(AES_KEY_SIZE)

        # Generate salt for key derivation
        salt = secrets.token_bytes(SALT_SIZE)

        # Create key object
        key = EncryptionKey(
            key_id=key_id,
            key_data=key_data,
            created_at=time.time(),
            salt=salt
        )

        # Store key
        self.keys[key_id] = key
        if self.current_key_id is None:
            self.current_key_id = key_id

        # Persist to storage
        self._save_keys()

        logger.info(f"Generated new encryption key: {key_id}")
        return key

    def rotate_key(self) -> EncryptionKey:
        """Rotate to new encryption key."""
        old_key_id = self.current_key_id
        new_key = self.generate_key()
        self.current_key_id = new_key.key_id

        logger.info(f"Rotated encryption key from {old_key_id} to {new_key.key_id}")
        return new_key

    def cleanup_old_keys(self, retain_count: int = 3):
        """Cleanup old keys, retaining specified count."""
        if len(self.keys) <= retain_count:
            return

        # Sort keys by creation time
        sorted_keys = sorted(
            self.keys.values(),
            key=lambda k: k.created_at,
            reverse=True
        )

        # Keep most recent keys
        keys_to_keep = sorted_keys[:retain_count]
        keys_to_remove = sorted_keys[retain_count:]

        # Remove old keys
        for key in keys_to_remove:
            del self.keys[key.key_id]
            logger.info(f"Removed old encryption key: {key.key_id}")

        self._save_keys()
```

#### Key Derivation for Password-Based Keys

```python
def derive_key_from_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """
    Derive encryption key from password using PBKDF2.

    Args:
        password: Password string
        salt: Salt bytes (generated if None)

    Returns:
        tuple: (derived_key, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(SALT_SIZE)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend()
    )

    key = kdf.derive(password.encode('utf-8'))
    return key, salt
```

### 3. Encryption Performance Optimization

#### Encryption Caching

```python
class PerformanceOptimizedEncryption:
    """Encryption with performance optimizations."""

    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self._cipher_cache = {}
        self._encryption_metrics = {
            "encrypt_count": 0,
            "decrypt_count": 0,
            "cache_hits": 0,
            "total_time": 0
        }
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    async def encrypt_batch(self, data_list: list[str | bytes]) -> list[EncryptedData]:
        """Batch encryption for improved performance."""
        tasks = []

        # Create encryption tasks
        for data in data_list:
            task = asyncio.create_task(self.encrypt(data))
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks)
        return results

    def _get_cached_cipher(self, key_id: str, iv: bytes) -> Cipher:
        """Get cached cipher object for performance."""
        cache_key = f"{key_id}:{base64.b64encode(iv).decode()}"

        if cache_key in self._cipher_cache:
            self._encryption_metrics["cache_hits"] += 1
            return self._cipher_cache[cache_key]

        # Create new cipher
        key = self.key_manager.get_key(key_id)
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(iv),
            backend=default_backend()
        )

        # Cache with size limit
        if len(self._cipher_cache) < 100:
            self._cipher_cache[cache_key] = cipher

        return cipher
```

#### Compression Integration

```python
class EncryptionWithCompression:
    """Encryption with transparent compression."""

    def __init__(self, encryption: AESEncryption, compression_threshold: int = 1024):
        self.encryption = encryption
        self.compression_threshold = compression_threshold

    async def encrypt_with_compression(self, data: str | bytes) -> EncryptedData:
        """Encrypt data with optional compression."""
        if isinstance(data, str):
            data = data.encode('utf-8')

        # Compress if data is large enough
        original_size = len(data)
        if original_size > self.compression_threshold:
            import zlib
            compressed_data = zlib.compress(data, level=6)
            compression_ratio = len(compressed_data) / original_size

            # Use compression if it provides significant savings
            if compression_ratio < 0.9:  # 10% or better compression
                data = b"COMPRESSED:" + compressed_data

        # Encrypt the (possibly compressed) data
        return await self.encryption.encrypt(data)

    async def decrypt_with_decompression(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt and decompress data."""
        decrypted_data = await self.encryption.decrypt(encrypted_data)

        # Check if data was compressed
        if decrypted_data.startswith(b"COMPRESSED:"):
            import zlib
            compressed_data = decrypted_data[11:]  # Remove "COMPRESSED:" prefix
            return zlib.decompress(compressed_data)

        return decrypted_data
```

## Access Control and Isolation

### 1. Project-Based Isolation

```python
class ProjectIsolationManager:
    """Manages project-based cache isolation."""

    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.project_permissions = {}

    def generate_project_key(self, project_name: str, base_key: str) -> str:
        """Generate project-isolated cache key."""
        # Create deterministic but isolated key
        project_hash = hashlib.sha256(project_name.encode()).hexdigest()[:16]
        return f"project:{project_hash}:{base_key}"

    async def check_project_access(self, user_id: str, project_name: str) -> bool:
        """Check if user has access to project cache."""
        project_perms = self.project_permissions.get(project_name, {})
        user_perms = project_perms.get(user_id, {})

        return user_perms.get("read", False) or user_perms.get("admin", False)

    async def isolated_cache_operation(
        self,
        operation: str,
        project_name: str,
        user_id: str,
        key: str,
        value: Any = None
    ) -> Any:
        """Perform cache operation with project isolation."""
        # Check access
        if not await self.check_project_access(user_id, project_name):
            raise PermissionError(f"User {user_id} does not have access to project {project_name}")

        # Generate isolated key
        isolated_key = self.generate_project_key(project_name, key)

        # Perform operation
        if operation == "get":
            return await self.cache_service.get(isolated_key)
        elif operation == "set":
            return await self.cache_service.set(isolated_key, value)
        elif operation == "delete":
            return await self.cache_service.delete(isolated_key)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
```

### 2. Session-Based Security

```python
class SessionSecurityManager:
    """Manages session-based cache security."""

    def __init__(self, cache_service: BaseCacheService, encryption: AESEncryption):
        self.cache_service = cache_service
        self.encryption = encryption
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour

    def create_session(self, user_id: str, project_context: dict) -> str:
        """Create new user session."""
        session_id = secrets.token_urlsafe(32)

        self.active_sessions[session_id] = {
            "user_id": user_id,
            "project_context": project_context,
            "created_at": time.time(),
            "last_activity": time.time()
        }

        return session_id

    async def validate_session(self, session_id: str) -> dict:
        """Validate and refresh session."""
        if session_id not in self.active_sessions:
            raise PermissionError("Invalid session")

        session = self.active_sessions[session_id]
        current_time = time.time()

        # Check session timeout
        if current_time - session["last_activity"] > self.session_timeout:
            del self.active_sessions[session_id]
            raise PermissionError("Session expired")

        # Update last activity
        session["last_activity"] = current_time

        return session

    async def secure_cache_get(self, session_id: str, key: str) -> Any:
        """Secure cache get with session validation."""
        session = await self.validate_session(session_id)

        # Generate session-specific key
        session_key = self._generate_session_key(session["user_id"], key)

        # Get encrypted data
        encrypted_data = await self.cache_service.get(session_key)
        if encrypted_data is None:
            return None

        # Decrypt if data is encrypted
        if isinstance(encrypted_data, dict) and "ciphertext" in encrypted_data:
            encrypted_obj = EncryptedData.from_dict(encrypted_data)
            decrypted_data = await self.encryption.decrypt(encrypted_obj)
            return json.loads(decrypted_data.decode('utf-8'))

        return encrypted_data

    def _generate_session_key(self, user_id: str, base_key: str) -> str:
        """Generate session-specific cache key."""
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return f"session:{user_hash}:{base_key}"
```

## Security Configuration

### 1. Environment-Based Security Settings

```bash
# Cache Security Configuration
CACHE_ENCRYPTION_ENABLED=true
CACHE_ENCRYPTION_KEY=your_base64_encoded_key
ENCRYPTION_KEY_STORE_PATH=/secure/path/to/keys
ENCRYPTION_KEY_ROTATION_INTERVAL=604800  # 7 days

# Redis Security
REDIS_PASSWORD=your_secure_password
REDIS_SSL_ENABLED=true
REDIS_SSL_CERT_PATH=/path/to/redis-cert.pem
REDIS_SSL_KEY_PATH=/path/to/redis-key.pem
REDIS_SSL_CA_CERT_PATH=/path/to/ca-cert.pem

# Access Control
CACHE_PROJECT_ISOLATION=true
CACHE_SESSION_TIMEOUT=3600
CACHE_ACCESS_LOGGING=true

# Security Audit
SECURITY_AUDIT_ENABLED=true
SECURITY_AUDIT_LOG_PATH=/var/log/cache-security.log
FAILED_ACCESS_THRESHOLD=5
FAILED_ACCESS_WINDOW=300  # 5 minutes
```

### 2. Runtime Security Configuration

```python
class SecurityConfiguration:
    """Security configuration management."""

    def __init__(self):
        self.encryption_enabled = self._get_bool_env("CACHE_ENCRYPTION_ENABLED", True)
        self.project_isolation = self._get_bool_env("CACHE_PROJECT_ISOLATION", True)
        self.access_logging = self._get_bool_env("CACHE_ACCESS_LOGGING", True)
        self.audit_enabled = self._get_bool_env("SECURITY_AUDIT_ENABLED", True)

        # Validate security configuration
        self._validate_security_config()

    def _validate_security_config(self):
        """Validate security configuration."""
        if self.encryption_enabled and not os.getenv("CACHE_ENCRYPTION_KEY"):
            logger.warning("Encryption enabled but no encryption key provided")

        if not self.project_isolation:
            logger.warning("Project isolation disabled - security risk in multi-tenant environments")

        if not self.access_logging:
            logger.warning("Access logging disabled - reduced audit capabilities")

    def get_security_level(self) -> str:
        """Determine current security level."""
        if self.encryption_enabled and self.project_isolation and self.access_logging:
            return "high"
        elif self.encryption_enabled and self.project_isolation:
            return "medium"
        elif self.encryption_enabled or self.project_isolation:
            return "low"
        else:
            return "minimal"
```

## Security Audit and Logging

### 1. Access Logging

```python
class SecurityAuditLogger:
    """Security audit logging system."""

    def __init__(self, log_path: str = None):
        self.log_path = log_path or os.getenv("SECURITY_AUDIT_LOG_PATH", "/var/log/cache-security.log")
        self.failed_attempts = {}
        self.failed_threshold = int(os.getenv("FAILED_ACCESS_THRESHOLD", "5"))
        self.failed_window = int(os.getenv("FAILED_ACCESS_WINDOW", "300"))

        # Setup security logger
        self.security_logger = self._setup_security_logger()

    def log_access_attempt(
        self,
        user_id: str,
        operation: str,
        resource: str,
        success: bool,
        ip_address: str = None
    ):
        """Log cache access attempt."""
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "operation": operation,
            "resource": resource,
            "success": success,
            "ip_address": ip_address
        }

        self.security_logger.info(json.dumps(log_entry))

        # Track failed attempts
        if not success:
            self._track_failed_attempt(user_id, ip_address)

    def _track_failed_attempt(self, user_id: str, ip_address: str):
        """Track failed access attempts for rate limiting."""
        current_time = time.time()
        key = f"{user_id}:{ip_address}"

        if key not in self.failed_attempts:
            self.failed_attempts[key] = []

        # Add current attempt
        self.failed_attempts[key].append(current_time)

        # Remove old attempts outside the window
        self.failed_attempts[key] = [
            timestamp for timestamp in self.failed_attempts[key]
            if current_time - timestamp <= self.failed_window
        ]

        # Check if threshold exceeded
        if len(self.failed_attempts[key]) >= self.failed_threshold:
            self._trigger_security_alert(user_id, ip_address)

    def _trigger_security_alert(self, user_id: str, ip_address: str):
        """Trigger security alert for suspicious activity."""
        alert = {
            "alert_type": "SUSPICIOUS_ACTIVITY",
            "user_id": user_id,
            "ip_address": ip_address,
            "failed_attempts": len(self.failed_attempts.get(f"{user_id}:{ip_address}", [])),
            "timestamp": time.time()
        }

        self.security_logger.error(json.dumps(alert))

        # Additional alerting (email, webhook, etc.) can be added here
```

### 2. Data Leakage Prevention

```python
class DataLeakagePreventionManager:
    """Prevents data leakage between projects and users."""

    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.scan_patterns = [
            r"password",
            r"secret",
            r"key",
            r"token",
            r"credential"
        ]

    async def scan_cache_entry(self, key: str, value: Any) -> dict:
        """Scan cache entry for sensitive data."""
        findings = {
            "sensitive_data_detected": False,
            "patterns_found": [],
            "severity": "low"
        }

        # Convert value to string for scanning
        if isinstance(value, dict):
            scan_text = json.dumps(value, default=str).lower()
        else:
            scan_text = str(value).lower()

        # Scan for sensitive patterns
        for pattern in self.scan_patterns:
            if re.search(pattern, scan_text):
                findings["sensitive_data_detected"] = True
                findings["patterns_found"].append(pattern)

        # Determine severity
        if len(findings["patterns_found"]) > 2:
            findings["severity"] = "high"
        elif len(findings["patterns_found"]) > 0:
            findings["severity"] = "medium"

        return findings

    async def sanitize_cache_entry(self, value: Any) -> Any:
        """Sanitize cache entry by removing sensitive data."""
        if isinstance(value, dict):
            sanitized = {}
            for k, v in value.items():
                if any(pattern in k.lower() for pattern in self.scan_patterns):
                    sanitized[k] = "[REDACTED]"
                else:
                    sanitized[k] = v
            return sanitized

        return value
```

## SSL/TLS Configuration for Redis

### 1. Redis SSL Setup

```yaml
# docker-compose.cache-secure.yml
version: '3.8'
services:
  redis-cache:
    image: redis:7-alpine
    ports:
      - "6380:6380"
    volumes:
      - redis_data:/data
      - ./certs:/tls
      - ./redis-secure.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    networks:
      - cache_network

networks:
  cache_network:
    driver: bridge

volumes:
  redis_data:
```

### 2. Redis Secure Configuration

```conf
# redis-secure.conf
port 0
tls-port 6380

# TLS Configuration
tls-cert-file /tls/redis.crt
tls-key-file /tls/redis.key
tls-ca-cert-file /tls/ca.crt

# Authentication
requirepass your_secure_password

# Security Settings
protected-mode yes
bind 127.0.0.1
tcp-keepalive 300

# Memory Security
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Log Security Events
loglevel notice
logfile /var/log/redis/redis.log
```

## Security Best Practices

### 1. Development Security Guidelines

#### Secure Coding Practices
- Always validate input before caching
- Use parameterized cache keys to prevent injection
- Implement proper error handling without information leakage
- Regular security code reviews

#### Key Management
- Never hardcode encryption keys in source code
- Use environment variables or secure key stores
- Implement automatic key rotation
- Maintain key backup and recovery procedures

#### Access Control
- Implement principle of least privilege
- Use session-based access controls
- Regular access review and cleanup
- Multi-factor authentication for administrative access

### 2. Operational Security

#### Monitoring and Alerting
- Monitor failed authentication attempts
- Alert on unusual access patterns
- Track cache hit/miss ratios for anomalies
- Regular security audit reviews

#### Backup and Recovery
- Encrypted backups of cache data
- Secure key backup procedures
- Disaster recovery testing
- Data retention policies

#### Network Security
- Use VPNs for remote access
- Implement network segmentation
- Regular security patching
- Firewall configuration review

### 3. Compliance Considerations

#### Data Protection Regulations
- GDPR compliance for EU data
- CCPA compliance for California residents
- Industry-specific requirements (HIPAA, PCI-DSS)
- Regular compliance audits

#### Data Retention
- Implement data retention policies
- Automatic data expiration
- Secure data deletion procedures
- Audit trail maintenance

## Security Testing

### 1. Security Test Suite

```python
class CacheSecurityTester:
    """Security testing for cache implementation."""

    async def test_encryption_strength(self):
        """Test encryption implementation strength."""
        # Test key generation randomness
        keys = [generate_encryption_key() for _ in range(100)]
        assert len(set(keys)) == 100, "Key generation not random enough"

        # Test encryption output randomness
        plaintext = "test data"
        ciphertexts = [await encrypt(plaintext) for _ in range(100)]
        assert len(set(ciphertexts)) == 100, "Encryption not producing unique outputs"

    async def test_access_controls(self):
        """Test access control implementation."""
        # Test project isolation
        project_a_key = generate_project_key("project_a", "test_key")
        project_b_key = generate_project_key("project_b", "test_key")
        assert project_a_key != project_b_key, "Project isolation failed"

        # Test unauthorized access
        with pytest.raises(PermissionError):
            await isolated_cache_operation("get", "project_a", "unauthorized_user", "test_key")

    async def test_data_leakage_prevention(self):
        """Test data leakage prevention mechanisms."""
        # Test cross-project data isolation
        await cache_service.set("project:a:test", "project_a_data")
        result = await cache_service.get("project:b:test")
        assert result is None, "Cross-project data leakage detected"
```

### 2. Penetration Testing Guidelines

#### Authentication Testing
- Test for weak password policies
- Verify session management security
- Check for authentication bypass vulnerabilities
- Test multi-factor authentication implementation

#### Authorization Testing
- Verify access control implementation
- Test for privilege escalation vulnerabilities
- Check for insecure direct object references
- Validate project isolation boundaries

#### Data Protection Testing
- Verify encryption implementation
- Test key management security
- Check for data exposure in logs
- Validate secure data transmission

## Security Incident Response

### 1. Incident Detection

```python
class SecurityIncidentDetector:
    """Detect and respond to security incidents."""

    def __init__(self):
        self.anomaly_thresholds = {
            "failed_auth_rate": 10,  # per minute
            "unusual_access_pattern": 5,  # standard deviations
            "data_access_volume": 1000,  # entries per minute
        }

    async def detect_anomalies(self, metrics: dict) -> list:
        """Detect security anomalies in cache access patterns."""
        anomalies = []

        # Check failed authentication rate
        if metrics["failed_auth_per_minute"] > self.anomaly_thresholds["failed_auth_rate"]:
            anomalies.append({
                "type": "high_failed_auth_rate",
                "severity": "high",
                "details": metrics["failed_auth_per_minute"]
            })

        # Check unusual access patterns
        if metrics["access_pattern_deviation"] > self.anomaly_thresholds["unusual_access_pattern"]:
            anomalies.append({
                "type": "unusual_access_pattern",
                "severity": "medium",
                "details": metrics["access_pattern_deviation"]
            })

        return anomalies
```

### 2. Response Procedures

#### Immediate Response
1. Isolate affected systems
2. Preserve evidence and logs
3. Assess impact and scope
4. Notify relevant stakeholders

#### Investigation Process
1. Analyze attack vectors
2. Identify compromised data
3. Determine root cause
4. Document findings

#### Recovery Actions
1. Patch vulnerabilities
2. Rotate compromised keys
3. Update security controls
4. Monitor for continued threats

#### Post-Incident Review
1. Document lessons learned
2. Update security procedures
3. Improve detection capabilities
4. Conduct security training
