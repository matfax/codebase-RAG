# Cache Configuration Guide

## Overview

This comprehensive guide covers all aspects of configuring the Query Caching Layer system, from basic setup to advanced performance tuning. Follow these guidelines to optimize your cache configuration for your specific use case.

## Configuration Architecture

### Configuration Hierarchy

```
Environment Variables (.env file)
        ↓
Default Configuration Values
        ↓
Runtime Configuration Validation
        ↓
Service-Specific Configuration
        ↓
Dynamic Configuration Updates
```

### Configuration Categories

1. **Core Cache Settings** - Basic cache behavior
2. **Redis Configuration** - Redis connection and performance
3. **Memory Cache Settings** - L1 cache configuration
4. **Cache Type Configuration** - Service-specific cache settings
5. **Security Settings** - Encryption and access control
6. **Performance Tuning** - Optimization parameters
7. **Monitoring and Metrics** - Observability configuration

## Basic Configuration

### Minimal Configuration (.env)

```bash
# Essential cache settings
CACHE_ENABLED=true
CACHE_LEVEL=BOTH  # Options: L1_MEMORY, L2_REDIS, BOTH

# Redis connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_secure_password

# Memory cache
MEMORY_CACHE_MAX_SIZE=1000
MEMORY_CACHE_MAX_MEMORY_MB=256

# Basic security
CACHE_ENCRYPTION_ENABLED=false
```

### Quick Start Configuration

```python
# config/cache_config.py
from src.config.cache_config import CacheConfig, CacheLevel

# Create basic configuration
config = CacheConfig(
    enabled=True,
    cache_level=CacheLevel.BOTH,
    redis=RedisConfig(
        host="localhost",
        port=6379,
        password="your_password"
    ),
    memory=MemoryCacheConfig(
        max_size=1000,
        max_memory_mb=256,
        ttl_seconds=3600
    )
)
```

## Environment Variable Reference

### Core Cache Settings

```bash
# Cache System Control
CACHE_ENABLED=true                    # Enable/disable entire cache system
CACHE_LEVEL=BOTH                      # L1_MEMORY, L2_REDIS, or BOTH
CACHE_WRITE_STRATEGY=WRITE_THROUGH    # WRITE_THROUGH, WRITE_BACK, WRITE_AROUND

# Global Cache Settings
CACHE_DEFAULT_TTL=3600                # Default TTL in seconds (1 hour)
CACHE_MAX_KEY_LENGTH=250              # Maximum cache key length
CACHE_KEY_PREFIX=codebase_rag         # Key namespace prefix

# Performance Settings
CACHE_BATCH_SIZE=100                  # Batch operation size
CACHE_PARALLEL_OPERATIONS=4           # Concurrent operations
CACHE_CONNECTION_POOL_SIZE=10         # Connection pool size
```

### Redis Configuration

```bash
# Connection Settings
REDIS_HOST=localhost                  # Redis server hostname
REDIS_PORT=6379                       # Redis server port
REDIS_PASSWORD=your_password          # Redis authentication password
REDIS_DB=0                           # Redis database number

# Connection Pool Settings
REDIS_MAX_CONNECTIONS=10              # Maximum connections in pool
REDIS_CONNECTION_TIMEOUT=5.0          # Connection timeout (seconds)
REDIS_SOCKET_TIMEOUT=5.0             # Socket timeout (seconds)
REDIS_RETRY_ON_TIMEOUT=true          # Retry on timeout
REDIS_MAX_RETRIES=3                  # Maximum retry attempts

# SSL/TLS Settings
REDIS_SSL_ENABLED=false              # Enable SSL/TLS
REDIS_SSL_CERT_PATH=/path/to/cert    # Client certificate path
REDIS_SSL_KEY_PATH=/path/to/key      # Client key path
REDIS_SSL_CA_CERT_PATH=/path/to/ca   # CA certificate path
```

### Memory Cache Configuration

```bash
# Memory Cache Size
MEMORY_CACHE_MAX_SIZE=1000           # Maximum number of entries
MEMORY_CACHE_MAX_MEMORY_MB=256       # Maximum memory usage (MB)
MEMORY_CACHE_TTL=3600                # Default TTL (seconds)

# Memory Cache Behavior
MEMORY_CACHE_EVICTION_POLICY=LRU     # LRU, LFU, FIFO, RANDOM
MEMORY_CACHE_CLEANUP_INTERVAL=300    # Cleanup interval (seconds)
```

### Cache Type Specific Configuration

```bash
# Embedding Cache (for ML embeddings)
EMBEDDING_CACHE_TTL=7200             # 2 hours - stable embeddings
EMBEDDING_CACHE_COMPRESSION=true     # Compress large embeddings
EMBEDDING_CACHE_ENCRYPTION=false     # Encrypt sensitive embeddings

# Search Cache (for search results)
SEARCH_CACHE_TTL=1800                # 30 minutes - fresh results
SEARCH_CACHE_MAX_SIZE=5000           # Large result sets
SEARCH_CACHE_COMPRESSION=true        # Compress search results

# Project Cache (for project metadata)
PROJECT_CACHE_TTL=3600               # 1 hour - stable metadata
PROJECT_CACHE_MAX_SIZE=1000          # Project information
PROJECT_CACHE_ENCRYPTION=false       # Encrypt sensitive project data

# File Cache (for parsed files)
FILE_CACHE_TTL=1800                  # 30 minutes - file changes
FILE_CACHE_MAX_SIZE=10000            # Large number of files
FILE_CACHE_COMPRESSION=true          # Compress parsed content
```

### Security Configuration

```bash
# Encryption Settings
CACHE_ENCRYPTION_ENABLED=false       # Enable data encryption
CACHE_ENCRYPTION_KEY=base64_key       # Base64 encoded encryption key
ENCRYPTION_KEY_STORE_PATH=.cache_keys # Key storage path
ENCRYPTION_KEY_ROTATION_INTERVAL=604800 # 7 days

# Access Control
CACHE_PROJECT_ISOLATION=true         # Enable project isolation
CACHE_SESSION_TIMEOUT=3600           # Session timeout (seconds)
CACHE_ACCESS_LOGGING=true            # Log cache access

# Security Audit
SECURITY_AUDIT_ENABLED=true          # Enable security auditing
SECURITY_AUDIT_LOG_PATH=/var/log/cache-security.log
FAILED_ACCESS_THRESHOLD=5            # Failed attempts before alert
FAILED_ACCESS_WINDOW=300             # Time window for failed attempts
```

### Monitoring and Metrics

```bash
# Metrics Collection
CACHE_METRICS_ENABLED=true           # Enable metrics collection
CACHE_HEALTH_CHECK_INTERVAL=60       # Health check interval (seconds)
CACHE_STATS_COLLECTION_INTERVAL=300  # Stats collection interval

# Performance Monitoring
CACHE_PERFORMANCE_MONITORING=true    # Enable performance monitoring
CACHE_LATENCY_THRESHOLD=50           # Latency alert threshold (ms)
CACHE_HIT_RATE_THRESHOLD=0.6         # Minimum acceptable hit rate

# Debugging
CACHE_DEBUG_MODE=false               # Enable debug mode
CACHE_LOG_LEVEL=INFO                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
CACHE_LOG_OPERATIONS=false           # Log individual cache operations
```

## Configuration Profiles

### Development Profile

```bash
# .env.development
CACHE_ENABLED=true
CACHE_LEVEL=BOTH
CACHE_DEBUG_MODE=true
CACHE_LOG_LEVEL=DEBUG

# Small development setup
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=dev_password

# Limited memory for development
MEMORY_CACHE_MAX_SIZE=100
MEMORY_CACHE_MAX_MEMORY_MB=64

# Short TTLs for development
CACHE_DEFAULT_TTL=300
EMBEDDING_CACHE_TTL=600
SEARCH_CACHE_TTL=300

# Disabled security for development
CACHE_ENCRYPTION_ENABLED=false
CACHE_PROJECT_ISOLATION=false
CACHE_ACCESS_LOGGING=false
```

### Testing Profile

```bash
# .env.testing
CACHE_ENABLED=true
CACHE_LEVEL=L1_MEMORY  # Memory only for tests
CACHE_DEBUG_MODE=true

# No Redis for testing
REDIS_HOST=localhost
REDIS_PORT=6380  # Different port

# Small memory footprint
MEMORY_CACHE_MAX_SIZE=50
MEMORY_CACHE_MAX_MEMORY_MB=32

# Very short TTLs for testing
CACHE_DEFAULT_TTL=60
EMBEDDING_CACHE_TTL=120
SEARCH_CACHE_TTL=60

# Minimal security for testing
CACHE_ENCRYPTION_ENABLED=false
CACHE_PROJECT_ISOLATION=true  # Test isolation
```

### Production Profile

```bash
# .env.production
CACHE_ENABLED=true
CACHE_LEVEL=BOTH
CACHE_DEBUG_MODE=false
CACHE_LOG_LEVEL=INFO

# Production Redis with SSL
REDIS_HOST=redis.production.com
REDIS_PORT=6380
REDIS_PASSWORD=secure_production_password
REDIS_SSL_ENABLED=true
REDIS_SSL_CERT_PATH=/etc/ssl/certs/redis-client.crt
REDIS_SSL_KEY_PATH=/etc/ssl/private/redis-client.key
REDIS_SSL_CA_CERT_PATH=/etc/ssl/certs/ca.crt

# Large memory allocation
MEMORY_CACHE_MAX_SIZE=10000
MEMORY_CACHE_MAX_MEMORY_MB=2048
REDIS_MAX_CONNECTIONS=50

# Production TTL settings
CACHE_DEFAULT_TTL=3600
EMBEDDING_CACHE_TTL=7200
SEARCH_CACHE_TTL=1800
PROJECT_CACHE_TTL=3600
FILE_CACHE_TTL=1800

# Full security enabled
CACHE_ENCRYPTION_ENABLED=true
CACHE_ENCRYPTION_KEY=your_base64_production_key
CACHE_PROJECT_ISOLATION=true
CACHE_ACCESS_LOGGING=true
SECURITY_AUDIT_ENABLED=true

# Production monitoring
CACHE_METRICS_ENABLED=true
CACHE_PERFORMANCE_MONITORING=true
CACHE_HEALTH_CHECK_INTERVAL=30
```

### High-Performance Profile

```bash
# .env.high-performance
CACHE_ENABLED=true
CACHE_LEVEL=BOTH
CACHE_WRITE_STRATEGY=WRITE_BACK  # Better performance

# Optimized Redis settings
REDIS_HOST=redis-cluster.internal
REDIS_PORT=6379
REDIS_MAX_CONNECTIONS=100
REDIS_CONNECTION_TIMEOUT=2.0
REDIS_SOCKET_TIMEOUT=2.0

# Large memory allocation
MEMORY_CACHE_MAX_SIZE=50000
MEMORY_CACHE_MAX_MEMORY_MB=8192
MEMORY_CACHE_CLEANUP_INTERVAL=60

# Optimized batch settings
CACHE_BATCH_SIZE=500
CACHE_PARALLEL_OPERATIONS=16
CACHE_CONNECTION_POOL_SIZE=50

# Longer TTLs for performance
EMBEDDING_CACHE_TTL=14400  # 4 hours
SEARCH_CACHE_TTL=3600      # 1 hour
PROJECT_CACHE_TTL=7200     # 2 hours

# Performance optimizations
CACHE_COMPRESSION_ENABLED=true
CACHE_COMPRESSION_ALGORITHM=lz4
```

## Advanced Configuration

### Dynamic Configuration Management

```python
class DynamicCacheConfig:
    """Dynamic cache configuration with runtime updates."""

    def __init__(self):
        self.config = CacheConfig.from_env()
        self.update_callbacks = []

    def register_update_callback(self, callback):
        """Register callback for configuration updates."""
        self.update_callbacks.append(callback)

    async def update_config(self, updates: dict):
        """Update configuration at runtime."""
        # Validate updates
        self._validate_updates(updates)

        # Apply updates
        for key, value in updates.items():
            setattr(self.config, key, value)

        # Notify callbacks
        for callback in self.update_callbacks:
            await callback(self.config, updates)

    def _validate_updates(self, updates: dict):
        """Validate configuration updates."""
        # Validate TTL values
        if 'default_ttl' in updates:
            if updates['default_ttl'] <= 0:
                raise ValueError("TTL must be positive")

        # Validate memory settings
        if 'memory_max_size' in updates:
            if updates['memory_max_size'] <= 0:
                raise ValueError("Memory size must be positive")

# Usage
config_manager = DynamicCacheConfig()

# Register update handler
async def handle_config_update(config, updates):
    if 'default_ttl' in updates:
        # Update cache TTL settings
        await cache_service.update_ttl_settings(updates['default_ttl'])

config_manager.register_update_callback(handle_config_update)

# Update configuration at runtime
await config_manager.update_config({'default_ttl': 7200})
```

### Environment-Specific Overrides

```python
class EnvironmentConfigManager:
    """Manage environment-specific configuration overrides."""

    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.base_config = CacheConfig.from_env()
        self.apply_environment_overrides()

    def apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        overrides = self._get_environment_overrides()

        for key, value in overrides.items():
            if hasattr(self.base_config, key):
                setattr(self.base_config, key, value)

    def _get_environment_overrides(self) -> dict:
        """Get environment-specific configuration overrides."""
        overrides = {
            'development': {
                'debug_mode': True,
                'log_level': 'DEBUG',
                'cache_level': CacheLevel.L1_MEMORY,
                'default_ttl': 300,
                'encryption_enabled': False
            },
            'testing': {
                'debug_mode': True,
                'log_level': 'DEBUG',
                'cache_level': CacheLevel.L1_MEMORY,
                'default_ttl': 60,
                'encryption_enabled': False,
                'metrics_enabled': False
            },
            'staging': {
                'debug_mode': False,
                'log_level': 'INFO',
                'cache_level': CacheLevel.BOTH,
                'default_ttl': 1800,
                'encryption_enabled': True,
                'project_isolation': True
            },
            'production': {
                'debug_mode': False,
                'log_level': 'WARNING',
                'cache_level': CacheLevel.BOTH,
                'default_ttl': 3600,
                'encryption_enabled': True,
                'project_isolation': True,
                'metrics_enabled': True,
                'health_check_interval': 30
            }
        }

        return overrides.get(self.environment, {})
```

### Configuration Validation

```python
class CacheConfigValidator:
    """Comprehensive cache configuration validation."""

    def __init__(self):
        self.validation_rules = {
            'redis': self._validate_redis_config,
            'memory': self._validate_memory_config,
            'security': self._validate_security_config,
            'performance': self._validate_performance_config
        }

    def validate_config(self, config: CacheConfig) -> list[str]:
        """Validate complete configuration and return warnings/errors."""
        issues = []

        for category, validator in self.validation_rules.items():
            try:
                category_issues = validator(config)
                issues.extend(category_issues)
            except Exception as e:
                issues.append(f"Validation error in {category}: {e}")

        return issues

    def _validate_redis_config(self, config: CacheConfig) -> list[str]:
        """Validate Redis configuration."""
        issues = []

        # Check connection settings
        if config.redis.connection_timeout <= 0:
            issues.append("Redis connection timeout must be positive")

        if config.redis.max_connections <= 0:
            issues.append("Redis max connections must be positive")

        # Check SSL configuration
        if config.redis.ssl_enabled:
            if not config.redis.ssl_cert_path:
                issues.append("SSL enabled but no certificate path provided")

            if config.redis.ssl_cert_path and not os.path.exists(config.redis.ssl_cert_path):
                issues.append(f"SSL certificate not found: {config.redis.ssl_cert_path}")

        # Security recommendations
        if not config.redis.password:
            issues.append("WARNING: Redis password not set - security risk")

        return issues

    def _validate_memory_config(self, config: CacheConfig) -> list[str]:
        """Validate memory configuration."""
        issues = []

        # Check memory limits
        if config.memory.max_memory_mb <= 0:
            issues.append("Memory cache max memory must be positive")

        if config.memory.max_size <= 0:
            issues.append("Memory cache max size must be positive")

        # Check for reasonable limits
        if config.memory.max_memory_mb > 8192:  # 8GB
            issues.append("WARNING: Memory cache allocation very large (>8GB)")

        if config.memory.cleanup_interval <= 0:
            issues.append("Memory cleanup interval must be positive")

        return issues

    def _validate_security_config(self, config: CacheConfig) -> list[str]:
        """Validate security configuration."""
        issues = []

        # Encryption validation
        if config.encryption_enabled:
            if not config.encryption_key:
                issues.append("Encryption enabled but no encryption key provided")

            if config.encryption_key and len(config.encryption_key) < 32:
                issues.append("WARNING: Encryption key appears to be too short")

        # Production security recommendations
        environment = os.getenv('ENVIRONMENT', 'development')
        if environment == 'production':
            if not config.encryption_enabled:
                issues.append("WARNING: Encryption disabled in production")

            if not config.project_isolation:
                issues.append("WARNING: Project isolation disabled in production")

            if not config.access_logging:
                issues.append("WARNING: Access logging disabled in production")

        return issues

    def _validate_performance_config(self, config: CacheConfig) -> list[str]:
        """Validate performance configuration."""
        issues = []

        # Check TTL settings
        if config.default_ttl <= 0:
            issues.append("Default TTL must be positive")

        # Check batch settings
        if config.batch_size <= 0:
            issues.append("Batch size must be positive")

        if config.parallel_operations <= 0:
            issues.append("Parallel operations must be positive")

        # Performance recommendations
        if config.batch_size > 1000:
            issues.append("WARNING: Large batch size may impact performance")

        if config.parallel_operations > 32:
            issues.append("WARNING: High parallel operations may overwhelm system")

        return issues

# Usage
validator = CacheConfigValidator()
config = CacheConfig.from_env()
issues = validator.validate_config(config)

if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

## Configuration Templates

### Docker Compose Configuration

```yaml
# docker-compose.cache.yml
version: '3.8'

services:
  redis-cache:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    networks:
      - cache_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  cache-monitor:
    image: redisinsight/redisinsight:latest
    container_name: cache-monitor
    ports:
      - "8001:8001"
    environment:
      - RIPORT=8001
      - RIHOST=0.0.0.0
    depends_on:
      - redis-cache
    networks:
      - cache_network
    restart: unless-stopped

networks:
  cache_network:
    driver: bridge

volumes:
  redis_data:
    driver: local
```

### Redis Configuration Template

```conf
# redis.conf
# Basic Configuration
port 6379
bind 0.0.0.0
protected-mode yes
requirepass ${REDIS_PASSWORD}

# Memory Management
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# Networking
tcp-keepalive 300
timeout 0
tcp-nodelay yes

# Performance
databases 16
lua-time-limit 5000

# Logging
loglevel notice
logfile ""
syslog-enabled no

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command SHUTDOWN SHUTDOWN_CACHE
```

### Kubernetes ConfigMap

```yaml
# k8s-cache-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cache-config
  namespace: codebase-rag
data:
  CACHE_ENABLED: "true"
  CACHE_LEVEL: "BOTH"
  CACHE_WRITE_STRATEGY: "WRITE_THROUGH"

  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  REDIS_MAX_CONNECTIONS: "50"
  REDIS_CONNECTION_TIMEOUT: "5.0"

  MEMORY_CACHE_MAX_SIZE: "10000"
  MEMORY_CACHE_MAX_MEMORY_MB: "1024"

  EMBEDDING_CACHE_TTL: "7200"
  SEARCH_CACHE_TTL: "1800"
  PROJECT_CACHE_TTL: "3600"

  CACHE_METRICS_ENABLED: "true"
  CACHE_HEALTH_CHECK_INTERVAL: "60"

---
apiVersion: v1
kind: Secret
metadata:
  name: cache-secrets
  namespace: codebase-rag
type: Opaque
data:
  REDIS_PASSWORD: <base64-encoded-password>
  CACHE_ENCRYPTION_KEY: <base64-encoded-encryption-key>
```

## Configuration Management Tools

### Configuration CLI Tool

```python
#!/usr/bin/env python3
"""Cache configuration management CLI tool."""

import argparse
import json
import sys
from pathlib import Path
from src.config.cache_config import CacheConfig

class CacheConfigCLI:
    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        parser = argparse.ArgumentParser(description='Cache Configuration Management')
        subparsers = parser.add_subparsers(dest='command', help='Commands')

        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate configuration')
        validate_parser.add_argument('--env-file', help='Environment file to validate')

        # Show command
        show_parser = subparsers.add_parser('show', help='Show current configuration')
        show_parser.add_argument('--format', choices=['json', 'yaml', 'env'], default='json')

        # Set command
        set_parser = subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key')
        set_parser.add_argument('value', help='Configuration value')

        # Generate command
        generate_parser = subparsers.add_parser('generate', help='Generate configuration template')
        generate_parser.add_argument('profile', choices=['dev', 'test', 'staging', 'prod'])
        generate_parser.add_argument('--output', help='Output file path')

        return parser

    def run(self, args=None):
        """Run the CLI tool."""
        args = self.parser.parse_args(args)

        if args.command == 'validate':
            return self._validate_config(args)
        elif args.command == 'show':
            return self._show_config(args)
        elif args.command == 'set':
            return self._set_config(args)
        elif args.command == 'generate':
            return self._generate_template(args)
        else:
            self.parser.print_help()
            return 1

    def _validate_config(self, args):
        """Validate configuration."""
        try:
            if args.env_file:
                # Load from specific env file
                import dotenv
                dotenv.load_dotenv(args.env_file)

            config = CacheConfig.from_env()
            validator = CacheConfigValidator()
            issues = validator.validate_config(config)

            if not issues:
                print("✓ Configuration is valid")
                return 0
            else:
                print("Configuration issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                return 1

        except Exception as e:
            print(f"Error validating configuration: {e}")
            return 1

    def _show_config(self, args):
        """Show current configuration."""
        try:
            config = CacheConfig.from_env()

            if args.format == 'json':
                print(json.dumps(config.to_dict(), indent=2))
            elif args.format == 'yaml':
                import yaml
                print(yaml.dump(config.to_dict(), default_flow_style=False))
            elif args.format == 'env':
                self._print_env_format(config)

            return 0

        except Exception as e:
            print(f"Error showing configuration: {e}")
            return 1

    def _generate_template(self, args):
        """Generate configuration template."""
        templates = {
            'dev': self._get_dev_template(),
            'test': self._get_test_template(),
            'staging': self._get_staging_template(),
            'prod': self._get_prod_template()
        }

        template = templates[args.profile]

        if args.output:
            with open(args.output, 'w') as f:
                f.write(template)
            print(f"Template written to {args.output}")
        else:
            print(template)

        return 0

if __name__ == '__main__':
    cli = CacheConfigCLI()
    sys.exit(cli.run())
```

### Configuration Monitoring

```python
class ConfigurationMonitor:
    """Monitor configuration changes and validate updates."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.current_config = self._load_config()
        self.change_callbacks = []

    def start_monitoring(self):
        """Start monitoring configuration file for changes."""
        import asyncio
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class ConfigChangeHandler(FileSystemEventHandler):
            def __init__(self, monitor):
                self.monitor = monitor

            def on_modified(self, event):
                if event.src_path == self.monitor.config_path:
                    asyncio.create_task(self.monitor._handle_config_change())

        observer = Observer()
        observer.schedule(
            ConfigChangeHandler(self),
            str(Path(self.config_path).parent),
            recursive=False
        )
        observer.start()

        return observer

    async def _handle_config_change(self):
        """Handle configuration file changes."""
        try:
            new_config = self._load_config()
            changes = self._detect_changes(self.current_config, new_config)

            if changes:
                # Validate new configuration
                validator = CacheConfigValidator()
                issues = validator.validate_config(new_config)

                if issues:
                    logger.warning(f"Configuration issues detected: {issues}")

                # Notify callbacks
                for callback in self.change_callbacks:
                    await callback(changes, new_config)

                self.current_config = new_config
                logger.info(f"Configuration updated: {len(changes)} changes")

        except Exception as e:
            logger.error(f"Error handling configuration change: {e}")
```

## Best Practices

### Configuration Security

1. **Environment Variables**
   - Use environment variables for sensitive data
   - Never commit passwords or keys to version control
   - Use secrets management systems in production

2. **Configuration Validation**
   - Always validate configuration on startup
   - Use type checking and range validation
   - Implement configuration schema validation

3. **Access Control**
   - Restrict access to configuration files
   - Use proper file permissions (600 for .env files)
   - Audit configuration changes

### Performance Optimization

1. **Memory Allocation**
   - Size memory cache based on available RAM
   - Monitor memory usage and adjust as needed
   - Use memory pressure indicators

2. **Connection Pooling**
   - Size connection pools based on concurrency needs
   - Monitor connection pool utilization
   - Adjust timeouts based on network conditions

3. **TTL Settings**
   - Set TTL based on data volatility
   - Use longer TTLs for stable data
   - Consider business requirements

### Operational Considerations

1. **Configuration Management**
   - Use version control for configuration files
   - Implement configuration deployment pipelines
   - Test configuration changes in staging first

2. **Monitoring**
   - Monitor configuration-related metrics
   - Set up alerts for configuration issues
   - Regular configuration audits

3. **Documentation**
   - Document all configuration parameters
   - Maintain environment-specific documentation
   - Keep configuration examples up-to-date

## Related Documentation

### Configuration and Management
- [Cache Deployment and Setup](cache-deployment-setup.md) - Deployment and initial setup guide
- [Cache Management and Maintenance](cache-management-maintenance.md) - Ongoing management procedures
- [Cache Monitoring and Metrics](cache-monitoring-metrics.md) - Monitoring setup and metrics reference
- [Cache Development and Extension](cache-development-extension.md) - Guide for developers extending the cache system

### Architecture and Design
- [Cache Architecture Overview](cache-architecture.md) - High-level architecture documentation
- [Cache Integration Patterns](cache-integration-patterns.md) - Detailed integration patterns
- [Cache Performance Optimization](cache-performance-optimization.md) - Performance optimization guide
- [Cache Security and Encryption](cache-security-encryption.md) - Security and encryption details
- [Cache Troubleshooting Guide](cache-troubleshooting.md) - Troubleshooting procedures

### Other Guides
- [Deployment Guide](../DEPLOYMENT_GUIDE.md) - Overall system deployment
- [Security Review](../SECURITY_REVIEW.md) - Security best practices
- [Architecture Deep Dive](../ARCHITECTURE_DEEP_DIVE.md) - In-depth system architecture
- [Best Practices](../BEST_PRACTICES.md) - General best practices
