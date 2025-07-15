"""
Unit tests for configuration management - Wave 15.1.5
Tests configuration loading, validation, and dynamic updates.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.config.cache_config import (
    CacheConfig,
    ConfigValidationError,
    InvalidationConfig,
    PerformanceConfig,
    RedisConfig,
    SecurityConfig,
    load_cache_config,
    merge_configs,
    validate_config,
)
from src.config.dynamic_config import ConfigChangeEvent, ConfigSource, ConfigWatcher, DynamicConfigManager
from src.config.environment_config import EnvironmentConfig, get_env_config, load_env_file, validate_env_vars


class TestCacheConfiguration:
    """Test cache configuration management."""

    def test_default_cache_config(self):
        """Test default cache configuration values."""
        config = CacheConfig()

        assert config.enabled is True
        assert config.default_ttl == 3600
        assert config.max_memory_mb == 1024
        assert config.eviction_policy == "lru"
        assert config.namespace_separator == ":"

    def test_redis_config_validation(self):
        """Test Redis configuration validation."""
        # Valid config
        valid_config = RedisConfig(
            host="localhost", port=6379, password="secret", db=0, ssl=False, connection_timeout=10, max_connections=50
        )
        assert valid_config.is_valid() is True

        # Invalid port
        with pytest.raises(ConfigValidationError):
            RedisConfig(host="localhost", port=99999)

        # Invalid timeout
        with pytest.raises(ConfigValidationError):
            RedisConfig(host="localhost", port=6379, connection_timeout=-1)

    def test_security_config(self):
        """Test security configuration settings."""
        config = SecurityConfig(
            encryption_enabled=True,
            encryption_algorithm="AES-256-GCM",
            key_rotation_interval=86400,
            audit_enabled=True,
            audit_retention_days=90,
            access_control_enabled=True,
        )

        assert config.encryption_enabled is True
        assert config.requires_encryption() is True
        assert config.key_rotation_interval == 86400

        # Test with disabled encryption
        config.encryption_enabled = False
        assert config.requires_encryption() is False

    def test_performance_config(self):
        """Test performance configuration settings."""
        config = PerformanceConfig(
            batch_size=100,
            pipeline_enabled=True,
            compression_enabled=True,
            compression_threshold=1024,
            async_operations=True,
            connection_pool_size=20,
            retry_attempts=3,
            retry_delay=0.1,
        )

        assert config.should_compress(2048) is True
        assert config.should_compress(512) is False
        assert config.get_retry_config() == {"attempts": 3, "delay": 0.1, "backoff": "exponential"}

    def test_invalidation_config(self):
        """Test invalidation configuration settings."""
        config = InvalidationConfig(
            cascade_enabled=True,
            max_cascade_depth=5,
            batch_invalidation_size=1000,
            file_monitoring_enabled=True,
            file_monitoring_interval=1.0,
            debounce_interval=0.5,
            pattern_cache_enabled=True,
        )

        assert config.cascade_enabled is True
        assert config.max_cascade_depth == 5
        assert config.should_debounce(0.3) is True  # Less than debounce interval
        assert config.should_debounce(1.0) is False


class TestConfigurationLoading:
    """Test configuration loading from various sources."""

    def test_load_from_file(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"cache": {"enabled": True, "default_ttl": 7200, "redis": {"host": "redis.example.com", "port": 6380}}}
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = load_cache_config(config_file)
            assert config.default_ttl == 7200
            assert config.redis.host == "redis.example.com"
            assert config.redis.port == 6380
        finally:
            os.unlink(config_file)

    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "CACHE_ENABLED": "true",
            "CACHE_DEFAULT_TTL": "1800",
            "REDIS_HOST": "env-redis-host",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": "env-secret",
            "CACHE_ENCRYPTION_ENABLED": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = load_cache_config()

            assert config.enabled is True
            assert config.default_ttl == 1800
            assert config.redis.host == "env-redis-host"
            assert config.redis.password == "env-secret"
            assert config.security.encryption_enabled is True

    def test_config_precedence(self):
        """Test configuration precedence (env > file > defaults)."""
        # Create config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"cache": {"default_ttl": 3600, "max_memory_mb": 512}}, f)
            config_file = f.name

        # Set environment variable
        with patch.dict(os.environ, {"CACHE_DEFAULT_TTL": "7200"}):
            try:
                config = load_cache_config(config_file)

                # Env should override file
                assert config.default_ttl == 7200
                # File should override default
                assert config.max_memory_mb == 512
            finally:
                os.unlink(config_file)

    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {"cache": {"enabled": True, "default_ttl": 3600, "redis": {"host": "localhost", "port": 6379}}}

        override_config = {"cache": {"default_ttl": 7200, "redis": {"host": "redis.example.com"}}}

        merged = merge_configs(base_config, override_config)

        assert merged["cache"]["enabled"] is True  # Preserved from base
        assert merged["cache"]["default_ttl"] == 7200  # Overridden
        assert merged["cache"]["redis"]["host"] == "redis.example.com"  # Overridden
        assert merged["cache"]["redis"]["port"] == 6379  # Preserved from base


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_validate_required_fields(self):
        """Test validation of required configuration fields."""
        # Valid config
        valid_config = {"cache": {"enabled": True, "redis": {"host": "localhost", "port": 6379}}}
        assert validate_config(valid_config) is True

        # Missing required field
        invalid_config = {"cache": {"enabled": True, "redis": {"port": 6379}}}  # Missing host
        with pytest.raises(ConfigValidationError):
            validate_config(invalid_config)

    def test_validate_value_ranges(self):
        """Test validation of configuration value ranges."""
        # Test TTL range
        with pytest.raises(ConfigValidationError):
            config = CacheConfig(default_ttl=-1)  # Negative TTL

        # Test memory limit
        with pytest.raises(ConfigValidationError):
            config = CacheConfig(max_memory_mb=0)  # Zero memory

        # Test connection pool size
        with pytest.raises(ConfigValidationError):
            config = PerformanceConfig(connection_pool_size=1001)  # Too large

    def test_validate_dependencies(self):
        """Test validation of configuration dependencies."""
        # Encryption requires key management
        config = SecurityConfig(encryption_enabled=True, key_rotation_interval=0)  # Invalid when encryption enabled

        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate()
        assert "key rotation" in str(exc_info.value)

    def test_validate_custom_rules(self):
        """Test custom validation rules."""
        config = CacheConfig()

        # Add custom validator
        def validate_namespace(config):
            if len(config.namespace_separator) > 1:
                raise ConfigValidationError("Namespace separator must be single character")

        config.add_validator(validate_namespace)

        # Test with invalid separator
        config.namespace_separator = "::"
        with pytest.raises(ConfigValidationError):
            config.validate()


class TestDynamicConfiguration:
    """Test dynamic configuration updates."""

    @pytest.fixture
    def dynamic_config(self):
        """Create dynamic configuration manager."""
        return DynamicConfigManager()

    def test_config_hot_reload(self, dynamic_config):
        """Test hot reload of configuration changes."""
        initial_config = CacheConfig(default_ttl=3600)
        dynamic_config.set_config(initial_config)

        # Register change handler
        changes_received = []

        def on_change(event: ConfigChangeEvent):
            changes_received.append(event)

        dynamic_config.on_change(on_change)

        # Update configuration
        updated_config = CacheConfig(default_ttl=7200)
        dynamic_config.update_config(updated_config)

        assert len(changes_received) == 1
        assert changes_received[0].field == "default_ttl"
        assert changes_received[0].old_value == 3600
        assert changes_received[0].new_value == 7200

    def test_config_watcher(self, dynamic_config):
        """Test configuration file watcher."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"cache": {"default_ttl": 3600}}, f)
            config_file = f.name

        try:
            watcher = ConfigWatcher(config_file)
            watcher.start()

            # Simulate file change
            with open(config_file, "w") as f:
                json.dump({"cache": {"default_ttl": 7200}}, f)

            # Wait for change detection
            import time

            time.sleep(0.1)

            changes = watcher.get_changes()
            assert len(changes) > 0

            watcher.stop()
        finally:
            os.unlink(config_file)

    def test_config_rollback(self, dynamic_config):
        """Test configuration rollback on error."""
        initial_config = CacheConfig(default_ttl=3600)
        dynamic_config.set_config(initial_config)

        # Enable rollback
        dynamic_config.enable_rollback()

        # Try invalid update
        try:
            invalid_config = CacheConfig(default_ttl=-1)
            dynamic_config.update_config(invalid_config)
        except ConfigValidationError:
            # Should rollback
            pass

        # Config should be rolled back
        current = dynamic_config.get_config()
        assert current.default_ttl == 3600

    def test_config_sources_priority(self, dynamic_config):
        """Test multiple configuration sources with priority."""
        # Add config sources
        dynamic_config.add_source(ConfigSource(name="file", priority=1, loader=lambda: {"default_ttl": 3600}))
        dynamic_config.add_source(ConfigSource(name="env", priority=2, loader=lambda: {"default_ttl": 7200}))
        dynamic_config.add_source(ConfigSource(name="api", priority=3, loader=lambda: {"default_ttl": 1800}))

        # Load from all sources
        config = dynamic_config.load_from_sources()

        # Highest priority should win
        assert config["default_ttl"] == 1800  # From API source


class TestEnvironmentConfiguration:
    """Test environment-specific configuration."""

    def test_environment_detection(self):
        """Test automatic environment detection."""
        # Test development
        with patch.dict(os.environ, {"ENV": "development"}):
            env = EnvironmentConfig.detect()
            assert env.name == "development"
            assert env.is_development is True

        # Test production
        with patch.dict(os.environ, {"ENV": "production"}):
            env = EnvironmentConfig.detect()
            assert env.name == "production"
            assert env.is_production is True

    def test_environment_specific_defaults(self):
        """Test environment-specific default values."""
        # Development defaults
        dev_config = EnvironmentConfig(name="development")
        dev_defaults = dev_config.get_defaults()
        assert dev_defaults["debug"] is True
        assert dev_defaults["cache_ttl"] == 300  # Shorter TTL in dev

        # Production defaults
        prod_config = EnvironmentConfig(name="production")
        prod_defaults = prod_config.get_defaults()
        assert prod_defaults["debug"] is False
        assert prod_defaults["cache_ttl"] == 3600  # Longer TTL in prod

    def test_environment_validation(self):
        """Test environment-specific validation rules."""
        # Production requires encryption
        prod_config = SecurityConfig(encryption_enabled=False)

        with patch.dict(os.environ, {"ENV": "production"}):
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_config({"security": prod_config}, environment="production")
            assert "encryption required in production" in str(exc_info.value)


class TestConfigurationExport:
    """Test configuration export and serialization."""

    def test_export_to_dict(self):
        """Test exporting configuration to dictionary."""
        config = CacheConfig(enabled=True, default_ttl=3600, redis=RedisConfig(host="localhost", port=6379))

        exported = config.to_dict()

        assert exported["enabled"] is True
        assert exported["default_ttl"] == 3600
        assert exported["redis"]["host"] == "localhost"

    def test_export_to_env_vars(self):
        """Test exporting configuration to environment variables."""
        config = CacheConfig(enabled=True, default_ttl=3600, redis=RedisConfig(host="redis.example.com", port=6380))

        env_vars = config.to_env_vars(prefix="CACHE_")

        assert env_vars["CACHE_ENABLED"] == "true"
        assert env_vars["CACHE_DEFAULT_TTL"] == "3600"
        assert env_vars["CACHE_REDIS_HOST"] == "redis.example.com"
        assert env_vars["CACHE_REDIS_PORT"] == "6380"

    def test_export_schema(self):
        """Test configuration schema export."""
        schema = CacheConfig.export_schema()

        assert "properties" in schema
        assert "enabled" in schema["properties"]
        assert schema["properties"]["enabled"]["type"] == "boolean"
        assert "default_ttl" in schema["properties"]
        assert schema["properties"]["default_ttl"]["type"] == "integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
