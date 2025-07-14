"""
Cache configuration management for the Codebase RAG MCP Server.

This module provides comprehensive configuration management for the cache system,
including environment variable handling, validation, and default values.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union


class CacheConfigError(Exception):
    """Custom exception for cache configuration errors."""

    pass


class CacheLevel(Enum):
    """Cache levels for multi-tier architecture."""

    L1_MEMORY = "L1_MEMORY"
    L2_REDIS = "L2_REDIS"
    BOTH = "BOTH"


class CachePolicy(Enum):
    """Cache eviction policies."""

    LRU = "LRU"
    LFU = "LFU"
    FIFO = "FIFO"
    RANDOM = "RANDOM"


class CacheWriteStrategy(Enum):
    """Cache write strategies."""

    WRITE_THROUGH = "WRITE_THROUGH"
    WRITE_BACK = "WRITE_BACK"
    WRITE_AROUND = "WRITE_AROUND"


@dataclass
class RedisConfig:
    """Redis-specific configuration."""

    host: str = "localhost"
    port: int = 6379
    password: str | None = None
    db: int = 0
    max_connections: int = 50
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    retry_on_timeout: bool = True
    retry_backoff_factor: float = 1.5
    max_retries: int = 3
    ssl_enabled: bool = False
    ssl_cert_path: str | None = None
    ssl_key_path: str | None = None
    ssl_ca_cert_path: str | None = None


@dataclass
class MemoryCacheConfig:
    """In-memory cache configuration."""

    max_size: int = 1000
    ttl_seconds: int = 3600
    eviction_policy: CachePolicy = CachePolicy.LRU
    cleanup_interval: int = 300
    max_memory_mb: int = 256


@dataclass
class CacheTypeConfig:
    """Configuration for specific cache types."""

    enabled: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    compression_enabled: bool = True
    encryption_enabled: bool = False

    # Type-specific settings
    embedding_cache_ttl: int = 7200  # 2 hours for embeddings
    search_cache_ttl: int = 1800  # 30 minutes for search results
    project_cache_ttl: int = 3600  # 1 hour for project data
    file_cache_ttl: int = 1800  # 30 minutes for file processing


@dataclass
class CacheConfig:
    """
    Comprehensive cache configuration for the Codebase RAG MCP Server.

    This class manages all cache-related configuration including Redis settings,
    memory cache settings, TTL values, and cache behavior parameters.
    """

    # Core cache settings
    enabled: bool = True
    cache_level: CacheLevel = CacheLevel.BOTH
    write_strategy: CacheWriteStrategy = CacheWriteStrategy.WRITE_THROUGH

    # Redis configuration
    redis: RedisConfig = field(default_factory=RedisConfig)

    # Memory cache configuration
    memory: MemoryCacheConfig = field(default_factory=MemoryCacheConfig)

    # Cache type configurations
    embedding_cache: CacheTypeConfig = field(default_factory=CacheTypeConfig)
    search_cache: CacheTypeConfig = field(default_factory=CacheTypeConfig)
    project_cache: CacheTypeConfig = field(default_factory=CacheTypeConfig)
    file_cache: CacheTypeConfig = field(default_factory=CacheTypeConfig)

    # Global cache settings
    default_ttl: int = 3600
    max_key_length: int = 250
    key_prefix: str = "codebase_rag"

    # Performance settings
    batch_size: int = 100
    parallel_operations: int = 4
    connection_pool_size: int = 10

    # Monitoring and metrics
    metrics_enabled: bool = True
    health_check_interval: int = 60
    stats_collection_interval: int = 300

    # Security settings
    encryption_key: str | None = None
    encryption_enabled: bool = False

    # Debugging and logging
    debug_mode: bool = False
    log_level: str = "INFO"
    log_cache_operations: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """
        Create cache configuration from environment variables.

        Returns:
            CacheConfig: Configuration instance with values from environment

        Raises:
            CacheConfigError: If configuration validation fails
        """
        config = cls()

        # Core cache settings
        config.enabled = cls._get_bool_env("CACHE_ENABLED", config.enabled)
        config.cache_level = cls._get_enum_env("CACHE_LEVEL", CacheLevel, config.cache_level)
        config.write_strategy = cls._get_enum_env("CACHE_WRITE_STRATEGY", CacheWriteStrategy, config.write_strategy)

        # Redis configuration
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = cls._get_int_env("REDIS_PORT", config.redis.port)
        config.redis.password = os.getenv("REDIS_PASSWORD", config.redis.password)
        config.redis.db = cls._get_int_env("REDIS_DB", config.redis.db)
        config.redis.max_connections = cls._get_int_env("REDIS_MAX_CONNECTIONS", config.redis.max_connections)
        config.redis.connection_timeout = cls._get_float_env("REDIS_CONNECTION_TIMEOUT", config.redis.connection_timeout)
        config.redis.socket_timeout = cls._get_float_env("REDIS_SOCKET_TIMEOUT", config.redis.socket_timeout)
        config.redis.retry_on_timeout = cls._get_bool_env("REDIS_RETRY_ON_TIMEOUT", config.redis.retry_on_timeout)
        config.redis.max_retries = cls._get_int_env("REDIS_MAX_RETRIES", config.redis.max_retries)
        config.redis.ssl_enabled = cls._get_bool_env("REDIS_SSL_ENABLED", config.redis.ssl_enabled)
        config.redis.ssl_cert_path = os.getenv("REDIS_SSL_CERT_PATH", config.redis.ssl_cert_path)
        config.redis.ssl_key_path = os.getenv("REDIS_SSL_KEY_PATH", config.redis.ssl_key_path)
        config.redis.ssl_ca_cert_path = os.getenv("REDIS_SSL_CA_CERT_PATH", config.redis.ssl_ca_cert_path)

        # Memory cache configuration
        config.memory.max_size = cls._get_int_env("MEMORY_CACHE_MAX_SIZE", config.memory.max_size)
        config.memory.ttl_seconds = cls._get_int_env("MEMORY_CACHE_TTL", config.memory.ttl_seconds)
        config.memory.eviction_policy = cls._get_enum_env("MEMORY_CACHE_EVICTION_POLICY", CachePolicy, config.memory.eviction_policy)
        config.memory.cleanup_interval = cls._get_int_env("MEMORY_CACHE_CLEANUP_INTERVAL", config.memory.cleanup_interval)
        config.memory.max_memory_mb = cls._get_int_env("MEMORY_CACHE_MAX_MEMORY_MB", config.memory.max_memory_mb)

        # Cache type configurations
        config.embedding_cache.ttl_seconds = cls._get_int_env("EMBEDDING_CACHE_TTL", config.embedding_cache.ttl_seconds)
        config.search_cache.ttl_seconds = cls._get_int_env("SEARCH_CACHE_TTL", config.search_cache.ttl_seconds)
        config.project_cache.ttl_seconds = cls._get_int_env("PROJECT_CACHE_TTL", config.project_cache.ttl_seconds)
        config.file_cache.ttl_seconds = cls._get_int_env("FILE_CACHE_TTL", config.file_cache.ttl_seconds)

        # Global cache settings
        config.default_ttl = cls._get_int_env("CACHE_DEFAULT_TTL", config.default_ttl)
        config.max_key_length = cls._get_int_env("CACHE_MAX_KEY_LENGTH", config.max_key_length)
        config.key_prefix = os.getenv("CACHE_KEY_PREFIX", config.key_prefix)

        # Performance settings
        config.batch_size = cls._get_int_env("CACHE_BATCH_SIZE", config.batch_size)
        config.parallel_operations = cls._get_int_env("CACHE_PARALLEL_OPERATIONS", config.parallel_operations)
        config.connection_pool_size = cls._get_int_env("CACHE_CONNECTION_POOL_SIZE", config.connection_pool_size)

        # Monitoring and metrics
        config.metrics_enabled = cls._get_bool_env("CACHE_METRICS_ENABLED", config.metrics_enabled)
        config.health_check_interval = cls._get_int_env("CACHE_HEALTH_CHECK_INTERVAL", config.health_check_interval)
        config.stats_collection_interval = cls._get_int_env("CACHE_STATS_COLLECTION_INTERVAL", config.stats_collection_interval)

        # Security settings
        config.encryption_key = os.getenv("CACHE_ENCRYPTION_KEY", config.encryption_key)
        config.encryption_enabled = cls._get_bool_env("CACHE_ENCRYPTION_ENABLED", config.encryption_enabled)

        # Debugging and logging
        config.debug_mode = cls._get_bool_env("CACHE_DEBUG_MODE", config.debug_mode)
        config.log_level = os.getenv("CACHE_LOG_LEVEL", config.log_level)
        config.log_cache_operations = cls._get_bool_env("CACHE_LOG_OPERATIONS", config.log_cache_operations)

        return config

    def _validate_config(self) -> None:
        """
        Validate the configuration settings.

        Raises:
            CacheConfigError: If configuration is invalid
        """
        errors = []

        # Validate Redis configuration
        if self.redis.port < 1 or self.redis.port > 65535:
            errors.append(f"Invalid Redis port: {self.redis.port}")

        if self.redis.connection_timeout <= 0:
            errors.append(f"Invalid Redis connection timeout: {self.redis.connection_timeout}")

        if self.redis.max_connections <= 0:
            errors.append(f"Invalid Redis max connections: {self.redis.max_connections}")

        if self.redis.max_retries < 0:
            errors.append(f"Invalid Redis max retries: {self.redis.max_retries}")

        # Validate SSL configuration
        if self.redis.ssl_enabled:
            if self.redis.ssl_cert_path and not Path(self.redis.ssl_cert_path).exists():
                errors.append(f"Redis SSL certificate not found: {self.redis.ssl_cert_path}")
            if self.redis.ssl_key_path and not Path(self.redis.ssl_key_path).exists():
                errors.append(f"Redis SSL key not found: {self.redis.ssl_key_path}")
            if self.redis.ssl_ca_cert_path and not Path(self.redis.ssl_ca_cert_path).exists():
                errors.append(f"Redis SSL CA certificate not found: {self.redis.ssl_ca_cert_path}")

        # Validate memory cache configuration
        if self.memory.max_size <= 0:
            errors.append(f"Invalid memory cache max size: {self.memory.max_size}")

        if self.memory.ttl_seconds <= 0:
            errors.append(f"Invalid memory cache TTL: {self.memory.ttl_seconds}")

        if self.memory.max_memory_mb <= 0:
            errors.append(f"Invalid memory cache max memory: {self.memory.max_memory_mb}")

        # Validate TTL values
        if self.default_ttl <= 0:
            errors.append(f"Invalid default TTL: {self.default_ttl}")

        # Validate performance settings
        if self.batch_size <= 0:
            errors.append(f"Invalid batch size: {self.batch_size}")

        if self.parallel_operations <= 0:
            errors.append(f"Invalid parallel operations: {self.parallel_operations}")

        if self.connection_pool_size <= 0:
            errors.append(f"Invalid connection pool size: {self.connection_pool_size}")

        # Validate key length
        if self.max_key_length <= 0:
            errors.append(f"Invalid max key length: {self.max_key_length}")

        # Validate encryption settings
        if self.encryption_enabled and not self.encryption_key:
            errors.append("Encryption enabled but no encryption key provided")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.log_level}. Must be one of {valid_log_levels}")

        if errors:
            raise CacheConfigError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))

    def get_cache_config(self, cache_type: str) -> CacheTypeConfig:
        """
        Get configuration for a specific cache type.

        Args:
            cache_type: The cache type (embedding, search, project, file)

        Returns:
            CacheTypeConfig: Configuration for the specified cache type

        Raises:
            CacheConfigError: If cache type is not supported
        """
        cache_configs = {
            "embedding": self.embedding_cache,
            "search": self.search_cache,
            "project": self.project_cache,
            "file": self.file_cache,
        }

        if cache_type not in cache_configs:
            raise CacheConfigError(f"Unsupported cache type: {cache_type}")

        return cache_configs[cache_type]

    def get_redis_url(self) -> str:
        """
        Get Redis connection URL.

        Returns:
            str: Redis connection URL
        """
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return asdict(self)

    @staticmethod
    def _get_int_env(key: str, default: int) -> int:
        """Get integer value from environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _get_float_env(key: str, default: float) -> float:
        """Get float value from environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            return default

    @staticmethod
    def _get_enum_env(key: str, enum_class: type, default: Any) -> Any:
        """Get enum value from environment variable."""
        value = os.getenv(key)
        if value:
            try:
                return enum_class(value.upper())
            except ValueError:
                return default
        return default


def get_cache_config() -> CacheConfig:
    """
    Get the cache configuration instance.

    Returns:
        CacheConfig: The cache configuration instance
    """
    return CacheConfig.from_env()


def validate_cache_config(config: CacheConfig | None = None) -> bool:
    """
    Validate cache configuration.

    Args:
        config: Cache configuration to validate (defaults to environment config)

    Returns:
        bool: True if configuration is valid

    Raises:
        CacheConfigError: If configuration is invalid
    """
    if config is None:
        config = get_cache_config()

    try:
        config._validate_config()
        return True
    except CacheConfigError:
        raise


# Global configuration instance
_cache_config: CacheConfig | None = None


def get_global_cache_config() -> CacheConfig:
    """
    Get the global cache configuration instance.

    Returns:
        CacheConfig: The global cache configuration instance
    """
    global _cache_config
    if _cache_config is None:
        _cache_config = CacheConfig.from_env()
    return _cache_config


def reset_global_cache_config() -> None:
    """Reset the global cache configuration (mainly for testing)."""
    global _cache_config
    _cache_config = None
