"""
Core cache service implementation for the Codebase RAG MCP Server.

This module provides the foundational cache service architecture with abstract base classes,
Redis connection management, multi-tier caching, and comprehensive cache operations.
"""

import asyncio
import logging
import sys
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from threading import Lock
from typing import Any, Optional, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from src.config.cache_config import CacheConfig, CacheLevel, CacheWriteStrategy, get_global_cache_config
from src.utils.cache_eviction_policies import EvictionOptimizer, EvictionPolicy, EvictionPolicyFactory
from src.utils.telemetry import get_telemetry_manager, trace_cache_method, trace_cache_operation


class CacheError(Exception):
    """Base exception for cache-related errors."""

    pass


class CacheConnectionError(CacheError):
    """Exception raised when cache connection fails."""

    pass


class CacheOperationError(CacheError):
    """Exception raised when cache operation fails."""

    pass


class CacheHealthStatus(Enum):
    """Cache health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


@dataclass
class CacheStats:
    """Cache statistics data structure."""

    hit_count: int = 0
    miss_count: int = 0
    set_count: int = 0
    delete_count: int = 0
    error_count: int = 0
    total_operations: int = 0
    last_reset: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_gets = self.hit_count + self.miss_count
        return self.hit_count / total_gets if total_gets > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    def reset(self) -> None:
        """Reset all statistics."""
        self.hit_count = 0
        self.miss_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.error_count = 0
        self.total_operations = 0
        self.last_reset = time.time()


@dataclass
class CacheHealthInfo:
    """Cache health information."""

    status: CacheHealthStatus
    redis_connected: bool = False
    redis_ping_time: float | None = None
    memory_usage: dict[str, Any] | None = None
    connection_pool_stats: dict[str, Any] | None = None
    last_error: str | None = None
    check_timestamp: float = field(default_factory=time.time)


class BaseCacheService(ABC):
    """
    Abstract base class for cache services.

    This class defines the interface that all cache services must implement,
    including basic cache operations, health monitoring, and statistics.
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the base cache service."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)
        self.stats = CacheStats()
        self._health_info = CacheHealthInfo(status=CacheHealthStatus.DISCONNECTED)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the cache service."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the cache service."""
        pass

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in the cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the cache."""
        pass

    @abstractmethod
    async def set_batch(self, items: dict[str, Any], ttl: int | None = None) -> dict[str, bool]:
        """Set multiple values in the cache."""
        pass

    @abstractmethod
    async def delete_batch(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values from the cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_health(self) -> CacheHealthInfo:
        """Get cache health information."""
        pass

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.stats.reset()

    def _update_stats(self, operation: str, success: bool = True) -> None:
        """Update cache statistics."""
        self.stats.total_operations += 1

        if operation == "get":
            if success:
                self.stats.hit_count += 1
            else:
                self.stats.miss_count += 1
        elif operation == "set":
            self.stats.set_count += 1
        elif operation == "delete":
            self.stats.delete_count += 1

        if not success:
            self.stats.error_count += 1


class RedisConnectionManager:
    """
    Redis connection manager with connection pooling and health monitoring.
    """

    def __init__(self, config: CacheConfig):
        """Initialize Redis connection manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._pool: ConnectionPool | None = None
        self._redis: redis.Redis | None = None
        self._health_check_task: asyncio.Task | None = None
        self._health_status = CacheHealthStatus.DISCONNECTED
        self._last_health_check = 0.0
        self._connection_attempts = 0
        self._max_connection_attempts = 5

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            # Create connection pool with SSL handling
            pool_kwargs = {
                "max_connections": self.config.redis.max_connections,
                "socket_timeout": self.config.redis.socket_timeout,
                "socket_connect_timeout": self.config.redis.connection_timeout,
                "retry_on_timeout": self.config.redis.retry_on_timeout,
                "retry_on_error": [ConnectionError, TimeoutError],
                "health_check_interval": self.config.health_check_interval,
            }

            # Add SSL parameters only if SSL is enabled
            if self.config.redis.ssl_enabled:
                pool_kwargs.update(
                    {
                        "ssl_cert_reqs": "required",
                        "ssl_ca_certs": self.config.redis.ssl_ca_cert_path,
                        "ssl_certfile": self.config.redis.ssl_cert_path,
                        "ssl_keyfile": self.config.redis.ssl_key_path,
                    }
                )

            self._pool = ConnectionPool.from_url(self.config.get_redis_url(), **pool_kwargs)

            # Create Redis client
            self._redis = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._test_connection()

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            self.logger.info("Redis connection manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis connection: {e}")
            self._health_status = CacheHealthStatus.UNHEALTHY
            raise CacheConnectionError(f"Failed to initialize Redis connection: {e}")

    async def shutdown(self) -> None:
        """Shutdown Redis connection manager."""
        try:
            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Close Redis connection
            if self._redis:
                await self._redis.aclose()

            # Close connection pool
            if self._pool:
                await self._pool.disconnect()

            self.logger.info("Redis connection manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during Redis shutdown: {e}")

    async def get_redis(self) -> redis.Redis:
        """Get Redis client instance."""
        if not self._redis:
            raise CacheConnectionError("Redis connection not initialized")
        return self._redis

    async def _test_connection(self) -> None:
        """Test Redis connection."""
        if not self._redis:
            raise CacheConnectionError("Redis client not initialized")

        try:
            start_time = time.time()
            await self._redis.ping()
            ping_time = time.time() - start_time

            self._health_status = CacheHealthStatus.HEALTHY
            self._connection_attempts = 0
            self.logger.debug(f"Redis connection test successful (ping: {ping_time:.3f}s)")

        except Exception as e:
            self._connection_attempts += 1
            self._health_status = CacheHealthStatus.UNHEALTHY
            self.logger.warning(f"Redis connection test failed: {e}")

            if self._connection_attempts >= self._max_connection_attempts:
                raise CacheConnectionError(f"Redis connection failed after {self._connection_attempts} attempts")

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._test_connection()
                self._last_health_check = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                self._health_status = CacheHealthStatus.UNHEALTHY

    async def get_health_info(self) -> CacheHealthInfo:
        """Get Redis health information."""
        redis_connected = self._health_status == CacheHealthStatus.HEALTHY
        ping_time = None
        memory_usage = None
        connection_pool_stats = None
        last_error = None

        try:
            if self._redis and redis_connected:
                # Test ping time
                start_time = time.time()
                await self._redis.ping()
                ping_time = time.time() - start_time

                # Get memory usage
                info = await self._redis.info("memory")
                memory_usage = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "N/A"),
                    "used_memory_peak": info.get("used_memory_peak", 0),
                    "used_memory_peak_human": info.get("used_memory_peak_human", "N/A"),
                }

                # Get connection pool stats
                if self._pool:
                    connection_pool_stats = {
                        "created_connections": self._pool.created_connections,
                        "available_connections": len(self._pool._available_connections),
                        "in_use_connections": len(self._pool._in_use_connections),
                        "max_connections": self._pool.max_connections,
                    }

        except Exception as e:
            last_error = str(e)
            self.logger.warning(f"Failed to get Redis health info: {e}")

        return CacheHealthInfo(
            status=self._health_status,
            redis_connected=redis_connected,
            redis_ping_time=ping_time,
            memory_usage=memory_usage,
            connection_pool_stats=connection_pool_stats,
            last_error=last_error,
        )

    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        return self._health_status == CacheHealthStatus.HEALTHY


class RedisCacheService(BaseCacheService):
    """
    Redis-based cache service implementation.

    This service provides async cache operations using Redis as the backend,
    with connection pooling, health monitoring, and comprehensive error handling.
    """

    def __init__(self, config: CacheConfig | None = None, redis_manager: RedisConnectionManager | None = None):
        """Initialize Redis cache service."""
        super().__init__(config)
        self.redis_manager = redis_manager or RedisConnectionManager(self.config)
        self._owns_redis_manager = redis_manager is None

    async def initialize(self) -> None:
        """Initialize Redis cache service."""
        if self._owns_redis_manager:
            await self.redis_manager.initialize()
        self.logger.info("Redis cache service initialized")

    async def shutdown(self) -> None:
        """Shutdown Redis cache service."""
        if self._owns_redis_manager:
            await self.redis_manager.shutdown()
        self.logger.info("Redis cache service shutdown")

    async def get(self, key: str) -> Any | None:
        """Get a value from Redis cache."""
        telemetry = get_telemetry_manager()

        with trace_cache_operation(
            operation="get",
            cache_name=self.config.key_prefix,
            cache_key=key,
            additional_attributes={"cache.type": "redis", "cache.level": "l2"},
        ):
            try:
                redis_client = await self.redis_manager.get_redis()

                # Add key prefix
                prefixed_key = f"{self.config.key_prefix}:{key}"

                # Get value from Redis
                value = await redis_client.get(prefixed_key)

                if value is not None:
                    # Deserialize value from Redis
                    try:
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        if isinstance(value, str):
                            # Try to deserialize as JSON, fallback to string
                            try:
                                import json

                                deserialized_value = json.loads(value)
                            except json.JSONDecodeError:
                                deserialized_value = value
                        else:
                            deserialized_value = value
                    except Exception:
                        deserialized_value = value

                    self._update_stats("get", success=True)
                    # Record telemetry hit
                    telemetry.record_cache_operation(
                        operation="get",
                        cache_name=self.config.key_prefix,
                        hit=True,
                        cache_size=len(value) if isinstance(value, Union[str, bytes]) else None,
                    )
                    return deserialized_value
                else:
                    self._update_stats("get", success=False)
                    # Record telemetry miss
                    telemetry.record_cache_operation(operation="get", cache_name=self.config.key_prefix, hit=False)
                    return None

            except Exception as e:
                self._update_stats("get", success=False)
                # Record telemetry error
                telemetry.record_cache_operation(operation="get", cache_name=self.config.key_prefix, error=True)
                self.logger.error(f"Failed to get key '{key}': {e}")
                raise CacheOperationError(f"Failed to get key '{key}': {e}")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in Redis cache."""
        telemetry = get_telemetry_manager()

        with trace_cache_operation(
            operation="set",
            cache_name=self.config.key_prefix,
            cache_key=key,
            additional_attributes={"cache.type": "redis", "cache.level": "l2"},
        ):
            try:
                redis_client = await self.redis_manager.get_redis()

                # Add key prefix
                prefixed_key = f"{self.config.key_prefix}:{key}"

                # Use provided TTL or default
                cache_ttl = ttl or self.config.default_ttl

                # Serialize value for Redis storage
                if isinstance(value, Union[dict, list]):
                    import json

                    serialized_value = json.dumps(value)
                elif isinstance(value, Union[str, bytes] | Union[int, float]):
                    serialized_value = value
                else:
                    import json

                    serialized_value = json.dumps(str(value))

                # Set value in Redis
                result = await redis_client.setex(prefixed_key, cache_ttl, serialized_value)

                self._update_stats("set", success=bool(result))

                # Record telemetry
                telemetry.record_cache_operation(
                    operation="set",
                    cache_name=self.config.key_prefix,
                    cache_size=len(value) if isinstance(value, Union[str, bytes]) else None,
                    additional_attributes={"cache.ttl": cache_ttl},
                )

                return bool(result)

            except Exception as e:
                self._update_stats("set", success=False)
                # Record telemetry error
                telemetry.record_cache_operation(operation="set", cache_name=self.config.key_prefix, error=True)
                self.logger.error(f"Failed to set key '{key}': {e}")
                raise CacheOperationError(f"Failed to set key '{key}': {e}")

    async def delete(self, key: str) -> bool:
        """Delete a value from Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Add key prefix
            prefixed_key = f"{self.config.key_prefix}:{key}"

            # Delete from Redis
            result = await redis_client.delete(prefixed_key)

            self._update_stats("delete", success=bool(result))
            return bool(result)

        except Exception as e:
            self._update_stats("delete", success=False)
            self.logger.error(f"Failed to delete key '{key}': {e}")
            raise CacheOperationError(f"Failed to delete key '{key}': {e}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Add key prefix
            prefixed_key = f"{self.config.key_prefix}:{key}"

            # Check existence in Redis
            result = await redis_client.exists(prefixed_key)
            return bool(result)

        except Exception as e:
            self.logger.error(f"Failed to check existence of key '{key}': {e}")
            raise CacheOperationError(f"Failed to check existence of key '{key}': {e}")

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Add key prefix to all keys
            prefixed_keys = [f"{self.config.key_prefix}:{key}" for key in keys]

            # Get values from Redis
            values = await redis_client.mget(prefixed_keys)

            # Build result dictionary
            result = {}
            for i, (key, value) in enumerate(zip(keys, values, strict=False)):
                if value is not None:
                    # Deserialize value from Redis
                    try:
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        if isinstance(value, str):
                            # Try to deserialize as JSON, fallback to string
                            try:
                                import json

                                deserialized_value = json.loads(value)
                            except json.JSONDecodeError:
                                deserialized_value = value
                        else:
                            deserialized_value = value
                    except Exception:
                        deserialized_value = value

                    result[key] = deserialized_value
                    self._update_stats("get", success=True)
                else:
                    self._update_stats("get", success=False)

            return result

        except Exception as e:
            self.logger.error(f"Failed to get batch keys: {e}")
            raise CacheOperationError(f"Failed to get batch keys: {e}")

    async def set_batch(self, items: dict[str, Any], ttl: int | None = None) -> dict[str, bool]:
        """Set multiple values in Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Use provided TTL or default
            cache_ttl = ttl or self.config.default_ttl

            # Prepare pipeline for batch operations
            async with redis_client.pipeline() as pipe:
                for key, value in items.items():
                    prefixed_key = f"{self.config.key_prefix}:{key}"

                    # Serialize value for Redis storage
                    if isinstance(value, Union[dict, list]):
                        import json

                        serialized_value = json.dumps(value)
                    elif isinstance(value, Union[str, bytes] | Union[int, float]):
                        serialized_value = value
                    else:
                        import json

                        serialized_value = json.dumps(str(value))

                    pipe.setex(prefixed_key, cache_ttl, serialized_value)

                # Execute pipeline
                results = await pipe.execute()

            # Build result dictionary
            result = {}
            for i, (key, success) in enumerate(zip(items.keys(), results, strict=False)):
                result[key] = bool(success)
                self._update_stats("set", success=bool(success))

            return result

        except Exception as e:
            self.logger.error(f"Failed to set batch items: {e}")
            raise CacheOperationError(f"Failed to set batch items: {e}")

    async def delete_batch(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values from Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Add key prefix to all keys
            prefixed_keys = [f"{self.config.key_prefix}:{key}" for key in keys]

            # Delete from Redis
            result = await redis_client.delete(*prefixed_keys)

            # Build result dictionary (Redis returns total count, not per-key)
            results = {}
            for key in keys:
                results[key] = result > 0  # Approximation
                self._update_stats("delete", success=result > 0)

            return results

        except Exception as e:
            self.logger.error(f"Failed to delete batch keys: {e}")
            raise CacheOperationError(f"Failed to delete batch keys: {e}")

    async def clear(self) -> bool:
        """Clear all cache entries with the configured prefix."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Get all keys with prefix
            pattern = f"{self.config.key_prefix}:*"
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                # Delete all keys
                result = await redis_client.delete(*keys)
                self.logger.info(f"Cleared {result} cache entries")
                return result > 0
            else:
                self.logger.info("No cache entries to clear")
                return True

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise CacheOperationError(f"Failed to clear cache: {e}")

    async def get_health(self) -> CacheHealthInfo:
        """Get Redis cache health information."""
        return await self.redis_manager.get_health_info()

    @asynccontextmanager
    async def get_redis_client(self):
        """Get Redis client context manager."""
        redis_client = await self.redis_manager.get_redis()
        try:
            yield redis_client
        finally:
            # Connection is managed by the pool, no need to close
            pass


@dataclass
class CacheEntry:
    """In-memory cache entry with metadata."""

    value: Any
    timestamp: float
    ttl: int | None = None
    access_count: int = 0
    size: int = 0

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.timestamp = time.time()
        self.access_count += 1

    @property
    def age(self) -> float:
        """Get age of the entry in seconds."""
        return time.time() - self.timestamp


class LRUMemoryCache:
    """
    Thread-safe memory cache with advanced eviction policies.

    This class provides an in-memory cache with configurable eviction policies,
    TTL support, size-based eviction, and performance optimization.
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 256,
        default_ttl: int = 3600,
        eviction_policy: EvictionPolicy = EvictionPolicy.ADAPTIVE,
    ):
        """Initialize memory cache with advanced eviction."""
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._total_size = 0
        self._stats = CacheStats()
        self.logger = logging.getLogger(__name__)

        # Advanced eviction policy
        self.eviction_policy_type = eviction_policy
        self.eviction_policy = EvictionPolicyFactory.create_policy(eviction_policy, max_size, self.logger)
        self.eviction_optimizer = EvictionOptimizer(self.logger)
        self._access_log: list[dict] = []
        self._optimization_interval = 10000  # Optimize every 10k operations
        self._operation_count = 0

    def get(self, key: str) -> Any | None:
        """Get a value from the cache using advanced eviction policy."""
        telemetry = get_telemetry_manager()

        with trace_cache_operation(
            operation="get", cache_name="memory_cache", cache_key=key, additional_attributes={"cache.type": "memory", "cache.level": "l1"}
        ):
            with self._lock:
                self._operation_count += 1

                # Log access for optimization
                self._log_access(key, "get")

                # Use eviction policy for get operation
                value = self.eviction_policy.get(key)

                if value is not None:
                    # Check if expired in our cache
                    if key in self._cache:
                        entry = self._cache[key]
                        if entry.is_expired():
                            self._remove_entry(key)
                            self.eviction_policy.remove(key)
                            self._stats.miss_count += 1
                            self._stats.total_operations += 1
                            telemetry.record_cache_operation(operation="get", cache_name="memory_cache", hit=False)
                            return None

                        # Update access
                        entry.touch()
                        self._stats.hit_count += 1
                        self._stats.total_operations += 1
                        telemetry.record_cache_operation(operation="get", cache_name="memory_cache", hit=True, cache_size=entry.size)
                        return entry.value

                # Cache miss
                self._stats.miss_count += 1
                self._stats.total_operations += 1
                telemetry.record_cache_operation(operation="get", cache_name="memory_cache", hit=False)

                # Periodic optimization
                if self._operation_count % self._optimization_interval == 0:
                    self._optimize_eviction_policy()

                return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in the cache using advanced eviction policy."""
        with self._lock:
            try:
                self._operation_count += 1

                # Log access for optimization
                self._log_access(key, "set")

                # Calculate size (rough estimation)
                size = self._estimate_size(value)

                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                    self.eviction_policy.remove(key)

                # Use eviction policy to handle capacity
                success = self.eviction_policy.put(key, value, size)

                if success:
                    # Create new entry in our cache
                    entry = CacheEntry(value=value, timestamp=time.time(), ttl=ttl or self.default_ttl, size=size)
                    self._cache[key] = entry
                    self._total_size += size

                    self._stats.set_count += 1
                    self._stats.total_operations += 1
                    return True
                else:
                    self._stats.error_count += 1
                    self._stats.total_operations += 1
                    return False

            except Exception as e:
                self.logger.error(f"Failed to set cache entry: {e}")
                self._stats.error_count += 1
                self._stats.total_operations += 1
                return False

    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats.delete_count += 1
                self._stats.total_operations += 1
                return True
            else:
                self._stats.total_operations += 1
                return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                return False

            return True

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_info(self) -> dict[str, Any]:
        """Get cache information."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_bytes": self._total_size,
                "memory_usage_mb": self._total_size / (1024 * 1024),
                "max_memory_mb": self.max_memory_mb,
                "hit_rate": self._stats.hit_rate,
                "miss_rate": self._stats.miss_rate,
                "total_operations": self._stats.total_operations,
            }

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache (internal method)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size -= entry.size

    def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict entries if needed to make room for new entry."""
        # Check size limit
        while len(self._cache) >= self.max_size or self._total_size + new_entry_size > self.max_memory_bytes:
            if not self._cache:
                break

            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        try:
            return sys.getsizeof(value)
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, Union[int, float]):
                return 8
            elif isinstance(value, Union[list, tuple]):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                return 1024  # Default estimate

    def _log_access(self, key: str, operation: str) -> None:
        """Log access for eviction policy optimization."""
        current_time = time.time()
        self._access_log.append({"key": key, "operation": operation, "timestamp": current_time})

        # Keep only recent access log (last 1000 operations)
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]

    def _optimize_eviction_policy(self) -> None:
        """Optimize eviction policy based on access patterns."""
        try:
            # Analyze workload characteristics
            workload_stats = self.eviction_optimizer.analyze_workload_characteristics(self._access_log)

            # Get optimization recommendations
            recommendations = self.eviction_optimizer.optimize_policy_parameters(self.eviction_policy, workload_stats)

            # Log recommendations
            if recommendations.get("recommendations"):
                self.logger.info(f"Cache optimization recommendations: {recommendations}")

                # Auto-apply some recommendations if beneficial
                for rec in recommendations["recommendations"]:
                    if rec["type"] == "policy" and rec.get("suggested_policy"):
                        suggested_policy = rec["suggested_policy"].upper()
                        if hasattr(EvictionPolicy, suggested_policy):
                            new_policy_type = getattr(EvictionPolicy, suggested_policy)
                            if new_policy_type != self.eviction_policy_type:
                                self._switch_eviction_policy(new_policy_type)
                                break

        except Exception as e:
            self.logger.error(f"Error optimizing eviction policy: {e}")

    def _switch_eviction_policy(self, new_policy_type: EvictionPolicy) -> None:
        """Switch to a new eviction policy."""
        try:
            # Create new policy
            new_policy = EvictionPolicyFactory.create_policy(new_policy_type, self.max_size, self.logger)

            # Migrate current entries
            for key, entry in self._cache.items():
                if not entry.is_expired():
                    new_policy.put(key, entry.value, entry.size)

            # Update policy
            old_policy = self.eviction_policy_type.value
            self.eviction_policy = new_policy
            self.eviction_policy_type = new_policy_type

            self.logger.info(f"Switched eviction policy from {old_policy} to {new_policy_type.value}")

        except Exception as e:
            self.logger.error(f"Error switching eviction policy: {e}")

    def get_eviction_stats(self) -> dict[str, Any]:
        """Get eviction policy statistics."""
        base_stats = self.eviction_policy.get_stats()
        base_stats.update(
            {
                "policy_type": self.eviction_policy_type.value,
                "available_policies": EvictionPolicyFactory.get_available_policies(),
                "optimization_enabled": True,
                "operation_count": self._operation_count,
                "access_log_size": len(self._access_log),
            }
        )
        return base_stats

    def configure_eviction_policy(self, policy_type: str = None, auto_optimize: bool = None) -> dict[str, Any]:
        """Configure eviction policy settings."""
        result = {"success": False, "message": ""}

        try:
            if policy_type:
                policy_enum = None
                for policy in EvictionPolicy:
                    if policy.value.upper() == policy_type.upper():
                        policy_enum = policy
                        break

                if policy_enum and policy_enum != self.eviction_policy_type:
                    self._switch_eviction_policy(policy_enum)
                    result["success"] = True
                    result["message"] = f"Switched to {policy_enum.value} eviction policy"
                elif policy_enum == self.eviction_policy_type:
                    result["success"] = True
                    result["message"] = f"Already using {policy_enum.value} eviction policy"
                else:
                    result["message"] = f"Unknown eviction policy: {policy_type}"

            if auto_optimize is not None:
                # Note: auto_optimize is always enabled in this implementation
                result["auto_optimize"] = True

            # Return current configuration
            result["current_config"] = {
                "policy": self.eviction_policy_type.value,
                "auto_optimize": True,
                "available_policies": EvictionPolicyFactory.get_available_policies(),
            }

            return result

        except Exception as e:
            result["message"] = f"Error configuring eviction policy: {e}"
            return result


class MultiTierCacheService(BaseCacheService):
    """
    Multi-tier cache service with L1 (memory) and L2 (Redis) layers.

    This service implements a comprehensive caching strategy with:
    - L1: Fast in-memory cache for frequently accessed data
    - L2: Persistent Redis cache for larger storage
    - Cache coherency between layers
    - Configurable write strategies (write-through, write-back)
    - Promotion and demotion logic
    - Advanced cache warming and optimization strategies
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize multi-tier cache service."""
        super().__init__(config)

        # Initialize L1 memory cache
        self.l1_cache = LRUMemoryCache(
            max_size=self.config.memory.max_size, max_memory_mb=self.config.memory.max_memory_mb, default_ttl=self.config.memory.ttl_seconds
        )

        # Initialize L2 Redis cache with shared connection manager
        self.l2_cache = None
        self._shared_redis_manager = None

        # Cache coherency tracking
        self._dirty_keys: set[str] = set()
        self._promotion_threshold = 3  # Access count threshold for promotion
        self._cleanup_task: asyncio.Task | None = None

        # Cache warming and optimization
        self._warmup_task: asyncio.Task | None = None
        self._access_patterns: dict[str, dict] = {}  # Track access patterns for intelligent warming
        self._warming_strategies = {
            "aggressive": self._aggressive_warmup,
            "conservative": self._conservative_warmup,
            "predictive": self._predictive_warmup,
            "adaptive": self._adaptive_warmup,
        }
        self._current_warming_strategy = "adaptive"
        self._warming_enabled = True
        self._warming_batch_size = 50
        self._warming_delay = 1.0  # Delay between warming batches

    async def initialize(self) -> None:
        """Initialize multi-tier cache service."""
        # Initialize L2 cache if enabled
        if self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
            self._shared_redis_manager = await get_redis_connection_manager()
            self.l2_cache = RedisCacheService(self.config, self._shared_redis_manager)
            await self.l2_cache.initialize()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Start cache warming task if enabled
        if self._warming_enabled:
            self._warmup_task = asyncio.create_task(self._warmup_loop())

        self.logger.info("Multi-tier cache service initialized")

    async def shutdown(self) -> None:
        """Shutdown multi-tier cache service."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel warmup task
        if self._warmup_task:
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass

        # Flush dirty keys if using write-back strategy
        if self.config.write_strategy == CacheWriteStrategy.WRITE_BACK:
            await self._flush_dirty_keys()

        # Shutdown L2 cache
        if self.l2_cache:
            await self.l2_cache.shutdown()

        self.logger.info("Multi-tier cache service shutdown")

    async def get(self, key: str) -> Any | None:
        """Get a value from the multi-tier cache."""
        try:
            # Track access pattern for cache warming optimization
            self._track_access_pattern(key)

            # Try L1 first
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                value = self.l1_cache.get(key)
                if value is not None:
                    self._update_stats("get", success=True)
                    return value

            # Try L2 if L1 miss
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                value = await self.l2_cache.get(key)
                if value is not None:
                    # Promote to L1 if frequently accessed
                    if self.config.cache_level == CacheLevel.BOTH:
                        self.l1_cache.set(key, value, self.config.memory.ttl_seconds)

                    self._update_stats("get", success=True)
                    return value

            # Cache miss
            self._update_stats("get", success=False)
            return None

        except Exception as e:
            self._update_stats("get", success=False)
            self.logger.error(f"Failed to get key '{key}': {e}")
            raise CacheOperationError(f"Failed to get key '{key}': {e}")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in the multi-tier cache."""
        try:
            success = True

            # Set in L1 if enabled
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                l1_success = self.l1_cache.set(key, value, ttl)
                success = success and l1_success

            # Set in L2 based on write strategy
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                if self.config.write_strategy == CacheWriteStrategy.WRITE_THROUGH:
                    # Write immediately to L2
                    l2_success = await self.l2_cache.set(key, value, ttl)
                    success = success and l2_success
                elif self.config.write_strategy == CacheWriteStrategy.WRITE_BACK:
                    # Mark as dirty for later write
                    self._dirty_keys.add(key)
                # WRITE_AROUND: Skip L2 cache

            self._update_stats("set", success=success)
            return success

        except Exception as e:
            self._update_stats("set", success=False)
            self.logger.error(f"Failed to set key '{key}': {e}")
            raise CacheOperationError(f"Failed to set key '{key}': {e}")

    async def delete(self, key: str) -> bool:
        """Delete a value from the multi-tier cache."""
        try:
            success = True

            # Delete from L1
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                l1_success = self.l1_cache.delete(key)
                success = success and l1_success

            # Delete from L2
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                l2_success = await self.l2_cache.delete(key)
                success = success and l2_success

            # Remove from dirty keys
            self._dirty_keys.discard(key)

            self._update_stats("delete", success=success)
            return success

        except Exception as e:
            self._update_stats("delete", success=False)
            self.logger.error(f"Failed to delete key '{key}': {e}")
            raise CacheOperationError(f"Failed to delete key '{key}': {e}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists in the multi-tier cache."""
        try:
            # Check L1 first
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                if self.l1_cache.exists(key):
                    return True

            # Check L2
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                if await self.l2_cache.exists(key):
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to check existence of key '{key}': {e}")
            raise CacheOperationError(f"Failed to check existence of key '{key}': {e}")

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from the multi-tier cache."""
        try:
            result = {}
            remaining_keys = list(keys)

            # Try L1 first
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                l1_hits = {}
                for key in remaining_keys[:]:
                    value = self.l1_cache.get(key)
                    if value is not None:
                        l1_hits[key] = value
                        result[key] = value
                        remaining_keys.remove(key)
                        self._update_stats("get", success=True)

            # Try L2 for remaining keys
            if self.l2_cache and remaining_keys and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                l2_results = await self.l2_cache.get_batch(remaining_keys)

                for key, value in l2_results.items():
                    result[key] = value
                    # Promote to L1 if both tiers enabled
                    if self.config.cache_level == CacheLevel.BOTH:
                        self.l1_cache.set(key, value, self.config.memory.ttl_seconds)
                    self._update_stats("get", success=True)

            # Update stats for misses
            for key in remaining_keys:
                if key not in result:
                    self._update_stats("get", success=False)

            return result

        except Exception as e:
            self.logger.error(f"Failed to get batch keys: {e}")
            raise CacheOperationError(f"Failed to get batch keys: {e}")

    async def set_batch(self, items: dict[str, Any], ttl: int | None = None) -> dict[str, bool]:
        """Set multiple values in the multi-tier cache."""
        try:
            result = {}

            # Set in L1 if enabled
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                for key, value in items.items():
                    l1_success = self.l1_cache.set(key, value, ttl)
                    result[key] = l1_success

            # Set in L2 based on write strategy
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                if self.config.write_strategy == CacheWriteStrategy.WRITE_THROUGH:
                    l2_results = await self.l2_cache.set_batch(items, ttl)
                    for key, success in l2_results.items():
                        result[key] = result.get(key, True) and success
                elif self.config.write_strategy == CacheWriteStrategy.WRITE_BACK:
                    # Mark all as dirty
                    self._dirty_keys.update(items.keys())

            # Update stats
            for key, success in result.items():
                self._update_stats("set", success=success)

            return result

        except Exception as e:
            self.logger.error(f"Failed to set batch items: {e}")
            raise CacheOperationError(f"Failed to set batch items: {e}")

    async def delete_batch(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values from the multi-tier cache."""
        try:
            result = {}

            # Delete from L1
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                for key in keys:
                    l1_success = self.l1_cache.delete(key)
                    result[key] = l1_success

            # Delete from L2
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                l2_results = await self.l2_cache.delete_batch(keys)
                for key, success in l2_results.items():
                    result[key] = result.get(key, True) and success

            # Remove from dirty keys
            for key in keys:
                self._dirty_keys.discard(key)

            # Update stats
            for key, success in result.items():
                self._update_stats("delete", success=success)

            return result

        except Exception as e:
            self.logger.error(f"Failed to delete batch keys: {e}")
            raise CacheOperationError(f"Failed to delete batch keys: {e}")

    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            success = True

            # Clear L1
            if self.config.cache_level in [CacheLevel.L1_MEMORY, CacheLevel.BOTH]:
                self.l1_cache.clear()

            # Clear L2
            if self.l2_cache and self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
                l2_success = await self.l2_cache.clear()
                success = success and l2_success

            # Clear dirty keys
            self._dirty_keys.clear()

            return success

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise CacheOperationError(f"Failed to clear cache: {e}")

    async def get_health(self) -> CacheHealthInfo:
        """Get multi-tier cache health information."""
        if self.l2_cache:
            return await self.l2_cache.get_health()
        else:
            return CacheHealthInfo(
                status=CacheHealthStatus.HEALTHY,
                redis_connected=False,
                redis_ping_time=None,
                memory_usage=self.l1_cache.get_info(),
            )

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.config.memory.cleanup_interval)

                # Cleanup expired entries from L1
                expired_count = self.l1_cache.cleanup_expired()
                if expired_count > 0:
                    self.logger.debug(f"Cleaned up {expired_count} expired L1 entries")

                # Flush dirty keys if using write-back strategy
                if self.config.write_strategy == CacheWriteStrategy.WRITE_BACK:
                    await self._flush_dirty_keys()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")

    async def _flush_dirty_keys(self) -> None:
        """Flush dirty keys to L2 cache."""
        if not self.l2_cache or not self._dirty_keys:
            return

        try:
            # Get values from L1 for dirty keys
            items_to_flush = {}
            for key in list(self._dirty_keys):
                value = self.l1_cache.get(key)
                if value is not None:
                    items_to_flush[key] = value

            if items_to_flush:
                # Flush to L2
                await self.l2_cache.set_batch(items_to_flush)
                self.logger.debug(f"Flushed {len(items_to_flush)} dirty keys to L2")

            # Clear dirty keys
            self._dirty_keys.clear()

        except Exception as e:
            self.logger.error(f"Failed to flush dirty keys: {e}")

    def get_tier_stats(self) -> dict[str, Any]:
        """Get statistics for each cache tier."""
        stats = {
            "l1_stats": self.l1_cache.get_stats(),
            "l1_info": self.l1_cache.get_info(),
            "l2_stats": self.l2_cache.get_stats() if self.l2_cache else None,
            "dirty_keys_count": len(self._dirty_keys),
            "write_strategy": self.config.write_strategy.value,
            "cache_level": self.config.cache_level.value,
        }
        return stats

    async def check_cache_coherency(self) -> dict[str, Any]:
        """
        Check cache coherency between L1 and L2 tiers.

        Returns:
            Dictionary with coherency check results
        """
        try:
            coherency_result = {
                "coherent": True,
                "l1_l2_consistent": True,
                "stale_entries": 0,
                "mismatched_keys": [],
                "checked_keys": 0,
                "timestamp": time.time(),
            }

            if not self.l2_cache:
                # No L2 cache, so coherency is trivial
                return coherency_result

            # Get sample of keys to check (for performance)
            l1_keys = list(self.l1_cache._cache.keys())[:100]  # Limit to 100 keys
            coherency_result["checked_keys"] = len(l1_keys)

            for key in l1_keys:
                try:
                    # Get values from both tiers
                    l1_entry = self.l1_cache.get(key)
                    l2_entry = await self.l2_cache.get(key)

                    # Compare values if both exist
                    if l1_entry is not None and l2_entry is not None:
                        if not self._entries_equal(l1_entry, l2_entry):
                            coherency_result["l1_l2_consistent"] = False
                            coherency_result["mismatched_keys"].append(key)
                            coherency_result["stale_entries"] += 1

                except Exception as e:
                    self.logger.warning(f"Error checking coherency for key {key}: {e}")
                    coherency_result["stale_entries"] += 1

            # Determine overall coherency
            coherency_result["coherent"] = coherency_result["l1_l2_consistent"] and coherency_result["stale_entries"] == 0

            return coherency_result

        except Exception as e:
            self.logger.error(f"Error checking cache coherency: {e}")
            return {"coherent": False, "l1_l2_consistent": False, "stale_entries": -1, "error": str(e), "timestamp": time.time()}

    def _entries_equal(self, entry1: Any, entry2: Any) -> bool:
        """Check if two cache entries are equal."""
        try:
            # Handle CacheEntry objects
            if hasattr(entry1, "data") and hasattr(entry2, "data"):
                return entry1.data == entry2.data
            else:
                return entry1 == entry2
        except Exception:
            return False

    # Cache Warming and Optimization Methods

    def _track_access_pattern(self, key: str) -> None:
        """Track access patterns for intelligent cache warming."""
        current_time = time.time()

        if key not in self._access_patterns:
            self._access_patterns[key] = {"access_count": 0, "last_access": current_time, "access_times": [], "frequency_score": 0.0}

        pattern = self._access_patterns[key]
        pattern["access_count"] += 1
        pattern["access_times"].append(current_time)
        pattern["last_access"] = current_time

        # Keep only recent access times (last 24 hours)
        recent_cutoff = current_time - 86400  # 24 hours
        pattern["access_times"] = [t for t in pattern["access_times"] if t > recent_cutoff]

        # Calculate frequency score
        if len(pattern["access_times"]) > 1:
            time_span = max(pattern["access_times"]) - min(pattern["access_times"])
            if time_span > 0:
                pattern["frequency_score"] = len(pattern["access_times"]) / time_span * 3600  # accesses per hour

        # Limit pattern storage to most recent 1000 keys
        if len(self._access_patterns) > 1000:
            oldest_keys = sorted(self._access_patterns.keys(), key=lambda k: self._access_patterns[k]["last_access"])[:100]
            for old_key in oldest_keys:
                del self._access_patterns[old_key]

    async def _warmup_loop(self) -> None:
        """Main cache warming loop."""
        while True:
            try:
                await asyncio.sleep(self.config.memory.cleanup_interval * 2)  # Run warming less frequently

                if self._warming_enabled and self._current_warming_strategy in self._warming_strategies:
                    await self._warming_strategies[self._current_warming_strategy]()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache warming loop error: {e}")

    async def _aggressive_warmup(self) -> None:
        """Aggressive warming strategy - warm most accessed keys."""
        try:
            # Get top accessed keys
            top_keys = self._get_top_accessed_keys(self._warming_batch_size * 2)

            # Warm keys in batches
            for i in range(0, len(top_keys), self._warming_batch_size):
                batch = top_keys[i : i + self._warming_batch_size]
                await self._warm_keys_batch(batch)
                await asyncio.sleep(self._warming_delay)

        except Exception as e:
            self.logger.error(f"Aggressive warmup error: {e}")

    async def _conservative_warmup(self) -> None:
        """Conservative warming strategy - warm only highly frequent keys."""
        try:
            # Get keys with high frequency scores
            frequent_keys = [
                key for key, pattern in self._access_patterns.items() if pattern["frequency_score"] > 10.0  # More than 10 accesses per hour
            ]

            # Sort by frequency and take top keys
            frequent_keys.sort(key=lambda k: self._access_patterns[k]["frequency_score"], reverse=True)
            batch = frequent_keys[: self._warming_batch_size // 2]

            if batch:
                await self._warm_keys_batch(batch)

        except Exception as e:
            self.logger.error(f"Conservative warmup error: {e}")

    async def _predictive_warmup(self) -> None:
        """Predictive warming strategy - predict which keys will be accessed."""
        try:
            current_time = time.time()
            predictive_keys = []

            # Analyze access patterns to predict future accesses
            for key, pattern in self._access_patterns.items():
                # Calculate time since last access
                time_since_access = current_time - pattern["last_access"]

                # Predict based on historical frequency and recency
                if len(pattern["access_times"]) >= 3:
                    avg_interval = self._calculate_average_interval(pattern["access_times"])

                    # If average interval suggests next access is due soon
                    if time_since_access >= avg_interval * 0.8:
                        predictive_keys.append((key, pattern["frequency_score"]))

            # Sort by prediction confidence (frequency score)
            predictive_keys.sort(key=lambda x: x[1], reverse=True)
            keys_to_warm = [key for key, _ in predictive_keys[: self._warming_batch_size]]

            if keys_to_warm:
                await self._warm_keys_batch(keys_to_warm)

        except Exception as e:
            self.logger.error(f"Predictive warmup error: {e}")

    async def _adaptive_warmup(self) -> None:
        """Adaptive warming strategy - combines multiple approaches."""
        try:
            current_hit_rate = self.l1_cache.get_stats().hit_rate

            # Adapt strategy based on current hit rate
            if current_hit_rate < 0.6:
                # Low hit rate - use aggressive warming
                await self._aggressive_warmup()
            elif current_hit_rate < 0.8:
                # Medium hit rate - use predictive warming
                await self._predictive_warmup()
            else:
                # High hit rate - use conservative warming
                await self._conservative_warmup()

        except Exception as e:
            self.logger.error(f"Adaptive warmup error: {e}")

    def _get_top_accessed_keys(self, count: int) -> list[str]:
        """Get top accessed keys sorted by access frequency."""
        # Sort keys by combined score of frequency and recency
        current_time = time.time()
        scored_keys = []

        for key, pattern in self._access_patterns.items():
            recency_score = 1.0 / (1.0 + (current_time - pattern["last_access"]) / 3600)  # Higher for recent
            combined_score = pattern["frequency_score"] * recency_score
            scored_keys.append((key, combined_score))

        scored_keys.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in scored_keys[:count]]

    def _calculate_average_interval(self, access_times: list[float]) -> float:
        """Calculate average interval between accesses."""
        if len(access_times) < 2:
            return 3600.0  # Default to 1 hour

        intervals = []
        for i in range(1, len(access_times)):
            intervals.append(access_times[i] - access_times[i - 1])

        return sum(intervals) / len(intervals)

    async def _warm_keys_batch(self, keys: list[str]) -> None:
        """Warm a batch of keys from L2 to L1 cache."""
        if not self.l2_cache or self.config.cache_level != CacheLevel.BOTH:
            return

        try:
            # Get values from L2 for keys not in L1
            keys_to_warm = []
            for key in keys:
                if not self.l1_cache.exists(key):
                    keys_to_warm.append(key)

            if keys_to_warm:
                # Batch get from L2
                l2_values = await self.l2_cache.get_batch(keys_to_warm)

                # Promote to L1
                for key, value in l2_values.items():
                    self.l1_cache.set(key, value, self.config.memory.ttl_seconds)

                self.logger.debug(f"Warmed {len(l2_values)} keys to L1 cache")

        except Exception as e:
            self.logger.error(f"Error warming keys batch: {e}")

    async def trigger_cache_warmup(self, strategy: str = None, keys: list[str] = None) -> dict[str, Any]:
        """
        Manually trigger cache warmup.

        Args:
            strategy: Warming strategy to use ('aggressive', 'conservative', 'predictive', 'adaptive')
            keys: Specific keys to warm (optional)

        Returns:
            Dictionary with warmup results
        """
        try:
            warmup_start = time.time()

            if keys:
                # Warm specific keys
                await self._warm_keys_batch(keys)
                return {"strategy": "manual", "keys_warmed": len(keys), "duration": time.time() - warmup_start, "success": True}
            else:
                # Use specified or current strategy
                strategy = strategy or self._current_warming_strategy
                if strategy in self._warming_strategies:
                    await self._warming_strategies[strategy]()
                    return {"strategy": strategy, "duration": time.time() - warmup_start, "success": True}
                else:
                    return {"strategy": strategy, "success": False, "error": f"Unknown warming strategy: {strategy}"}

        except Exception as e:
            return {"strategy": strategy or "unknown", "success": False, "error": str(e), "duration": time.time() - warmup_start}

    def configure_cache_warming(
        self, enabled: bool = None, strategy: str = None, batch_size: int = None, delay: float = None
    ) -> dict[str, Any]:
        """
        Configure cache warming parameters.

        Args:
            enabled: Enable/disable cache warming
            strategy: Warming strategy to use
            batch_size: Number of keys to warm per batch
            delay: Delay between warming batches

        Returns:
            Current warming configuration
        """
        if enabled is not None:
            self._warming_enabled = enabled

        if strategy is not None and strategy in self._warming_strategies:
            self._current_warming_strategy = strategy

        if batch_size is not None and batch_size > 0:
            self._warming_batch_size = batch_size

        if delay is not None and delay >= 0:
            self._warming_delay = delay

        return {
            "enabled": self._warming_enabled,
            "strategy": self._current_warming_strategy,
            "batch_size": self._warming_batch_size,
            "delay": self._warming_delay,
            "available_strategies": list(self._warming_strategies.keys()),
        }

    def get_warming_stats(self) -> dict[str, Any]:
        """Get cache warming statistics and access patterns."""
        current_time = time.time()

        # Calculate pattern statistics
        total_patterns = len(self._access_patterns)
        high_frequency_count = sum(1 for p in self._access_patterns.values() if p["frequency_score"] > 10.0)
        recent_access_count = sum(1 for p in self._access_patterns.values() if current_time - p["last_access"] < 3600)

        # Get top accessed keys
        top_keys = self._get_top_accessed_keys(10)

        return {
            "warming_enabled": self._warming_enabled,
            "current_strategy": self._current_warming_strategy,
            "total_tracked_patterns": total_patterns,
            "high_frequency_keys": high_frequency_count,
            "recently_accessed_keys": recent_access_count,
            "top_accessed_keys": top_keys,
            "batch_size": self._warming_batch_size,
            "warming_delay": self._warming_delay,
        }


# Global cache service and Redis connection manager instances
_cache_service: RedisCacheService | MultiTierCacheService | None = None
_redis_connection_manager: RedisConnectionManager | None = None


async def get_redis_connection_manager() -> RedisConnectionManager:
    """Get the global Redis connection manager instance."""
    global _redis_connection_manager
    if _redis_connection_manager is None:
        config = get_global_cache_config()
        _redis_connection_manager = RedisConnectionManager(config)
        await _redis_connection_manager.initialize()
    return _redis_connection_manager


async def get_cache_service() -> RedisCacheService | MultiTierCacheService:
    """
    Get the global cache service instance.

    Returns:
        Union[RedisCacheService, MultiTierCacheService]: The global cache service instance
    """
    global _cache_service
    if _cache_service is None:
        config = get_global_cache_config()
        if config.cache_level == CacheLevel.BOTH:
            _cache_service = MultiTierCacheService(config)
        else:
            redis_manager = await get_redis_connection_manager()
            _cache_service = RedisCacheService(config, redis_manager)
        await _cache_service.initialize()
    return _cache_service


async def shutdown_cache_service() -> None:
    """Shutdown the global cache service."""
    global _cache_service, _redis_connection_manager
    if _cache_service:
        await _cache_service.shutdown()
        _cache_service = None
    if _redis_connection_manager:
        await _redis_connection_manager.shutdown()
        _redis_connection_manager = None
