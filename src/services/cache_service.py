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
from typing import Any, Dict, List, Optional, Tuple, Union

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from ..config.cache_config import CacheConfig, CacheLevel, CacheWriteStrategy, get_global_cache_config


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
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self.config.get_redis_url(),
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout,
                socket_connect_timeout=self.config.redis.connection_timeout,
                retry_on_timeout=self.config.redis.retry_on_timeout,
                retry_on_error=[ConnectionError, TimeoutError],
                health_check_interval=self.config.health_check_interval,
                ssl=self.config.redis.ssl_enabled,
                ssl_cert_reqs=None if not self.config.redis.ssl_enabled else "required",
                ssl_ca_certs=self.config.redis.ssl_ca_cert_path,
                ssl_certfile=self.config.redis.ssl_cert_path,
                ssl_keyfile=self.config.redis.ssl_key_path,
            )

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

    def __init__(self, config: CacheConfig | None = None):
        """Initialize Redis cache service."""
        super().__init__(config)
        self.redis_manager = RedisConnectionManager(self.config)

    async def initialize(self) -> None:
        """Initialize Redis cache service."""
        await self.redis_manager.initialize()
        self.logger.info("Redis cache service initialized")

    async def shutdown(self) -> None:
        """Shutdown Redis cache service."""
        await self.redis_manager.shutdown()
        self.logger.info("Redis cache service shutdown")

    async def get(self, key: str) -> Any | None:
        """Get a value from Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Add key prefix
            prefixed_key = f"{self.config.key_prefix}:{key}"

            # Get value from Redis
            value = await redis_client.get(prefixed_key)

            if value is not None:
                self._update_stats("get", success=True)
                # Note: Deserialization will be handled by cache_utils
                return value
            else:
                self._update_stats("get", success=False)
                return None

        except Exception as e:
            self._update_stats("get", success=False)
            self.logger.error(f"Failed to get key '{key}': {e}")
            raise CacheOperationError(f"Failed to get key '{key}': {e}")

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in Redis cache."""
        try:
            redis_client = await self.redis_manager.get_redis()

            # Add key prefix
            prefixed_key = f"{self.config.key_prefix}:{key}"

            # Use provided TTL or default
            cache_ttl = ttl or self.config.default_ttl

            # Set value in Redis
            # Note: Serialization will be handled by cache_utils
            result = await redis_client.setex(prefixed_key, cache_ttl, value)

            self._update_stats("set", success=bool(result))
            return bool(result)

        except Exception as e:
            self._update_stats("set", success=False)
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
                    result[key] = value
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
                    pipe.setex(prefixed_key, cache_ttl, value)

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
    Thread-safe LRU memory cache implementation.

    This class provides an in-memory cache with LRU (Least Recently Used) eviction
    policy, TTL support, and size-based eviction.
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 256, default_ttl: int = 3600):
        """Initialize LRU memory cache."""
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._total_size = 0
        self._stats = CacheStats()
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Any | None:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.miss_count += 1
                self._stats.total_operations += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self._stats.miss_count += 1
                self._stats.total_operations += 1
                return None

            # Update access and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)

            self._stats.hit_count += 1
            self._stats.total_operations += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in the cache."""
        with self._lock:
            try:
                # Calculate size (rough estimation)
                size = self._estimate_size(value)

                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)

                # Check if we need to evict entries
                self._evict_if_needed(size)

                # Create new entry
                entry = CacheEntry(value=value, timestamp=time.time(), ttl=ttl or self.default_ttl, size=size)

                # Add to cache
                self._cache[key] = entry
                self._total_size += size

                self._stats.set_count += 1
                self._stats.total_operations += 1
                return True

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
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                return 1024  # Default estimate


class MultiTierCacheService(BaseCacheService):
    """
    Multi-tier cache service with L1 (memory) and L2 (Redis) layers.

    This service implements a comprehensive caching strategy with:
    - L1: Fast in-memory cache for frequently accessed data
    - L2: Persistent Redis cache for larger storage
    - Cache coherency between layers
    - Configurable write strategies (write-through, write-back)
    - Promotion and demotion logic
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize multi-tier cache service."""
        super().__init__(config)

        # Initialize L1 memory cache
        self.l1_cache = LRUMemoryCache(
            max_size=self.config.memory.max_size, max_memory_mb=self.config.memory.max_memory_mb, default_ttl=self.config.memory.ttl_seconds
        )

        # Initialize L2 Redis cache
        self.l2_cache = RedisCacheService(config) if self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH] else None

        # Cache coherency tracking
        self._dirty_keys: set[str] = set()
        self._promotion_threshold = 3  # Access count threshold for promotion
        self._cleanup_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize multi-tier cache service."""
        # Initialize L2 cache if enabled
        if self.l2_cache:
            await self.l2_cache.initialize()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

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


# Global cache service instance
_cache_service: RedisCacheService | MultiTierCacheService | None = None


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
            _cache_service = RedisCacheService(config)
        await _cache_service.initialize()
    return _cache_service


async def shutdown_cache_service() -> None:
    """Shutdown the global cache service."""
    global _cache_service
    if _cache_service:
        await _cache_service.shutdown()
        _cache_service = None
