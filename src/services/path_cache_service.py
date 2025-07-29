"""
Path Cache Service for Wave 2.0 Task 2.8 - Path Relationship Caching.

This service implements intelligent caching mechanisms to avoid recomputing path
relationships. It provides multi-level caching, intelligent invalidation, and
performance optimization for path-based operations in the PathRAG system.
"""

import asyncio
import builtins
import hashlib
import logging
import pickle
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathType,
    RelationalPathCollection,
)


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""

    L1_MEMORY = "l1_memory"  # In-memory cache (fastest)
    L2_REDIS = "l2_redis"  # Redis cache (fast, persistent)
    L3_DISK = "l3_disk"  # Disk cache (slower, persistent)


class CacheStrategy(Enum):
    """Cache replacement strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive replacement strategy


class InvalidationReason(Enum):
    """Reasons for cache invalidation."""

    MANUAL = "manual"  # Manual invalidation
    TIMEOUT = "timeout"  # TTL expiration
    DEPENDENCY_CHANGE = "dependency"  # Dependency changed
    STORAGE_CHANGE = "storage"  # Underlying storage changed
    MEMORY_PRESSURE = "memory_pressure"  # Memory pressure eviction


@dataclass
class CacheEntry:
    """Entry in the path cache."""

    # Core data
    cache_key: str  # Unique cache key
    data: Any  # Cached data
    data_type: str  # Type of cached data

    # Timing information
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    expires_at: float | None = None  # TTL expiration time

    # Access patterns
    access_count: int = 0  # Number of accesses
    hit_count: int = 0  # Number of cache hits

    # Metadata
    data_size: int = 0  # Size of cached data in bytes
    compression_ratio: float = 1.0  # Compression ratio if compressed
    dependencies: set[str] = field(default_factory=set)  # Dependencies for invalidation

    # Quality metrics
    computation_cost: float = 0.0  # Cost to recompute (0-1)
    staleness_tolerance: float = 300.0  # Max staleness in seconds

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
        self.hit_count += 1

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_stale(self) -> bool:
        """Check if entry is stale."""
        return (time.time() - self.last_modified) > self.staleness_tolerance

    def get_age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    def get_staleness(self) -> float:
        """Get staleness of entry in seconds."""
        return time.time() - self.last_modified


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    # Hit/miss statistics
    total_requests: int = 0  # Total cache requests
    cache_hits: int = 0  # Number of cache hits
    cache_misses: int = 0  # Number of cache misses
    hit_rate: float = 0.0  # Cache hit rate (0-1)

    # Storage statistics
    total_entries: int = 0  # Total cached entries
    total_memory_usage: int = 0  # Total memory usage in bytes
    average_entry_size: float = 0.0  # Average entry size

    # Performance statistics
    average_lookup_time_ms: float = 0.0  # Average lookup time
    average_store_time_ms: float = 0.0  # Average store time

    # Eviction statistics
    total_evictions: int = 0  # Total evictions
    evictions_by_reason: dict[InvalidationReason, int] = field(default_factory=dict)

    # Level-specific statistics
    l1_hits: int = 0  # L1 cache hits
    l2_hits: int = 0  # L2 cache hits
    l3_hits: int = 0  # L3 cache hits

    def calculate_hit_rate(self):
        """Calculate and update hit rate."""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
        else:
            self.hit_rate = 0.0

    def is_performing_well(self) -> bool:
        """Check if cache is performing well."""
        return self.hit_rate > 0.7 and self.average_lookup_time_ms < 10.0


@dataclass
class CacheConfig:
    """Configuration for path cache service."""

    # Cache sizes (number of entries)
    l1_max_size: int = 1000  # L1 memory cache size
    l2_max_size: int = 10000  # L2 Redis cache size (if available)
    l3_max_size: int = 100000  # L3 disk cache size

    # Memory limits (bytes)
    l1_max_memory: int = 100 * 1024 * 1024  # 100MB for L1
    l2_max_memory: int = 1024 * 1024 * 1024  # 1GB for L2
    l3_max_memory: int = 10 * 1024 * 1024 * 1024  # 10GB for L3

    # TTL settings (seconds)
    default_ttl: float = 3600  # 1 hour default TTL
    path_data_ttl: float = 7200  # 2 hours for path data
    relationship_ttl: float = 1800  # 30 minutes for relationships
    similarity_ttl: float = 900  # 15 minutes for similarity scores

    # Cache strategies
    l1_strategy: CacheStrategy = CacheStrategy.LRU
    l2_strategy: CacheStrategy = CacheStrategy.LFU
    l3_strategy: CacheStrategy = CacheStrategy.FIFO

    # Performance settings
    enable_compression: bool = True  # Enable data compression
    compression_threshold: int = 1024  # Compress data larger than 1KB
    enable_async_operations: bool = True  # Enable async cache operations

    # Quality settings
    staleness_check_interval: float = 300  # Check staleness every 5 minutes
    auto_refresh_threshold: float = 0.8  # Auto-refresh when 80% stale
    preemptive_eviction: bool = True  # Evict before memory limits

    # Redis settings (if available)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0


class PathCacheService:
    """
    Advanced multi-level caching service for path relationships and computed data.
    Provides intelligent caching with configurable strategies, automatic invalidation,
    and performance optimization for PathRAG operations.

    Key features:
    - Multi-level hierarchical caching (L1/L2/L3)
    - Intelligent cache replacement strategies
    - Automatic invalidation based on dependencies
    - Compression for large data
    - Performance monitoring and optimization
    - Async operations for better throughput
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize the path cache service.

        Args:
            config: Cache configuration options
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or CacheConfig()

        # Multi-level cache storage
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # Memory cache
        self._l2_cache: Any | None = None  # Redis cache (if available)
        self._l3_cache: dict[str, CacheEntry] = {}  # Disk cache

        # Cache statistics
        self._stats = CacheStats()

        # Dependency tracking
        self._dependency_map: dict[str, set[str]] = defaultdict(set)  # dependency -> cache_keys
        self._reverse_dependencies: dict[str, set[str]] = defaultdict(set)  # cache_key -> dependencies

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._stats_update_task: asyncio.Task | None = None

        # Performance tracking
        self._operation_times: dict[str, list[float]] = defaultdict(list)

        # Initialize Redis if available
        self._init_redis_cache()

        # Start background tasks
        self._start_background_tasks()

    async def get(self, cache_key: str, data_type: str | None = None) -> Any | None:
        """
        Get data from cache.

        Args:
            cache_key: Unique cache key
            data_type: Optional data type filter

        Returns:
            Cached data if found, None otherwise
        """
        start_time = time.time()

        try:
            self._stats.total_requests += 1

            # Try L1 cache first (fastest)
            if cache_key in self._l1_cache:
                entry = self._l1_cache[cache_key]

                if not entry.is_expired() and (data_type is None or entry.data_type == data_type):
                    entry.update_access()
                    self._move_to_front_l1(cache_key)

                    self._stats.cache_hits += 1
                    self._stats.l1_hits += 1

                    lookup_time = (time.time() - start_time) * 1000
                    self._operation_times["lookup"].append(lookup_time)

                    return entry.data

            # Try L2 cache (Redis)
            if self._l2_cache is not None:
                l2_data = await self._get_from_l2(cache_key, data_type)
                if l2_data is not None:
                    # Promote to L1
                    await self._promote_to_l1(cache_key, l2_data, data_type or "unknown")

                    self._stats.cache_hits += 1
                    self._stats.l2_hits += 1

                    lookup_time = (time.time() - start_time) * 1000
                    self._operation_times["lookup"].append(lookup_time)

                    return l2_data

            # Try L3 cache (disk)
            if cache_key in self._l3_cache:
                entry = self._l3_cache[cache_key]

                if not entry.is_expired() and (data_type is None or entry.data_type == data_type):
                    entry.update_access()

                    # Promote to higher levels
                    await self._promote_to_l1(cache_key, entry.data, entry.data_type)
                    if self._l2_cache is not None:
                        await self._store_in_l2(cache_key, entry.data, entry.data_type, entry.expires_at)

                    self._stats.cache_hits += 1
                    self._stats.l3_hits += 1

                    lookup_time = (time.time() - start_time) * 1000
                    self._operation_times["lookup"].append(lookup_time)

                    return entry.data

            # Cache miss
            self._stats.cache_misses += 1

            lookup_time = (time.time() - start_time) * 1000
            self._operation_times["lookup"].append(lookup_time)

            return None

        except Exception as e:
            self.logger.error(f"Cache get failed for key {cache_key}: {str(e)}")
            return None

    async def set(
        self,
        cache_key: str,
        data: Any,
        data_type: str,
        ttl: float | None = None,
        dependencies: set[str] | None = None,
        computation_cost: float = 0.5,
    ) -> bool:
        """
        Store data in cache.

        Args:
            cache_key: Unique cache key
            data: Data to cache
            data_type: Type of data being cached
            ttl: Time to live in seconds
            dependencies: Set of dependencies for invalidation
            computation_cost: Cost to recompute (0-1)

        Returns:
            True if stored successfully, False otherwise
        """
        start_time = time.time()

        try:
            # Calculate TTL
            if ttl is None:
                ttl = self._get_default_ttl(data_type)

            expires_at = time.time() + ttl if ttl > 0 else None

            # Calculate data size
            data_size = self._calculate_data_size(data)

            # Create cache entry
            entry = CacheEntry(
                cache_key=cache_key,
                data=data,
                data_type=data_type,
                expires_at=expires_at,
                data_size=data_size,
                computation_cost=computation_cost,
                dependencies=dependencies or set(),
            )

            # Store in appropriate cache levels
            stored = False

            # Always try L1 first
            if await self._store_in_l1(cache_key, entry):
                stored = True

            # Store in L2 if available and data is valuable
            if self._l2_cache is not None and computation_cost > 0.3:
                await self._store_in_l2(cache_key, data, data_type, expires_at)

            # Store in L3 for long-term caching
            if computation_cost > 0.5:
                await self._store_in_l3(cache_key, entry)

            # Update dependency tracking
            if dependencies:
                self._update_dependencies(cache_key, dependencies)

            store_time = (time.time() - start_time) * 1000
            self._operation_times["store"].append(store_time)

            return stored

        except Exception as e:
            self.logger.error(f"Cache set failed for key {cache_key}: {str(e)}")
            return False

    async def invalidate(self, cache_key: str, reason: InvalidationReason = InvalidationReason.MANUAL) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            cache_key: Cache key to invalidate
            reason: Reason for invalidation

        Returns:
            True if invalidated successfully, False otherwise
        """
        try:
            invalidated = False

            # Remove from L1
            if cache_key in self._l1_cache:
                del self._l1_cache[cache_key]
                invalidated = True

            # Remove from L2
            if self._l2_cache is not None:
                await self._remove_from_l2(cache_key)
                invalidated = True

            # Remove from L3
            if cache_key in self._l3_cache:
                del self._l3_cache[cache_key]
                invalidated = True

            # Update dependency tracking
            self._remove_dependencies(cache_key)

            # Update statistics
            if invalidated:
                self._stats.total_evictions += 1
                self._stats.evictions_by_reason[reason] = self._stats.evictions_by_reason.get(reason, 0) + 1

            return invalidated

        except Exception as e:
            self.logger.error(f"Cache invalidation failed for key {cache_key}: {str(e)}")
            return False

    async def invalidate_by_dependency(self, dependency: str, reason: InvalidationReason = InvalidationReason.DEPENDENCY_CHANGE) -> int:
        """
        Invalidate all cache entries that depend on a specific dependency.

        Args:
            dependency: Dependency that changed
            reason: Reason for invalidation

        Returns:
            Number of entries invalidated
        """
        try:
            dependent_keys = self._dependency_map.get(dependency, set()).copy()
            invalidated_count = 0

            for cache_key in dependent_keys:
                if await self.invalidate(cache_key, reason):
                    invalidated_count += 1

            self.logger.info(f"Invalidated {invalidated_count} entries for dependency: {dependency}")

            return invalidated_count

        except Exception as e:
            self.logger.error(f"Dependency invalidation failed for {dependency}: {str(e)}")
            return 0

    async def get_path_relationships(self, path_id: str, relationship_type: str = "all") -> dict[str, Any] | None:
        """
        Get cached path relationships.

        Args:
            path_id: ID of the path
            relationship_type: Type of relationships to get

        Returns:
            Cached relationship data if available
        """
        cache_key = f"relationships:{path_id}:{relationship_type}"
        return await self.get(cache_key, "relationships")

    async def set_path_relationships(
        self, path_id: str, relationships: dict[str, Any], relationship_type: str = "all", ttl: float | None = None
    ) -> bool:
        """
        Cache path relationships.

        Args:
            path_id: ID of the path
            relationships: Relationship data to cache
            relationship_type: Type of relationships
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        cache_key = f"relationships:{path_id}:{relationship_type}"
        dependencies = {f"path:{path_id}"}

        return await self.set(
            cache_key,
            relationships,
            "relationships",
            ttl or self.config.relationship_ttl,
            dependencies,
            computation_cost=0.7,  # Relationships are expensive to compute
        )

    async def get_path_similarity(self, path_id1: str, path_id2: str, similarity_type: str = "structural") -> float | None:
        """
        Get cached path similarity score.

        Args:
            path_id1: First path ID
            path_id2: Second path ID
            similarity_type: Type of similarity

        Returns:
            Cached similarity score if available
        """
        # Normalize order for consistent caching
        if path_id1 > path_id2:
            path_id1, path_id2 = path_id2, path_id1

        cache_key = f"similarity:{path_id1}:{path_id2}:{similarity_type}"
        return await self.get(cache_key, "similarity")

    async def set_path_similarity(
        self, path_id1: str, path_id2: str, similarity_score: float, similarity_type: str = "structural", ttl: float | None = None
    ) -> bool:
        """
        Cache path similarity score.

        Args:
            path_id1: First path ID
            path_id2: Second path ID
            similarity_score: Similarity score to cache
            similarity_type: Type of similarity
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        # Normalize order for consistent caching
        if path_id1 > path_id2:
            path_id1, path_id2 = path_id2, path_id1

        cache_key = f"similarity:{path_id1}:{path_id2}:{similarity_type}"
        dependencies = {f"path:{path_id1}", f"path:{path_id2}"}

        return await self.set(
            cache_key,
            similarity_score,
            "similarity",
            ttl or self.config.similarity_ttl,
            dependencies,
            computation_cost=0.6,  # Similarity calculations are moderately expensive
        )

    async def get_collection_cache(self, collection_id: str, cache_type: str) -> Any | None:
        """
        Get cached collection data.

        Args:
            collection_id: ID of the collection
            cache_type: Type of cached data

        Returns:
            Cached collection data if available
        """
        cache_key = f"collection:{collection_id}:{cache_type}"
        return await self.get(cache_key, "collection")

    async def set_collection_cache(self, collection_id: str, data: Any, cache_type: str, ttl: float | None = None) -> bool:
        """
        Cache collection data.

        Args:
            collection_id: ID of the collection
            data: Data to cache
            cache_type: Type of data being cached
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        cache_key = f"collection:{collection_id}:{cache_type}"
        dependencies = {f"collection:{collection_id}"}

        return await self.set(
            cache_key,
            data,
            "collection",
            ttl or self.config.default_ttl,
            dependencies,
            computation_cost=0.8,  # Collection operations are expensive
        )

    async def clear_all(self) -> bool:
        """
        Clear all cache levels.

        Returns:
            True if cleared successfully
        """
        try:
            # Clear L1
            self._l1_cache.clear()

            # Clear L2
            if self._l2_cache is not None:
                await self._clear_l2()

            # Clear L3
            self._l3_cache.clear()

            # Clear dependency tracking
            self._dependency_map.clear()
            self._reverse_dependencies.clear()

            # Reset statistics
            self._stats = CacheStats()

            self.logger.info("All cache levels cleared")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            return False

    async def get_stats(self) -> CacheStats:
        """
        Get current cache statistics.

        Returns:
            CacheStats object with current statistics
        """
        # Update statistics
        self._stats.total_entries = len(self._l1_cache) + len(self._l3_cache)

        # Calculate memory usage
        l1_memory = sum(entry.data_size for entry in self._l1_cache.values())
        l3_memory = sum(entry.data_size for entry in self._l3_cache.values())
        self._stats.total_memory_usage = l1_memory + l3_memory

        if self._stats.total_entries > 0:
            self._stats.average_entry_size = self._stats.total_memory_usage / self._stats.total_entries

        # Calculate average operation times
        if self._operation_times["lookup"]:
            self._stats.average_lookup_time_ms = np.mean(self._operation_times["lookup"])

        if self._operation_times["store"]:
            self._stats.average_store_time_ms = np.mean(self._operation_times["store"])

        # Update hit rate
        self._stats.calculate_hit_rate()

        return self._stats

    async def optimize_cache(self) -> dict[str, Any]:
        """
        Optimize cache performance.

        Returns:
            Dictionary with optimization results
        """
        try:
            optimization_results = {"l1_evicted": 0, "l2_evicted": 0, "l3_evicted": 0, "memory_freed": 0, "performance_improved": False}

            # Clean expired entries
            expired_count = await self._cleanup_expired_entries()
            optimization_results["expired_cleaned"] = expired_count

            # Optimize L1 cache
            if len(self._l1_cache) > self.config.l1_max_size * 0.8:
                l1_evicted = await self._optimize_l1_cache()
                optimization_results["l1_evicted"] = l1_evicted

            # Optimize L3 cache
            if len(self._l3_cache) > self.config.l3_max_size * 0.8:
                l3_evicted = await self._optimize_l3_cache()
                optimization_results["l3_evicted"] = l3_evicted

            # Check performance improvement
            stats = await self.get_stats()
            if stats.hit_rate > 0.7 and stats.average_lookup_time_ms < 10.0:
                optimization_results["performance_improved"] = True

            self.logger.info(f"Cache optimization completed: {optimization_results}")

            return optimization_results

        except Exception as e:
            self.logger.error(f"Cache optimization failed: {str(e)}")
            return {"error": str(e)}

    # Private methods

    def _init_redis_cache(self):
        """Initialize Redis cache if available."""
        try:
            import redis

            self._l2_cache = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=False,
            )

            # Test connection
            self._l2_cache.ping()
            self.logger.info("Redis cache initialized successfully")

        except Exception as e:
            self.logger.warning(f"Redis cache not available: {str(e)}")
            self._l2_cache = None

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.config.enable_async_operations:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._stats_update_task = asyncio.create_task(self._periodic_stats_update())

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.staleness_check_interval)
                await self._cleanup_expired_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic cleanup failed: {str(e)}")

    async def _periodic_stats_update(self):
        """Periodic statistics update."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                await self.get_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats update failed: {str(e)}")

    async def _cleanup_expired_entries(self) -> int:
        """Clean up expired entries from all cache levels."""
        expired_count = 0

        # Clean L1
        expired_keys = [key for key, entry in self._l1_cache.items() if entry.is_expired()]

        for key in expired_keys:
            await self.invalidate(key, InvalidationReason.TIMEOUT)
            expired_count += 1

        # Clean L3
        expired_keys = [key for key, entry in self._l3_cache.items() if entry.is_expired()]

        for key in expired_keys:
            await self.invalidate(key, InvalidationReason.TIMEOUT)
            expired_count += 1

        return expired_count

    def _get_default_ttl(self, data_type: str) -> float:
        """Get default TTL for a data type."""
        ttl_map = {
            "path_data": self.config.path_data_ttl,
            "relationships": self.config.relationship_ttl,
            "similarity": self.config.similarity_ttl,
            "collection": self.config.default_ttl,
        }

        return ttl_map.get(data_type, self.config.default_ttl)

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            if self.config.enable_compression and hasattr(data, "__sizeof__"):
                return data.__sizeof__()
            else:
                # Rough approximation using pickle
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                return len(serialized)
        except Exception:
            # Fallback estimate
            return 1024  # 1KB default

    async def _store_in_l1(self, cache_key: str, entry: CacheEntry) -> bool:
        """Store entry in L1 cache."""
        try:
            # Check memory limits
            if len(self._l1_cache) >= self.config.l1_max_size:
                await self._evict_from_l1()

            # Store entry
            self._l1_cache[cache_key] = entry

            return True

        except Exception as e:
            self.logger.error(f"L1 store failed: {str(e)}")
            return False

    async def _evict_from_l1(self):
        """Evict entries from L1 cache based on strategy."""
        if self.config.l1_strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self._l1_cache:
                evicted_key = next(iter(self._l1_cache))
                del self._l1_cache[evicted_key]

        elif self.config.l1_strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self._l1_cache:
                min_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].access_count)
                del self._l1_cache[min_key]

        # Add other strategies as needed

    def _move_to_front_l1(self, cache_key: str):
        """Move entry to front of L1 cache (for LRU)."""
        if cache_key in self._l1_cache:
            entry = self._l1_cache.pop(cache_key)
            self._l1_cache[cache_key] = entry

    async def _get_from_l2(self, cache_key: str, data_type: str | None) -> Any | None:
        """Get data from L2 (Redis) cache."""
        if self._l2_cache is None:
            return None

        try:
            serialized_data = self._l2_cache.get(cache_key)
            if serialized_data:
                return pickle.loads(serialized_data)
            return None

        except Exception as e:
            self.logger.error(f"L2 get failed: {str(e)}")
            return None

    async def _store_in_l2(self, cache_key: str, data: Any, data_type: str, expires_at: float | None) -> bool:
        """Store data in L2 (Redis) cache."""
        if self._l2_cache is None:
            return False

        try:
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            if expires_at:
                ttl = max(1, int(expires_at - time.time()))
                self._l2_cache.setex(cache_key, ttl, serialized_data)
            else:
                self._l2_cache.set(cache_key, serialized_data)

            return True

        except Exception as e:
            self.logger.error(f"L2 store failed: {str(e)}")
            return False

    async def _remove_from_l2(self, cache_key: str) -> bool:
        """Remove data from L2 (Redis) cache."""
        if self._l2_cache is None:
            return False

        try:
            self._l2_cache.delete(cache_key)
            return True

        except Exception as e:
            self.logger.error(f"L2 remove failed: {str(e)}")
            return False

    async def _clear_l2(self) -> bool:
        """Clear L2 (Redis) cache."""
        if self._l2_cache is None:
            return False

        try:
            self._l2_cache.flushdb()
            return True

        except Exception as e:
            self.logger.error(f"L2 clear failed: {str(e)}")
            return False

    async def _store_in_l3(self, cache_key: str, entry: CacheEntry) -> bool:
        """Store entry in L3 (disk) cache."""
        try:
            # Check size limits
            if len(self._l3_cache) >= self.config.l3_max_size:
                await self._evict_from_l3()

            # Store entry
            self._l3_cache[cache_key] = entry

            return True

        except Exception as e:
            self.logger.error(f"L3 store failed: {str(e)}")
            return False

    async def _evict_from_l3(self):
        """Evict entries from L3 cache."""
        if self.config.l3_strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            if self._l3_cache:
                oldest_key = min(self._l3_cache.keys(), key=lambda k: self._l3_cache[k].created_at)
                del self._l3_cache[oldest_key]

    async def _promote_to_l1(self, cache_key: str, data: Any, data_type: str):
        """Promote data to L1 cache."""
        entry = CacheEntry(cache_key=cache_key, data=data, data_type=data_type, data_size=self._calculate_data_size(data))

        await self._store_in_l1(cache_key, entry)

    def _update_dependencies(self, cache_key: str, dependencies: builtins.set[str]):
        """Update dependency tracking."""
        # Clear old dependencies
        if cache_key in self._reverse_dependencies:
            old_deps = self._reverse_dependencies[cache_key]
            for dep in old_deps:
                self._dependency_map[dep].discard(cache_key)

        # Set new dependencies
        self._reverse_dependencies[cache_key] = dependencies.copy()
        for dep in dependencies:
            self._dependency_map[dep].add(cache_key)

    def _remove_dependencies(self, cache_key: str):
        """Remove dependency tracking for a cache key."""
        if cache_key in self._reverse_dependencies:
            dependencies = self._reverse_dependencies[cache_key]
            for dep in dependencies:
                self._dependency_map[dep].discard(cache_key)
            del self._reverse_dependencies[cache_key]

    async def _optimize_l1_cache(self) -> int:
        """Optimize L1 cache and return number of evicted entries."""
        evicted_count = 0
        target_size = int(self.config.l1_max_size * 0.7)  # Reduce to 70%

        while len(self._l1_cache) > target_size:
            await self._evict_from_l1()
            evicted_count += 1

        return evicted_count

    async def _optimize_l3_cache(self) -> int:
        """Optimize L3 cache and return number of evicted entries."""
        evicted_count = 0
        target_size = int(self.config.l3_max_size * 0.7)  # Reduce to 70%

        while len(self._l3_cache) > target_size:
            await self._evict_from_l3()
            evicted_count += 1

        return evicted_count

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._stats_update_task:
                self._stats_update_task.cancel()
        except Exception:
            pass


# Factory function
def create_path_cache_service(config: CacheConfig | None = None) -> PathCacheService:
    """
    Factory function to create a PathCacheService instance.

    Args:
        config: Optional cache configuration

    Returns:
        Configured PathCacheService instance
    """
    return PathCacheService(config)
