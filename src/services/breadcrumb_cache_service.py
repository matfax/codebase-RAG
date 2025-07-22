"""
Enhanced breadcrumb resolution caching service with TTL and file modification tracking.

This service provides sophisticated caching for breadcrumb resolution results with
automatic invalidation based on file modification times and configurable TTL policies.
Designed specifically for the enhanced function call detection system performance optimization.
"""

import asyncio
import logging
import sys
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Union

from src.models.breadcrumb_cache_models import (
    BreadcrumbCacheConfig,
    BreadcrumbCacheEntry,
    CacheStats,
    FileModificationTracker,
)


class BreadcrumbCacheService:
    """
    Advanced caching service for breadcrumb resolution with TTL and dependency tracking.
    """

    def __init__(self, config: BreadcrumbCacheConfig | None = None):
        """
        Initialize the breadcrumb cache service.

        Args:
            config: Cache configuration, defaults to environment-based config
        """
        self.config = config or BreadcrumbCacheConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Cache storage with LRU ordering
        self._cache: OrderedDict[str, BreadcrumbCacheEntry] = OrderedDict()
        self._stats = CacheStats()

        # File dependency tracking
        self._file_trackers: dict[str, FileModificationTracker] = {}
        self._cache_to_files: dict[str, set[str]] = {}  # cache_key -> file_paths
        self._file_to_caches: dict[str, set[str]] = {}  # file_path -> cache_keys

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._is_running = False

        if self.config.enabled:
            self.logger.info(f"BreadcrumbCacheService initialized with config: {self.config}")

    async def start(self):
        """Start the cache service and background cleanup task."""
        if not self.config.enabled:
            return

        self._is_running = True

        # Start background cleanup task
        if self.config.cleanup_interval_seconds > 0:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())

        self.logger.info("BreadcrumbCacheService started")

    async def stop(self):
        """Stop the cache service and cleanup task."""
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self.logger.info("BreadcrumbCacheService stopped")

    async def get(self, cache_key: str) -> Any:
        """
        Get a cached result.

        Args:
            cache_key: Cache key to look up

        Returns:
            Cached result or None if not found/invalid
        """
        if not self.config.enabled:
            return None

        entry = self._cache.get(cache_key)
        if entry is None:
            self._stats.record_miss()
            return None

        # Check if entry is still valid
        if not entry.is_valid():
            self.logger.debug(f"Cache entry {cache_key} is invalid, removing")
            await self._remove_entry(cache_key)
            self._stats.record_miss()
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        entry.touch()
        self._stats.record_hit()

        self.logger.debug(f"Cache hit for key: {cache_key}")
        return entry.result

    async def put(
        self,
        cache_key: str,
        result: Any,
        file_dependencies: list[str] | None = None,
        confidence_score: float = 1.0,
        custom_ttl: float | None = None,
    ) -> bool:
        """
        Store a result in the cache.

        Args:
            cache_key: Cache key
            result: Result to cache
            file_dependencies: List of file paths this result depends on
            confidence_score: Confidence score for TTL calculation
            custom_ttl: Custom TTL override

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.config.enabled:
            return False

        try:
            # Calculate TTL
            ttl = custom_ttl or self.config.get_ttl_for_confidence(confidence_score)

            # Create file trackers for dependencies
            trackers = []
            if file_dependencies and self.config.enable_dependency_tracking:
                for file_path in file_dependencies:
                    tracker = await self._get_or_create_file_tracker(file_path)
                    trackers.append(tracker)

            # Create cache entry
            entry = BreadcrumbCacheEntry(
                cache_key=cache_key,
                result=result,
                timestamp=time.time(),
                ttl_seconds=ttl,
                file_dependencies=trackers,
                confidence_score=confidence_score,
            )

            # Check cache size and evict if needed
            await self._ensure_cache_capacity()

            # Store entry
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)  # Mark as most recently used

            # Update dependency mappings
            if file_dependencies:
                self._cache_to_files[cache_key] = set(file_dependencies)
                for file_path in file_dependencies:
                    if file_path not in self._file_to_caches:
                        self._file_to_caches[file_path] = set()
                    self._file_to_caches[file_path].add(cache_key)

            # Update stats
            self._stats.total_entries = len(self._cache)

            self.logger.debug(f"Cached result for key: {cache_key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache result for key {cache_key}: {e}")
            return False

    async def invalidate_by_key(self, cache_key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            cache_key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        if cache_key in self._cache:
            await self._remove_entry(cache_key)
            self._stats.record_invalidation()
            self.logger.debug(f"Invalidated cache entry: {cache_key}")
            return True
        return False

    async def invalidate_by_file(self, file_path: str) -> int:
        """
        Invalidate all cache entries that depend on a file.

        Args:
            file_path: File path that was modified

        Returns:
            Number of cache entries invalidated
        """
        if not self.config.enable_dependency_tracking:
            return 0

        cache_keys = self._file_to_caches.get(file_path, set()).copy()
        invalidated_count = 0

        for cache_key in cache_keys:
            if await self.invalidate_by_key(cache_key):
                invalidated_count += 1

        if invalidated_count > 0:
            self.logger.info(f"Invalidated {invalidated_count} cache entries due to file change: {file_path}")

        return invalidated_count

    async def invalidate_stale_entries(self) -> int:
        """
        Invalidate all stale cache entries.

        Returns:
            Number of entries invalidated
        """
        stale_keys = []

        for cache_key, entry in self._cache.items():
            if not entry.is_valid():
                stale_keys.append(cache_key)

        invalidated_count = 0
        for cache_key in stale_keys:
            if await self.invalidate_by_key(cache_key):
                invalidated_count += 1

        if invalidated_count > 0:
            self.logger.info(f"Invalidated {invalidated_count} stale cache entries")

        return invalidated_count

    async def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._cache_to_files.clear()
        self._file_to_caches.clear()
        self._file_trackers.clear()
        self._stats = CacheStats()

        self.logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        self._stats.total_entries = len(self._cache)

        # Calculate memory usage estimation
        try:
            memory_usage = sys.getsizeof(self._cache)
            for entry in self._cache.values():
                memory_usage += sys.getsizeof(entry)
                memory_usage += sys.getsizeof(entry.result)
            self._stats.memory_usage_bytes = memory_usage
        except Exception:
            self._stats.memory_usage_bytes = 0

        # Calculate average TTL
        if self._cache:
            total_ttl = sum(entry.ttl_seconds for entry in self._cache.values())
            self._stats.average_ttl_seconds = total_ttl / len(self._cache)

        return self._stats.to_dict()

    async def get_cache_info(self) -> dict[str, Any]:
        """Get detailed cache information."""
        stats = self.get_stats()

        # Add configuration info
        info = {
            "stats": stats,
            "config": {
                "enabled": self.config.enabled,
                "max_entries": self.config.max_entries,
                "default_ttl_seconds": self.config.default_ttl_seconds,
                "eviction_policy": self.config.eviction_policy,
                "memory_limit_mb": self.config.memory_limit_mb,
            },
            "cache_details": {
                "total_files_tracked": len(self._file_trackers),
                "oldest_entry_age": await self._get_oldest_entry_age(),
                "newest_entry_age": await self._get_newest_entry_age(),
            },
        }

        return info

    async def _get_or_create_file_tracker(self, file_path: str) -> FileModificationTracker:
        """Get or create a file modification tracker."""
        if file_path not in self._file_trackers:
            try:
                from pathlib import Path

                path = Path(file_path)
                mtime = path.stat().st_mtime if path.exists() else 0.0

                self._file_trackers[file_path] = FileModificationTracker(file_path=file_path, last_modified=mtime)
            except Exception as e:
                self.logger.warning(f"Failed to create file tracker for {file_path}: {e}")
                self._file_trackers[file_path] = FileModificationTracker(file_path=file_path, last_modified=0.0)

        return self._file_trackers[file_path]

    async def _remove_entry(self, cache_key: str):
        """Remove a cache entry and clean up dependencies."""
        if cache_key not in self._cache:
            return

        # Remove from cache
        del self._cache[cache_key]

        # Clean up dependency mappings
        if cache_key in self._cache_to_files:
            file_paths = self._cache_to_files[cache_key]
            for file_path in file_paths:
                if file_path in self._file_to_caches:
                    self._file_to_caches[file_path].discard(cache_key)
                    # Remove file tracker if no more caches depend on it
                    if not self._file_to_caches[file_path]:
                        del self._file_to_caches[file_path]
                        self._file_trackers.pop(file_path, None)

            del self._cache_to_files[cache_key]

    async def _ensure_cache_capacity(self):
        """Ensure cache doesn't exceed capacity limits."""
        # Check entry count limit
        while len(self._cache) >= self.config.max_entries:
            # Remove least recently used entry
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
            self._stats.record_eviction()

        # Check memory limit (simplified estimation)
        if self.config.memory_limit_mb > 0:
            current_memory_mb = self._stats.memory_usage_bytes / (1024 * 1024)
            while current_memory_mb > self.config.memory_limit_mb and len(self._cache) > 0:
                oldest_key = next(iter(self._cache))
                await self._remove_entry(oldest_key)
                self._stats.record_eviction()
                current_memory_mb = self._stats.memory_usage_bytes / (1024 * 1024)

    async def _background_cleanup(self):
        """Background task for periodic cache cleanup."""
        self.logger.info("Started background cache cleanup task")

        while self._is_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)

                if not self._is_running:
                    break

                # Clean up expired and stale entries
                await self.invalidate_stale_entries()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        self.logger.info("Background cache cleanup task stopped")

    async def _get_oldest_entry_age(self) -> float:
        """Get age of oldest cache entry in seconds."""
        if not self._cache:
            return 0.0

        oldest_entry = next(iter(self._cache.values()))
        return oldest_entry.get_age_seconds()

    async def _get_newest_entry_age(self) -> float:
        """Get age of newest cache entry in seconds."""
        if not self._cache:
            return 0.0

        newest_entry = next(reversed(self._cache.values()))
        return newest_entry.get_age_seconds()
