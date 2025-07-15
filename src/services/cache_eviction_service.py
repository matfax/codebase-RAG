"""
Intelligent cache eviction policies service.

This module provides advanced cache eviction strategies including LRU, LFU, TTL-based,
memory-pressure-aware, and custom eviction policies with configurable strategies
per cache type and performance-optimized algorithms.
"""

import asyncio
import logging
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Optional, Union

from ..utils.memory_utils import (
    CacheMemoryEvent,
    MemoryPressureLevel,
    SystemMemoryPressure,
    get_system_memory_pressure,
    track_cache_memory_event,
)


class EvictionStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based
    MEMORY_PRESSURE = "memory_pressure"  # Memory pressure aware
    RANDOM = "random"  # Random eviction
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on cache behavior
    CUSTOM = "custom"  # Custom eviction policy


class EvictionTrigger(Enum):
    """Eviction trigger conditions."""

    SIZE_LIMIT = "size_limit"  # Cache size limit reached
    MEMORY_PRESSURE = "memory_pressure"  # System memory pressure
    TTL_EXPIRED = "ttl_expired"  # TTL expiration
    MANUAL = "manual"  # Manual eviction request
    BATCH_CLEANUP = "batch_cleanup"  # Batch cleanup operation


@dataclass
class CacheEntry:
    """Enhanced cache entry with eviction metadata."""

    key: str
    value: Any
    timestamp: float
    last_access: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: float | None = None
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.last_access == 0:
            self.last_access = self.timestamp

    @property
    def age(self) -> float:
        """Get age of the entry in seconds."""
        return time.time() - self.timestamp

    @property
    def idle_time(self) -> float:
        """Get idle time since last access in seconds."""
        return time.time() - self.last_access

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return self.age > self.ttl

    @property
    def frequency_score(self) -> float:
        """Calculate frequency score for LFU eviction."""
        return self.access_count / (self.age + 1)

    @property
    def recency_score(self) -> float:
        """Calculate recency score for LRU eviction."""
        return 1.0 / (self.idle_time + 1)

    def touch(self) -> None:
        """Update access information."""
        self.last_access = time.time()
        self.access_count += 1


@dataclass
class EvictionStats:
    """Statistics for cache eviction operations."""

    total_evictions: int = 0
    evictions_by_strategy: dict[str, int] = field(default_factory=dict)
    evictions_by_trigger: dict[str, int] = field(default_factory=dict)
    total_size_evicted: int = 0
    avg_eviction_time: float = 0.0
    last_eviction_time: float = 0.0
    memory_pressure_evictions: int = 0
    ttl_evictions: int = 0
    size_limit_evictions: int = 0

    def record_eviction(self, strategy: EvictionStrategy, trigger: EvictionTrigger, size_bytes: int, eviction_time: float) -> None:
        """Record an eviction event."""
        self.total_evictions += 1
        self.evictions_by_strategy[strategy.value] = self.evictions_by_strategy.get(strategy.value, 0) + 1
        self.evictions_by_trigger[trigger.value] = self.evictions_by_trigger.get(trigger.value, 0) + 1
        self.total_size_evicted += size_bytes
        self.last_eviction_time = time.time()

        # Update average eviction time
        if self.total_evictions > 1:
            self.avg_eviction_time = (self.avg_eviction_time * (self.total_evictions - 1) + eviction_time) / self.total_evictions
        else:
            self.avg_eviction_time = eviction_time

        # Update specific counters
        if trigger == EvictionTrigger.MEMORY_PRESSURE:
            self.memory_pressure_evictions += 1
        elif trigger == EvictionTrigger.TTL_EXPIRED:
            self.ttl_evictions += 1
        elif trigger == EvictionTrigger.SIZE_LIMIT:
            self.size_limit_evictions += 1


@dataclass
class EvictionConfig:
    """Configuration for cache eviction policies."""

    # Primary strategy
    primary_strategy: EvictionStrategy = EvictionStrategy.LRU

    # Fallback strategies
    fallback_strategies: list[EvictionStrategy] = field(default_factory=lambda: [EvictionStrategy.TTL, EvictionStrategy.RANDOM])

    # Memory pressure thresholds
    memory_pressure_threshold: float = 0.8  # 80% memory usage
    critical_memory_threshold: float = 0.9  # 90% memory usage

    # Eviction batch sizes
    batch_size: int = 100  # Number of entries to evict at once
    aggressive_batch_size: int = 500  # Size for aggressive eviction

    # TTL settings
    ttl_check_interval: float = 300.0  # 5 minutes
    ttl_batch_size: int = 1000  # TTL cleanup batch size

    # Adaptive settings
    adaptive_threshold: int = 1000  # Minimum entries for adaptive behavior
    adaptive_sample_size: int = 100  # Sample size for adaptive analysis

    # Performance settings
    max_eviction_time: float = 1.0  # Maximum time for eviction operation
    parallel_eviction: bool = True  # Enable parallel eviction

    # Custom policy settings
    custom_policy_func: Callable[[list[CacheEntry]], list[CacheEntry]] | None = None


class BaseEvictionPolicy(ABC):
    """Base class for cache eviction policies."""

    def __init__(self, config: EvictionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select candidates for eviction."""
        pass

    @abstractmethod
    def get_strategy(self) -> EvictionStrategy:
        """Get the eviction strategy."""
        pass

    def should_evict(self, entry: CacheEntry) -> bool:
        """Check if an entry should be evicted."""
        return entry.is_expired

    def calculate_priority(self, entry: CacheEntry) -> float:
        """Calculate eviction priority (lower = higher priority for eviction)."""
        return entry.priority


class LRUEvictionPolicy(BaseEvictionPolicy):
    """Least Recently Used eviction policy."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.LRU

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select LRU candidates for eviction."""
        if not entries:
            return []

        # Sort by last access time (oldest first)
        sorted_entries = sorted(entries, key=lambda e: e.last_access)
        return sorted_entries[:target_count]


class LFUEvictionPolicy(BaseEvictionPolicy):
    """Least Frequently Used eviction policy."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.LFU

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select LFU candidates for eviction."""
        if not entries:
            return []

        # Sort by frequency score (lowest first)
        sorted_entries = sorted(entries, key=lambda e: e.frequency_score)
        return sorted_entries[:target_count]


class TTLEvictionPolicy(BaseEvictionPolicy):
    """Time To Live based eviction policy."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.TTL

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select TTL candidates for eviction."""
        if not entries:
            return []

        # First, get all expired entries
        expired_entries = [e for e in entries if e.is_expired]

        if len(expired_entries) >= target_count:
            return expired_entries[:target_count]

        # If not enough expired entries, select by remaining TTL
        non_expired = [e for e in entries if not e.is_expired and e.ttl is not None]
        remaining_ttl = sorted(non_expired, key=lambda e: e.ttl - e.age)

        needed = target_count - len(expired_entries)
        return expired_entries + remaining_ttl[:needed]


class MemoryPressureEvictionPolicy(BaseEvictionPolicy):
    """Memory pressure aware eviction policy."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.MEMORY_PRESSURE

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select candidates based on memory pressure."""
        if not entries:
            return []

        pressure = get_system_memory_pressure()

        # Adjust eviction aggressiveness based on pressure
        if pressure.level == MemoryPressureLevel.CRITICAL:
            # Critical pressure: evict large entries first
            sorted_entries = sorted(entries, key=lambda e: e.size_bytes, reverse=True)
        elif pressure.level == MemoryPressureLevel.HIGH:
            # High pressure: balance size and age
            sorted_entries = sorted(entries, key=lambda e: (e.size_bytes * e.age), reverse=True)
        else:
            # Low/moderate pressure: standard LRU
            sorted_entries = sorted(entries, key=lambda e: e.last_access)

        return sorted_entries[:target_count]


class RandomEvictionPolicy(BaseEvictionPolicy):
    """Random eviction policy."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.RANDOM

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select random candidates for eviction."""
        if not entries:
            return []

        import random

        return random.sample(entries, min(target_count, len(entries)))


class FIFOEvictionPolicy(BaseEvictionPolicy):
    """First In, First Out eviction policy."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.FIFO

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select FIFO candidates for eviction."""
        if not entries:
            return []

        # Sort by creation timestamp (oldest first)
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        return sorted_entries[:target_count]


class AdaptiveEvictionPolicy(BaseEvictionPolicy):
    """Adaptive eviction policy that adjusts based on cache behavior."""

    def __init__(self, config: EvictionConfig):
        super().__init__(config)
        self.lru_policy = LRUEvictionPolicy(config)
        self.lfu_policy = LFUEvictionPolicy(config)
        self.ttl_policy = TTLEvictionPolicy(config)
        self.memory_policy = MemoryPressureEvictionPolicy(config)

        # Adaptive metrics
        self.hit_rate_history: list[float] = []
        self.strategy_performance: dict[EvictionStrategy, float] = {}
        self.current_strategy = EvictionStrategy.LRU
        self.adaptation_interval = 100  # Number of operations between adaptations
        self.operation_count = 0

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.ADAPTIVE

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select candidates using adaptive strategy."""
        if not entries:
            return []

        # Check if we should adapt strategy
        if self.operation_count % self.adaptation_interval == 0:
            await self._adapt_strategy(entries)

        self.operation_count += 1

        # Use current best strategy
        if self.current_strategy == EvictionStrategy.LRU:
            return await self.lru_policy.select_candidates(entries, target_count)
        elif self.current_strategy == EvictionStrategy.LFU:
            return await self.lfu_policy.select_candidates(entries, target_count)
        elif self.current_strategy == EvictionStrategy.TTL:
            return await self.ttl_policy.select_candidates(entries, target_count)
        elif self.current_strategy == EvictionStrategy.MEMORY_PRESSURE:
            return await self.memory_policy.select_candidates(entries, target_count)
        else:
            return await self.lru_policy.select_candidates(entries, target_count)

    async def _adapt_strategy(self, entries: list[CacheEntry]) -> None:
        """Adapt the eviction strategy based on cache behavior."""
        if len(entries) < self.config.adaptive_threshold:
            return

        # Analyze cache patterns
        memory_pressure = get_system_memory_pressure()

        # Check if we have many expired entries
        expired_ratio = sum(1 for e in entries if e.is_expired) / len(entries)

        # Check access patterns
        avg_access_count = sum(e.access_count for e in entries) / len(entries)

        # Decide on strategy
        if memory_pressure.level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            self.current_strategy = EvictionStrategy.MEMORY_PRESSURE
        elif expired_ratio > 0.2:  # 20% expired entries
            self.current_strategy = EvictionStrategy.TTL
        elif avg_access_count > 5:  # High access frequency
            self.current_strategy = EvictionStrategy.LFU
        else:
            self.current_strategy = EvictionStrategy.LRU

        self.logger.debug(f"Adapted eviction strategy to: {self.current_strategy.value}")


class CustomEvictionPolicy(BaseEvictionPolicy):
    """Custom eviction policy using user-defined function."""

    def get_strategy(self) -> EvictionStrategy:
        return EvictionStrategy.CUSTOM

    async def select_candidates(self, entries: list[CacheEntry], target_count: int) -> list[CacheEntry]:
        """Select candidates using custom policy function."""
        if not entries or not self.config.custom_policy_func:
            return []

        try:
            return self.config.custom_policy_func(entries)[:target_count]
        except Exception as e:
            self.logger.error(f"Custom eviction policy failed: {e}")
            # Fallback to LRU
            lru_policy = LRUEvictionPolicy(self.config)
            return await lru_policy.select_candidates(entries, target_count)


class CacheEvictionService:
    """
    Intelligent cache eviction service with multiple strategies and policies.

    This service provides comprehensive cache eviction capabilities including:
    - Multiple eviction strategies (LRU, LFU, TTL, memory-pressure, etc.)
    - Configurable eviction policies per cache type
    - Memory pressure-triggered eviction
    - Performance-optimized eviction algorithms
    - Adaptive eviction based on cache behavior
    """

    def __init__(self, config: EvictionConfig | None = None):
        self.config = config or EvictionConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize eviction policies
        self.policies: dict[EvictionStrategy, BaseEvictionPolicy] = {
            EvictionStrategy.LRU: LRUEvictionPolicy(self.config),
            EvictionStrategy.LFU: LFUEvictionPolicy(self.config),
            EvictionStrategy.TTL: TTLEvictionPolicy(self.config),
            EvictionStrategy.MEMORY_PRESSURE: MemoryPressureEvictionPolicy(self.config),
            EvictionStrategy.RANDOM: RandomEvictionPolicy(self.config),
            EvictionStrategy.FIFO: FIFOEvictionPolicy(self.config),
            EvictionStrategy.ADAPTIVE: AdaptiveEvictionPolicy(self.config),
            EvictionStrategy.CUSTOM: CustomEvictionPolicy(self.config),
        }

        # Cache configurations per cache type
        self.cache_configs: dict[str, EvictionConfig] = {}

        # Statistics
        self.stats = EvictionStats()

        # Background tasks
        self.ttl_cleanup_task: asyncio.Task | None = None
        self.memory_monitor_task: asyncio.Task | None = None

        # Cache registry
        self.cache_registry: dict[str, weakref.ref] = {}
        self.cache_entries: dict[str, dict[str, CacheEntry]] = defaultdict(dict)

        # Locks
        self.eviction_lock = Lock()
        self.stats_lock = Lock()

        # Performance monitoring
        self.eviction_times: list[float] = []
        self.max_eviction_time_history = 100

    async def initialize(self) -> None:
        """Initialize the eviction service."""
        # Start background tasks
        self.ttl_cleanup_task = asyncio.create_task(self._ttl_cleanup_loop())
        self.memory_monitor_task = asyncio.create_task(self._memory_monitor_loop())

        self.logger.info("Cache eviction service initialized")

    async def shutdown(self) -> None:
        """Shutdown the eviction service."""
        # Cancel background tasks
        if self.ttl_cleanup_task:
            self.ttl_cleanup_task.cancel()
            try:
                await self.ttl_cleanup_task
            except asyncio.CancelledError:
                pass

        if self.memory_monitor_task:
            self.memory_monitor_task.cancel()
            try:
                await self.memory_monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Cache eviction service shutdown")

    def register_cache(self, cache_name: str, cache_instance: Any, config: EvictionConfig | None = None) -> None:
        """Register a cache for eviction management."""
        self.cache_registry[cache_name] = weakref.ref(cache_instance)
        if config:
            self.cache_configs[cache_name] = config

        self.logger.info(f"Registered cache '{cache_name}' for eviction management")

    def unregister_cache(self, cache_name: str) -> None:
        """Unregister a cache from eviction management."""
        self.cache_registry.pop(cache_name, None)
        self.cache_configs.pop(cache_name, None)
        self.cache_entries.pop(cache_name, None)

        self.logger.info(f"Unregistered cache '{cache_name}' from eviction management")

    def track_cache_entry(self, cache_name: str, key: str, value: Any, size_bytes: int, ttl: float | None = None) -> None:
        """Track a cache entry for eviction purposes."""
        current_time = time.time()

        entry = CacheEntry(key=key, value=value, timestamp=current_time, last_access=current_time, size_bytes=size_bytes, ttl=ttl)

        self.cache_entries[cache_name][key] = entry

        # Track memory allocation
        track_cache_memory_event(cache_name, CacheMemoryEvent.ALLOCATION, size_bytes / (1024 * 1024), {"key": key})  # Convert to MB

    def update_cache_entry_access(self, cache_name: str, key: str) -> None:
        """Update cache entry access information."""
        if cache_name in self.cache_entries and key in self.cache_entries[cache_name]:
            self.cache_entries[cache_name][key].touch()

    def remove_cache_entry(self, cache_name: str, key: str) -> None:
        """Remove a cache entry from tracking."""
        if cache_name in self.cache_entries and key in self.cache_entries[cache_name]:
            entry = self.cache_entries[cache_name].pop(key)

            # Track memory deallocation
            track_cache_memory_event(
                cache_name,
                CacheMemoryEvent.DEALLOCATION,
                entry.size_bytes / (1024 * 1024),
                {"key": key},  # Convert to MB
            )

    async def evict_entries(
        self,
        cache_name: str,
        target_count: int,
        strategy: EvictionStrategy | None = None,
        trigger: EvictionTrigger = EvictionTrigger.MANUAL,
    ) -> list[str]:
        """Evict cache entries using specified strategy."""
        start_time = time.time()

        with self.eviction_lock:
            # Get cache configuration
            config = self.cache_configs.get(cache_name, self.config)
            eviction_strategy = strategy or config.primary_strategy

            # Get cache entries
            entries = list(self.cache_entries.get(cache_name, {}).values())

            if not entries:
                return []

            # Select eviction policy
            policy = self.policies.get(eviction_strategy)
            if not policy:
                self.logger.warning(f"Unknown eviction strategy: {eviction_strategy}")
                policy = self.policies[EvictionStrategy.LRU]

            # Select candidates for eviction
            candidates = await policy.select_candidates(entries, target_count)

            if not candidates:
                return []

            # Evict candidates
            evicted_keys = []
            total_size_evicted = 0

            for candidate in candidates:
                try:
                    # Remove from tracking
                    self.remove_cache_entry(cache_name, candidate.key)

                    # Get actual cache and remove entry
                    cache_ref = self.cache_registry.get(cache_name)
                    if cache_ref:
                        cache_instance = cache_ref()
                        if cache_instance and hasattr(cache_instance, "delete"):
                            if asyncio.iscoroutinefunction(cache_instance.delete):
                                await cache_instance.delete(candidate.key)
                            else:
                                cache_instance.delete(candidate.key)

                    evicted_keys.append(candidate.key)
                    total_size_evicted += candidate.size_bytes

                    # Track eviction event
                    track_cache_memory_event(
                        cache_name,
                        CacheMemoryEvent.EVICTION,
                        candidate.size_bytes / (1024 * 1024),  # Convert to MB
                        {"key": candidate.key, "strategy": eviction_strategy.value},
                    )

                except Exception as e:
                    self.logger.error(f"Failed to evict entry {candidate.key}: {e}")

            # Record statistics
            eviction_time = time.time() - start_time
            self.stats.record_eviction(eviction_strategy, trigger, total_size_evicted, eviction_time)

            # Track eviction performance
            self.eviction_times.append(eviction_time)
            if len(self.eviction_times) > self.max_eviction_time_history:
                self.eviction_times = self.eviction_times[-50:]

            self.logger.info(
                f"Evicted {len(evicted_keys)} entries from cache '{cache_name}' "
                f"using {eviction_strategy.value} strategy in {eviction_time:.3f}s"
            )

            return evicted_keys

    async def handle_memory_pressure(self, cache_name: str, pressure_level: MemoryPressureLevel) -> int:
        """Handle memory pressure by evicting entries."""
        config = self.cache_configs.get(cache_name, self.config)

        # Determine eviction count based on pressure level
        if pressure_level == MemoryPressureLevel.CRITICAL:
            target_count = config.aggressive_batch_size
        elif pressure_level == MemoryPressureLevel.HIGH:
            target_count = config.batch_size * 2
        else:
            target_count = config.batch_size

        # Limit to available entries
        available_entries = len(self.cache_entries.get(cache_name, {}))
        target_count = min(target_count, available_entries)

        if target_count == 0:
            return 0

        # Evict entries
        evicted_keys = await self.evict_entries(cache_name, target_count, EvictionStrategy.MEMORY_PRESSURE, EvictionTrigger.MEMORY_PRESSURE)

        return len(evicted_keys)

    async def cleanup_expired_entries(self, cache_name: str | None = None) -> dict[str, int]:
        """Clean up expired entries from cache(s)."""
        cleanup_results = {}

        cache_names = [cache_name] if cache_name else list(self.cache_entries.keys())

        for name in cache_names:
            entries = self.cache_entries.get(name, {})
            expired_keys = []

            for key, entry in entries.items():
                if entry.is_expired:
                    expired_keys.append(key)

            if expired_keys:
                evicted_keys = await self.evict_entries(name, len(expired_keys), EvictionStrategy.TTL, EvictionTrigger.TTL_EXPIRED)
                cleanup_results[name] = len(evicted_keys)
            else:
                cleanup_results[name] = 0

        return cleanup_results

    async def _ttl_cleanup_loop(self) -> None:
        """Background TTL cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.config.ttl_check_interval)

                # Clean up expired entries
                results = await self.cleanup_expired_entries()

                total_cleaned = sum(results.values())
                if total_cleaned > 0:
                    self.logger.debug(f"TTL cleanup removed {total_cleaned} expired entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"TTL cleanup error: {e}")

    async def _memory_monitor_loop(self) -> None:
        """Background memory monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check system memory pressure
                pressure = get_system_memory_pressure()

                if pressure.level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    self.logger.warning(f"Memory pressure detected: {pressure.level.value}")

                    # Handle memory pressure for all caches
                    for cache_name in self.cache_entries:
                        try:
                            evicted_count = await self.handle_memory_pressure(cache_name, pressure.level)
                            if evicted_count > 0:
                                self.logger.info(f"Memory pressure evicted {evicted_count} entries from '{cache_name}'")
                        except Exception as e:
                            self.logger.error(f"Memory pressure handling failed for '{cache_name}': {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

    def get_cache_info(self, cache_name: str) -> dict[str, Any]:
        """Get information about a cache."""
        entries = self.cache_entries.get(cache_name, {})

        if not entries:
            return {"cache_name": cache_name, "entry_count": 0}

        total_size = sum(entry.size_bytes for entry in entries.values())
        total_access_count = sum(entry.access_count for entry in entries.values())
        expired_count = sum(1 for entry in entries.values() if entry.is_expired)

        return {
            "cache_name": cache_name,
            "entry_count": len(entries),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_access_count": total_access_count,
            "expired_count": expired_count,
            "avg_access_count": total_access_count / len(entries) if entries else 0,
            "config": self.cache_configs.get(cache_name, self.config).__dict__,
        }

    def get_eviction_stats(self) -> EvictionStats:
        """Get eviction statistics."""
        return self.stats

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for eviction operations."""
        if not self.eviction_times:
            return {"avg_eviction_time": 0.0, "max_eviction_time": 0.0, "min_eviction_time": 0.0}

        return {
            "avg_eviction_time": sum(self.eviction_times) / len(self.eviction_times),
            "max_eviction_time": max(self.eviction_times),
            "min_eviction_time": min(self.eviction_times),
            "total_evictions": len(self.eviction_times),
            "eviction_times_history": self.eviction_times[-10:],  # Last 10 evictions
        }

    def reset_stats(self) -> None:
        """Reset eviction statistics."""
        with self.stats_lock:
            self.stats = EvictionStats()
            self.eviction_times = []

        self.logger.info("Eviction statistics reset")


# Global eviction service instance
_eviction_service: CacheEvictionService | None = None


async def get_eviction_service(config: EvictionConfig | None = None) -> CacheEvictionService:
    """Get the global cache eviction service instance."""
    global _eviction_service
    if _eviction_service is None:
        _eviction_service = CacheEvictionService(config)
        await _eviction_service.initialize()
    return _eviction_service


async def shutdown_eviction_service() -> None:
    """Shutdown the global cache eviction service."""
    global _eviction_service
    if _eviction_service:
        await _eviction_service.shutdown()
        _eviction_service = None
