"""
Cache memory pressure handling service.

This service monitors cache memory usage and coordinates memory pressure responses
across all cache services, providing intelligent cache eviction and adaptive sizing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..utils.memory_utils import (
    CacheMemoryEvent,
    CacheMemoryStats,
    MemoryPressureLevel,
    SystemMemoryPressure,
    add_memory_pressure_callback,
    get_cache_memory_stats,
    get_system_memory_pressure,
    register_cache_service,
    track_cache_memory_event,
    unregister_cache_service,
)


class EvictionStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live based
    SIZE = "size"  # Largest items first
    RANDOM = "random"  # Random eviction
    ADAPTIVE = "adaptive"  # Adaptive strategy based on access patterns


class PressureResponseLevel(Enum):
    """Memory pressure response levels."""

    NONE = "none"  # No action needed
    GENTLE = "gentle"  # Light eviction
    MODERATE = "moderate"  # Moderate eviction
    AGGRESSIVE = "aggressive"  # Heavy eviction
    CRITICAL = "critical"  # Emergency eviction


@dataclass
class CacheEvictionConfig:
    """Configuration for cache eviction behavior."""

    strategy: EvictionStrategy = EvictionStrategy.ADAPTIVE
    batch_size: int = 100
    max_eviction_percentage: float = 0.5  # Maximum 50% of cache can be evicted
    min_retention_count: int = 10  # Minimum items to retain
    ttl_threshold_seconds: float = 300.0  # 5 minutes
    frequency_threshold: int = 5  # Minimum access frequency to retain
    size_threshold_mb: float = 10.0  # Large item threshold
    adaptive_window_seconds: float = 3600.0  # 1 hour window for adaptive strategy


@dataclass
class PressureResponse:
    """Memory pressure response configuration."""

    level: PressureResponseLevel
    eviction_percentage: float
    strategy_override: EvictionStrategy | None = None
    batch_size_multiplier: float = 1.0
    delay_seconds: float = 0.0
    description: str = ""


@dataclass
class CacheMemoryPressureStats:
    """Cache memory pressure handling statistics."""

    total_pressure_events: int = 0
    responses_by_level: dict[PressureResponseLevel, int] = field(default_factory=dict)
    total_evictions: int = 0
    evicted_items_by_strategy: dict[EvictionStrategy, int] = field(default_factory=dict)
    evicted_size_mb: float = 0.0
    response_time_seconds: float = 0.0
    last_pressure_time: float = 0.0
    adaptive_strategy_switches: int = 0
    emergency_cleanups: int = 0

    def record_pressure_event(
        self, level: PressureResponseLevel, evictions: int, strategy: EvictionStrategy, size_mb: float, response_time: float
    ) -> None:
        """Record a pressure response event."""
        self.total_pressure_events += 1
        self.responses_by_level[level] = self.responses_by_level.get(level, 0) + 1
        self.total_evictions += evictions
        self.evicted_items_by_strategy[strategy] = self.evicted_items_by_strategy.get(strategy, 0) + evictions
        self.evicted_size_mb += size_mb
        self.response_time_seconds = response_time
        self.last_pressure_time = time.time()


class CacheMemoryPressureService:
    """Service for handling cache memory pressure across all cache services."""

    def __init__(self):
        """Initialize the cache memory pressure service."""
        self.logger = logging.getLogger(__name__)
        self._registered_caches: dict[str, Any] = {}
        self._eviction_configs: dict[str, CacheEvictionConfig] = {}
        self._pressure_responses: dict[MemoryPressureLevel, PressureResponse] = {}
        self._stats = CacheMemoryPressureStats()
        self._monitoring_task: asyncio.Task | None = None
        self._pressure_handling_enabled = True
        self._response_in_progress: set[str] = set()
        self._global_eviction_config = CacheEvictionConfig()

        # Initialize default pressure responses
        self._initialize_default_responses()

        # Register for memory pressure callbacks
        add_memory_pressure_callback(self._handle_memory_pressure)

    def _initialize_default_responses(self) -> None:
        """Initialize default memory pressure responses."""
        self._pressure_responses = {
            MemoryPressureLevel.LOW: PressureResponse(
                level=PressureResponseLevel.NONE, eviction_percentage=0.0, description="No action needed - memory usage is normal"
            ),
            MemoryPressureLevel.MODERATE: PressureResponse(
                level=PressureResponseLevel.GENTLE,
                eviction_percentage=0.05,  # 5% eviction
                strategy_override=EvictionStrategy.LRU,
                batch_size_multiplier=0.5,
                delay_seconds=1.0,
                description="Gentle eviction - remove least recently used items",
            ),
            MemoryPressureLevel.HIGH: PressureResponse(
                level=PressureResponseLevel.MODERATE,
                eviction_percentage=0.15,  # 15% eviction
                strategy_override=EvictionStrategy.ADAPTIVE,
                batch_size_multiplier=1.0,
                delay_seconds=0.5,
                description="Moderate eviction - adaptive strategy with increased urgency",
            ),
            MemoryPressureLevel.CRITICAL: PressureResponse(
                level=PressureResponseLevel.CRITICAL,
                eviction_percentage=0.30,  # 30% eviction
                strategy_override=EvictionStrategy.SIZE,
                batch_size_multiplier=2.0,
                delay_seconds=0.0,
                description="Critical eviction - remove largest items immediately",
            ),
        }

    async def initialize(self) -> None:
        """Initialize the memory pressure service."""
        self.logger.info("Initializing cache memory pressure service")

        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Cache memory pressure service initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the memory pressure service."""
        self.logger.info("Shutting down cache memory pressure service")

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Cache memory pressure service shutdown complete")

    def register_cache(self, cache_name: str, cache_service: Any, eviction_config: CacheEvictionConfig | None = None) -> None:
        """Register a cache service for memory pressure handling.

        Args:
            cache_name: Unique cache name
            cache_service: Cache service instance
            eviction_config: Optional eviction configuration
        """
        self._registered_caches[cache_name] = cache_service
        self._eviction_configs[cache_name] = eviction_config or CacheEvictionConfig()

        # Register with memory utils
        register_cache_service(cache_name, cache_service)

        self.logger.info(f"Registered cache '{cache_name}' for memory pressure handling")

    def unregister_cache(self, cache_name: str) -> None:
        """Unregister a cache service.

        Args:
            cache_name: Cache name to unregister
        """
        self._registered_caches.pop(cache_name, None)
        self._eviction_configs.pop(cache_name, None)

        # Unregister from memory utils
        unregister_cache_service(cache_name)

        self.logger.info(f"Unregistered cache '{cache_name}' from memory pressure handling")

    def update_eviction_config(self, cache_name: str, config: CacheEvictionConfig) -> None:
        """Update eviction configuration for a cache.

        Args:
            cache_name: Cache name
            config: New eviction configuration
        """
        if cache_name in self._eviction_configs:
            self._eviction_configs[cache_name] = config
            self.logger.info(f"Updated eviction config for cache '{cache_name}'")
        else:
            self.logger.warning(f"Cache '{cache_name}' not registered for eviction config update")

    def update_pressure_response(self, pressure_level: MemoryPressureLevel, response: PressureResponse) -> None:
        """Update memory pressure response configuration.

        Args:
            pressure_level: Memory pressure level
            response: New response configuration
        """
        self._pressure_responses[pressure_level] = response
        self.logger.info(f"Updated pressure response for level {pressure_level.value}")

    def enable_pressure_handling(self) -> None:
        """Enable memory pressure handling."""
        self._pressure_handling_enabled = True
        self.logger.info("Memory pressure handling enabled")

    def disable_pressure_handling(self) -> None:
        """Disable memory pressure handling."""
        self._pressure_handling_enabled = False
        self.logger.info("Memory pressure handling disabled")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for proactive pressure handling."""
        while True:
            try:
                # Check system memory pressure
                system_pressure = get_system_memory_pressure()

                # Check individual cache memory usage
                for cache_name in self._registered_caches:
                    cache_stats = get_cache_memory_stats(cache_name)
                    if isinstance(cache_stats, CacheMemoryStats):
                        await self._check_cache_pressure(cache_name, cache_stats)

                # Check for system-wide pressure
                if system_pressure.level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    await self._handle_system_pressure(system_pressure)

                # Sleep for monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in memory pressure monitoring loop: {e}")
                await asyncio.sleep(5)  # Short sleep before retry

    async def _check_cache_pressure(self, cache_name: str, stats: CacheMemoryStats) -> None:
        """Check individual cache memory pressure."""

        # Check if cache is growing too fast
        if stats.allocation_rate_mb_per_sec > 5.0:  # 5 MB/s allocation rate
            await self._handle_cache_pressure(cache_name, MemoryPressureLevel.MODERATE, "High allocation rate detected")

        # Check if cache has excessive pressure events
        if stats.pressure_events > 10:
            await self._handle_cache_pressure(cache_name, MemoryPressureLevel.HIGH, "Excessive pressure events detected")

    async def _handle_system_pressure(self, system_pressure: SystemMemoryPressure) -> None:
        """Handle system-wide memory pressure."""
        if not self._pressure_handling_enabled:
            return

        self.logger.warning(f"System memory pressure: {system_pressure.level.value} " f"({system_pressure.usage_percent:.1f}%)")

        # Handle pressure for all caches
        for cache_name in self._registered_caches:
            if cache_name not in self._response_in_progress:
                await self._handle_cache_pressure(cache_name, system_pressure.level, "System memory pressure")

    async def _handle_memory_pressure(self, cache_name: str, pressure_level: MemoryPressureLevel) -> None:
        """Handle memory pressure callback from memory utils."""
        if not self._pressure_handling_enabled:
            return

        await self._handle_cache_pressure(cache_name, pressure_level, "Memory pressure callback")

    async def _handle_cache_pressure(self, cache_name: str, pressure_level: MemoryPressureLevel, reason: str) -> None:
        """Handle memory pressure for a specific cache."""
        if cache_name in self._response_in_progress:
            self.logger.debug(f"Pressure response already in progress for cache '{cache_name}'")
            return

        self._response_in_progress.add(cache_name)

        try:
            response = self._pressure_responses.get(pressure_level)
            if not response or response.level == PressureResponseLevel.NONE:
                return

            self.logger.warning(f"Handling {pressure_level.value} pressure for cache '{cache_name}': {reason}")

            start_time = time.time()

            # Apply response delay if configured
            if response.delay_seconds > 0:
                await asyncio.sleep(response.delay_seconds)

            # Perform eviction
            evicted_count, evicted_size_mb = await self._perform_eviction(cache_name, response)

            # Record statistics
            response_time = time.time() - start_time
            strategy = response.strategy_override or self._eviction_configs[cache_name].strategy

            self._stats.record_pressure_event(response.level, evicted_count, strategy, evicted_size_mb, response_time)

            # Track memory event
            track_cache_memory_event(
                cache_name,
                CacheMemoryEvent.PRESSURE,
                evicted_size_mb,
                {
                    "pressure_level": pressure_level.value,
                    "response_level": response.level.value,
                    "evicted_count": evicted_count,
                    "evicted_size_mb": evicted_size_mb,
                    "strategy": strategy.value,
                    "reason": reason,
                    "response_time": response_time,
                },
            )

            self.logger.info(
                f"Pressure response complete for cache '{cache_name}': "
                f"evicted {evicted_count} items ({evicted_size_mb:.1f}MB) "
                f"in {response_time:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"Error handling pressure for cache '{cache_name}': {e}")
        finally:
            self._response_in_progress.discard(cache_name)

    async def _perform_eviction(self, cache_name: str, response: PressureResponse) -> tuple[int, float]:
        """Perform cache eviction based on response configuration.

        Args:
            cache_name: Cache name
            response: Pressure response configuration

        Returns:
            Tuple of (evicted_count, evicted_size_mb)
        """
        cache_service = self._registered_caches.get(cache_name)
        if not cache_service:
            return 0, 0.0

        config = self._eviction_configs.get(cache_name, self._global_eviction_config)
        strategy = response.strategy_override or config.strategy

        # Calculate eviction parameters
        batch_size = int(config.batch_size * response.batch_size_multiplier)

        # Get current cache stats
        cache_stats = get_cache_memory_stats(cache_name)
        if not isinstance(cache_stats, CacheMemoryStats):
            return 0, 0.0

        # Calculate target eviction count
        if hasattr(cache_service, "get_cache_size"):
            total_items = await cache_service.get_cache_size()
        else:
            total_items = 1000  # Default estimate

        max_eviction = int(total_items * min(response.eviction_percentage, config.max_eviction_percentage))
        target_eviction = min(max_eviction, max(0, total_items - config.min_retention_count))

        if target_eviction <= 0:
            return 0, 0.0

        # Perform eviction based on strategy
        evicted_count = 0
        evicted_size_mb = 0.0

        if strategy == EvictionStrategy.LRU:
            evicted_count, evicted_size_mb = await self._evict_lru(cache_service, target_eviction, batch_size)
        elif strategy == EvictionStrategy.LFU:
            evicted_count, evicted_size_mb = await self._evict_lfu(cache_service, target_eviction, batch_size)
        elif strategy == EvictionStrategy.TTL:
            evicted_count, evicted_size_mb = await self._evict_ttl(cache_service, target_eviction, batch_size, config)
        elif strategy == EvictionStrategy.SIZE:
            evicted_count, evicted_size_mb = await self._evict_size(cache_service, target_eviction, batch_size, config)
        elif strategy == EvictionStrategy.RANDOM:
            evicted_count, evicted_size_mb = await self._evict_random(cache_service, target_eviction, batch_size)
        elif strategy == EvictionStrategy.ADAPTIVE:
            evicted_count, evicted_size_mb = await self._evict_adaptive(cache_service, target_eviction, batch_size, config)

        return evicted_count, evicted_size_mb

    async def _evict_lru(self, cache_service: Any, target_eviction: int, batch_size: int) -> tuple[int, float]:
        """Evict least recently used items."""
        # Implementation depends on cache service interface
        # This is a placeholder that should be adapted to actual cache service API
        if hasattr(cache_service, "evict_lru"):
            return await cache_service.evict_lru(target_eviction, batch_size)
        return 0, 0.0

    async def _evict_lfu(self, cache_service: Any, target_eviction: int, batch_size: int) -> tuple[int, float]:
        """Evict least frequently used items."""
        if hasattr(cache_service, "evict_lfu"):
            return await cache_service.evict_lfu(target_eviction, batch_size)
        return 0, 0.0

    async def _evict_ttl(self, cache_service: Any, target_eviction: int, batch_size: int, config: CacheEvictionConfig) -> tuple[int, float]:
        """Evict items based on TTL."""
        if hasattr(cache_service, "evict_expired"):
            return await cache_service.evict_expired(target_eviction, batch_size, config.ttl_threshold_seconds)
        return 0, 0.0

    async def _evict_size(
        self, cache_service: Any, target_eviction: int, batch_size: int, config: CacheEvictionConfig
    ) -> tuple[int, float]:
        """Evict largest items first."""
        if hasattr(cache_service, "evict_largest"):
            return await cache_service.evict_largest(target_eviction, batch_size, config.size_threshold_mb)
        return 0, 0.0

    async def _evict_random(self, cache_service: Any, target_eviction: int, batch_size: int) -> tuple[int, float]:
        """Evict random items."""
        if hasattr(cache_service, "evict_random"):
            return await cache_service.evict_random(target_eviction, batch_size)
        return 0, 0.0

    async def _evict_adaptive(
        self, cache_service: Any, target_eviction: int, batch_size: int, config: CacheEvictionConfig
    ) -> tuple[int, float]:
        """Evict items using adaptive strategy."""
        if hasattr(cache_service, "evict_adaptive"):
            return await cache_service.evict_adaptive(target_eviction, batch_size, config.adaptive_window_seconds)

        # Fallback to LRU if adaptive is not available
        return await self._evict_lru(cache_service, target_eviction, batch_size)

    def get_pressure_stats(self) -> CacheMemoryPressureStats:
        """Get memory pressure handling statistics."""
        return self._stats

    def get_registered_caches(self) -> list[str]:
        """Get list of registered cache names."""
        return list(self._registered_caches.keys())

    def get_eviction_config(self, cache_name: str) -> CacheEvictionConfig | None:
        """Get eviction configuration for a cache."""
        return self._eviction_configs.get(cache_name)

    def get_pressure_response(self, pressure_level: MemoryPressureLevel) -> PressureResponse | None:
        """Get pressure response configuration for a level."""
        return self._pressure_responses.get(pressure_level)

    async def manual_eviction(
        self, cache_name: str, strategy: EvictionStrategy, count: int, reason: str = "Manual eviction"
    ) -> tuple[int, float]:
        """Manually trigger cache eviction.

        Args:
            cache_name: Cache name
            strategy: Eviction strategy to use
            count: Number of items to evict
            reason: Reason for eviction

        Returns:
            Tuple of (evicted_count, evicted_size_mb)
        """
        if cache_name not in self._registered_caches:
            raise ValueError(f"Cache '{cache_name}' not registered")

        self.logger.info(f"Manual eviction requested for cache '{cache_name}': {reason}")

        # Create temporary response configuration
        response = PressureResponse(
            level=PressureResponseLevel.MODERATE,
            eviction_percentage=1.0,  # Will be overridden by count
            strategy_override=strategy,
            description=reason,
        )

        # Perform eviction
        cache_service = self._registered_caches[cache_name]
        config = self._eviction_configs.get(cache_name, self._global_eviction_config)

        # Override target eviction count
        original_method = self._perform_eviction

        async def manual_eviction_wrapper(cache_name: str, response: PressureResponse) -> tuple[int, float]:
            return await self._perform_manual_eviction(cache_service, strategy, count, config.batch_size)

        # Temporarily replace method
        self._perform_eviction = manual_eviction_wrapper

        try:
            evicted_count, evicted_size_mb = await self._perform_eviction(cache_name, response)

            # Track manual eviction event
            track_cache_memory_event(
                cache_name,
                CacheMemoryEvent.EVICTION,
                evicted_size_mb,
                {"type": "manual", "strategy": strategy.value, "evicted_count": evicted_count, "reason": reason},
            )

            self.logger.info(
                f"Manual eviction completed for cache '{cache_name}': " f"evicted {evicted_count} items ({evicted_size_mb:.1f}MB)"
            )

            return evicted_count, evicted_size_mb

        finally:
            # Restore original method
            self._perform_eviction = original_method

    async def _perform_manual_eviction(
        self, cache_service: Any, strategy: EvictionStrategy, count: int, batch_size: int
    ) -> tuple[int, float]:
        """Perform manual eviction with specific count."""
        if strategy == EvictionStrategy.LRU:
            return await self._evict_lru(cache_service, count, batch_size)
        elif strategy == EvictionStrategy.LFU:
            return await self._evict_lfu(cache_service, count, batch_size)
        elif strategy == EvictionStrategy.SIZE:
            return await self._evict_size(cache_service, count, batch_size, CacheEvictionConfig())
        elif strategy == EvictionStrategy.RANDOM:
            return await self._evict_random(cache_service, count, batch_size)
        else:
            return await self._evict_lru(cache_service, count, batch_size)  # Default to LRU

    def reset_stats(self) -> None:
        """Reset pressure handling statistics."""
        self._stats = CacheMemoryPressureStats()
        self.logger.info("Reset cache memory pressure statistics")

    def get_status_report(self) -> dict[str, Any]:
        """Get comprehensive status report."""
        return {
            "enabled": self._pressure_handling_enabled,
            "registered_caches": len(self._registered_caches),
            "active_responses": len(self._response_in_progress),
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "statistics": {
                "total_pressure_events": self._stats.total_pressure_events,
                "total_evictions": self._stats.total_evictions,
                "evicted_size_mb": self._stats.evicted_size_mb,
                "last_pressure_time": self._stats.last_pressure_time,
                "responses_by_level": {level.value: count for level, count in self._stats.responses_by_level.items()},
                "evictions_by_strategy": {strategy.value: count for strategy, count in self._stats.evicted_items_by_strategy.items()},
            },
            "configuration": {
                "pressure_responses": {
                    level.value: {
                        "response_level": response.level.value,
                        "eviction_percentage": response.eviction_percentage,
                        "strategy_override": response.strategy_override.value if response.strategy_override else None,
                        "description": response.description,
                    }
                    for level, response in self._pressure_responses.items()
                }
            },
        }


# Global cache memory pressure service instance
_cache_memory_pressure_service: CacheMemoryPressureService | None = None


def get_cache_memory_pressure_service() -> CacheMemoryPressureService:
    """Get the global cache memory pressure service instance."""
    global _cache_memory_pressure_service
    if _cache_memory_pressure_service is None:
        _cache_memory_pressure_service = CacheMemoryPressureService()
    return _cache_memory_pressure_service


async def initialize_cache_memory_pressure_service() -> None:
    """Initialize the global cache memory pressure service."""
    service = get_cache_memory_pressure_service()
    await service.initialize()


async def shutdown_cache_memory_pressure_service() -> None:
    """Shutdown the global cache memory pressure service."""
    global _cache_memory_pressure_service
    if _cache_memory_pressure_service:
        await _cache_memory_pressure_service.shutdown()
        _cache_memory_pressure_service = None
