"""
Adaptive cache sizing service for memory-aware cache management.

This service dynamically adjusts cache sizes based on system memory availability,
usage patterns, and performance metrics to optimize memory utilization.
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
    get_cache_memory_stats,
    get_system_memory_pressure,
    track_cache_memory_event,
)


class SizingStrategy(Enum):
    """Cache sizing strategies."""

    FIXED = "fixed"  # Fixed size based on configuration
    PROPORTIONAL = "proportional"  # Proportional to available memory
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns
    PERFORMANCE = "performance"  # Performance-based sizing
    HYBRID = "hybrid"  # Combination of strategies


class SizingTrigger(Enum):
    """Triggers for cache resizing."""

    MEMORY_PRESSURE = "memory_pressure"
    USAGE_PATTERN = "usage_pattern"
    PERFORMANCE = "performance"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class CacheSizingConfig:
    """Configuration for adaptive cache sizing."""

    strategy: SizingStrategy = SizingStrategy.ADAPTIVE
    min_size_mb: float = 10.0  # Minimum cache size in MB
    max_size_mb: float = 1000.0  # Maximum cache size in MB
    target_memory_percentage: float = 0.2  # Target 20% of available memory
    growth_factor: float = 1.5  # Growth factor for scaling up
    shrink_factor: float = 0.8  # Shrink factor for scaling down
    resize_threshold_mb: float = 50.0  # Minimum change to trigger resize
    adaptation_window_seconds: float = 300.0  # 5 minutes adaptation window
    performance_weight: float = 0.3  # Weight for performance metrics
    memory_weight: float = 0.4  # Weight for memory metrics
    usage_weight: float = 0.3  # Weight for usage patterns


@dataclass
class CacheUsageMetrics:
    """Cache usage metrics for adaptive sizing."""

    hit_rate: float = 0.0
    miss_rate: float = 0.0
    access_frequency: float = 0.0  # Accesses per second
    average_response_time: float = 0.0  # Average response time in ms
    memory_efficiency: float = 0.0  # Memory efficiency ratio
    eviction_rate: float = 0.0  # Evictions per second
    growth_rate: float = 0.0  # Size growth rate
    fragmentation: float = 0.0  # Memory fragmentation ratio
    last_updated: float = field(default_factory=time.time)

    def update_from_cache_stats(self, stats: CacheMemoryStats) -> None:
        """Update metrics from cache memory statistics."""
        self.memory_efficiency = stats.memory_efficiency
        self.eviction_rate = stats.eviction_count / max(1, time.time() - stats.last_cleanup_time)
        self.growth_rate = stats.allocation_rate_mb_per_sec - stats.deallocation_rate_mb_per_sec
        self.fragmentation = stats.fragmentation_ratio
        self.last_updated = time.time()


@dataclass
class SizingDecision:
    """Cache sizing decision with reasoning."""

    cache_name: str
    current_size_mb: float
    target_size_mb: float
    strategy: SizingStrategy
    trigger: SizingTrigger
    reasoning: str
    confidence: float  # 0.0 to 1.0
    memory_pressure: MemoryPressureLevel
    expected_impact: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AdaptiveSizingStats:
    """Statistics for adaptive cache sizing."""

    total_resizes: int = 0
    successful_resizes: int = 0
    failed_resizes: int = 0
    size_increases: int = 0
    size_decreases: int = 0
    total_memory_saved_mb: float = 0.0
    total_memory_allocated_mb: float = 0.0
    average_resize_time_seconds: float = 0.0
    last_resize_time: float = 0.0
    performance_improvements: int = 0
    performance_degradations: int = 0

    def record_resize(self, size_change_mb: float, duration_seconds: float, successful: bool) -> None:
        """Record a resize operation."""
        self.total_resizes += 1
        if successful:
            self.successful_resizes += 1
            if size_change_mb > 0:
                self.size_increases += 1
                self.total_memory_allocated_mb += size_change_mb
            else:
                self.size_decreases += 1
                self.total_memory_saved_mb += abs(size_change_mb)
        else:
            self.failed_resizes += 1

        self.average_resize_time_seconds = (
            self.average_resize_time_seconds * (self.total_resizes - 1) + duration_seconds
        ) / self.total_resizes
        self.last_resize_time = time.time()


class AdaptiveCacheSizingService:
    """Service for adaptive cache sizing based on memory availability and usage patterns."""

    def __init__(self):
        """Initialize the adaptive cache sizing service."""
        self.logger = logging.getLogger(__name__)
        self._registered_caches: dict[str, Any] = {}
        self._sizing_configs: dict[str, CacheSizingConfig] = {}
        self._usage_metrics: dict[str, CacheUsageMetrics] = {}
        self._sizing_history: list[SizingDecision] = []
        self._stats = AdaptiveSizingStats()

        # Service state
        self._sizing_enabled = True
        self._monitoring_task: asyncio.Task | None = None
        self._resize_in_progress: set[str] = set()
        self._default_config = CacheSizingConfig()

        # Adaptive parameters
        self._memory_history: list[tuple[float, float]] = []  # (timestamp, available_memory_mb)
        self._performance_history: dict[str, list[tuple[float, float]]] = {}  # cache_name -> [(timestamp, metric)]

    async def initialize(self) -> None:
        """Initialize the adaptive cache sizing service."""
        self.logger.info("Initializing adaptive cache sizing service")

        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Adaptive cache sizing service initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the adaptive cache sizing service."""
        self.logger.info("Shutting down adaptive cache sizing service")

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Adaptive cache sizing service shutdown complete")

    def register_cache(self, cache_name: str, cache_service: Any, sizing_config: CacheSizingConfig | None = None) -> None:
        """Register a cache for adaptive sizing.

        Args:
            cache_name: Unique cache name
            cache_service: Cache service instance
            sizing_config: Optional sizing configuration
        """
        self._registered_caches[cache_name] = cache_service
        self._sizing_configs[cache_name] = sizing_config or CacheSizingConfig()
        self._usage_metrics[cache_name] = CacheUsageMetrics()
        self._performance_history[cache_name] = []

        self.logger.info(f"Registered cache '{cache_name}' for adaptive sizing")

    def unregister_cache(self, cache_name: str) -> None:
        """Unregister a cache from adaptive sizing.

        Args:
            cache_name: Cache name to unregister
        """
        self._registered_caches.pop(cache_name, None)
        self._sizing_configs.pop(cache_name, None)
        self._usage_metrics.pop(cache_name, None)
        self._performance_history.pop(cache_name, None)

        self.logger.info(f"Unregistered cache '{cache_name}' from adaptive sizing")

    def update_sizing_config(self, cache_name: str, config: CacheSizingConfig) -> None:
        """Update sizing configuration for a cache.

        Args:
            cache_name: Cache name
            config: New sizing configuration
        """
        if cache_name in self._sizing_configs:
            self._sizing_configs[cache_name] = config
            self.logger.info(f"Updated sizing config for cache '{cache_name}'")
        else:
            self.logger.warning(f"Cache '{cache_name}' not registered for sizing config update")

    def enable_sizing(self) -> None:
        """Enable adaptive cache sizing."""
        self._sizing_enabled = True
        self.logger.info("Adaptive cache sizing enabled")

    def disable_sizing(self) -> None:
        """Disable adaptive cache sizing."""
        self._sizing_enabled = False
        self.logger.info("Adaptive cache sizing disabled")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for adaptive sizing."""
        while True:
            try:
                # Update memory history
                system_pressure = get_system_memory_pressure()
                current_time = time.time()
                self._memory_history.append((current_time, system_pressure.available_mb))

                # Limit history size
                if len(self._memory_history) > 1000:
                    self._memory_history = self._memory_history[-500:]

                # Check each registered cache for sizing opportunities
                for cache_name in self._registered_caches:
                    await self._check_cache_sizing(cache_name)

                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in adaptive sizing monitoring loop: {e}")
                await asyncio.sleep(10)  # Short sleep before retry

    async def _check_cache_sizing(self, cache_name: str) -> None:
        """Check if cache needs resizing based on current conditions."""
        if not self._sizing_enabled or cache_name in self._resize_in_progress:
            return

        # Get current cache statistics
        cache_stats = get_cache_memory_stats(cache_name)
        if not isinstance(cache_stats, CacheMemoryStats):
            return

        # Update usage metrics
        metrics = self._usage_metrics[cache_name]
        metrics.update_from_cache_stats(cache_stats)

        # Get system memory pressure
        system_pressure = get_system_memory_pressure()

        # Determine if resizing is needed
        sizing_decision = await self._analyze_sizing_need(cache_name, cache_stats, system_pressure)

        if (
            sizing_decision
            and abs(sizing_decision.target_size_mb - sizing_decision.current_size_mb) > self._sizing_configs[cache_name].resize_threshold_mb
        ):
            await self._perform_resize(sizing_decision)

    async def _analyze_sizing_need(
        self, cache_name: str, cache_stats: CacheMemoryStats, system_pressure: SystemMemoryPressure
    ) -> SizingDecision | None:
        """Analyze whether cache needs resizing and determine target size."""
        config = self._sizing_configs[cache_name]
        metrics = self._usage_metrics[cache_name]
        current_size_mb = cache_stats.current_size_mb

        # Determine sizing strategy
        if config.strategy == SizingStrategy.FIXED:
            return None  # Fixed size, no resizing

        target_size_mb = current_size_mb
        trigger = SizingTrigger.SCHEDULED
        reasoning = "No change needed"
        confidence = 0.5

        # Memory pressure based sizing
        if system_pressure.level == MemoryPressureLevel.CRITICAL:
            target_size_mb = max(config.min_size_mb, current_size_mb * config.shrink_factor)
            trigger = SizingTrigger.MEMORY_PRESSURE
            reasoning = "Critical memory pressure detected"
            confidence = 0.9
        elif system_pressure.level == MemoryPressureLevel.HIGH:
            target_size_mb = max(config.min_size_mb, current_size_mb * 0.9)
            trigger = SizingTrigger.MEMORY_PRESSURE
            reasoning = "High memory pressure detected"
            confidence = 0.8
        elif system_pressure.level == MemoryPressureLevel.LOW:
            # Consider growing if performance metrics suggest it
            if metrics.hit_rate < 0.8 and metrics.eviction_rate > 0.1:
                available_memory_mb = system_pressure.available_mb
                target_memory_mb = available_memory_mb * config.target_memory_percentage
                target_size_mb = min(config.max_size_mb, max(current_size_mb * config.growth_factor, target_memory_mb))
                trigger = SizingTrigger.PERFORMANCE
                reasoning = "Low hit rate and high eviction rate suggest need for larger cache"
                confidence = 0.7

        # Adaptive strategy considerations
        if config.strategy == SizingStrategy.ADAPTIVE:
            adaptive_target = await self._calculate_adaptive_size(cache_name, cache_stats, system_pressure)
            if adaptive_target != current_size_mb:
                target_size_mb = adaptive_target
                trigger = SizingTrigger.USAGE_PATTERN
                reasoning = "Adaptive algorithm suggests size adjustment"
                confidence = 0.6

        # Performance-based adjustments
        if config.strategy in [SizingStrategy.PERFORMANCE, SizingStrategy.HYBRID]:
            performance_target = await self._calculate_performance_based_size(cache_name, metrics)
            if performance_target != current_size_mb:
                target_size_mb = performance_target
                trigger = SizingTrigger.PERFORMANCE
                reasoning = "Performance metrics suggest size adjustment"
                confidence = 0.7

        # Apply constraints
        target_size_mb = max(config.min_size_mb, min(config.max_size_mb, target_size_mb))

        # Only return decision if there's a significant change
        if abs(target_size_mb - current_size_mb) > config.resize_threshold_mb:
            return SizingDecision(
                cache_name=cache_name,
                current_size_mb=current_size_mb,
                target_size_mb=target_size_mb,
                strategy=config.strategy,
                trigger=trigger,
                reasoning=reasoning,
                confidence=confidence,
                memory_pressure=system_pressure.level,
                expected_impact=self._predict_impact(current_size_mb, target_size_mb, metrics),
            )

        return None

    async def _calculate_adaptive_size(
        self, cache_name: str, cache_stats: CacheMemoryStats, system_pressure: SystemMemoryPressure
    ) -> float:
        """Calculate adaptive cache size based on usage patterns and memory availability."""
        config = self._sizing_configs[cache_name]
        metrics = self._usage_metrics[cache_name]
        current_size_mb = cache_stats.current_size_mb

        # Base size calculation
        base_size = current_size_mb

        # Adjust based on memory availability
        available_memory_mb = system_pressure.available_mb
        memory_factor = min(1.0, available_memory_mb / 1000.0)  # Scale based on available memory

        # Adjust based on usage patterns
        usage_factor = 1.0
        if metrics.hit_rate > 0.9:
            usage_factor = 1.2  # High hit rate, can grow
        elif metrics.hit_rate < 0.5:
            usage_factor = 0.8  # Low hit rate, should shrink

        # Adjust based on eviction patterns
        eviction_factor = 1.0
        if metrics.eviction_rate > 0.1:
            eviction_factor = 1.3  # High eviction, need more space
        elif metrics.eviction_rate < 0.01:
            eviction_factor = 0.9  # Low eviction, can shrink

        # Calculate adaptive size
        adaptive_size = base_size * memory_factor * usage_factor * eviction_factor

        # Apply constraints
        return max(config.min_size_mb, min(config.max_size_mb, adaptive_size))

    async def _calculate_performance_based_size(self, cache_name: str, metrics: CacheUsageMetrics) -> float:
        """Calculate cache size based on performance metrics."""
        config = self._sizing_configs[cache_name]
        cache_stats = get_cache_memory_stats(cache_name)

        if not isinstance(cache_stats, CacheMemoryStats):
            return config.min_size_mb

        current_size_mb = cache_stats.current_size_mb

        # Performance scoring
        performance_score = 0.0

        # Hit rate contribution
        if metrics.hit_rate > 0.9:
            performance_score += 0.3
        elif metrics.hit_rate < 0.5:
            performance_score -= 0.3

        # Response time contribution
        if metrics.average_response_time < 10.0:  # < 10ms
            performance_score += 0.2
        elif metrics.average_response_time > 100.0:  # > 100ms
            performance_score -= 0.2

        # Memory efficiency contribution
        if metrics.memory_efficiency > 0.8:
            performance_score += 0.2
        elif metrics.memory_efficiency < 0.3:
            performance_score -= 0.2

        # Fragmentation contribution
        if metrics.fragmentation < 0.1:
            performance_score += 0.1
        elif metrics.fragmentation > 0.5:
            performance_score -= 0.1

        # Calculate size adjustment
        if performance_score > 0.2:
            target_size = current_size_mb * config.growth_factor
        elif performance_score < -0.2:
            target_size = current_size_mb * config.shrink_factor
        else:
            target_size = current_size_mb

        return max(config.min_size_mb, min(config.max_size_mb, target_size))

    def _predict_impact(self, current_size_mb: float, target_size_mb: float, metrics: CacheUsageMetrics) -> str:
        """Predict the impact of resizing on cache performance."""
        if target_size_mb > current_size_mb:
            return f"Expected improvements: higher hit rate (current: {metrics.hit_rate:.2f}), reduced evictions"
        elif target_size_mb < current_size_mb:
            return f"Expected changes: memory savings of {current_size_mb - target_size_mb:.1f}MB, possible hit rate reduction"
        else:
            return "No significant impact expected"

    async def _perform_resize(self, decision: SizingDecision) -> None:
        """Perform cache resize operation."""
        cache_name = decision.cache_name

        if cache_name in self._resize_in_progress:
            return

        self._resize_in_progress.add(cache_name)

        try:
            start_time = time.time()
            cache_service = self._registered_caches[cache_name]

            self.logger.info(
                f"Resizing cache '{cache_name}' from {decision.current_size_mb:.1f}MB "
                f"to {decision.target_size_mb:.1f}MB - {decision.reasoning}"
            )

            # Perform the resize
            success = await self._resize_cache(cache_service, decision.target_size_mb)

            # Record the operation
            duration = time.time() - start_time
            size_change = decision.target_size_mb - decision.current_size_mb

            self._stats.record_resize(size_change, duration, success)

            # Track memory event
            track_cache_memory_event(
                cache_name,
                CacheMemoryEvent.ALLOCATION if size_change > 0 else CacheMemoryEvent.DEALLOCATION,
                abs(size_change),
                {
                    "resize_type": "adaptive",
                    "trigger": decision.trigger.value,
                    "strategy": decision.strategy.value,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                    "success": success,
                    "duration_seconds": duration,
                },
            )

            # Store decision in history
            self._sizing_history.append(decision)

            # Limit history size
            if len(self._sizing_history) > 1000:
                self._sizing_history = self._sizing_history[-500:]

            if success:
                self.logger.info(f"Successfully resized cache '{cache_name}' in {duration:.2f}s")
            else:
                self.logger.warning(f"Failed to resize cache '{cache_name}' after {duration:.2f}s")

        except Exception as e:
            self.logger.error(f"Error resizing cache '{cache_name}': {e}")
        finally:
            self._resize_in_progress.discard(cache_name)

    async def _resize_cache(self, cache_service: Any, target_size_mb: float) -> bool:
        """Resize cache service to target size."""
        try:
            if hasattr(cache_service, "resize"):
                return await cache_service.resize(target_size_mb)
            elif hasattr(cache_service, "set_max_size"):
                return await cache_service.set_max_size(target_size_mb)
            else:
                self.logger.warning("Cache service does not support resizing")
                return False
        except Exception as e:
            self.logger.error(f"Error during cache resize: {e}")
            return False

    async def manual_resize(self, cache_name: str, target_size_mb: float, reason: str = "Manual resize") -> bool:
        """Manually resize a cache.

        Args:
            cache_name: Cache name
            target_size_mb: Target size in MB
            reason: Reason for manual resize

        Returns:
            bool: True if resize was successful
        """
        if cache_name not in self._registered_caches:
            raise ValueError(f"Cache '{cache_name}' not registered")

        config = self._sizing_configs[cache_name]
        current_size_mb = get_cache_memory_stats(cache_name).current_size_mb

        # Apply constraints
        target_size_mb = max(config.min_size_mb, min(config.max_size_mb, target_size_mb))

        decision = SizingDecision(
            cache_name=cache_name,
            current_size_mb=current_size_mb,
            target_size_mb=target_size_mb,
            strategy=SizingStrategy.FIXED,
            trigger=SizingTrigger.MANUAL,
            reasoning=reason,
            confidence=1.0,
            memory_pressure=get_system_memory_pressure().level,
            expected_impact=f"Manual resize from {current_size_mb:.1f}MB to {target_size_mb:.1f}MB",
        )

        await self._perform_resize(decision)

        return decision.target_size_mb == target_size_mb

    def get_sizing_stats(self) -> AdaptiveSizingStats:
        """Get adaptive sizing statistics."""
        return self._stats

    def get_sizing_history(self, cache_name: str | None = None, limit: int = 100) -> list[SizingDecision]:
        """Get sizing decision history.

        Args:
            cache_name: Optional cache name to filter by
            limit: Maximum number of decisions to return

        Returns:
            List of sizing decisions
        """
        history = self._sizing_history

        if cache_name:
            history = [d for d in history if d.cache_name == cache_name]

        return history[-limit:]

    def get_cache_metrics(self, cache_name: str) -> CacheUsageMetrics | None:
        """Get usage metrics for a cache."""
        return self._usage_metrics.get(cache_name)

    def get_registered_caches(self) -> list[str]:
        """Get list of registered cache names."""
        return list(self._registered_caches.keys())

    def get_sizing_config(self, cache_name: str) -> CacheSizingConfig | None:
        """Get sizing configuration for a cache."""
        return self._sizing_configs.get(cache_name)

    def predict_optimal_size(self, cache_name: str) -> float | None:
        """Predict optimal cache size based on current metrics.

        Args:
            cache_name: Cache name

        Returns:
            Predicted optimal size in MB, or None if not available
        """
        if cache_name not in self._registered_caches:
            return None

        cache_stats = get_cache_memory_stats(cache_name)
        if not isinstance(cache_stats, CacheMemoryStats):
            return None

        system_pressure = get_system_memory_pressure()

        # Use adaptive calculation
        return await self._calculate_adaptive_size(cache_name, cache_stats, system_pressure)

    def get_status_report(self) -> dict[str, Any]:
        """Get comprehensive status report."""
        current_time = time.time()

        # Calculate recent activity
        recent_decisions = [d for d in self._sizing_history if current_time - d.timestamp <= 3600]  # Last hour

        cache_status = {}
        for cache_name in self._registered_caches:
            cache_stats = get_cache_memory_stats(cache_name)
            metrics = self._usage_metrics.get(cache_name)
            config = self._sizing_configs.get(cache_name)

            if isinstance(cache_stats, CacheMemoryStats) and metrics and config:
                cache_status[cache_name] = {
                    "current_size_mb": cache_stats.current_size_mb,
                    "min_size_mb": config.min_size_mb,
                    "max_size_mb": config.max_size_mb,
                    "strategy": config.strategy.value,
                    "hit_rate": metrics.hit_rate,
                    "memory_efficiency": metrics.memory_efficiency,
                    "eviction_rate": metrics.eviction_rate,
                    "fragmentation": metrics.fragmentation,
                    "resize_in_progress": cache_name in self._resize_in_progress,
                }

        return {
            "enabled": self._sizing_enabled,
            "registered_caches": len(self._registered_caches),
            "active_resizes": len(self._resize_in_progress),
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "statistics": {
                "total_resizes": self._stats.total_resizes,
                "successful_resizes": self._stats.successful_resizes,
                "failed_resizes": self._stats.failed_resizes,
                "size_increases": self._stats.size_increases,
                "size_decreases": self._stats.size_decreases,
                "total_memory_saved_mb": self._stats.total_memory_saved_mb,
                "total_memory_allocated_mb": self._stats.total_memory_allocated_mb,
                "average_resize_time_seconds": self._stats.average_resize_time_seconds,
                "recent_decisions": len(recent_decisions),
            },
            "cache_status": cache_status,
            "system_memory": {
                "available_mb": get_system_memory_pressure().available_mb,
                "pressure_level": get_system_memory_pressure().level.value,
                "total_mb": get_system_memory_pressure().total_mb,
            },
        }

    def reset_stats(self) -> None:
        """Reset sizing statistics."""
        self._stats = AdaptiveSizingStats()
        self._sizing_history.clear()
        self._memory_history.clear()
        for cache_name in self._performance_history:
            self._performance_history[cache_name].clear()

        self.logger.info("Reset adaptive sizing statistics")


# Global adaptive cache sizing service instance
_adaptive_sizing_service: AdaptiveCacheSizingService | None = None


def get_adaptive_sizing_service() -> AdaptiveCacheSizingService:
    """Get the global adaptive cache sizing service instance."""
    global _adaptive_sizing_service
    if _adaptive_sizing_service is None:
        _adaptive_sizing_service = AdaptiveCacheSizingService()
    return _adaptive_sizing_service


async def initialize_adaptive_sizing_service() -> None:
    """Initialize the global adaptive cache sizing service."""
    service = get_adaptive_sizing_service()
    await service.initialize()


async def shutdown_adaptive_sizing_service() -> None:
    """Shutdown the global adaptive cache sizing service."""
    global _adaptive_sizing_service
    if _adaptive_sizing_service:
        await _adaptive_sizing_service.shutdown()
        _adaptive_sizing_service = None
