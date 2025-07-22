"""
Cache garbage collection coordinator service.

This service coordinates cache eviction with Python's garbage collection system
to optimize memory usage and prevent memory fragmentation.
"""

import asyncio
import gc
import logging
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from ..utils.memory_utils import (
    CacheMemoryEvent,
    CacheMemoryStats,
    MemoryPressureLevel,
    SystemMemoryPressure,
    get_cache_memory_stats,
    get_system_memory_pressure,
    track_cache_memory_event,
)


class GCTrigger(Enum):
    """Garbage collection trigger types."""

    MANUAL = "manual"
    MEMORY_PRESSURE = "memory_pressure"
    SCHEDULED = "scheduled"
    CACHE_EVICTION = "cache_eviction"
    ALLOCATION_THRESHOLD = "allocation_threshold"
    FRAGMENTATION = "fragmentation"


class GCStrategy(Enum):
    """Garbage collection strategies."""

    INCREMENTAL = "incremental"  # Frequent small collections
    GENERATIONAL = "generational"  # Standard generational GC
    FULL_COLLECTION = "full_collection"  # Complete collection cycle
    ADAPTIVE = "adaptive"  # Adaptive based on memory pressure
    COORDINATED = "coordinated"  # Coordinated with cache eviction


@dataclass
class GCCoordinationConfig:
    """Configuration for garbage collection coordination."""

    strategy: GCStrategy = GCStrategy.ADAPTIVE
    enable_coordination: bool = True
    gc_threshold_mb: float = 100.0  # Trigger GC when memory pressure exceeds this
    max_gc_frequency_seconds: float = 30.0  # Maximum GC frequency
    min_gc_frequency_seconds: float = 300.0  # Minimum GC frequency
    coordination_delay_seconds: float = 1.0  # Delay between cache eviction and GC
    enable_weak_references: bool = True  # Enable weak reference tracking
    enable_fragmentation_detection: bool = True  # Enable fragmentation detection
    fragmentation_threshold: float = 0.3  # Fragmentation threshold for GC trigger


@dataclass
class GCCoordinationStats:
    """Statistics for garbage collection coordination."""

    total_gc_runs: int = 0
    coordinated_gc_runs: int = 0
    manual_gc_runs: int = 0
    scheduled_gc_runs: int = 0
    pressure_triggered_gc_runs: int = 0
    total_objects_collected: int = 0
    total_memory_freed_mb: float = 0.0
    average_gc_time_seconds: float = 0.0
    last_gc_time: float = 0.0
    weak_references_cleaned: int = 0
    fragmentation_events: int = 0

    def record_gc_run(
        self,
        trigger: GCTrigger,
        objects_collected: int,
        memory_freed_mb: float,
        duration_seconds: float,
        weak_refs_cleaned: int = 0,
    ) -> None:
        """Record a garbage collection run."""
        self.total_gc_runs += 1
        self.total_objects_collected += objects_collected
        self.total_memory_freed_mb += memory_freed_mb
        self.weak_references_cleaned += weak_refs_cleaned

        # Update trigger-specific counters
        if trigger == GCTrigger.MANUAL:
            self.manual_gc_runs += 1
        elif trigger == GCTrigger.MEMORY_PRESSURE:
            self.pressure_triggered_gc_runs += 1
        elif trigger == GCTrigger.SCHEDULED:
            self.scheduled_gc_runs += 1
        elif trigger == GCTrigger.CACHE_EVICTION:
            self.coordinated_gc_runs += 1

        # Update averages
        self.average_gc_time_seconds = (self.average_gc_time_seconds * (self.total_gc_runs - 1) + duration_seconds) / self.total_gc_runs
        self.last_gc_time = time.time()


@dataclass
class GCEvent:
    """Garbage collection event information."""

    trigger: GCTrigger
    timestamp: float
    objects_before: int
    objects_after: int
    memory_before_mb: float
    memory_after_mb: float
    duration_seconds: float
    generation: int
    weak_refs_cleaned: int
    fragmentation_before: float
    fragmentation_after: float
    cache_evictions_triggered: int = 0


class CacheGCCoordinatorService:
    """Service for coordinating cache eviction with garbage collection."""

    def __init__(self):
        """Initialize the cache GC coordinator service."""
        self.logger = logging.getLogger(__name__)
        self._config = GCCoordinationConfig()
        self._stats = GCCoordinationStats()

        # Service state
        self._coordination_enabled = True
        self._monitoring_task: asyncio.Task | None = None
        self._gc_in_progress = False
        self._weak_refs: dict[str, weakref.WeakSet] = {}
        self._gc_history: list[GCEvent] = []

        # Callbacks and coordination
        self._cache_eviction_callbacks: list[Callable] = []
        self._pre_gc_callbacks: list[Callable] = []
        self._post_gc_callbacks: list[Callable] = []

        # Timing and thresholds
        self._last_gc_time = 0.0
        self._memory_baseline_mb = 0.0
        self._allocation_counter = 0
        self._allocation_threshold = 10000  # Objects allocated since last GC

    async def initialize(self) -> None:
        """Initialize the cache GC coordinator service."""
        self.logger.info("Initializing cache GC coordinator service")

        # Set up garbage collection configuration
        self._configure_gc()

        # Start monitoring task
        if self._config.enable_coordination:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Initialize weak reference tracking
        if self._config.enable_weak_references:
            self._initialize_weak_reference_tracking()

        # Record baseline memory usage
        self._memory_baseline_mb = get_system_memory_pressure().current_usage_mb

        self.logger.info("Cache GC coordinator service initialized successfully")

    async def shutdown(self) -> None:
        """Shutdown the cache GC coordinator service."""
        self.logger.info("Shutting down cache GC coordinator service")

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Cleanup weak references
        self._cleanup_weak_references()

        self.logger.info("Cache GC coordinator service shutdown complete")

    def _configure_gc(self) -> None:
        """Configure garbage collection settings."""
        # Set GC thresholds based on strategy
        if self._config.strategy == GCStrategy.INCREMENTAL:
            gc.set_threshold(500, 10, 5)  # More frequent collections
        elif self._config.strategy == GCStrategy.GENERATIONAL:
            gc.set_threshold(700, 10, 10)  # Default settings
        elif self._config.strategy == GCStrategy.ADAPTIVE:
            gc.set_threshold(1000, 15, 15)  # Less frequent, adaptive

        # Enable garbage collection
        gc.enable()

        self.logger.info(f"Configured garbage collection with strategy: {self._config.strategy.value}")

    def _initialize_weak_reference_tracking(self) -> None:
        """Initialize weak reference tracking for cache objects."""
        self._weak_refs = {
            "cache_objects": weakref.WeakSet(),
            "cache_entries": weakref.WeakSet(),
            "cache_metadata": weakref.WeakSet(),
        }

        self.logger.info("Initialized weak reference tracking")

    def register_cache_object(self, obj: Any, category: str = "cache_objects") -> None:
        """Register a cache object for weak reference tracking."""
        if self._config.enable_weak_references and category in self._weak_refs:
            self._weak_refs[category].add(obj)

    def register_cache_eviction_callback(self, callback: Callable) -> None:
        """Register a callback to be called before cache eviction."""
        self._cache_eviction_callbacks.append(callback)

    def register_pre_gc_callback(self, callback: Callable) -> None:
        """Register a callback to be called before garbage collection."""
        self._pre_gc_callbacks.append(callback)

    def register_post_gc_callback(self, callback: Callable) -> None:
        """Register a callback to be called after garbage collection."""
        self._post_gc_callbacks.append(callback)

    def update_config(self, config: GCCoordinationConfig) -> None:
        """Update coordination configuration."""
        self._config = config
        self._configure_gc()
        self.logger.info("Updated GC coordination configuration")

    def enable_coordination(self) -> None:
        """Enable garbage collection coordination."""
        self._coordination_enabled = True
        self.logger.info("Enabled GC coordination")

    def disable_coordination(self) -> None:
        """Disable garbage collection coordination."""
        self._coordination_enabled = False
        self.logger.info("Disabled GC coordination")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for GC coordination."""
        while True:
            try:
                # Check system memory pressure
                system_pressure = get_system_memory_pressure()

                # Determine if GC is needed
                gc_needed = await self._should_trigger_gc(system_pressure)

                if gc_needed:
                    await self._perform_coordinated_gc(GCTrigger.MEMORY_PRESSURE)

                # Check for fragmentation
                if self._config.enable_fragmentation_detection:
                    await self._check_fragmentation()

                # Sleep based on memory pressure
                sleep_time = self._calculate_monitoring_interval(system_pressure)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in GC coordination monitoring loop: {e}")
                await asyncio.sleep(10)  # Short sleep before retry

    async def _should_trigger_gc(self, system_pressure: SystemMemoryPressure) -> bool:
        """Determine if garbage collection should be triggered."""
        if not self._coordination_enabled or self._gc_in_progress:
            return False

        current_time = time.time()

        # Check minimum frequency
        if current_time - self._last_gc_time < self._config.max_gc_frequency_seconds:
            return False

        # Check memory pressure threshold
        if system_pressure.level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            return True

        # Check allocation threshold
        if self._allocation_counter >= self._allocation_threshold:
            return True

        # Check maximum frequency
        if current_time - self._last_gc_time >= self._config.min_gc_frequency_seconds:
            return True

        return False

    async def _check_fragmentation(self) -> None:
        """Check for memory fragmentation and trigger GC if needed."""
        # Get current memory statistics
        stats = gc.get_stats()
        if not stats:
            return

        # Calculate fragmentation ratio (simplified)
        gen0_stats = stats[0]
        fragmentation = gen0_stats.get("uncollectable", 0) / max(gen0_stats.get("collections", 1), 1)

        if fragmentation > self._config.fragmentation_threshold:
            self._stats.fragmentation_events += 1
            await self._perform_coordinated_gc(GCTrigger.FRAGMENTATION)

    def _calculate_monitoring_interval(self, system_pressure: SystemMemoryPressure) -> float:
        """Calculate monitoring interval based on memory pressure."""
        if system_pressure.level == MemoryPressureLevel.CRITICAL:
            return 5.0  # Check every 5 seconds
        elif system_pressure.level == MemoryPressureLevel.HIGH:
            return 15.0  # Check every 15 seconds
        elif system_pressure.level == MemoryPressureLevel.MODERATE:
            return 30.0  # Check every 30 seconds
        else:
            return 60.0  # Check every minute

    async def _perform_coordinated_gc(self, trigger: GCTrigger) -> GCEvent:
        """Perform coordinated garbage collection."""
        if self._gc_in_progress:
            return None

        self._gc_in_progress = True
        start_time = time.time()

        try:
            # Get pre-GC statistics
            objects_before = len(gc.get_objects())
            memory_before = get_system_memory_pressure().current_usage_mb
            fragmentation_before = self._calculate_fragmentation()

            # Call pre-GC callbacks
            for callback in self._pre_gc_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error(f"Error in pre-GC callback: {e}")

            # Trigger cache eviction if coordinated
            cache_evictions = 0
            if trigger == GCTrigger.CACHE_EVICTION:
                cache_evictions = await self._trigger_cache_eviction()

                # Wait for coordination delay
                if self._config.coordination_delay_seconds > 0:
                    await asyncio.sleep(self._config.coordination_delay_seconds)

            # Perform garbage collection
            collected_objects = self._perform_gc(trigger)

            # Clean up weak references
            weak_refs_cleaned = self._cleanup_weak_references()

            # Get post-GC statistics
            objects_after = len(gc.get_objects())
            memory_after = get_system_memory_pressure().current_usage_mb
            fragmentation_after = self._calculate_fragmentation()
            duration = time.time() - start_time

            # Create GC event
            gc_event = GCEvent(
                trigger=trigger,
                timestamp=start_time,
                objects_before=objects_before,
                objects_after=objects_after,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                duration_seconds=duration,
                generation=2,  # Full collection
                weak_refs_cleaned=weak_refs_cleaned,
                fragmentation_before=fragmentation_before,
                fragmentation_after=fragmentation_after,
                cache_evictions_triggered=cache_evictions,
            )

            # Record statistics
            memory_freed = max(0, memory_before - memory_after)
            self._stats.record_gc_run(trigger, collected_objects, memory_freed, duration, weak_refs_cleaned)

            # Store event in history
            self._gc_history.append(gc_event)
            if len(self._gc_history) > 1000:
                self._gc_history = self._gc_history[-500:]

            # Track memory event
            track_cache_memory_event(
                "gc_coordinator",
                CacheMemoryEvent.DEALLOCATION,
                memory_freed,
                {
                    "trigger": trigger.value,
                    "objects_collected": collected_objects,
                    "weak_refs_cleaned": weak_refs_cleaned,
                    "duration_seconds": duration,
                    "fragmentation_improvement": fragmentation_before - fragmentation_after,
                },
            )

            # Call post-GC callbacks
            for callback in self._post_gc_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(gc_event)
                    else:
                        callback(gc_event)
                except Exception as e:
                    self.logger.error(f"Error in post-GC callback: {e}")

            # Update counters
            self._last_gc_time = time.time()
            self._allocation_counter = 0

            self.logger.info(
                f"Coordinated GC completed - Trigger: {trigger.value}, "
                f"Objects collected: {collected_objects}, "
                f"Memory freed: {memory_freed:.1f}MB, "
                f"Duration: {duration:.2f}s"
            )

            return gc_event

        except Exception as e:
            self.logger.error(f"Error during coordinated GC: {e}")
            return None
        finally:
            self._gc_in_progress = False

    def _perform_gc(self, trigger: GCTrigger) -> int:
        """Perform garbage collection based on strategy."""
        if self._config.strategy == GCStrategy.INCREMENTAL:
            return gc.collect(0)  # Collect generation 0 only
        elif self._config.strategy == GCStrategy.GENERATIONAL:
            return gc.collect(1)  # Collect generations 0 and 1
        elif self._config.strategy == GCStrategy.FULL_COLLECTION:
            return gc.collect(2)  # Full collection
        elif self._config.strategy == GCStrategy.ADAPTIVE:
            # Adaptive strategy based on trigger
            if trigger == GCTrigger.MEMORY_PRESSURE:
                return gc.collect(2)  # Full collection for pressure
            else:
                return gc.collect(1)  # Generational for other triggers
        else:
            return gc.collect()  # Default collection

    async def _trigger_cache_eviction(self) -> int:
        """Trigger cache eviction before garbage collection."""
        evictions = 0

        for callback in self._cache_eviction_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback()
                else:
                    result = callback()

                if isinstance(result, int):
                    evictions += result

            except Exception as e:
                self.logger.error(f"Error in cache eviction callback: {e}")

        return evictions

    def _cleanup_weak_references(self) -> int:
        """Clean up weak references and return count of cleaned references."""
        cleaned_count = 0

        for category, weak_set in self._weak_refs.items():
            try:
                # Force cleanup of dead references
                initial_count = len(weak_set)
                # Create a new set to force cleanup
                new_set = weakref.WeakSet()
                for obj in weak_set:
                    new_set.add(obj)
                self._weak_refs[category] = new_set
                cleaned_count += initial_count - len(new_set)
            except Exception as e:
                self.logger.error(f"Error cleaning weak references for {category}: {e}")

        return cleaned_count

    def _calculate_fragmentation(self) -> float:
        """Calculate memory fragmentation ratio."""
        try:
            stats = gc.get_stats()
            if not stats:
                return 0.0

            # Simple fragmentation calculation based on uncollectable objects
            gen0_stats = stats[0]
            collections = gen0_stats.get("collections", 1)
            uncollectable = gen0_stats.get("uncollectable", 0)

            return uncollectable / max(collections, 1)
        except Exception:
            return 0.0

    async def manual_gc(self, reason: str = "Manual GC trigger") -> GCEvent:
        """Manually trigger garbage collection."""
        self.logger.info(f"Manual GC triggered: {reason}")
        return await self._perform_coordinated_gc(GCTrigger.MANUAL)

    async def coordinate_with_cache_eviction(self, cache_name: str) -> GCEvent:
        """Coordinate GC with cache eviction."""
        self.logger.info(f"Coordinating GC with cache eviction for '{cache_name}'")
        return await self._perform_coordinated_gc(GCTrigger.CACHE_EVICTION)

    def get_gc_stats(self) -> GCCoordinationStats:
        """Get garbage collection coordination statistics."""
        return self._stats

    def get_gc_history(self, limit: int = 100) -> list[GCEvent]:
        """Get garbage collection history."""
        return self._gc_history[-limit:]

    def get_weak_reference_stats(self) -> dict[str, int]:
        """Get weak reference statistics."""
        return {category: len(weak_set) for category, weak_set in self._weak_refs.items()}

    def get_config(self) -> GCCoordinationConfig:
        """Get current coordination configuration."""
        return self._config

    def get_status_report(self) -> dict[str, Any]:
        """Get comprehensive status report."""
        current_time = time.time()

        # Calculate recent activity
        recent_events = [e for e in self._gc_history if current_time - e.timestamp <= 3600]

        # Get garbage collection statistics
        gc_stats = gc.get_stats()

        return {
            "enabled": self._coordination_enabled,
            "gc_in_progress": self._gc_in_progress,
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "configuration": {
                "strategy": self._config.strategy.value,
                "coordination_enabled": self._config.enable_coordination,
                "weak_references_enabled": self._config.enable_weak_references,
                "fragmentation_detection_enabled": self._config.enable_fragmentation_detection,
                "gc_threshold_mb": self._config.gc_threshold_mb,
                "coordination_delay_seconds": self._config.coordination_delay_seconds,
            },
            "statistics": {
                "total_gc_runs": self._stats.total_gc_runs,
                "coordinated_gc_runs": self._stats.coordinated_gc_runs,
                "manual_gc_runs": self._stats.manual_gc_runs,
                "pressure_triggered_gc_runs": self._stats.pressure_triggered_gc_runs,
                "total_objects_collected": self._stats.total_objects_collected,
                "total_memory_freed_mb": self._stats.total_memory_freed_mb,
                "average_gc_time_seconds": self._stats.average_gc_time_seconds,
                "weak_references_cleaned": self._stats.weak_references_cleaned,
                "fragmentation_events": self._stats.fragmentation_events,
                "recent_events": len(recent_events),
                "last_gc_time": self._stats.last_gc_time,
            },
            "system_gc_stats": gc_stats,
            "weak_reference_counts": self.get_weak_reference_stats(),
            "current_fragmentation": self._calculate_fragmentation(),
            "allocation_counter": self._allocation_counter,
            "time_since_last_gc": current_time - self._last_gc_time,
        }

    def reset_stats(self) -> None:
        """Reset coordination statistics."""
        self._stats = GCCoordinationStats()
        self._gc_history.clear()
        self._allocation_counter = 0
        self._last_gc_time = 0.0

        self.logger.info("Reset GC coordination statistics")


# Global cache GC coordinator service instance
_gc_coordinator_service: CacheGCCoordinatorService | None = None


def get_gc_coordinator_service() -> CacheGCCoordinatorService:
    """Get the global cache GC coordinator service instance."""
    global _gc_coordinator_service
    if _gc_coordinator_service is None:
        _gc_coordinator_service = CacheGCCoordinatorService()
    return _gc_coordinator_service


async def initialize_gc_coordinator_service() -> None:
    """Initialize the global cache GC coordinator service."""
    service = get_gc_coordinator_service()
    await service.initialize()


async def shutdown_gc_coordinator_service() -> None:
    """Shutdown the global cache GC coordinator service."""
    global _gc_coordinator_service
    if _gc_coordinator_service:
        await _gc_coordinator_service.shutdown()
        _gc_coordinator_service = None
