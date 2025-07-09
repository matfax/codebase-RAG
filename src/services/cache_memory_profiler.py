"""
Cache memory usage profiler service.

This module provides comprehensive cache memory profiling capabilities including
memory usage pattern analysis, allocation/deallocation tracking, memory hotspots
detection, and performance profiling for cache operations.
"""

import asyncio
import logging
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Optional, Union

from ..utils.memory_utils import (
    CacheMemoryEvent,
    get_memory_stats,
    get_total_cache_memory_usage,
    track_cache_memory_event,
)


class ProfilingLevel(Enum):
    """Profiling detail levels."""

    BASIC = "basic"  # Basic memory usage tracking
    DETAILED = "detailed"  # Detailed memory patterns
    COMPREHENSIVE = "comprehensive"  # Full profiling with stack traces


class MemoryEventType(Enum):
    """Types of memory events to track."""

    ALLOCATION = "allocation"
    DEALLOCATION = "deallocation"
    RESIZE = "resize"
    CLEANUP = "cleanup"
    EVICTION = "eviction"
    PRESSURE = "pressure"


@dataclass
class MemoryAllocation:
    """Represents a memory allocation event."""

    cache_name: str
    key: str
    size_bytes: int
    timestamp: float
    event_type: MemoryEventType
    thread_id: int
    stack_trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        """Size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    @property
    def age(self) -> float:
        """Age in seconds."""
        return time.time() - self.timestamp


@dataclass
class MemoryProfile:
    """Memory profile for a cache."""

    cache_name: str
    start_time: float
    end_time: float
    total_allocations: int = 0
    total_deallocations: int = 0
    total_allocated_bytes: int = 0
    total_deallocated_bytes: int = 0
    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0
    allocation_rate_mb_per_sec: float = 0.0
    deallocation_rate_mb_per_sec: float = 0.0
    average_allocation_size_bytes: float = 0.0
    memory_efficiency_ratio: float = 0.0
    fragmentation_ratio: float = 0.0
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    events: list[MemoryAllocation] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Profile duration in seconds."""
        return self.end_time - self.start_time

    @property
    def net_memory_bytes(self) -> int:
        """Net memory usage (allocated - deallocated)."""
        return self.total_allocated_bytes - self.total_deallocated_bytes

    @property
    def memory_turnover_ratio(self) -> float:
        """Memory turnover ratio (deallocated / allocated)."""
        if self.total_allocated_bytes == 0:
            return 0.0
        return self.total_deallocated_bytes / self.total_allocated_bytes

    def update_rates(self) -> None:
        """Update allocation and deallocation rates."""
        if self.duration_seconds > 0:
            self.allocation_rate_mb_per_sec = (self.total_allocated_bytes / (1024 * 1024)) / self.duration_seconds
            self.deallocation_rate_mb_per_sec = (self.total_deallocated_bytes / (1024 * 1024)) / self.duration_seconds

    def update_efficiency(self) -> None:
        """Update memory efficiency metrics."""
        if self.total_allocated_bytes > 0:
            self.memory_efficiency_ratio = self.current_memory_bytes / self.total_allocated_bytes
            self.average_allocation_size_bytes = self.total_allocated_bytes / max(1, self.total_allocations)


@dataclass
class MemoryHotspot:
    """Memory hotspot information."""

    cache_name: str
    key_pattern: str
    allocation_count: int
    total_size_bytes: int
    average_size_bytes: float
    stack_trace: list[str]
    first_seen: float
    last_seen: float

    @property
    def duration(self) -> float:
        """Hotspot duration in seconds."""
        return self.last_seen - self.first_seen

    @property
    def allocation_rate(self) -> float:
        """Allocations per second."""
        if self.duration > 0:
            return self.allocation_count / self.duration
        return 0.0


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time."""

    timestamp: float
    total_system_memory_mb: float
    available_system_memory_mb: float
    process_memory_mb: float
    total_cache_memory_mb: float
    cache_memory_breakdown: dict[str, float] = field(default_factory=dict)
    memory_pressure_level: str = "normal"
    gc_stats: dict[str, Any] = field(default_factory=dict)


class CacheMemoryProfiler:
    """
    Comprehensive cache memory profiler.

    This service provides detailed memory profiling capabilities including:
    - Memory allocation/deallocation tracking
    - Memory usage pattern analysis
    - Memory hotspots detection
    - Performance profiling
    - Memory leak detection
    - Memory fragmentation analysis
    """

    def __init__(self, profiling_level: ProfilingLevel = ProfilingLevel.DETAILED):
        self.profiling_level = profiling_level
        self.logger = logging.getLogger(__name__)

        # Profiling state
        self.is_profiling = False
        self.profiling_lock = Lock()

        # Event tracking
        self.memory_events: deque[MemoryAllocation] = deque(maxlen=10000)
        self.event_lock = Lock()

        # Cache profiles
        self.cache_profiles: dict[str, MemoryProfile] = {}
        self.active_profiles: dict[str, MemoryProfile] = {}

        # Memory snapshots
        self.snapshots: deque[MemorySnapshot] = deque(maxlen=1000)
        self.snapshot_interval = 30.0  # seconds

        # Background tasks
        self.snapshot_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None

        # Hotspot detection
        self.hotspot_threshold = 10  # Minimum allocations to be considered a hotspot
        self.hotspot_window = 300.0  # 5 minutes
        self.detected_hotspots: dict[str, MemoryHotspot] = {}

        # Stack trace collection
        self.collect_stack_traces = profiling_level in [ProfilingLevel.DETAILED, ProfilingLevel.COMPREHENSIVE]

        # Performance metrics
        self.allocation_times: deque[float] = deque(maxlen=1000)
        self.deallocation_times: deque[float] = deque(maxlen=1000)

    async def initialize(self) -> None:
        """Initialize the memory profiler."""
        self.is_profiling = True

        # Start tracemalloc if comprehensive profiling is enabled
        if self.profiling_level == ProfilingLevel.COMPREHENSIVE:
            tracemalloc.start()
            self.logger.info("Started tracemalloc for comprehensive profiling")

        # Start background tasks
        self.snapshot_task = asyncio.create_task(self._snapshot_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info(f"Cache memory profiler initialized with {self.profiling_level.value} profiling level")

    async def shutdown(self) -> None:
        """Shutdown the memory profiler."""
        self.is_profiling = False

        # Cancel background tasks
        if self.snapshot_task:
            self.snapshot_task.cancel()
            try:
                await self.snapshot_task
            except asyncio.CancelledError:
                pass

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop tracemalloc
        if self.profiling_level == ProfilingLevel.COMPREHENSIVE:
            tracemalloc.stop()
            self.logger.info("Stopped tracemalloc")

        self.logger.info("Cache memory profiler shutdown")

    def start_cache_profiling(self, cache_name: str) -> None:
        """Start profiling for a specific cache."""
        with self.profiling_lock:
            if cache_name not in self.active_profiles:
                self.active_profiles[cache_name] = MemoryProfile(cache_name=cache_name, start_time=time.time(), end_time=0.0)
                self.logger.info(f"Started profiling for cache '{cache_name}'")

    def stop_cache_profiling(self, cache_name: str) -> MemoryProfile | None:
        """Stop profiling for a specific cache and return the profile."""
        with self.profiling_lock:
            if cache_name in self.active_profiles:
                profile = self.active_profiles.pop(cache_name)
                profile.end_time = time.time()
                profile.update_rates()
                profile.update_efficiency()
                self.cache_profiles[cache_name] = profile
                self.logger.info(f"Stopped profiling for cache '{cache_name}'")
                return profile
            return None

    def track_allocation(
        self,
        cache_name: str,
        key: str,
        size_bytes: int,
        event_type: MemoryEventType = MemoryEventType.ALLOCATION,
        metadata: dict | None = None,
    ) -> None:
        """Track a memory allocation event."""
        if not self.is_profiling:
            return

        current_time = time.time()
        thread_id = asyncio.current_task().get_coro().__name__ if asyncio.current_task() else 0

        # Collect stack trace if enabled
        stack_trace = []
        if self.collect_stack_traces:
            if self.profiling_level == ProfilingLevel.COMPREHENSIVE and tracemalloc.is_tracing():
                current_trace = tracemalloc.get_traced_memory()
                stack_trace = [f"Memory usage: {current_trace[0]} bytes"]
            else:
                # Get simplified stack trace
                import inspect

                frame = inspect.currentframe()
                try:
                    for _ in range(3):  # Skip 3 frames to get to caller
                        frame = frame.f_back
                        if frame is None:
                            break
                    if frame:
                        stack_trace = [f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"]
                finally:
                    del frame

        # Create allocation event
        allocation = MemoryAllocation(
            cache_name=cache_name,
            key=key,
            size_bytes=size_bytes,
            timestamp=current_time,
            event_type=event_type,
            thread_id=thread_id,
            stack_trace=stack_trace,
            metadata=metadata or {},
        )

        # Add to events queue
        with self.event_lock:
            self.memory_events.append(allocation)

        # Update active profile if exists
        with self.profiling_lock:
            if cache_name in self.active_profiles:
                profile = self.active_profiles[cache_name]
                if event_type == MemoryEventType.ALLOCATION:
                    profile.total_allocations += 1
                    profile.total_allocated_bytes += size_bytes
                    profile.current_memory_bytes += size_bytes
                    profile.peak_memory_bytes = max(profile.peak_memory_bytes, profile.current_memory_bytes)
                elif event_type == MemoryEventType.DEALLOCATION:
                    profile.total_deallocations += 1
                    profile.total_deallocated_bytes += size_bytes
                    profile.current_memory_bytes = max(0, profile.current_memory_bytes - size_bytes)

        # Check for hotspots
        self._check_for_hotspots(allocation)

        # Track memory event in global system
        track_cache_memory_event(
            cache_name,
            CacheMemoryEvent.ALLOCATION if event_type == MemoryEventType.ALLOCATION else CacheMemoryEvent.DEALLOCATION,
            size_bytes / (1024 * 1024),
            metadata,
        )

    def track_deallocation(self, cache_name: str, key: str, size_bytes: int, metadata: dict | None = None) -> None:
        """Track a memory deallocation event."""
        self.track_allocation(cache_name, key, size_bytes, MemoryEventType.DEALLOCATION, metadata)

    def _check_for_hotspots(self, allocation: MemoryAllocation) -> None:
        """Check for memory hotspots."""
        if allocation.event_type != MemoryEventType.ALLOCATION:
            return

        current_time = time.time()
        hotspot_key = f"{allocation.cache_name}:{allocation.key[:50]}"  # Truncate key for pattern matching

        if hotspot_key in self.detected_hotspots:
            hotspot = self.detected_hotspots[hotspot_key]
            hotspot.allocation_count += 1
            hotspot.total_size_bytes += allocation.size_bytes
            hotspot.average_size_bytes = hotspot.total_size_bytes / hotspot.allocation_count
            hotspot.last_seen = current_time
        else:
            self.detected_hotspots[hotspot_key] = MemoryHotspot(
                cache_name=allocation.cache_name,
                key_pattern=allocation.key[:50],
                allocation_count=1,
                total_size_bytes=allocation.size_bytes,
                average_size_bytes=allocation.size_bytes,
                stack_trace=allocation.stack_trace,
                first_seen=current_time,
                last_seen=current_time,
            )

        # Clean up old hotspots
        cutoff_time = current_time - self.hotspot_window
        expired_hotspots = [key for key, hotspot in self.detected_hotspots.items() if hotspot.last_seen < cutoff_time]
        for key in expired_hotspots:
            del self.detected_hotspots[key]

    @asynccontextmanager
    async def profile_operation(self, cache_name: str, operation_name: str) -> AsyncIterator[dict]:
        """Context manager for profiling cache operations."""
        start_time = time.time()
        start_memory = get_total_cache_memory_usage()

        operation_context = {"cache_name": cache_name, "operation": operation_name, "start_time": start_time}

        try:
            yield operation_context
        finally:
            end_time = time.time()
            end_memory = get_total_cache_memory_usage()
            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            # Record operation metrics
            operation_context.update(
                {
                    "end_time": end_time,
                    "duration": duration,
                    "memory_delta_mb": memory_delta,
                    "memory_start_mb": start_memory,
                    "memory_end_mb": end_memory,
                }
            )

            # Track in appropriate deque
            if "allocation" in operation_name.lower():
                self.allocation_times.append(duration)
            elif "deallocation" in operation_name.lower():
                self.deallocation_times.append(duration)

            self.logger.debug(
                f"Operation '{operation_name}' on cache '{cache_name}' took {duration:.3f}s, memory delta: {memory_delta:.2f}MB"
            )

    async def take_memory_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        current_time = time.time()
        memory_stats = get_memory_stats()
        cache_memory = get_total_cache_memory_usage()

        # Get cache memory breakdown
        cache_breakdown = {}
        with self.profiling_lock:
            for cache_name, profile in self.active_profiles.items():
                cache_breakdown[cache_name] = profile.current_memory_bytes / (1024 * 1024)

        # Get GC stats
        import gc

        gc_stats = {"counts": gc.get_count(), "stats": gc.get_stats() if hasattr(gc, "get_stats") else {}}

        snapshot = MemorySnapshot(
            timestamp=current_time,
            total_system_memory_mb=memory_stats["system_memory"]["total_mb"],
            available_system_memory_mb=memory_stats["system_memory"]["available_mb"],
            process_memory_mb=memory_stats["rss_mb"],
            total_cache_memory_mb=cache_memory,
            cache_memory_breakdown=cache_breakdown,
            gc_stats=gc_stats,
        )

        # Add to snapshots
        self.snapshots.append(snapshot)

        return snapshot

    def get_memory_trend(self, cache_name: str | None = None, window_minutes: int = 60) -> dict[str, Any]:
        """Get memory usage trend over time."""
        current_time = time.time()
        window_seconds = window_minutes * 60
        cutoff_time = current_time - window_seconds

        # Filter snapshots within window
        relevant_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not relevant_snapshots:
            return {"error": "No snapshots available for the specified time window"}

        # Calculate trends
        if cache_name:
            # Cache-specific trend
            cache_memory_values = [s.cache_memory_breakdown.get(cache_name, 0) for s in relevant_snapshots]
            if not any(cache_memory_values):
                return {"error": f"No memory data available for cache '{cache_name}'"}

            return {
                "cache_name": cache_name,
                "time_window_minutes": window_minutes,
                "data_points": len(relevant_snapshots),
                "memory_values_mb": cache_memory_values,
                "min_memory_mb": min(cache_memory_values),
                "max_memory_mb": max(cache_memory_values),
                "avg_memory_mb": sum(cache_memory_values) / len(cache_memory_values),
                "current_memory_mb": cache_memory_values[-1] if cache_memory_values else 0,
                "trend": "increasing" if cache_memory_values[-1] > cache_memory_values[0] else "decreasing",
            }
        else:
            # System-wide trend
            total_cache_values = [s.total_cache_memory_mb for s in relevant_snapshots]
            process_memory_values = [s.process_memory_mb for s in relevant_snapshots]

            return {
                "time_window_minutes": window_minutes,
                "data_points": len(relevant_snapshots),
                "total_cache_memory_mb": {
                    "values": total_cache_values,
                    "min": min(total_cache_values),
                    "max": max(total_cache_values),
                    "avg": sum(total_cache_values) / len(total_cache_values),
                    "current": total_cache_values[-1] if total_cache_values else 0,
                },
                "process_memory_mb": {
                    "values": process_memory_values,
                    "min": min(process_memory_values),
                    "max": max(process_memory_values),
                    "avg": sum(process_memory_values) / len(process_memory_values),
                    "current": process_memory_values[-1] if process_memory_values else 0,
                },
            }

    def get_allocation_patterns(self, cache_name: str | None = None, window_minutes: int = 60) -> dict[str, Any]:
        """Analyze allocation patterns."""
        current_time = time.time()
        window_seconds = window_minutes * 60
        cutoff_time = current_time - window_seconds

        # Filter events within window
        with self.event_lock:
            relevant_events = [
                e for e in self.memory_events if e.timestamp >= cutoff_time and (cache_name is None or e.cache_name == cache_name)
            ]

        if not relevant_events:
            return {"error": "No allocation events in the specified time window"}

        # Analyze patterns
        allocations = [e for e in relevant_events if e.event_type == MemoryEventType.ALLOCATION]
        deallocations = [e for e in relevant_events if e.event_type == MemoryEventType.DEALLOCATION]

        # Size distribution
        allocation_sizes = [e.size_bytes for e in allocations]
        deallocation_sizes = [e.size_bytes for e in deallocations]

        # Calculate allocation rate (per minute)
        allocation_rate = len(allocations) / (window_minutes) if window_minutes > 0 else 0
        deallocation_rate = len(deallocations) / (window_minutes) if window_minutes > 0 else 0

        # Cache breakdown
        cache_breakdown = defaultdict(lambda: {"allocations": 0, "deallocations": 0, "allocated_bytes": 0, "deallocated_bytes": 0})
        for event in relevant_events:
            cache_stats = cache_breakdown[event.cache_name]
            if event.event_type == MemoryEventType.ALLOCATION:
                cache_stats["allocations"] += 1
                cache_stats["allocated_bytes"] += event.size_bytes
            elif event.event_type == MemoryEventType.DEALLOCATION:
                cache_stats["deallocations"] += 1
                cache_stats["deallocated_bytes"] += event.size_bytes

        return {
            "cache_name": cache_name,
            "time_window_minutes": window_minutes,
            "total_events": len(relevant_events),
            "allocations": {
                "count": len(allocations),
                "rate_per_minute": allocation_rate,
                "total_bytes": sum(allocation_sizes),
                "avg_size_bytes": sum(allocation_sizes) / len(allocation_sizes) if allocation_sizes else 0,
                "min_size_bytes": min(allocation_sizes) if allocation_sizes else 0,
                "max_size_bytes": max(allocation_sizes) if allocation_sizes else 0,
            },
            "deallocations": {
                "count": len(deallocations),
                "rate_per_minute": deallocation_rate,
                "total_bytes": sum(deallocation_sizes),
                "avg_size_bytes": sum(deallocation_sizes) / len(deallocation_sizes) if deallocation_sizes else 0,
                "min_size_bytes": min(deallocation_sizes) if deallocation_sizes else 0,
                "max_size_bytes": max(deallocation_sizes) if deallocation_sizes else 0,
            },
            "cache_breakdown": dict(cache_breakdown),
            "net_allocation_bytes": sum(allocation_sizes) - sum(deallocation_sizes),
        }

    def get_memory_hotspots(self, cache_name: str | None = None, min_allocations: int = 5) -> list[dict[str, Any]]:
        """Get memory hotspots."""
        hotspots = []
        for hotspot in self.detected_hotspots.values():
            if hotspot.allocation_count >= min_allocations and (cache_name is None or hotspot.cache_name == cache_name):
                hotspots.append(
                    {
                        "cache_name": hotspot.cache_name,
                        "key_pattern": hotspot.key_pattern,
                        "allocation_count": hotspot.allocation_count,
                        "total_size_mb": hotspot.total_size_bytes / (1024 * 1024),
                        "average_size_bytes": hotspot.average_size_bytes,
                        "allocation_rate_per_sec": hotspot.allocation_rate,
                        "duration_seconds": hotspot.duration,
                        "first_seen": hotspot.first_seen,
                        "last_seen": hotspot.last_seen,
                        "stack_trace": hotspot.stack_trace,
                    }
                )

        # Sort by total size descending
        hotspots.sort(key=lambda x: x["total_size_mb"], reverse=True)
        return hotspots

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for memory operations."""
        allocation_times = list(self.allocation_times)
        deallocation_times = list(self.deallocation_times)

        def calculate_stats(times: list[float]) -> dict[str, float]:
            if not times:
                return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0, "p99": 0.0}

            sorted_times = sorted(times)
            count = len(sorted_times)
            p95_idx = int(count * 0.95)
            p99_idx = int(count * 0.99)

            return {
                "count": count,
                "avg": sum(sorted_times) / count,
                "min": sorted_times[0],
                "max": sorted_times[-1],
                "p95": sorted_times[p95_idx] if p95_idx < count else sorted_times[-1],
                "p99": sorted_times[p99_idx] if p99_idx < count else sorted_times[-1],
            }

        return {
            "allocation_times": calculate_stats(allocation_times),
            "deallocation_times": calculate_stats(deallocation_times),
            "total_events_tracked": len(self.memory_events),
            "active_profiles": len(self.active_profiles),
            "completed_profiles": len(self.cache_profiles),
            "memory_snapshots": len(self.snapshots),
            "detected_hotspots": len(self.detected_hotspots),
        }

    def get_cache_profile(self, cache_name: str) -> dict[str, Any] | None:
        """Get detailed profile for a specific cache."""
        profile = self.cache_profiles.get(cache_name) or self.active_profiles.get(cache_name)
        if not profile:
            return None

        return {
            "cache_name": profile.cache_name,
            "start_time": profile.start_time,
            "end_time": profile.end_time,
            "duration_seconds": profile.duration_seconds,
            "is_active": cache_name in self.active_profiles,
            "memory_usage": {
                "total_allocated_bytes": profile.total_allocated_bytes,
                "total_deallocated_bytes": profile.total_deallocated_bytes,
                "net_memory_bytes": profile.net_memory_bytes,
                "current_memory_bytes": profile.current_memory_bytes,
                "peak_memory_bytes": profile.peak_memory_bytes,
                "total_allocated_mb": profile.total_allocated_bytes / (1024 * 1024),
                "total_deallocated_mb": profile.total_deallocated_bytes / (1024 * 1024),
                "net_memory_mb": profile.net_memory_bytes / (1024 * 1024),
                "current_memory_mb": profile.current_memory_bytes / (1024 * 1024),
                "peak_memory_mb": profile.peak_memory_bytes / (1024 * 1024),
            },
            "allocation_stats": {
                "total_allocations": profile.total_allocations,
                "total_deallocations": profile.total_deallocations,
                "allocation_rate_mb_per_sec": profile.allocation_rate_mb_per_sec,
                "deallocation_rate_mb_per_sec": profile.deallocation_rate_mb_per_sec,
                "average_allocation_size_bytes": profile.average_allocation_size_bytes,
                "memory_turnover_ratio": profile.memory_turnover_ratio,
            },
            "efficiency_metrics": {
                "memory_efficiency_ratio": profile.memory_efficiency_ratio,
                "fragmentation_ratio": profile.fragmentation_ratio,
            },
            "recent_events": len([e for e in self.memory_events if e.cache_name == cache_name and e.age < 300]),  # Last 5 minutes
        }

    async def _snapshot_loop(self) -> None:
        """Background task for taking memory snapshots."""
        while self.is_profiling:
            try:
                await asyncio.sleep(self.snapshot_interval)
                await self.take_memory_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in snapshot loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old data."""
        while self.is_profiling:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                # Clean up old events (keep last 1 hour)
                cutoff_time = time.time() - 3600
                with self.event_lock:
                    self.memory_events = deque([e for e in self.memory_events if e.timestamp >= cutoff_time], maxlen=10000)

                # Clean up old profiles (keep last 24 hours)
                profile_cutoff = time.time() - 86400
                old_profiles = [name for name, profile in self.cache_profiles.items() if profile.end_time < profile_cutoff]
                for name in old_profiles:
                    del self.cache_profiles[name]

                if old_profiles:
                    self.logger.debug(f"Cleaned up {len(old_profiles)} old profiles")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    def reset_profiling_data(self) -> None:
        """Reset all profiling data."""
        with self.profiling_lock:
            self.cache_profiles.clear()
            self.active_profiles.clear()

        with self.event_lock:
            self.memory_events.clear()

        self.snapshots.clear()
        self.detected_hotspots.clear()
        self.allocation_times.clear()
        self.deallocation_times.clear()

        self.logger.info("All profiling data has been reset")


# Global profiler instance
_memory_profiler: CacheMemoryProfiler | None = None


async def get_memory_profiler(profiling_level: ProfilingLevel = ProfilingLevel.DETAILED) -> CacheMemoryProfiler:
    """Get the global memory profiler instance."""
    global _memory_profiler
    if _memory_profiler is None:
        _memory_profiler = CacheMemoryProfiler(profiling_level)
        await _memory_profiler.initialize()
    return _memory_profiler


async def shutdown_memory_profiler() -> None:
    """Shutdown the global memory profiler."""
    global _memory_profiler
    if _memory_profiler:
        await _memory_profiler.shutdown()
        _memory_profiler = None
