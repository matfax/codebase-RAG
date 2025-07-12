"""
Cache memory leak detection service for identifying and analyzing memory leaks in cache operations.

This module provides comprehensive memory leak detection capabilities including:
- Memory usage tracking over time
- Leak pattern detection and analysis
- Threshold-based leak detection
- Memory growth trend analysis
- Automatic leak reporting and alerting
"""

import asyncio
import gc
import logging
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from weakref import WeakSet

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class LeakSeverity(Enum):
    """Severity levels for memory leaks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LeakType(Enum):
    """Types of memory leaks that can be detected."""

    GRADUAL_GROWTH = "gradual_growth"  # Slow, steady memory growth
    RAPID_GROWTH = "rapid_growth"  # Fast memory growth
    PERIODIC_SPIKE = "periodic_spike"  # Periodic memory spikes
    SUSTAINED_HIGH = "sustained_high"  # Sustained high memory usage
    CACHE_BLOAT = "cache_bloat"  # Cache growing beyond expected limits
    FRAGMENTATION = "fragmentation"  # Memory fragmentation issues


@dataclass
class MemorySnapshot:
    """Represents a point-in-time memory snapshot."""

    timestamp: datetime
    total_memory_mb: float
    rss_memory_mb: float
    vms_memory_mb: float
    cache_memory_mb: float
    cache_entry_count: int
    cache_name: str
    thread_id: int
    gc_stats: dict[str, Any]
    tracemalloc_stats: dict[str, Any] | None = None


@dataclass
class MemoryLeak:
    """Represents a detected memory leak."""

    leak_id: str
    cache_name: str
    leak_type: LeakType
    severity: LeakSeverity
    detected_at: datetime
    start_memory_mb: float
    current_memory_mb: float
    memory_growth_mb: float
    growth_rate_mb_per_minute: float
    duration_minutes: float
    snapshots: list[MemorySnapshot]
    stack_traces: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakDetectionConfig:
    """Configuration for memory leak detection."""

    # Detection thresholds
    memory_growth_threshold_mb: float = 100.0  # MB growth before flagging
    growth_rate_threshold_mb_per_min: float = 5.0  # MB/min growth rate threshold
    sustained_high_threshold_mb: float = 500.0  # Sustained high memory threshold

    # Time windows
    detection_window_minutes: int = 30  # Time window for leak detection
    rapid_growth_window_minutes: int = 5  # Window for rapid growth detection
    baseline_calculation_minutes: int = 60  # Window for baseline calculation

    # Snapshot configuration
    snapshot_interval_seconds: int = 30  # Interval between snapshots
    max_snapshots_per_cache: int = 1000  # Max snapshots to keep per cache

    # Advanced detection
    enable_tracemalloc: bool = True  # Enable tracemalloc for stack traces
    enable_gc_monitoring: bool = True  # Enable garbage collection monitoring
    statistical_analysis: bool = True  # Enable statistical analysis

    # Cleanup
    auto_cleanup_old_data: bool = True  # Auto cleanup old detection data
    cleanup_interval_hours: int = 24  # Hours between cleanup cycles


class CacheMemoryLeakDetector:
    """
    Advanced memory leak detector for cache operations.

    Provides comprehensive memory leak detection including:
    - Real-time memory monitoring
    - Pattern-based leak detection
    - Statistical analysis of memory trends
    - Automatic alerting and reporting
    """

    def __init__(self, config: LeakDetectionConfig | None = None):
        """Initialize the memory leak detector."""
        self.config = config or LeakDetectionConfig()
        self._memory_snapshots: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_snapshots_per_cache))
        self._detected_leaks: dict[str, list[MemoryLeak]] = defaultdict(list)
        self._cache_baselines: dict[str, float] = {}
        self._monitoring_active: dict[str, bool] = defaultdict(bool)
        self._monitoring_tasks: dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()

        # Setup tracemalloc if enabled
        if self.config.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames in stack traces

        # Global monitoring state
        self._global_monitoring = False
        self._global_task: asyncio.Task | None = None

        logger.info("Cache memory leak detector initialized")

    async def start_monitoring(self, cache_name: str) -> None:
        """Start memory leak monitoring for a specific cache."""
        async with asyncio.Lock():
            if self._monitoring_active[cache_name]:
                logger.warning(f"Memory leak monitoring already active for cache: {cache_name}")
                return

            self._monitoring_active[cache_name] = True

            # Create monitoring task
            task = asyncio.create_task(self._monitor_cache_memory(cache_name))
            self._monitoring_tasks[cache_name] = task

            logger.info(f"Started memory leak monitoring for cache: {cache_name}")

    async def stop_monitoring(self, cache_name: str) -> None:
        """Stop memory leak monitoring for a specific cache."""
        async with asyncio.Lock():
            if not self._monitoring_active.get(cache_name, False):
                return

            self._monitoring_active[cache_name] = False

            # Cancel monitoring task
            if cache_name in self._monitoring_tasks:
                task = self._monitoring_tasks.pop(cache_name)
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            logger.info(f"Stopped memory leak monitoring for cache: {cache_name}")

    async def start_global_monitoring(self) -> None:
        """Start global memory leak monitoring for all caches."""
        async with asyncio.Lock():
            if self._global_monitoring:
                logger.warning("Global memory leak monitoring already active")
                return

            self._global_monitoring = True
            self._global_task = asyncio.create_task(self._global_memory_monitor())

            logger.info("Started global memory leak monitoring")

    async def stop_global_monitoring(self) -> None:
        """Stop global memory leak monitoring."""
        async with asyncio.Lock():
            if not self._global_monitoring:
                return

            self._global_monitoring = False

            if self._global_task and not self._global_task.done():
                self._global_task.cancel()
                try:
                    await self._global_task
                except asyncio.CancelledError:
                    pass

            logger.info("Stopped global memory leak monitoring")

    async def take_snapshot(self, cache_name: str, cache_entry_count: int = 0, cache_memory_mb: float = 0.0) -> MemorySnapshot:
        """Take a memory snapshot for the specified cache."""
        timestamp = datetime.now()
        thread_id = threading.get_ident()

        # Get memory stats
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                total_memory_mb = memory_info.rss / (1024 * 1024)
                rss_memory_mb = memory_info.rss / (1024 * 1024)
                vms_memory_mb = memory_info.vms / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Failed to get memory info: {e}")
                total_memory_mb = rss_memory_mb = vms_memory_mb = 0.0
        else:
            total_memory_mb = rss_memory_mb = vms_memory_mb = 0.0

        # Get garbage collection stats
        gc_stats = {}
        if self.config.enable_gc_monitoring:
            try:
                gc_stats = {
                    "counts": gc.get_count(),
                    "threshold": gc.get_threshold(),
                    "collected": gc.collect(0),  # Only collect generation 0
                }
            except Exception as e:
                logger.debug(f"Failed to get GC stats: {e}")

        # Get tracemalloc stats
        tracemalloc_stats = None
        if self.config.enable_tracemalloc and tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_stats = {"current_mb": current / (1024 * 1024), "peak_mb": peak / (1024 * 1024)}
            except Exception as e:
                logger.debug(f"Failed to get tracemalloc stats: {e}")

        snapshot = MemorySnapshot(
            timestamp=timestamp,
            total_memory_mb=total_memory_mb,
            rss_memory_mb=rss_memory_mb,
            vms_memory_mb=vms_memory_mb,
            cache_memory_mb=cache_memory_mb,
            cache_entry_count=cache_entry_count,
            cache_name=cache_name,
            thread_id=thread_id,
            gc_stats=gc_stats,
            tracemalloc_stats=tracemalloc_stats,
        )

        # Store snapshot
        with self._lock:
            self._memory_snapshots[cache_name].append(snapshot)

        return snapshot

    async def analyze_memory_leaks(self, cache_name: str) -> list[MemoryLeak]:
        """Analyze memory snapshots to detect potential memory leaks."""
        with self._lock:
            snapshots = list(self._memory_snapshots.get(cache_name, []))

        if len(snapshots) < 3:
            return []  # Need at least 3 snapshots for analysis

        detected_leaks = []

        # Gradual growth detection
        gradual_leak = await self._detect_gradual_growth(cache_name, snapshots)
        if gradual_leak:
            detected_leaks.append(gradual_leak)

        # Rapid growth detection
        rapid_leak = await self._detect_rapid_growth(cache_name, snapshots)
        if rapid_leak:
            detected_leaks.append(rapid_leak)

        # Sustained high usage detection
        sustained_leak = await self._detect_sustained_high_usage(cache_name, snapshots)
        if sustained_leak:
            detected_leaks.append(sustained_leak)

        # Periodic spike detection
        periodic_leak = await self._detect_periodic_spikes(cache_name, snapshots)
        if periodic_leak:
            detected_leaks.append(periodic_leak)

        # Store detected leaks
        if detected_leaks:
            with self._lock:
                self._detected_leaks[cache_name].extend(detected_leaks)

        return detected_leaks

    async def _detect_gradual_growth(self, cache_name: str, snapshots: list[MemorySnapshot]) -> MemoryLeak | None:
        """Detect gradual memory growth patterns."""
        if len(snapshots) < 10:
            return None

        # Calculate baseline from first 25% of snapshots
        baseline_count = max(3, len(snapshots) // 4)
        baseline_memory = sum(s.total_memory_mb for s in snapshots[:baseline_count]) / baseline_count

        # Get recent memory usage (last 25% of snapshots)
        recent_count = max(3, len(snapshots) // 4)
        recent_memory = sum(s.total_memory_mb for s in snapshots[-recent_count:]) / recent_count

        memory_growth = recent_memory - baseline_memory

        if memory_growth < self.config.memory_growth_threshold_mb:
            return None

        # Calculate growth rate
        time_span = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60
        growth_rate = memory_growth / time_span if time_span > 0 else 0

        if growth_rate < self.config.growth_rate_threshold_mb_per_min:
            return None

        # Determine severity
        severity = self._calculate_leak_severity(memory_growth, growth_rate)

        # Generate recommendations
        recommendations = [
            "Monitor cache eviction policies and ensure they're working correctly",
            "Check for circular references or objects not being properly dereferenced",
            "Review cache size limits and consider implementing stricter bounds",
            "Analyze memory allocation patterns for optimization opportunities",
        ]

        leak = MemoryLeak(
            leak_id=f"{cache_name}_gradual_{int(time.time())}",
            cache_name=cache_name,
            leak_type=LeakType.GRADUAL_GROWTH,
            severity=severity,
            detected_at=datetime.now(),
            start_memory_mb=baseline_memory,
            current_memory_mb=recent_memory,
            memory_growth_mb=memory_growth,
            growth_rate_mb_per_minute=growth_rate,
            duration_minutes=time_span,
            snapshots=snapshots,
            recommendations=recommendations,
            metadata={"baseline_snapshots": baseline_count, "recent_snapshots": recent_count},
        )

        return leak

    async def _detect_rapid_growth(self, cache_name: str, snapshots: list[MemorySnapshot]) -> MemoryLeak | None:
        """Detect rapid memory growth patterns."""
        window_minutes = self.config.rapid_growth_window_minutes
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        # Get snapshots within the rapid growth window
        recent_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]

        if len(recent_snapshots) < 3:
            return None

        start_memory = recent_snapshots[0].total_memory_mb
        end_memory = recent_snapshots[-1].total_memory_mb
        memory_growth = end_memory - start_memory

        time_span = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds() / 60
        growth_rate = memory_growth / time_span if time_span > 0 else 0

        # Check if growth rate exceeds threshold
        rapid_threshold = self.config.growth_rate_threshold_mb_per_min * 3  # 3x normal threshold

        if growth_rate < rapid_threshold:
            return None

        severity = LeakSeverity.HIGH if growth_rate > rapid_threshold * 2 else LeakSeverity.MEDIUM

        recommendations = [
            "Investigate recent operations that may have caused rapid memory growth",
            "Check for batch operations or bulk data loading without proper cleanup",
            "Review memory allocation patterns during high-load periods",
            "Consider implementing circuit breakers for memory-intensive operations",
        ]

        leak = MemoryLeak(
            leak_id=f"{cache_name}_rapid_{int(time.time())}",
            cache_name=cache_name,
            leak_type=LeakType.RAPID_GROWTH,
            severity=severity,
            detected_at=datetime.now(),
            start_memory_mb=start_memory,
            current_memory_mb=end_memory,
            memory_growth_mb=memory_growth,
            growth_rate_mb_per_minute=growth_rate,
            duration_minutes=time_span,
            snapshots=recent_snapshots,
            recommendations=recommendations,
            metadata={"rapid_growth_window_minutes": window_minutes},
        )

        return leak

    async def _detect_sustained_high_usage(self, cache_name: str, snapshots: list[MemorySnapshot]) -> MemoryLeak | None:
        """Detect sustained high memory usage patterns."""
        if len(snapshots) < 10:
            return None

        # Check if memory usage has been consistently high
        high_usage_threshold = self.config.sustained_high_threshold_mb
        high_usage_snapshots = [s for s in snapshots if s.total_memory_mb > high_usage_threshold]

        # Need at least 80% of snapshots to be high usage
        high_usage_ratio = len(high_usage_snapshots) / len(snapshots)

        if high_usage_ratio < 0.8:
            return None

        avg_memory = sum(s.total_memory_mb for s in snapshots) / len(snapshots)
        time_span = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60

        severity = self._calculate_sustained_severity(avg_memory, high_usage_ratio)

        recommendations = [
            "Review cache size limits and consider reducing maximum cache size",
            "Implement more aggressive eviction policies",
            "Analyze cache hit ratios to ensure cache efficiency",
            "Consider partitioning cache across multiple instances",
        ]

        leak = MemoryLeak(
            leak_id=f"{cache_name}_sustained_{int(time.time())}",
            cache_name=cache_name,
            leak_type=LeakType.SUSTAINED_HIGH,
            severity=severity,
            detected_at=datetime.now(),
            start_memory_mb=snapshots[0].total_memory_mb,
            current_memory_mb=snapshots[-1].total_memory_mb,
            memory_growth_mb=avg_memory - snapshots[0].total_memory_mb,
            growth_rate_mb_per_minute=0.0,  # Not applicable for sustained high usage
            duration_minutes=time_span,
            snapshots=snapshots,
            recommendations=recommendations,
            metadata={"high_usage_ratio": high_usage_ratio, "average_memory_mb": avg_memory},
        )

        return leak

    async def _detect_periodic_spikes(self, cache_name: str, snapshots: list[MemorySnapshot]) -> MemoryLeak | None:
        """Detect periodic memory spike patterns."""
        if len(snapshots) < 20:
            return None

        # Calculate memory usage differences between consecutive snapshots
        memory_diffs = []
        for i in range(1, len(snapshots)):
            diff = snapshots[i].total_memory_mb - snapshots[i - 1].total_memory_mb
            memory_diffs.append(diff)

        # Find large positive spikes (>50MB increase)
        spike_threshold = 50.0
        spikes = [i for i, diff in enumerate(memory_diffs) if diff > spike_threshold]

        if len(spikes) < 3:
            return None

        # Check for periodicity
        spike_intervals = []
        for i in range(1, len(spikes)):
            interval = spikes[i] - spikes[i - 1]
            spike_intervals.append(interval)

        # If intervals are similar, it suggests periodic behavior
        if len(spike_intervals) > 0:
            avg_interval = sum(spike_intervals) / len(spike_intervals)
            interval_variance = sum((x - avg_interval) ** 2 for x in spike_intervals) / len(spike_intervals)

            # If variance is low, it's likely periodic
            if interval_variance < (avg_interval * 0.3) ** 2:  # 30% variance threshold
                severity = LeakSeverity.MEDIUM

                recommendations = [
                    "Investigate periodic operations that cause memory spikes",
                    "Review scheduled tasks or batch processing that may cause spikes",
                    "Consider smoothing out memory allocation patterns",
                    "Implement memory pre-allocation for predictable spike patterns",
                ]

                time_span = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60

                leak = MemoryLeak(
                    leak_id=f"{cache_name}_periodic_{int(time.time())}",
                    cache_name=cache_name,
                    leak_type=LeakType.PERIODIC_SPIKE,
                    severity=severity,
                    detected_at=datetime.now(),
                    start_memory_mb=snapshots[0].total_memory_mb,
                    current_memory_mb=snapshots[-1].total_memory_mb,
                    memory_growth_mb=max(memory_diffs),
                    growth_rate_mb_per_minute=0.0,
                    duration_minutes=time_span,
                    snapshots=snapshots,
                    recommendations=recommendations,
                    metadata={"spike_count": len(spikes), "average_interval": avg_interval, "interval_variance": interval_variance},
                )

                return leak

        return None

    def _calculate_leak_severity(self, memory_growth_mb: float, growth_rate_mb_per_min: float) -> LeakSeverity:
        """Calculate the severity of a memory leak based on growth metrics."""
        if memory_growth_mb > 1000 or growth_rate_mb_per_min > 20:
            return LeakSeverity.CRITICAL
        elif memory_growth_mb > 500 or growth_rate_mb_per_min > 10:
            return LeakSeverity.HIGH
        elif memory_growth_mb > 200 or growth_rate_mb_per_min > 5:
            return LeakSeverity.MEDIUM
        else:
            return LeakSeverity.LOW

    def _calculate_sustained_severity(self, avg_memory_mb: float, high_usage_ratio: float) -> LeakSeverity:
        """Calculate severity for sustained high memory usage."""
        if avg_memory_mb > 2000 and high_usage_ratio > 0.95:
            return LeakSeverity.CRITICAL
        elif avg_memory_mb > 1000 and high_usage_ratio > 0.9:
            return LeakSeverity.HIGH
        elif avg_memory_mb > 500 and high_usage_ratio > 0.8:
            return LeakSeverity.MEDIUM
        else:
            return LeakSeverity.LOW

    async def _monitor_cache_memory(self, cache_name: str) -> None:
        """Background task to monitor memory for a specific cache."""
        logger.info(f"Starting memory monitoring loop for cache: {cache_name}")

        try:
            while self._monitoring_active.get(cache_name, False):
                # Take memory snapshot
                await self.take_snapshot(cache_name)

                # Analyze for leaks every 5 snapshots
                with self._lock:
                    snapshot_count = len(self._memory_snapshots.get(cache_name, []))

                if snapshot_count % 5 == 0:
                    leaks = await self.analyze_memory_leaks(cache_name)
                    if leaks:
                        for leak in leaks:
                            logger.warning(
                                f"Memory leak detected: {leak.leak_type.value} " f"in cache {cache_name}, severity: {leak.severity.value}"
                            )

                # Wait for next snapshot
                await asyncio.sleep(self.config.snapshot_interval_seconds)

        except asyncio.CancelledError:
            logger.info(f"Memory monitoring cancelled for cache: {cache_name}")
        except Exception as e:
            logger.error(f"Error in memory monitoring for cache {cache_name}: {e}")

    async def _global_memory_monitor(self) -> None:
        """Global memory monitoring task."""
        logger.info("Starting global memory monitoring loop")

        try:
            while self._global_monitoring:
                # Take global snapshot
                await self.take_snapshot("global", 0, 0.0)

                # Cleanup old data if enabled
                if self.config.auto_cleanup_old_data:
                    await self._cleanup_old_data()

                # Wait for next check
                await asyncio.sleep(self.config.snapshot_interval_seconds * 2)  # Less frequent than cache-specific

        except asyncio.CancelledError:
            logger.info("Global memory monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in global memory monitoring: {e}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.cleanup_interval_hours)

        with self._lock:
            for cache_name in list(self._memory_snapshots.keys()):
                # Clean old snapshots
                snapshots = self._memory_snapshots[cache_name]
                original_count = len(snapshots)

                # Keep only recent snapshots
                while snapshots and snapshots[0].timestamp < cutoff_time:
                    snapshots.popleft()

                cleaned_count = original_count - len(snapshots)
                if cleaned_count > 0:
                    logger.debug(f"Cleaned {cleaned_count} old snapshots for cache: {cache_name}")

                # Clean old leaks
                if cache_name in self._detected_leaks:
                    original_leak_count = len(self._detected_leaks[cache_name])
                    self._detected_leaks[cache_name] = [
                        leak for leak in self._detected_leaks[cache_name] if leak.detected_at >= cutoff_time
                    ]
                    cleaned_leak_count = original_leak_count - len(self._detected_leaks[cache_name])
                    if cleaned_leak_count > 0:
                        logger.debug(f"Cleaned {cleaned_leak_count} old leaks for cache: {cache_name}")

    def get_cache_snapshots(self, cache_name: str, limit: int | None = None) -> list[MemorySnapshot]:
        """Get memory snapshots for a specific cache."""
        with self._lock:
            snapshots = list(self._memory_snapshots.get(cache_name, []))

        if limit:
            return snapshots[-limit:]
        return snapshots

    def get_detected_leaks(self, cache_name: str | None = None, severity: LeakSeverity | None = None) -> list[MemoryLeak]:
        """Get detected memory leaks, optionally filtered by cache name and severity."""
        with self._lock:
            if cache_name:
                leaks = list(self._detected_leaks.get(cache_name, []))
            else:
                leaks = []
                for cache_leaks in self._detected_leaks.values():
                    leaks.extend(cache_leaks)

        if severity:
            leaks = [leak for leak in leaks if leak.severity == severity]

        return sorted(leaks, key=lambda x: x.detected_at, reverse=True)

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status."""
        with self._lock:
            cache_statuses = {}
            for cache_name, active in self._monitoring_active.items():
                snapshot_count = len(self._memory_snapshots.get(cache_name, []))
                leak_count = len(self._detected_leaks.get(cache_name, []))

                cache_statuses[cache_name] = {"monitoring_active": active, "snapshot_count": snapshot_count, "leak_count": leak_count}

        return {
            "global_monitoring": self._global_monitoring,
            "cache_monitoring": cache_statuses,
            "config": {
                "snapshot_interval_seconds": self.config.snapshot_interval_seconds,
                "memory_growth_threshold_mb": self.config.memory_growth_threshold_mb,
                "growth_rate_threshold_mb_per_min": self.config.growth_rate_threshold_mb_per_min,
            },
        }


# Global leak detector instance
_global_leak_detector: CacheMemoryLeakDetector | None = None


async def get_leak_detector(config: LeakDetectionConfig | None = None) -> CacheMemoryLeakDetector:
    """Get or create the global memory leak detector instance."""
    global _global_leak_detector

    if _global_leak_detector is None:
        _global_leak_detector = CacheMemoryLeakDetector(config)

    return _global_leak_detector


async def start_cache_leak_monitoring(cache_name: str) -> None:
    """Start memory leak monitoring for a specific cache."""
    detector = await get_leak_detector()
    await detector.start_monitoring(cache_name)


async def stop_cache_leak_monitoring(cache_name: str) -> None:
    """Stop memory leak monitoring for a specific cache."""
    detector = await get_leak_detector()
    await detector.stop_monitoring(cache_name)


async def analyze_cache_memory_leaks(cache_name: str) -> list[MemoryLeak]:
    """Analyze memory leaks for a specific cache."""
    detector = await get_leak_detector()
    return await detector.analyze_memory_leaks(cache_name)


async def take_cache_memory_snapshot(cache_name: str, cache_entry_count: int = 0, cache_memory_mb: float = 0.0) -> MemorySnapshot:
    """Take a memory snapshot for a specific cache."""
    detector = await get_leak_detector()
    return await detector.take_snapshot(cache_name, cache_entry_count, cache_memory_mb)
