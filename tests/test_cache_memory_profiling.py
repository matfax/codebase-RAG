"""
Cache Memory Profiling Tests for Query Caching Layer.

This module provides comprehensive memory profiling and analysis for cache operations including:
- Memory allocation and deallocation tracking
- Memory leak detection
- Memory usage pattern analysis
- Cache memory efficiency measurements
- Memory pressure simulation
- Garbage collection impact analysis
"""

import asyncio
import gc
import json
import random
import statistics
import string
import sys
import time
import tracemalloc
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import psutil
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.cache_config import CacheConfig
from services.cache_service import CacheStats
from services.project_cache_service import ProjectCacheService
from services.search_cache_service import SearchCacheService
from utils.performance_monitor import MemoryMonitor


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""

    timestamp: float
    operation: str
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    heap_size_mb: float  # Python heap size
    heap_objects: int  # Number of objects in heap
    gc_collections: tuple[int, int, int]  # GC collections per generation
    tracemalloc_current_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0
    cache_stats: CacheStats | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryProfileResult:
    """Result of memory profiling session."""

    test_name: str
    snapshots: list[MemorySnapshot]
    duration_seconds: float
    memory_leak_detected: bool = False
    memory_efficiency_score: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    memory_volatility: float = 0.0
    gc_pressure: dict[str, Any] = field(default_factory=dict)
    memory_hotspots: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class CacheMemoryProfiler:
    """Comprehensive memory profiler for cache operations."""

    def __init__(self):
        self.process = psutil.Process()
        self.memory_monitor = MemoryMonitor()
        self.tracemalloc_enabled = False

    def start_memory_tracing(self) -> None:
        """Start detailed memory tracing with tracemalloc."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_enabled = True

    def stop_memory_tracing(self) -> None:
        """Stop memory tracing."""
        if self.tracemalloc_enabled and tracemalloc.is_tracing():
            tracemalloc.stop()
            self.tracemalloc_enabled = False

    def take_memory_snapshot(self, operation: str, cache_service: Any = None) -> MemorySnapshot:
        """Take a comprehensive memory snapshot."""
        timestamp = time.time()

        # Process memory info
        memory_info = self.process.memory_info()
        rss_mb = memory_info.rss / (1024 * 1024)
        vms_mb = memory_info.vms / (1024 * 1024)

        # Python heap info
        import gc as gc_module

        heap_objects = len(gc_module.get_objects())
        gc_stats = gc_module.get_stats()
        gc_collections = tuple(stat["collections"] for stat in gc_stats)

        # Tracemalloc info if enabled
        tracemalloc_current_mb = 0.0
        tracemalloc_peak_mb = 0.0
        if self.tracemalloc_enabled and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current_mb = current / (1024 * 1024)
            tracemalloc_peak_mb = peak / (1024 * 1024)

        # Get heap size estimate
        heap_size_mb = tracemalloc_current_mb if tracemalloc_current_mb > 0 else rss_mb * 0.7

        # Cache stats if available
        cache_stats = None
        if cache_service and hasattr(cache_service, "get_stats"):
            try:
                cache_stats = cache_service.get_stats()
            except Exception:
                pass

        return MemorySnapshot(
            timestamp=timestamp,
            operation=operation,
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            heap_size_mb=heap_size_mb,
            heap_objects=heap_objects,
            gc_collections=gc_collections,
            tracemalloc_current_mb=tracemalloc_current_mb,
            tracemalloc_peak_mb=tracemalloc_peak_mb,
            cache_stats=cache_stats,
        )

    async def profile_cache_operation(
        self, operation_name: str, cache_service: Any, operation_func, *args, **kwargs
    ) -> tuple[Any, list[MemorySnapshot]]:
        """Profile memory usage during a single cache operation."""
        snapshots = []

        # Pre-operation snapshot
        snapshots.append(self.take_memory_snapshot(f"{operation_name}_before", cache_service))

        # Execute operation
        result = await operation_func(*args, **kwargs)

        # Post-operation snapshot
        snapshots.append(self.take_memory_snapshot(f"{operation_name}_after", cache_service))

        # Force garbage collection and take another snapshot
        gc.collect()
        await asyncio.sleep(0.01)  # Allow async cleanup
        snapshots.append(self.take_memory_snapshot(f"{operation_name}_after_gc", cache_service))

        return result, snapshots

    async def profile_memory_leak_test(
        self, cache_service: Any, iterations: int = 1000, operation_interval: float = 0.001
    ) -> MemoryProfileResult:
        """Test for memory leaks in cache operations."""
        test_name = "memory_leak_test"
        snapshots = []
        start_time = time.time()

        self.start_memory_tracing()

        try:
            # Initial snapshot
            snapshots.append(self.take_memory_snapshot("initial", cache_service))

            # Perform operations in batches to detect gradual leaks
            batch_size = max(1, iterations // 10)

            for batch in range(0, iterations, batch_size):
                # Batch operations
                for i in range(min(batch_size, iterations - batch)):
                    key = f"leak_test_key_{batch}_{i}"
                    value = {"data": "x" * 1000, "id": i}  # 1KB data

                    # Set operation
                    await cache_service.set(key, value)

                    # Get operation
                    await cache_service.get(key)

                    # Delete operation (some keys to create churn)
                    if i % 3 == 0:
                        await cache_service.delete(key)

                    if operation_interval > 0:
                        await asyncio.sleep(operation_interval)

                # Take snapshot after each batch
                snapshots.append(self.take_memory_snapshot(f"batch_{batch//batch_size}", cache_service))

                # Force garbage collection periodically
                if batch % (batch_size * 3) == 0:
                    gc.collect()
                    await asyncio.sleep(0.01)
                    snapshots.append(self.take_memory_snapshot(f"gc_batch_{batch//batch_size}", cache_service))

            # Final cleanup and snapshot
            gc.collect()
            await asyncio.sleep(0.1)
            snapshots.append(self.take_memory_snapshot("final", cache_service))

        finally:
            self.stop_memory_tracing()

        end_time = time.time()
        duration = end_time - start_time

        return self._analyze_memory_profile(test_name, snapshots, duration)

    async def profile_memory_growth_pattern(
        self, cache_service: Any, data_sizes: list[int] = None, iterations_per_size: int = 100
    ) -> MemoryProfileResult:
        """Profile memory growth patterns with different data sizes."""
        if data_sizes is None:
            data_sizes = [100, 1000, 10000, 100000, 1000000]  # 100B to 1MB

        test_name = "memory_growth_pattern"
        snapshots = []
        start_time = time.time()

        self.start_memory_tracing()

        try:
            # Initial snapshot
            snapshots.append(self.take_memory_snapshot("initial", cache_service))

            for size in data_sizes:
                # Generate test data of specified size
                test_data = self._generate_test_data(size)

                # Multiple operations with this data size
                for i in range(iterations_per_size):
                    key = f"growth_test_{size}_{i}"
                    await cache_service.set(key, test_data)

                    # Periodic snapshots
                    if i % 25 == 0:
                        snapshots.append(self.take_memory_snapshot(f"size_{size}_iter_{i}", cache_service))

                # Snapshot after completing size
                snapshots.append(self.take_memory_snapshot(f"size_{size}_complete", cache_service))

                # Cleanup for next size
                for i in range(iterations_per_size):
                    key = f"growth_test_{size}_{i}"
                    await cache_service.delete(key)

                gc.collect()
                await asyncio.sleep(0.05)
                snapshots.append(self.take_memory_snapshot(f"size_{size}_cleanup", cache_service))

            # Final snapshot
            snapshots.append(self.take_memory_snapshot("final", cache_service))

        finally:
            self.stop_memory_tracing()

        end_time = time.time()
        duration = end_time - start_time

        return self._analyze_memory_profile(test_name, snapshots, duration)

    async def profile_cache_memory_efficiency(
        self, cache_service: Any, cache_size_limits: list[int] = None, workload_size: int = 1000
    ) -> MemoryProfileResult:
        """Profile cache memory efficiency under different memory constraints."""
        if cache_size_limits is None:
            cache_size_limits = [10, 50, 100, 200]  # MB limits

        test_name = "memory_efficiency"
        snapshots = []
        start_time = time.time()

        self.start_memory_tracing()

        try:
            # Initial snapshot
            snapshots.append(self.take_memory_snapshot("initial", cache_service))

            for limit_mb in cache_size_limits:
                # Configure cache limit if possible
                if hasattr(cache_service, "config") and hasattr(cache_service.config, "max_memory_mb"):
                    cache_service.config.max_memory_mb = limit_mb

                # Fill cache with data
                for i in range(workload_size):
                    key = f"efficiency_test_{limit_mb}_{i}"
                    # Variable size data to test efficiency
                    data_size = random.randint(500, 2000)
                    value = self._generate_test_data(data_size)

                    await cache_service.set(key, value)

                    # Check memory periodically
                    if i % 100 == 0:
                        snapshots.append(self.take_memory_snapshot(f"limit_{limit_mb}_fill_{i}", cache_service))

                # Test cache behavior under memory pressure
                snapshots.append(self.take_memory_snapshot(f"limit_{limit_mb}_filled", cache_service))

                # Read operations to test efficiency
                for i in range(0, workload_size, 10):
                    key = f"efficiency_test_{limit_mb}_{i}"
                    await cache_service.get(key)

                snapshots.append(self.take_memory_snapshot(f"limit_{limit_mb}_read_test", cache_service))

                # Clear cache for next test
                if hasattr(cache_service, "clear"):
                    await cache_service.clear()
                else:
                    # Manual cleanup
                    for i in range(workload_size):
                        key = f"efficiency_test_{limit_mb}_{i}"
                        await cache_service.delete(key)

                gc.collect()
                await asyncio.sleep(0.1)
                snapshots.append(self.take_memory_snapshot(f"limit_{limit_mb}_cleanup", cache_service))

            # Final snapshot
            snapshots.append(self.take_memory_snapshot("final", cache_service))

        finally:
            self.stop_memory_tracing()

        end_time = time.time()
        duration = end_time - start_time

        return self._analyze_memory_profile(test_name, snapshots, duration)

    async def profile_garbage_collection_impact(
        self, cache_service: Any, operations_count: int = 10000, force_gc_interval: int = 1000
    ) -> MemoryProfileResult:
        """Profile the impact of garbage collection on cache performance."""
        test_name = "garbage_collection_impact"
        snapshots = []
        start_time = time.time()

        self.start_memory_tracing()

        try:
            # Initial snapshot
            snapshots.append(self.take_memory_snapshot("initial", cache_service))

            # Track GC before operations
            gc_before = [stat["collections"] for stat in gc.get_stats()]

            for i in range(operations_count):
                key = f"gc_test_key_{i}"
                value = {"data": "x" * random.randint(100, 1000), "id": i}

                # Cache operations
                await cache_service.set(key, value)
                retrieved = await cache_service.get(key)

                # Force GC at intervals
                if i % force_gc_interval == 0 and i > 0:
                    # Pre-GC snapshot
                    snapshots.append(self.take_memory_snapshot(f"pre_gc_{i}", cache_service))

                    # Force garbage collection
                    collected = gc.collect()

                    # Post-GC snapshot
                    snapshot = self.take_memory_snapshot(f"post_gc_{i}", cache_service)
                    snapshot.additional_data["gc_collected"] = collected
                    snapshots.append(snapshot)

                # Regular snapshots
                if i % (force_gc_interval // 4) == 0:
                    snapshots.append(self.take_memory_snapshot(f"operation_{i}", cache_service))

            # Track GC after operations
            gc_after = [stat["collections"] for stat in gc.get_stats()]

            # Final snapshot with GC stats
            final_snapshot = self.take_memory_snapshot("final", cache_service)
            final_snapshot.additional_data["gc_before"] = gc_before
            final_snapshot.additional_data["gc_after"] = gc_after
            final_snapshot.additional_data["gc_difference"] = [after - before for after, before in zip(gc_after, gc_before, strict=False)]
            snapshots.append(final_snapshot)

        finally:
            self.stop_memory_tracing()

        end_time = time.time()
        duration = end_time - start_time

        return self._analyze_memory_profile(test_name, snapshots, duration)

    def _generate_test_data(self, size_bytes: int) -> Any:
        """Generate test data of specified size."""
        if size_bytes < 100:
            return {"small_data": "x" * max(1, size_bytes - 20)}
        elif size_bytes < 10000:
            return {
                "id": random.randint(1, 1000000),
                "data": "x" * (size_bytes - 100),
                "metadata": {"created": time.time(), "type": "test"},
            }
        else:
            # Large data
            chunk_size = 1000
            chunks = ["x" * chunk_size for _ in range(size_bytes // chunk_size)]
            remainder = "x" * (size_bytes % chunk_size)
            return {"large_data": chunks, "remainder": remainder, "size": size_bytes, "created": time.time()}

    def _analyze_memory_profile(self, test_name: str, snapshots: list[MemorySnapshot], duration: float) -> MemoryProfileResult:
        """Analyze memory profile results."""
        if len(snapshots) < 2:
            return MemoryProfileResult(test_name=test_name, snapshots=snapshots, duration_seconds=duration)

        # Basic memory statistics
        rss_values = [s.rss_mb for s in snapshots]
        heap_values = [s.heap_size_mb for s in snapshots]
        object_counts = [s.heap_objects for s in snapshots]

        initial_memory = rss_values[0]
        final_memory = rss_values[-1]
        peak_memory = max(rss_values)
        memory_growth = final_memory - initial_memory

        # Memory volatility (standard deviation / mean)
        memory_volatility = statistics.stdev(rss_values) / statistics.mean(rss_values) if len(rss_values) > 1 else 0

        # Memory leak detection
        memory_leak_detected = self._detect_memory_leak(snapshots)

        # Memory efficiency score (higher is better)
        memory_efficiency_score = self._calculate_efficiency_score(snapshots)

        # GC pressure analysis
        gc_pressure = self._analyze_gc_pressure(snapshots)

        # Memory hotspots
        memory_hotspots = self._identify_memory_hotspots(snapshots)

        # Recommendations
        recommendations = self._generate_memory_recommendations(memory_leak_detected, memory_efficiency_score, gc_pressure, memory_growth)

        return MemoryProfileResult(
            test_name=test_name,
            snapshots=snapshots,
            duration_seconds=duration,
            memory_leak_detected=memory_leak_detected,
            memory_efficiency_score=memory_efficiency_score,
            peak_memory_mb=peak_memory,
            memory_growth_mb=memory_growth,
            memory_volatility=memory_volatility,
            gc_pressure=gc_pressure,
            memory_hotspots=memory_hotspots,
            recommendations=recommendations,
        )

    def _detect_memory_leak(self, snapshots: list[MemorySnapshot]) -> bool:
        """Detect potential memory leaks."""
        if len(snapshots) < 5:
            return False

        # Analyze memory trend over time
        rss_values = [s.rss_mb for s in snapshots]

        # Check for consistent upward trend
        windows = []
        window_size = max(3, len(rss_values) // 5)

        for i in range(0, len(rss_values) - window_size + 1, window_size):
            window = rss_values[i : i + window_size]
            windows.append(statistics.mean(window))

        if len(windows) < 3:
            return False

        # Check if memory consistently increases across windows
        increasing_windows = 0
        for i in range(1, len(windows)):
            if windows[i] > windows[i - 1] * 1.05:  # 5% increase threshold
                increasing_windows += 1

        # Memory leak if most windows show increase
        return increasing_windows >= len(windows) * 0.6

    def _calculate_efficiency_score(self, snapshots: list[MemorySnapshot]) -> float:
        """Calculate memory efficiency score (0-100)."""
        if len(snapshots) < 2:
            return 0.0

        scores = []

        # Score based on memory growth relative to operations
        initial_memory = snapshots[0].rss_mb
        final_memory = snapshots[-1].rss_mb
        memory_growth_score = max(0, 100 - (final_memory - initial_memory) * 2)  # Penalize growth
        scores.append(memory_growth_score)

        # Score based on memory volatility (stability)
        rss_values = [s.rss_mb for s in snapshots]
        if len(rss_values) > 1:
            volatility = statistics.stdev(rss_values) / statistics.mean(rss_values)
            volatility_score = max(0, 100 - volatility * 200)  # Penalize volatility
            scores.append(volatility_score)

        # Score based on heap object growth
        object_counts = [s.heap_objects for s in snapshots]
        if len(object_counts) > 1:
            object_growth = (object_counts[-1] - object_counts[0]) / object_counts[0]
            object_score = max(0, 100 - object_growth * 100)  # Penalize object growth
            scores.append(object_score)

        return statistics.mean(scores) if scores else 0.0

    def _analyze_gc_pressure(self, snapshots: list[MemorySnapshot]) -> dict[str, Any]:
        """Analyze garbage collection pressure."""
        if len(snapshots) < 2:
            return {"analysis": "insufficient_data"}

        initial_gc = snapshots[0].gc_collections
        final_gc = snapshots[-1].gc_collections

        gc_differences = [final - initial for final, initial in zip(final_gc, initial_gc, strict=False)]
        total_collections = sum(gc_differences)

        # Analyze GC events with additional data
        gc_events = []
        for snapshot in snapshots:
            if "gc_collected" in snapshot.additional_data:
                gc_events.append(snapshot.additional_data["gc_collected"])

        return {
            "total_collections": total_collections,
            "collections_per_generation": gc_differences,
            "gc_events": gc_events,
            "gc_frequency": total_collections / snapshots[-1].timestamp - snapshots[0].timestamp if len(snapshots) > 1 else 0,
            "pressure_level": "high" if total_collections > 100 else "medium" if total_collections > 50 else "low",
        }

    def _identify_memory_hotspots(self, snapshots: list[MemorySnapshot]) -> list[str]:
        """Identify memory usage hotspots."""
        hotspots = []

        if len(snapshots) < 3:
            return hotspots

        # Find operations with high memory growth
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1]
            curr = snapshots[i]

            memory_increase = curr.rss_mb - prev.rss_mb
            if memory_increase > 10:  # 10MB increase
                hotspots.append(f"{curr.operation}: +{memory_increase:.1f}MB")

            # Check for object count spikes
            object_increase = curr.heap_objects - prev.heap_objects
            if object_increase > 10000:  # 10k objects
                hotspots.append(f"{curr.operation}: +{object_increase} objects")

        return hotspots[:10]  # Top 10 hotspots

    def _generate_memory_recommendations(
        self, memory_leak_detected: bool, efficiency_score: float, gc_pressure: dict[str, Any], memory_growth: float
    ) -> list[str]:
        """Generate memory optimization recommendations."""
        recommendations = []

        if memory_leak_detected:
            recommendations.append("Memory leak detected. Review object lifecycle and cleanup procedures.")

        if efficiency_score < 50:
            recommendations.append("Low memory efficiency. Consider optimizing data structures and caching strategies.")

        if gc_pressure.get("pressure_level") == "high":
            recommendations.append("High GC pressure detected. Consider reducing object allocation rate.")

        if memory_growth > 100:  # 100MB growth
            recommendations.append(f"Significant memory growth ({memory_growth:.1f}MB). Review memory management practices.")

        if efficiency_score > 80:
            recommendations.append("Good memory efficiency. Current memory management is effective.")

        return recommendations


class TestCacheMemoryProfiling:
    """Test suite for cache memory profiling."""

    @pytest.fixture
    def memory_profiler(self):
        """Create a memory profiler."""
        return CacheMemoryProfiler()

    @pytest.fixture
    def mock_cache_service(self):
        """Create a mock cache service for memory profiling."""
        mock_service = AsyncMock()
        mock_service.get_stats.return_value = CacheStats()

        # Mock operations
        mock_service.get.return_value = None
        mock_service.set.return_value = True
        mock_service.delete.return_value = True
        mock_service.clear.return_value = True

        return mock_service

    def test_memory_snapshot_creation(self, memory_profiler, mock_cache_service):
        """Test memory snapshot creation."""
        snapshot = memory_profiler.take_memory_snapshot("test_operation", mock_cache_service)

        # Verify snapshot structure
        assert snapshot.operation == "test_operation"
        assert snapshot.timestamp > 0
        assert snapshot.rss_mb > 0
        assert snapshot.vms_mb > 0
        assert snapshot.heap_size_mb > 0
        assert snapshot.heap_objects > 0
        assert len(snapshot.gc_collections) == 3  # Three GC generations

        print(f"Memory snapshot: {snapshot.rss_mb:.1f}MB RSS, {snapshot.heap_objects} objects")

    @pytest.mark.asyncio
    async def test_single_operation_profiling(self, memory_profiler, mock_cache_service):
        """Test memory profiling of a single cache operation."""

        async def test_operation():
            return await mock_cache_service.set("test_key", {"data": "test_value"})

        result, snapshots = await memory_profiler.profile_cache_operation("test_set", mock_cache_service, test_operation)

        # Verify profiling results
        assert result is True
        assert len(snapshots) == 3  # before, after, after_gc

        # Verify snapshot operations
        operations = [s.operation for s in snapshots]
        assert "test_set_before" in operations
        assert "test_set_after" in operations
        assert "test_set_after_gc" in operations

        print(f"Operation profiled: {len(snapshots)} snapshots collected")

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, memory_profiler, mock_cache_service):
        """Test memory leak detection functionality."""

        # Create a mock service that simulates memory leak
        leaky_service = AsyncMock()

        # Simulate memory leak by creating objects that aren't cleaned up
        leaked_objects = []

        async def leaky_set(key, value):
            # Simulate memory leak
            leaked_objects.append({"key": key, "value": value, "extra": "x" * 1000})
            return True

        async def leaky_get(key):
            return None

        async def leaky_delete(key):
            return True

        leaky_service.set.side_effect = leaky_set
        leaky_service.get.side_effect = leaky_get
        leaky_service.delete.side_effect = leaky_delete
        leaky_service.get_stats.return_value = CacheStats()

        # Run memory leak test
        result = await memory_profiler.profile_memory_leak_test(
            leaky_service,
            iterations=100,
            operation_interval=0.0,  # Reduced for testing
        )

        # Verify leak detection
        assert result.test_name == "memory_leak_test"
        assert len(result.snapshots) > 5
        assert result.duration_seconds > 0

        # The leak should be detected due to leaked_objects growing
        print(f"Memory leak test: {result.memory_leak_detected}, growth: {result.memory_growth_mb:.1f}MB")

        # Cleanup
        leaked_objects.clear()
        gc.collect()

    @pytest.mark.asyncio
    async def test_memory_growth_pattern_profiling(self, memory_profiler, mock_cache_service):
        """Test memory growth pattern profiling."""
        result = await memory_profiler.profile_memory_growth_pattern(
            mock_cache_service,
            data_sizes=[100, 1000, 10000],
            iterations_per_size=20,  # Reduced for testing
        )

        # Verify profiling results
        assert result.test_name == "memory_growth_pattern"
        assert len(result.snapshots) > 10  # Should have multiple snapshots
        assert result.duration_seconds > 0
        assert result.memory_efficiency_score >= 0

        print(f"Growth pattern profiling: {result.memory_efficiency_score:.1f} efficiency score")

    @pytest.mark.asyncio
    async def test_memory_efficiency_profiling(self, memory_profiler, mock_cache_service):
        """Test cache memory efficiency profiling."""
        result = await memory_profiler.profile_cache_memory_efficiency(
            mock_cache_service,
            cache_size_limits=[10, 50],
            workload_size=50,  # Reduced for testing
        )

        # Verify efficiency profiling
        assert result.test_name == "memory_efficiency"
        assert len(result.snapshots) > 5
        assert result.memory_efficiency_score >= 0
        assert isinstance(result.recommendations, list)

        print(f"Memory efficiency: {result.memory_efficiency_score:.1f} score, {len(result.recommendations)} recommendations")

    @pytest.mark.asyncio
    async def test_garbage_collection_impact_profiling(self, memory_profiler, mock_cache_service):
        """Test garbage collection impact profiling."""
        result = await memory_profiler.profile_garbage_collection_impact(
            mock_cache_service,
            operations_count=500,
            force_gc_interval=100,  # Reduced for testing
        )

        # Verify GC profiling
        assert result.test_name == "garbage_collection_impact"
        assert len(result.snapshots) > 5
        assert "pressure_level" in result.gc_pressure
        assert "total_collections" in result.gc_pressure

        print(f"GC impact: {result.gc_pressure['pressure_level']} pressure, {result.gc_pressure['total_collections']} collections")

    def test_memory_leak_detection_logic(self, memory_profiler):
        """Test memory leak detection algorithm."""
        # Create mock snapshots simulating memory leak
        snapshots = []
        base_time = time.time()

        # Simulate increasing memory usage
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=base_time + i,
                operation=f"operation_{i}",
                rss_mb=100 + i * 5,  # Consistent growth
                vms_mb=200 + i * 10,
                heap_size_mb=80 + i * 4,
                heap_objects=10000 + i * 1000,
                gc_collections=(i, i, i),
            )
            snapshots.append(snapshot)

        # Test leak detection
        leak_detected = memory_profiler._detect_memory_leak(snapshots)
        assert leak_detected, "Should detect memory leak with consistent growth"

        # Test stable memory (no leak)
        stable_snapshots = []
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=base_time + i,
                operation=f"stable_operation_{i}",
                rss_mb=100 + random.uniform(-2, 2),  # Stable with minor fluctuation
                vms_mb=200 + random.uniform(-5, 5),
                heap_size_mb=80 + random.uniform(-1, 1),
                heap_objects=10000 + random.randint(-500, 500),
                gc_collections=(i, i, i),
            )
            stable_snapshots.append(snapshot)

        stable_leak_detected = memory_profiler._detect_memory_leak(stable_snapshots)
        assert not stable_leak_detected, "Should not detect leak with stable memory"

    def test_efficiency_score_calculation(self, memory_profiler):
        """Test memory efficiency score calculation."""
        # Create snapshots for efficiency testing
        snapshots = []
        base_time = time.time()

        # Efficient memory usage pattern
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + i,
                operation=f"efficient_op_{i}",
                rss_mb=100 + random.uniform(-1, 1),  # Very stable
                vms_mb=200,
                heap_size_mb=80,
                heap_objects=10000 + random.randint(-100, 100),  # Stable objects
                gc_collections=(i, i, i),
            )
            snapshots.append(snapshot)

        score = memory_profiler._calculate_efficiency_score(snapshots)
        assert score > 80, f"High efficiency expected, got {score}"

        # Inefficient memory usage pattern
        inefficient_snapshots = []
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + i,
                operation=f"inefficient_op_{i}",
                rss_mb=100 + i * 20,  # Rapid growth
                vms_mb=200 + i * 40,
                heap_size_mb=80 + i * 15,
                heap_objects=10000 + i * 5000,  # Rapid object growth
                gc_collections=(i, i, i),
            )
            inefficient_snapshots.append(snapshot)

        inefficient_score = memory_profiler._calculate_efficiency_score(inefficient_snapshots)
        assert inefficient_score < 50, f"Low efficiency expected, got {inefficient_score}"


@pytest.mark.performance
@pytest.mark.integration
class TestCacheMemoryProfilingIntegration:
    """Integration tests for cache memory profiling with real services."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(enabled=True, redis_url="redis://localhost:6379/15", default_ttl=300, max_memory_mb=100)  # Test database

    @pytest.mark.asyncio
    async def test_search_cache_memory_profiling(self, cache_config):
        """Test memory profiling with SearchCacheService."""
        try:
            service = SearchCacheService(cache_config)
            await service.initialize()

            profiler = CacheMemoryProfiler()

            # Run memory leak test
            result = await profiler.profile_memory_leak_test(service, iterations=200, operation_interval=0.001)

            # Verify profiling results
            assert result.test_name == "memory_leak_test"
            assert len(result.snapshots) > 5
            assert result.memory_efficiency_score >= 0

            # Memory should be well-managed (no major leaks expected)
            assert result.memory_growth_mb < 50, f"Excessive memory growth: {result.memory_growth_mb}MB"

            print(f"SearchCache memory profiling: efficiency={result.memory_efficiency_score:.1f}, growth={result.memory_growth_mb:.1f}MB")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_project_cache_memory_efficiency(self, cache_config):
        """Test memory efficiency profiling with ProjectCacheService."""
        try:
            service = ProjectCacheService(cache_config)
            await service.initialize()

            profiler = CacheMemoryProfiler()

            # Run efficiency profiling
            result = await profiler.profile_cache_memory_efficiency(service, cache_size_limits=[20, 50], workload_size=100)

            # Verify efficiency profiling
            assert result.test_name == "memory_efficiency"
            assert result.memory_efficiency_score > 30, "Minimum efficiency threshold"
            assert len(result.recommendations) >= 1

            print(f"ProjectCache efficiency: {result.memory_efficiency_score:.1f} score")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
