"""
Unit tests for the cache memory profiler service.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cache_memory_profiler import (
    CacheMemoryProfiler,
    MemoryAllocation,
    MemoryEventType,
    MemoryHotspot,
    MemoryProfile,
    MemorySnapshot,
    ProfilingLevel,
    get_memory_profiler,
    shutdown_memory_profiler,
)


class TestMemoryAllocation:
    """Test memory allocation data structure."""

    def test_memory_allocation_creation(self):
        """Test memory allocation creation."""
        allocation = MemoryAllocation(
            cache_name="test_cache",
            key="test_key",
            size_bytes=1024,
            timestamp=time.time(),
            event_type=MemoryEventType.ALLOCATION,
            thread_id=12345,
        )

        assert allocation.cache_name == "test_cache"
        assert allocation.key == "test_key"
        assert allocation.size_bytes == 1024
        assert allocation.event_type == MemoryEventType.ALLOCATION
        assert allocation.thread_id == 12345

    def test_memory_allocation_properties(self):
        """Test memory allocation properties."""
        timestamp = time.time() - 10  # 10 seconds ago
        allocation = MemoryAllocation(
            cache_name="test_cache",
            key="test_key",
            size_bytes=1024 * 1024,  # 1MB
            timestamp=timestamp,
            event_type=MemoryEventType.ALLOCATION,
            thread_id=12345,
        )

        assert allocation.size_mb == 1.0
        assert allocation.age > 9  # Should be about 10 seconds
        assert allocation.age < 11


class TestMemoryProfile:
    """Test memory profile data structure."""

    def test_memory_profile_creation(self):
        """Test memory profile creation."""
        start_time = time.time()
        profile = MemoryProfile(cache_name="test_cache", start_time=start_time, end_time=start_time + 100)

        assert profile.cache_name == "test_cache"
        assert profile.start_time == start_time
        assert profile.end_time == start_time + 100
        assert profile.duration_seconds == 100

    def test_memory_profile_calculations(self):
        """Test memory profile calculations."""
        profile = MemoryProfile(cache_name="test_cache", start_time=time.time(), end_time=time.time() + 100)

        profile.total_allocated_bytes = 1024 * 1024  # 1MB
        profile.total_deallocated_bytes = 512 * 1024  # 512KB
        profile.total_allocations = 10
        profile.current_memory_bytes = 512 * 1024  # 512KB

        assert profile.net_memory_bytes == 512 * 1024
        assert profile.memory_turnover_ratio == 0.5

        profile.update_rates()
        assert profile.allocation_rate_mb_per_sec > 0
        assert profile.deallocation_rate_mb_per_sec > 0

        profile.update_efficiency()
        assert profile.memory_efficiency_ratio == 0.5
        assert profile.average_allocation_size_bytes == 1024 * 1024 / 10


class TestMemorySnapshot:
    """Test memory snapshot data structure."""

    def test_memory_snapshot_creation(self):
        """Test memory snapshot creation."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_system_memory_mb=8192,
            available_system_memory_mb=4096,
            process_memory_mb=256,
            total_cache_memory_mb=64,
            cache_memory_breakdown={"cache1": 32, "cache2": 32},
        )

        assert snapshot.total_system_memory_mb == 8192
        assert snapshot.available_system_memory_mb == 4096
        assert snapshot.process_memory_mb == 256
        assert snapshot.total_cache_memory_mb == 64
        assert snapshot.cache_memory_breakdown["cache1"] == 32
        assert snapshot.cache_memory_breakdown["cache2"] == 32


class TestMemoryHotspot:
    """Test memory hotspot data structure."""

    def test_memory_hotspot_creation(self):
        """Test memory hotspot creation."""
        start_time = time.time()
        hotspot = MemoryHotspot(
            cache_name="test_cache",
            key_pattern="test_pattern",
            allocation_count=100,
            total_size_bytes=1024 * 1024,  # 1MB
            average_size_bytes=1024 * 10,  # 10KB
            stack_trace=["trace1", "trace2"],
            first_seen=start_time,
            last_seen=start_time + 300,  # 5 minutes later
        )

        assert hotspot.cache_name == "test_cache"
        assert hotspot.key_pattern == "test_pattern"
        assert hotspot.allocation_count == 100
        assert hotspot.total_size_bytes == 1024 * 1024
        assert hotspot.average_size_bytes == 1024 * 10
        assert hotspot.duration == 300  # 5 minutes
        assert hotspot.allocation_rate == 100 / 300  # allocations per second


class TestCacheMemoryProfiler:
    """Test cache memory profiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = CacheMemoryProfiler(ProfilingLevel.DETAILED)

    @pytest.mark.asyncio
    async def test_profiler_initialization(self):
        """Test profiler initialization."""
        await self.profiler.initialize()

        assert self.profiler.is_profiling is True
        assert self.profiler.snapshot_task is not None
        assert self.profiler.cleanup_task is not None

        await self.profiler.shutdown()

    @pytest.mark.asyncio
    async def test_profiler_shutdown(self):
        """Test profiler shutdown."""
        await self.profiler.initialize()
        await self.profiler.shutdown()

        assert self.profiler.is_profiling is False

    def test_cache_profiling_lifecycle(self):
        """Test cache profiling start and stop."""
        cache_name = "test_cache"

        # Start profiling
        self.profiler.start_cache_profiling(cache_name)
        assert cache_name in self.profiler.active_profiles

        # Stop profiling
        profile = self.profiler.stop_cache_profiling(cache_name)
        assert profile is not None
        assert profile.cache_name == cache_name
        assert cache_name not in self.profiler.active_profiles
        assert cache_name in self.profiler.cache_profiles

    def test_allocation_tracking(self):
        """Test allocation tracking."""
        cache_name = "test_cache"
        self.profiler.start_cache_profiling(cache_name)

        # Track allocation
        self.profiler.track_allocation(cache_name, "key1", 1024)

        # Check that event was recorded
        assert len(self.profiler.memory_events) == 1
        event = self.profiler.memory_events[0]
        assert event.cache_name == cache_name
        assert event.key == "key1"
        assert event.size_bytes == 1024
        assert event.event_type == MemoryEventType.ALLOCATION

        # Check that profile was updated
        profile = self.profiler.active_profiles[cache_name]
        assert profile.total_allocations == 1
        assert profile.total_allocated_bytes == 1024
        assert profile.current_memory_bytes == 1024

    def test_deallocation_tracking(self):
        """Test deallocation tracking."""
        cache_name = "test_cache"
        self.profiler.start_cache_profiling(cache_name)

        # Track allocation then deallocation
        self.profiler.track_allocation(cache_name, "key1", 1024)
        self.profiler.track_deallocation(cache_name, "key1", 1024)

        # Check that both events were recorded
        assert len(self.profiler.memory_events) == 2
        assert self.profiler.memory_events[0].event_type == MemoryEventType.ALLOCATION
        assert self.profiler.memory_events[1].event_type == MemoryEventType.DEALLOCATION

        # Check that profile was updated
        profile = self.profiler.active_profiles[cache_name]
        assert profile.total_allocations == 1
        assert profile.total_deallocations == 1
        assert profile.total_allocated_bytes == 1024
        assert profile.total_deallocated_bytes == 1024
        assert profile.current_memory_bytes == 0

    def test_hotspot_detection(self):
        """Test memory hotspot detection."""
        cache_name = "test_cache"
        self.profiler.start_cache_profiling(cache_name)

        # Create multiple allocations for the same key pattern
        for i in range(10):
            self.profiler.track_allocation(cache_name, f"hotspot_key_{i}", 1024)

        # Check that hotspot was detected
        hotspots = self.profiler.get_memory_hotspots(cache_name, min_allocations=5)
        assert len(hotspots) > 0

        # Check hotspot details
        hotspot = hotspots[0]
        assert hotspot["cache_name"] == cache_name
        assert hotspot["allocation_count"] >= 5

    @pytest.mark.asyncio
    async def test_memory_snapshot(self):
        """Test memory snapshot functionality."""
        with patch("cache_memory_profiler.get_memory_stats") as mock_stats:
            mock_stats.return_value = {
                "system_memory": {"total_mb": 8192, "available_mb": 4096},
                "rss_mb": 256,
            }

            with patch("cache_memory_profiler.get_total_cache_memory_usage") as mock_cache_memory:
                mock_cache_memory.return_value = 64

                snapshot = await self.profiler.take_memory_snapshot()

                assert snapshot.total_system_memory_mb == 8192
                assert snapshot.available_system_memory_mb == 4096
                assert snapshot.process_memory_mb == 256
                assert snapshot.total_cache_memory_mb == 64
                assert len(self.profiler.snapshots) == 1

    @pytest.mark.asyncio
    async def test_profile_operation_context(self):
        """Test profile operation context manager."""
        cache_name = "test_cache"

        async with self.profiler.profile_operation(cache_name, "test_operation") as context:
            assert context["cache_name"] == cache_name
            assert context["operation"] == "test_operation"
            assert "start_time" in context
            await asyncio.sleep(0.1)  # Simulate some work

        assert "end_time" in context
        assert "duration" in context
        assert context["duration"] > 0.1

    def test_memory_trend_analysis(self):
        """Test memory trend analysis."""
        # Add some snapshots
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=time.time() - (i * 60),  # 1 minute intervals
                total_system_memory_mb=8192,
                available_system_memory_mb=4096,
                process_memory_mb=256,
                total_cache_memory_mb=64 + i * 8,  # Increasing cache memory
                cache_memory_breakdown={"test_cache": 32 + i * 4},
            )
            self.profiler.snapshots.append(snapshot)

        # Test system-wide trend
        trend = self.profiler.get_memory_trend()
        assert "total_cache_memory_mb" in trend
        assert "process_memory_mb" in trend
        assert trend["data_points"] == 5

        # Test cache-specific trend
        cache_trend = self.profiler.get_memory_trend("test_cache")
        assert cache_trend["cache_name"] == "test_cache"
        assert cache_trend["data_points"] == 5

    def test_allocation_pattern_analysis(self):
        """Test allocation pattern analysis."""
        cache_name = "test_cache"
        self.profiler.start_cache_profiling(cache_name)

        # Create allocation patterns
        for i in range(10):
            self.profiler.track_allocation(cache_name, f"key_{i}", 1024 * (i + 1))

        for i in range(5):
            self.profiler.track_deallocation(cache_name, f"key_{i}", 1024 * (i + 1))

        # Analyze patterns
        patterns = self.profiler.get_allocation_patterns(cache_name)
        assert "allocations" in patterns
        assert "deallocations" in patterns
        assert "cache_breakdown" in patterns

        assert patterns["allocations"]["count"] == 10
        assert patterns["deallocations"]["count"] == 5
        assert patterns["net_allocation_bytes"] > 0

    def test_performance_metrics(self):
        """Test performance metrics."""
        # Add some timing data
        for i in range(10):
            self.profiler.allocation_times.append(0.1 + i * 0.01)
            self.profiler.deallocation_times.append(0.05 + i * 0.005)

        metrics = self.profiler.get_performance_metrics()
        assert "allocation_times" in metrics
        assert "deallocation_times" in metrics
        assert "total_events_tracked" in metrics

        assert metrics["allocation_times"]["count"] == 10
        assert metrics["deallocation_times"]["count"] == 10
        assert metrics["allocation_times"]["avg"] > 0

    def test_cache_profile_retrieval(self):
        """Test cache profile retrieval."""
        cache_name = "test_cache"
        self.profiler.start_cache_profiling(cache_name)

        # Track some allocations
        self.profiler.track_allocation(cache_name, "key1", 1024)
        self.profiler.track_allocation(cache_name, "key2", 2048)

        # Get profile
        profile_data = self.profiler.get_cache_profile(cache_name)
        assert profile_data is not None
        assert profile_data["cache_name"] == cache_name
        assert profile_data["is_active"] is True
        assert profile_data["memory_usage"]["total_allocated_bytes"] == 3072
        assert profile_data["allocation_stats"]["total_allocations"] == 2

    def test_profiling_data_reset(self):
        """Test profiling data reset."""
        cache_name = "test_cache"
        self.profiler.start_cache_profiling(cache_name)

        # Add some data
        self.profiler.track_allocation(cache_name, "key1", 1024)
        self.profiler.snapshots.append(
            MemorySnapshot(
                timestamp=time.time(),
                total_system_memory_mb=8192,
                available_system_memory_mb=4096,
                process_memory_mb=256,
                total_cache_memory_mb=64,
            )
        )

        # Verify data exists
        assert len(self.profiler.memory_events) > 0
        assert len(self.profiler.active_profiles) > 0
        assert len(self.profiler.snapshots) > 0

        # Reset data
        self.profiler.reset_profiling_data()

        # Verify data is cleared
        assert len(self.profiler.memory_events) == 0
        assert len(self.profiler.active_profiles) == 0
        assert len(self.profiler.cache_profiles) == 0
        assert len(self.profiler.snapshots) == 0


class TestGlobalProfiler:
    """Test global profiler functions."""

    @pytest.mark.asyncio
    async def test_global_profiler_lifecycle(self):
        """Test global profiler lifecycle."""
        # Get profiler instance
        profiler = await get_memory_profiler()

        assert profiler is not None
        assert isinstance(profiler, CacheMemoryProfiler)
        assert profiler.is_profiling is True

        # Get same instance
        profiler2 = await get_memory_profiler()
        assert profiler is profiler2

        # Shutdown profiler
        await shutdown_memory_profiler()

        # Should create new instance
        profiler3 = await get_memory_profiler()
        assert profiler3 is not profiler

        await shutdown_memory_profiler()

    @pytest.mark.asyncio
    async def test_global_profiler_different_levels(self):
        """Test global profiler with different profiling levels."""
        # Test with basic profiling
        profiler_basic = await get_memory_profiler(ProfilingLevel.BASIC)
        assert profiler_basic.profiling_level == ProfilingLevel.BASIC

        await shutdown_memory_profiler()

        # Test with comprehensive profiling
        profiler_comprehensive = await get_memory_profiler(ProfilingLevel.COMPREHENSIVE)
        assert profiler_comprehensive.profiling_level == ProfilingLevel.COMPREHENSIVE

        await shutdown_memory_profiler()


class TestProfilingLevels:
    """Test different profiling levels."""

    def test_basic_profiling_level(self):
        """Test basic profiling level."""
        profiler = CacheMemoryProfiler(ProfilingLevel.BASIC)
        assert profiler.profiling_level == ProfilingLevel.BASIC
        assert profiler.collect_stack_traces is False

    def test_detailed_profiling_level(self):
        """Test detailed profiling level."""
        profiler = CacheMemoryProfiler(ProfilingLevel.DETAILED)
        assert profiler.profiling_level == ProfilingLevel.DETAILED
        assert profiler.collect_stack_traces is True

    def test_comprehensive_profiling_level(self):
        """Test comprehensive profiling level."""
        profiler = CacheMemoryProfiler(ProfilingLevel.COMPREHENSIVE)
        assert profiler.profiling_level == ProfilingLevel.COMPREHENSIVE
        assert profiler.collect_stack_traces is True


class TestMemoryEventTypes:
    """Test memory event types."""

    def test_memory_event_types(self):
        """Test all memory event types."""
        assert MemoryEventType.ALLOCATION.value == "allocation"
        assert MemoryEventType.DEALLOCATION.value == "deallocation"
        assert MemoryEventType.RESIZE.value == "resize"
        assert MemoryEventType.CLEANUP.value == "cleanup"
        assert MemoryEventType.EVICTION.value == "eviction"
        assert MemoryEventType.PRESSURE.value == "pressure"


if __name__ == "__main__":
    pytest.main([__file__])
