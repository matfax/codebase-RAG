"""
Comprehensive test suite for the cache memory leak detector.

Tests cover:
- Memory snapshot creation and storage
- Leak detection algorithms
- Configuration validation
- Monitoring lifecycle
- Performance and reliability
"""

import asyncio
import gc
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cache_memory_leak_detector import (
    CacheMemoryLeakDetector,
    LeakDetectionConfig,
    LeakSeverity,
    LeakType,
    MemoryLeak,
    MemorySnapshot,
    analyze_cache_memory_leaks,
    get_leak_detector,
    start_cache_leak_monitoring,
    stop_cache_leak_monitoring,
    take_cache_memory_snapshot,
)


class TestMemorySnapshot:
    """Test cases for MemorySnapshot data model."""

    def test_memory_snapshot_creation(self):
        """Test creating a memory snapshot with all fields."""
        timestamp = datetime.now()
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            total_memory_mb=256.5,
            rss_memory_mb=240.2,
            vms_memory_mb=512.8,
            cache_memory_mb=64.1,
            cache_entry_count=1000,
            cache_name="test_cache",
            thread_id=12345,
            gc_stats={"counts": [10, 5, 2]},
            tracemalloc_stats={"current_mb": 50.0, "peak_mb": 75.0},
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.total_memory_mb == 256.5
        assert snapshot.cache_name == "test_cache"
        assert snapshot.cache_entry_count == 1000
        assert snapshot.gc_stats["counts"] == [10, 5, 2]
        assert snapshot.tracemalloc_stats["current_mb"] == 50.0


class TestMemoryLeak:
    """Test cases for MemoryLeak data model."""

    def test_memory_leak_creation(self):
        """Test creating a memory leak with all fields."""
        timestamp = datetime.now()
        snapshots = []

        leak = MemoryLeak(
            leak_id="test_leak_001",
            cache_name="test_cache",
            leak_type=LeakType.GRADUAL_GROWTH,
            severity=LeakSeverity.MEDIUM,
            detected_at=timestamp,
            start_memory_mb=100.0,
            current_memory_mb=200.0,
            memory_growth_mb=100.0,
            growth_rate_mb_per_minute=2.5,
            duration_minutes=40.0,
            snapshots=snapshots,
            recommendations=["Check cache eviction"],
            metadata={"test": "value"},
        )

        assert leak.leak_id == "test_leak_001"
        assert leak.leak_type == LeakType.GRADUAL_GROWTH
        assert leak.severity == LeakSeverity.MEDIUM
        assert leak.memory_growth_mb == 100.0
        assert leak.growth_rate_mb_per_minute == 2.5
        assert "Check cache eviction" in leak.recommendations


class TestLeakDetectionConfig:
    """Test cases for LeakDetectionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LeakDetectionConfig()

        assert config.memory_growth_threshold_mb == 100.0
        assert config.growth_rate_threshold_mb_per_min == 5.0
        assert config.detection_window_minutes == 30
        assert config.snapshot_interval_seconds == 30
        assert config.enable_tracemalloc is True
        assert config.auto_cleanup_old_data is True

    def test_custom_config(self):
        """Test creating configuration with custom values."""
        config = LeakDetectionConfig(
            memory_growth_threshold_mb=50.0, growth_rate_threshold_mb_per_min=2.0, detection_window_minutes=15, enable_tracemalloc=False
        )

        assert config.memory_growth_threshold_mb == 50.0
        assert config.growth_rate_threshold_mb_per_min == 2.0
        assert config.detection_window_minutes == 15
        assert config.enable_tracemalloc is False


class TestCacheMemoryLeakDetector:
    """Test cases for CacheMemoryLeakDetector."""

    @pytest.fixture
    def detector(self):
        """Create a memory leak detector for testing."""
        config = LeakDetectionConfig(
            snapshot_interval_seconds=1,  # Fast intervals for testing
            enable_tracemalloc=False,  # Disable to avoid conflicts
            auto_cleanup_old_data=False,  # Disable for controlled testing
        )
        return CacheMemoryLeakDetector(config)

    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for consistent testing."""
        with patch("cache_memory_leak_detector.PSUTIL_AVAILABLE", True), patch("cache_memory_leak_detector.psutil") as mock_psutil:
            # Mock process and memory info
            mock_process = MagicMock()
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 256 * 1024 * 1024  # 256 MB
            mock_memory_info.vms = 512 * 1024 * 1024  # 512 MB
            mock_process.memory_info.return_value = mock_memory_info
            mock_psutil.Process.return_value = mock_process

            yield mock_psutil

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.config is not None
        assert detector._memory_snapshots == {}
        assert detector._detected_leaks == {}
        assert detector._monitoring_active == {}

    @pytest.mark.asyncio
    async def test_take_snapshot_basic(self, detector, mock_psutil):
        """Test taking a basic memory snapshot."""
        snapshot = await detector.take_snapshot("test_cache", 100, 50.0)

        assert snapshot.cache_name == "test_cache"
        assert snapshot.cache_entry_count == 100
        assert snapshot.cache_memory_mb == 50.0
        assert snapshot.total_memory_mb == 256.0  # From mock
        assert isinstance(snapshot.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_take_snapshot_without_psutil(self, detector):
        """Test taking snapshot when psutil is not available."""
        with patch("cache_memory_leak_detector.PSUTIL_AVAILABLE", False):
            snapshot = await detector.take_snapshot("test_cache", 100, 50.0)

            assert snapshot.cache_name == "test_cache"
            assert snapshot.total_memory_mb == 0.0
            assert snapshot.rss_memory_mb == 0.0
            assert snapshot.vms_memory_mb == 0.0

    @pytest.mark.asyncio
    async def test_snapshot_storage(self, detector, mock_psutil):
        """Test that snapshots are properly stored."""
        await detector.take_snapshot("cache1", 100, 50.0)
        await detector.take_snapshot("cache1", 110, 55.0)
        await detector.take_snapshot("cache2", 200, 100.0)

        cache1_snapshots = detector.get_cache_snapshots("cache1")
        cache2_snapshots = detector.get_cache_snapshots("cache2")

        assert len(cache1_snapshots) == 2
        assert len(cache2_snapshots) == 1
        assert cache1_snapshots[0].cache_entry_count == 100
        assert cache1_snapshots[1].cache_entry_count == 110
        assert cache2_snapshots[0].cache_entry_count == 200

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, detector):
        """Test starting and stopping monitoring."""
        cache_name = "test_cache"

        # Initially not monitoring
        status = detector.get_monitoring_status()
        assert not status["cache_monitoring"].get(cache_name, {}).get("monitoring_active", False)

        # Start monitoring
        await detector.start_monitoring(cache_name)
        status = detector.get_monitoring_status()
        assert status["cache_monitoring"][cache_name]["monitoring_active"] is True

        # Allow some monitoring to occur
        await asyncio.sleep(0.1)

        # Stop monitoring
        await detector.stop_monitoring(cache_name)
        status = detector.get_monitoring_status()
        assert status["cache_monitoring"][cache_name]["monitoring_active"] is False

    @pytest.mark.asyncio
    async def test_global_monitoring(self, detector):
        """Test global monitoring lifecycle."""
        # Initially not monitoring globally
        status = detector.get_monitoring_status()
        assert status["global_monitoring"] is False

        # Start global monitoring
        await detector.start_global_monitoring()
        status = detector.get_monitoring_status()
        assert status["global_monitoring"] is True

        # Allow some monitoring to occur
        await asyncio.sleep(0.1)

        # Stop global monitoring
        await detector.stop_global_monitoring()
        status = detector.get_monitoring_status()
        assert status["global_monitoring"] is False

    @pytest.mark.asyncio
    async def test_duplicate_monitoring_start(self, detector):
        """Test that starting monitoring twice doesn't create issues."""
        cache_name = "test_cache"

        await detector.start_monitoring(cache_name)
        await detector.start_monitoring(cache_name)  # Should not cause issues

        status = detector.get_monitoring_status()
        assert status["cache_monitoring"][cache_name]["monitoring_active"] is True

        await detector.stop_monitoring(cache_name)

    @pytest.mark.asyncio
    async def test_gradual_growth_detection(self, detector, mock_psutil):
        """Test detection of gradual memory growth."""
        cache_name = "test_cache"

        # Create snapshots showing gradual memory growth
        base_time = datetime.now()
        base_memory = 100.0

        # Simulate gradual growth over time
        for i in range(20):
            # Increase memory gradually
            memory_growth = i * 10  # 10MB per snapshot
            mock_psutil.Process.return_value.memory_info.return_value.rss = int((base_memory + memory_growth) * 1024 * 1024)

            # Create snapshot with timestamp spread over time
            snapshot = await detector.take_snapshot(cache_name, 100 + i * 10, 50.0 + i * 5)
            # Manually adjust timestamp for testing
            snapshot.timestamp = base_time + timedelta(minutes=i * 2)

            # Update the stored snapshot with adjusted timestamp
            detector._memory_snapshots[cache_name][-1] = snapshot

        # Analyze for leaks
        leaks = await detector.analyze_memory_leaks(cache_name)

        # Should detect gradual growth
        gradual_leaks = [leak for leak in leaks if leak.leak_type == LeakType.GRADUAL_GROWTH]
        assert len(gradual_leaks) > 0

        leak = gradual_leaks[0]
        assert leak.cache_name == cache_name
        assert leak.memory_growth_mb > 0
        assert leak.severity in [LeakSeverity.LOW, LeakSeverity.MEDIUM, LeakSeverity.HIGH]

    @pytest.mark.asyncio
    async def test_rapid_growth_detection(self, detector, mock_psutil):
        """Test detection of rapid memory growth."""
        cache_name = "test_cache"

        # Create snapshots showing rapid memory growth
        base_time = datetime.now()

        # First few snapshots with stable memory
        for i in range(5):
            mock_psutil.Process.return_value.memory_info.return_value.rss = int(100 * 1024 * 1024)
            snapshot = await detector.take_snapshot(cache_name, 100, 50.0)
            snapshot.timestamp = base_time + timedelta(minutes=i)
            detector._memory_snapshots[cache_name][-1] = snapshot

        # Then rapid growth
        for i in range(5, 10):
            # Rapid growth: 100MB per minute
            rapid_memory = 100 + (i - 4) * 100
            mock_psutil.Process.return_value.memory_info.return_value.rss = int(rapid_memory * 1024 * 1024)
            snapshot = await detector.take_snapshot(cache_name, 100 + i * 10, 50.0)
            snapshot.timestamp = base_time + timedelta(minutes=i)
            detector._memory_snapshots[cache_name][-1] = snapshot

        # Analyze for leaks
        leaks = await detector.analyze_memory_leaks(cache_name)

        # Should detect rapid growth
        rapid_leaks = [leak for leak in leaks if leak.leak_type == LeakType.RAPID_GROWTH]
        assert len(rapid_leaks) > 0

        leak = rapid_leaks[0]
        assert leak.cache_name == cache_name
        assert leak.growth_rate_mb_per_minute > 15  # Should be high growth rate

    @pytest.mark.asyncio
    async def test_sustained_high_usage_detection(self, detector, mock_psutil):
        """Test detection of sustained high memory usage."""
        cache_name = "test_cache"

        # Create snapshots showing sustained high memory usage
        high_memory = 600.0  # Above the 500MB threshold

        for i in range(15):
            mock_psutil.Process.return_value.memory_info.return_value.rss = int(high_memory * 1024 * 1024)
            await detector.take_snapshot(cache_name, 1000, high_memory / 2)

        # Analyze for leaks
        leaks = await detector.analyze_memory_leaks(cache_name)

        # Should detect sustained high usage
        sustained_leaks = [leak for leak in leaks if leak.leak_type == LeakType.SUSTAINED_HIGH]
        assert len(sustained_leaks) > 0

        leak = sustained_leaks[0]
        assert leak.cache_name == cache_name
        assert leak.severity in [LeakSeverity.MEDIUM, LeakSeverity.HIGH]

    @pytest.mark.asyncio
    async def test_periodic_spike_detection(self, detector, mock_psutil):
        """Test detection of periodic memory spikes."""
        cache_name = "test_cache"

        # Create snapshots showing periodic spikes
        base_memory = 100.0
        spike_memory = 200.0

        for i in range(30):
            # Every 5th snapshot is a spike
            if i % 5 == 0:
                memory = spike_memory
            else:
                memory = base_memory

            mock_psutil.Process.return_value.memory_info.return_value.rss = int(memory * 1024 * 1024)
            await detector.take_snapshot(cache_name, 100, 50.0)

        # Analyze for leaks
        leaks = await detector.analyze_memory_leaks(cache_name)

        # Should detect periodic spikes
        periodic_leaks = [leak for leak in leaks if leak.leak_type == LeakType.PERIODIC_SPIKE]
        assert len(periodic_leaks) > 0

        leak = periodic_leaks[0]
        assert leak.cache_name == cache_name
        assert leak.metadata["spike_count"] > 0

    @pytest.mark.asyncio
    async def test_no_false_positives(self, detector, mock_psutil):
        """Test that stable memory usage doesn't trigger false leaks."""
        cache_name = "test_cache"

        # Create snapshots with stable memory usage
        stable_memory = 200.0

        for i in range(10):
            # Small random variations (±5MB)
            memory = stable_memory + (i % 3 - 1) * 5
            mock_psutil.Process.return_value.memory_info.return_value.rss = int(memory * 1024 * 1024)
            await detector.take_snapshot(cache_name, 100, 50.0)

        # Analyze for leaks
        leaks = await detector.analyze_memory_leaks(cache_name)

        # Should not detect any leaks for stable usage
        assert len(leaks) == 0

    def test_severity_calculation(self, detector):
        """Test leak severity calculation."""
        # Test critical severity
        severity = detector._calculate_leak_severity(1500.0, 25.0)
        assert severity == LeakSeverity.CRITICAL

        # Test high severity
        severity = detector._calculate_leak_severity(700.0, 15.0)
        assert severity == LeakSeverity.HIGH

        # Test medium severity
        severity = detector._calculate_leak_severity(300.0, 7.0)
        assert severity == LeakSeverity.MEDIUM

        # Test low severity
        severity = detector._calculate_leak_severity(150.0, 3.0)
        assert severity == LeakSeverity.LOW

    def test_sustained_severity_calculation(self, detector):
        """Test sustained usage severity calculation."""
        # Test critical severity
        severity = detector._calculate_sustained_severity(2500.0, 0.96)
        assert severity == LeakSeverity.CRITICAL

        # Test high severity
        severity = detector._calculate_sustained_severity(1200.0, 0.92)
        assert severity == LeakSeverity.HIGH

        # Test medium severity
        severity = detector._calculate_sustained_severity(600.0, 0.85)
        assert severity == LeakSeverity.MEDIUM

        # Test low severity
        severity = detector._calculate_sustained_severity(300.0, 0.7)
        assert severity == LeakSeverity.LOW

    def test_get_detected_leaks_filtering(self, detector):
        """Test filtering of detected leaks."""
        cache_name = "test_cache"

        # Create test leaks with different severities
        leaks = [
            MemoryLeak(
                leak_id="leak1",
                cache_name=cache_name,
                leak_type=LeakType.GRADUAL_GROWTH,
                severity=LeakSeverity.HIGH,
                detected_at=datetime.now(),
                start_memory_mb=100,
                current_memory_mb=200,
                memory_growth_mb=100,
                growth_rate_mb_per_minute=5.0,
                duration_minutes=20,
                snapshots=[],
            ),
            MemoryLeak(
                leak_id="leak2",
                cache_name=cache_name,
                leak_type=LeakType.RAPID_GROWTH,
                severity=LeakSeverity.CRITICAL,
                detected_at=datetime.now(),
                start_memory_mb=100,
                current_memory_mb=300,
                memory_growth_mb=200,
                growth_rate_mb_per_minute=20.0,
                duration_minutes=10,
                snapshots=[],
            ),
        ]

        detector._detected_leaks[cache_name] = leaks

        # Test cache-specific filtering
        cache_leaks = detector.get_detected_leaks(cache_name)
        assert len(cache_leaks) == 2

        # Test severity filtering
        critical_leaks = detector.get_detected_leaks(severity=LeakSeverity.CRITICAL)
        assert len(critical_leaks) == 1
        assert critical_leaks[0].severity == LeakSeverity.CRITICAL

        # Test combined filtering
        cache_critical_leaks = detector.get_detected_leaks(cache_name, LeakSeverity.CRITICAL)
        assert len(cache_critical_leaks) == 1

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, detector, mock_psutil):
        """Test cleanup of old monitoring data."""
        cache_name = "test_cache"

        # Create old snapshots
        old_time = datetime.now() - timedelta(hours=25)  # Older than cleanup threshold
        for i in range(5):
            snapshot = await detector.take_snapshot(cache_name, 100, 50.0)
            snapshot.timestamp = old_time
            detector._memory_snapshots[cache_name][-1] = snapshot

        # Create recent snapshots
        for i in range(3):
            await detector.take_snapshot(cache_name, 100, 50.0)

        assert len(detector._memory_snapshots[cache_name]) == 8

        # Run cleanup
        await detector._cleanup_old_data()

        # Should only keep recent snapshots
        assert len(detector._memory_snapshots[cache_name]) == 3


class TestGlobalFunctions:
    """Test cases for global utility functions."""

    @pytest.mark.asyncio
    async def test_get_leak_detector_singleton(self):
        """Test that get_leak_detector returns the same instance."""
        detector1 = await get_leak_detector()
        detector2 = await get_leak_detector()

        assert detector1 is detector2

    @pytest.mark.asyncio
    async def test_start_stop_cache_monitoring_functions(self):
        """Test global start/stop monitoring functions."""
        cache_name = "test_cache"

        # Start monitoring
        await start_cache_leak_monitoring(cache_name)

        detector = await get_leak_detector()
        status = detector.get_monitoring_status()
        assert status["cache_monitoring"][cache_name]["monitoring_active"] is True

        # Stop monitoring
        await stop_cache_leak_monitoring(cache_name)

        status = detector.get_monitoring_status()
        assert status["cache_monitoring"][cache_name]["monitoring_active"] is False

    @pytest.mark.asyncio
    async def test_analyze_cache_memory_leaks_function(self):
        """Test global analyze function."""
        cache_name = "test_cache"

        # Create some test data
        detector = await get_leak_detector()
        await detector.take_snapshot(cache_name, 100, 50.0)

        # Call global function
        leaks = await analyze_cache_memory_leaks(cache_name)

        # Should return list (empty for insufficient data)
        assert isinstance(leaks, list)

    @pytest.mark.asyncio
    async def test_take_cache_memory_snapshot_function(self):
        """Test global snapshot function."""
        cache_name = "test_cache"

        with patch("cache_memory_leak_detector.PSUTIL_AVAILABLE", True), patch("cache_memory_leak_detector.psutil") as mock_psutil:
            mock_process = MagicMock()
            mock_memory_info = MagicMock()
            mock_memory_info.rss = 256 * 1024 * 1024
            mock_memory_info.vms = 512 * 1024 * 1024
            mock_process.memory_info.return_value = mock_memory_info
            mock_psutil.Process.return_value = mock_process

            snapshot = await take_cache_memory_snapshot(cache_name, 100, 50.0)

            assert snapshot.cache_name == cache_name
            assert snapshot.cache_entry_count == 100
            assert snapshot.cache_memory_mb == 50.0


class TestPerformanceAndReliability:
    """Test cases for performance and reliability."""

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, mock_psutil):
        """Test that the detector itself doesn't consume excessive memory."""
        config = LeakDetectionConfig(max_snapshots_per_cache=100)
        detector = CacheMemoryLeakDetector(config)

        # Take many snapshots
        for i in range(200):  # More than max_snapshots_per_cache
            await detector.take_snapshot("test_cache", i, i * 0.5)

        # Should not exceed max snapshots
        snapshots = detector.get_cache_snapshots("test_cache")
        assert len(snapshots) <= 100

    @pytest.mark.asyncio
    async def test_concurrent_access(self, mock_psutil):
        """Test concurrent access to detector."""
        detector = CacheMemoryLeakDetector()

        async def take_snapshots(cache_name: str, count: int):
            for i in range(count):
                await detector.take_snapshot(f"{cache_name}_{i}", i, i * 0.5)

        # Run concurrent snapshot taking
        tasks = [take_snapshots("cache1", 10), take_snapshots("cache2", 10), take_snapshots("cache3", 10)]

        await asyncio.gather(*tasks)

        # Should have snapshots for all caches
        assert len(detector._memory_snapshots) >= 30  # At least 30 different cache names

    @pytest.mark.asyncio
    async def test_error_handling_in_monitoring(self):
        """Test error handling during monitoring."""
        detector = CacheMemoryLeakDetector()

        # Mock an error in memory monitoring
        with patch.object(detector, "take_snapshot", side_effect=Exception("Test error")):
            # Start monitoring (should handle errors gracefully)
            await detector.start_monitoring("test_cache")

            # Allow some time for monitoring loop
            await asyncio.sleep(0.1)

            # Stop monitoring
            await detector.stop_monitoring("test_cache")

        # Should not crash and should be able to continue
        assert True  # If we reach here, error handling worked

    @pytest.mark.asyncio
    async def test_monitoring_task_cleanup(self):
        """Test that monitoring tasks are properly cleaned up."""
        detector = CacheMemoryLeakDetector()

        # Start monitoring
        await detector.start_monitoring("test_cache")

        # Verify task exists
        assert "test_cache" in detector._monitoring_tasks
        task = detector._monitoring_tasks["test_cache"]
        assert not task.done()

        # Stop monitoring
        await detector.stop_monitoring("test_cache")

        # Task should be cleaned up and cancelled
        assert task.done() or task.cancelled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
