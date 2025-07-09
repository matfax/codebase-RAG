"""
Tests for cache garbage collection coordinator service.

This module tests the GC coordination functionality including cache eviction
coordination, memory pressure handling, and garbage collection optimization.
"""

import asyncio
import gc
import time
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..utils.memory_utils import MemoryPressureLevel, SystemMemoryPressure
from .cache_gc_coordinator_service import (
    CacheGCCoordinatorService,
    GCCoordinationConfig,
    GCCoordinationStats,
    GCEvent,
    GCStrategy,
    GCTrigger,
    get_gc_coordinator_service,
    initialize_gc_coordinator_service,
    shutdown_gc_coordinator_service,
)


class MockCacheObject:
    """Mock cache object for testing."""

    def __init__(self, data: str = "test_data"):
        self.data = data
        self.size = len(data)

    def __del__(self):
        """Destructor for testing weak references."""
        pass


@pytest.fixture
def gc_coordinator():
    """Create a cache GC coordinator service."""
    return CacheGCCoordinatorService()


@pytest.fixture
def gc_config():
    """Create a test GC coordination configuration."""
    return GCCoordinationConfig(
        strategy=GCStrategy.ADAPTIVE,
        enable_coordination=True,
        gc_threshold_mb=50.0,
        max_gc_frequency_seconds=10.0,
        min_gc_frequency_seconds=60.0,
        coordination_delay_seconds=0.5,
        enable_weak_references=True,
        enable_fragmentation_detection=True,
        fragmentation_threshold=0.2,
    )


@pytest.fixture
def system_pressure():
    """Create mock system memory pressure."""
    return SystemMemoryPressure(
        level=MemoryPressureLevel.MODERATE,
        current_usage_mb=4000.0,
        available_mb=4000.0,
        total_mb=8000.0,
        cache_usage_mb=800.0,
        cache_usage_percent=10.0,
        recommendation="Normal memory usage",
    )


@pytest.mark.asyncio
async def test_service_initialization(gc_coordinator):
    """Test service initialization and shutdown."""
    # Test initialization
    await gc_coordinator.initialize()
    assert gc_coordinator._coordination_enabled is True
    assert gc_coordinator._monitoring_task is not None
    assert not gc_coordinator._monitoring_task.done()

    # Test shutdown
    await gc_coordinator.shutdown()
    assert gc_coordinator._monitoring_task.done()


@pytest.mark.asyncio
async def test_config_update(gc_coordinator, gc_config):
    """Test configuration updates."""
    # Update configuration
    gc_coordinator.update_config(gc_config)

    assert gc_coordinator._config == gc_config
    assert gc_coordinator._config.strategy == GCStrategy.ADAPTIVE
    assert gc_coordinator._config.gc_threshold_mb == 50.0


@pytest.mark.asyncio
async def test_coordination_enablement(gc_coordinator):
    """Test enabling and disabling coordination."""
    # Test disable
    gc_coordinator.disable_coordination()
    assert gc_coordinator._coordination_enabled is False

    # Test enable
    gc_coordinator.enable_coordination()
    assert gc_coordinator._coordination_enabled is True


@pytest.mark.asyncio
async def test_weak_reference_registration(gc_coordinator):
    """Test weak reference registration and tracking."""
    # Initialize service
    await gc_coordinator.initialize()

    # Create test objects
    obj1 = MockCacheObject("data1")
    obj2 = MockCacheObject("data2")

    # Register objects
    gc_coordinator.register_cache_object(obj1, "cache_objects")
    gc_coordinator.register_cache_object(obj2, "cache_entries")

    # Check registration
    weak_stats = gc_coordinator.get_weak_reference_stats()
    assert weak_stats["cache_objects"] == 1
    assert weak_stats["cache_entries"] == 1

    # Delete objects and check cleanup
    del obj1, obj2
    gc.collect()  # Force collection

    # Cleanup should reduce counts
    cleaned = gc_coordinator._cleanup_weak_references()
    assert cleaned >= 0  # Some references may have been cleaned

    await gc_coordinator.shutdown()


@pytest.mark.asyncio
async def test_callback_registration(gc_coordinator):
    """Test callback registration and execution."""
    # Create mock callbacks
    cache_eviction_callback = AsyncMock(return_value=5)
    pre_gc_callback = MagicMock()
    post_gc_callback = AsyncMock()

    # Register callbacks
    gc_coordinator.register_cache_eviction_callback(cache_eviction_callback)
    gc_coordinator.register_pre_gc_callback(pre_gc_callback)
    gc_coordinator.register_post_gc_callback(post_gc_callback)

    # Test callback execution during GC
    gc_event = await gc_coordinator.manual_gc("Test callbacks")

    # Verify callbacks were called
    cache_eviction_callback.assert_called_once()
    pre_gc_callback.assert_called_once()
    post_gc_callback.assert_called_once()

    # Verify GC event was created
    assert gc_event is not None
    assert gc_event.trigger == GCTrigger.MANUAL


@pytest.mark.asyncio
async def test_memory_pressure_trigger(gc_coordinator):
    """Test garbage collection triggered by memory pressure."""
    # Mock high memory pressure
    high_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.HIGH,
        current_usage_mb=7000.0,
        available_mb=1000.0,
        total_mb=8000.0,
        cache_usage_mb=1500.0,
        cache_usage_percent=18.75,
        recommendation="High memory usage",
    )

    # Mock the system pressure function
    with patch("src.services.cache_gc_coordinator_service.get_system_memory_pressure", return_value=high_pressure):
        # Test should trigger GC
        should_trigger = await gc_coordinator._should_trigger_gc(high_pressure)
        assert should_trigger is True

        # Perform coordinated GC
        gc_event = await gc_coordinator._perform_coordinated_gc(GCTrigger.MEMORY_PRESSURE)

        assert gc_event is not None
        assert gc_event.trigger == GCTrigger.MEMORY_PRESSURE
        assert gc_event.objects_before >= gc_event.objects_after


@pytest.mark.asyncio
async def test_fragmentation_detection(gc_coordinator, gc_config):
    """Test fragmentation detection and GC triggering."""
    # Enable fragmentation detection
    gc_config.enable_fragmentation_detection = True
    gc_config.fragmentation_threshold = 0.1
    gc_coordinator.update_config(gc_config)

    # Mock fragmentation calculation to return high fragmentation
    with patch.object(gc_coordinator, "_calculate_fragmentation", return_value=0.3):
        # Test fragmentation check
        await gc_coordinator._check_fragmentation()

        # Should have triggered a fragmentation event
        assert gc_coordinator._stats.fragmentation_events > 0


@pytest.mark.asyncio
async def test_coordinated_gc_with_cache_eviction(gc_coordinator):
    """Test coordinated GC with cache eviction."""
    # Register cache eviction callback
    eviction_callback = AsyncMock(return_value=10)
    gc_coordinator.register_cache_eviction_callback(eviction_callback)

    # Perform coordinated GC
    gc_event = await gc_coordinator.coordinate_with_cache_eviction("test_cache")

    # Verify eviction callback was called
    eviction_callback.assert_called_once()

    # Verify GC event
    assert gc_event is not None
    assert gc_event.trigger == GCTrigger.CACHE_EVICTION
    assert gc_event.cache_evictions_triggered == 10


@pytest.mark.asyncio
async def test_gc_strategies(gc_coordinator):
    """Test different garbage collection strategies."""
    strategies = [
        GCStrategy.INCREMENTAL,
        GCStrategy.GENERATIONAL,
        GCStrategy.FULL_COLLECTION,
        GCStrategy.ADAPTIVE,
    ]

    for strategy in strategies:
        config = GCCoordinationConfig(strategy=strategy)
        gc_coordinator.update_config(config)

        # Test GC with this strategy
        collected = gc_coordinator._perform_gc(GCTrigger.MANUAL)
        assert collected >= 0  # Should collect some objects or none


@pytest.mark.asyncio
async def test_gc_frequency_limits(gc_coordinator, gc_config):
    """Test GC frequency limits."""
    # Set short frequency limits for testing
    gc_config.max_gc_frequency_seconds = 1.0
    gc_config.min_gc_frequency_seconds = 3.0
    gc_coordinator.update_config(gc_config)

    # Mock normal memory pressure
    normal_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.LOW,
        current_usage_mb=2000.0,
        available_mb=6000.0,
        total_mb=8000.0,
        cache_usage_mb=400.0,
        cache_usage_percent=5.0,
        recommendation="Low memory usage",
    )

    # First GC should be allowed
    gc_coordinator._last_gc_time = 0.0
    should_trigger = await gc_coordinator._should_trigger_gc(normal_pressure)
    assert should_trigger is True

    # Immediate second GC should be blocked by max frequency
    gc_coordinator._last_gc_time = time.time()
    should_trigger = await gc_coordinator._should_trigger_gc(normal_pressure)
    assert should_trigger is False

    # After enough time, should be allowed again
    gc_coordinator._last_gc_time = time.time() - 4.0
    should_trigger = await gc_coordinator._should_trigger_gc(normal_pressure)
    assert should_trigger is True


@pytest.mark.asyncio
async def test_allocation_threshold_trigger(gc_coordinator):
    """Test GC triggering based on allocation threshold."""
    # Set allocation threshold
    gc_coordinator._allocation_threshold = 100
    gc_coordinator._allocation_counter = 150  # Exceed threshold

    # Mock normal memory pressure
    normal_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.LOW,
        current_usage_mb=2000.0,
        available_mb=6000.0,
        total_mb=8000.0,
        cache_usage_mb=400.0,
        cache_usage_percent=5.0,
        recommendation="Low memory usage",
    )

    # Should trigger due to allocation threshold
    should_trigger = await gc_coordinator._should_trigger_gc(normal_pressure)
    assert should_trigger is True


@pytest.mark.asyncio
async def test_monitoring_interval_calculation(gc_coordinator):
    """Test monitoring interval calculation based on memory pressure."""
    # Test different pressure levels
    critical_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.CRITICAL,
        current_usage_mb=7900.0,
        available_mb=100.0,
        total_mb=8000.0,
        cache_usage_mb=2000.0,
        cache_usage_percent=25.0,
        recommendation="Critical memory usage",
    )

    high_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.HIGH,
        current_usage_mb=7000.0,
        available_mb=1000.0,
        total_mb=8000.0,
        cache_usage_mb=1500.0,
        cache_usage_percent=18.75,
        recommendation="High memory usage",
    )

    moderate_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.MODERATE,
        current_usage_mb=5000.0,
        available_mb=3000.0,
        total_mb=8000.0,
        cache_usage_mb=1000.0,
        cache_usage_percent=12.5,
        recommendation="Moderate memory usage",
    )

    low_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.LOW,
        current_usage_mb=2000.0,
        available_mb=6000.0,
        total_mb=8000.0,
        cache_usage_mb=400.0,
        cache_usage_percent=5.0,
        recommendation="Low memory usage",
    )

    # Test intervals
    assert gc_coordinator._calculate_monitoring_interval(critical_pressure) == 5.0
    assert gc_coordinator._calculate_monitoring_interval(high_pressure) == 15.0
    assert gc_coordinator._calculate_monitoring_interval(moderate_pressure) == 30.0
    assert gc_coordinator._calculate_monitoring_interval(low_pressure) == 60.0


@pytest.mark.asyncio
async def test_gc_statistics_tracking(gc_coordinator):
    """Test GC statistics tracking and recording."""
    # Perform manual GC
    await gc_coordinator.manual_gc("Test statistics")

    # Check statistics were recorded
    stats = gc_coordinator.get_gc_stats()
    assert stats.total_gc_runs == 1
    assert stats.manual_gc_runs == 1
    assert stats.total_objects_collected >= 0
    assert stats.total_memory_freed_mb >= 0
    assert stats.last_gc_time > 0

    # Check history
    history = gc_coordinator.get_gc_history()
    assert len(history) == 1
    assert history[0].trigger == GCTrigger.MANUAL


@pytest.mark.asyncio
async def test_gc_event_creation(gc_coordinator):
    """Test GC event creation and tracking."""
    # Perform coordinated GC
    gc_event = await gc_coordinator._perform_coordinated_gc(GCTrigger.MANUAL)

    # Verify event structure
    assert gc_event is not None
    assert gc_event.trigger == GCTrigger.MANUAL
    assert gc_event.timestamp > 0
    assert gc_event.objects_before >= 0
    assert gc_event.objects_after >= 0
    assert gc_event.memory_before_mb >= 0
    assert gc_event.memory_after_mb >= 0
    assert gc_event.duration_seconds >= 0
    assert gc_event.generation >= 0
    assert gc_event.weak_refs_cleaned >= 0
    assert gc_event.fragmentation_before >= 0
    assert gc_event.fragmentation_after >= 0


@pytest.mark.asyncio
async def test_status_report(gc_coordinator):
    """Test status report generation."""
    # Initialize service
    await gc_coordinator.initialize()

    # Perform some operations
    await gc_coordinator.manual_gc("Test status report")

    # Get status report
    report = gc_coordinator.get_status_report()

    # Verify report structure
    assert "enabled" in report
    assert "gc_in_progress" in report
    assert "monitoring_active" in report
    assert "configuration" in report
    assert "statistics" in report
    assert "system_gc_stats" in report
    assert "weak_reference_counts" in report
    assert "current_fragmentation" in report
    assert "allocation_counter" in report
    assert "time_since_last_gc" in report

    # Verify some values
    assert report["enabled"] is True
    assert report["statistics"]["total_gc_runs"] == 1
    assert report["statistics"]["manual_gc_runs"] == 1

    await gc_coordinator.shutdown()


@pytest.mark.asyncio
async def test_stats_reset(gc_coordinator):
    """Test statistics reset functionality."""
    # Perform operations to generate statistics
    await gc_coordinator.manual_gc("Test reset")

    # Verify stats exist
    stats = gc_coordinator.get_gc_stats()
    assert stats.total_gc_runs > 0

    history = gc_coordinator.get_gc_history()
    assert len(history) > 0

    # Reset stats
    gc_coordinator.reset_stats()

    # Verify stats are reset
    stats = gc_coordinator.get_gc_stats()
    assert stats.total_gc_runs == 0
    assert stats.manual_gc_runs == 0
    assert stats.total_objects_collected == 0
    assert stats.total_memory_freed_mb == 0.0

    history = gc_coordinator.get_gc_history()
    assert len(history) == 0


@pytest.mark.asyncio
async def test_concurrent_gc_prevention(gc_coordinator):
    """Test prevention of concurrent GC operations."""
    # Start first GC
    task1 = asyncio.create_task(gc_coordinator.manual_gc("First GC"))

    # Try to start second GC immediately
    task2 = asyncio.create_task(gc_coordinator.manual_gc("Second GC"))

    # Wait for both to complete
    result1, result2 = await asyncio.gather(task1, task2)

    # Only one should have actually performed GC
    assert (result1 is not None) != (result2 is not None)  # Exactly one should be None

    # Statistics should show only one GC run
    stats = gc_coordinator.get_gc_stats()
    assert stats.total_gc_runs == 1


@pytest.mark.asyncio
async def test_error_handling_in_callbacks(gc_coordinator):
    """Test error handling in callback execution."""

    # Create callbacks that raise exceptions
    def failing_pre_gc_callback():
        raise Exception("Pre-GC callback failed")

    async def failing_post_gc_callback(gc_event):
        raise Exception("Post-GC callback failed")

    # Register failing callbacks
    gc_coordinator.register_pre_gc_callback(failing_pre_gc_callback)
    gc_coordinator.register_post_gc_callback(failing_post_gc_callback)

    # GC should still complete despite callback failures
    gc_event = await gc_coordinator.manual_gc("Test error handling")

    # Verify GC completed successfully
    assert gc_event is not None
    assert gc_event.trigger == GCTrigger.MANUAL


@pytest.mark.asyncio
async def test_global_service_functions():
    """Test global service functions."""
    # Test initialization
    await initialize_gc_coordinator_service()
    service = get_gc_coordinator_service()
    assert service is not None

    # Test shutdown
    await shutdown_gc_coordinator_service()

    # Service should still be accessible but shut down
    service = get_gc_coordinator_service()
    assert service is not None


@pytest.mark.asyncio
async def test_monitoring_loop_error_handling(gc_coordinator):
    """Test error handling in monitoring loop."""
    # Initialize service
    await gc_coordinator.initialize()

    # Mock get_system_memory_pressure to raise exception
    with patch("src.services.cache_gc_coordinator_service.get_system_memory_pressure", side_effect=Exception("Monitoring error")):
        # Let monitoring loop run for a short time
        await asyncio.sleep(0.1)

        # Service should still be running
        assert not gc_coordinator._monitoring_task.done()

    # Cleanup
    await gc_coordinator.shutdown()


@pytest.mark.asyncio
async def test_fragmentation_calculation(gc_coordinator):
    """Test fragmentation calculation."""
    # Test with mock GC stats
    mock_stats = [
        {"collections": 100, "uncollectable": 10},
        {"collections": 50, "uncollectable": 2},
        {"collections": 25, "uncollectable": 1},
    ]

    with patch("gc.get_stats", return_value=mock_stats):
        fragmentation = gc_coordinator._calculate_fragmentation()

        # Should calculate based on generation 0
        expected = 10 / 100
        assert fragmentation == expected


@pytest.mark.asyncio
async def test_gc_configuration_strategies(gc_coordinator):
    """Test different GC configuration strategies."""
    # Test incremental strategy
    incremental_config = GCCoordinationConfig(strategy=GCStrategy.INCREMENTAL)
    gc_coordinator.update_config(incremental_config)

    # Mock gc.set_threshold to verify it's called with correct values
    with patch("gc.set_threshold") as mock_set_threshold:
        gc_coordinator._configure_gc()
        mock_set_threshold.assert_called_with(500, 10, 5)

    # Test generational strategy
    generational_config = GCCoordinationConfig(strategy=GCStrategy.GENERATIONAL)
    gc_coordinator.update_config(generational_config)

    with patch("gc.set_threshold") as mock_set_threshold:
        gc_coordinator._configure_gc()
        mock_set_threshold.assert_called_with(700, 10, 10)

    # Test adaptive strategy
    adaptive_config = GCCoordinationConfig(strategy=GCStrategy.ADAPTIVE)
    gc_coordinator.update_config(adaptive_config)

    with patch("gc.set_threshold") as mock_set_threshold:
        gc_coordinator._configure_gc()
        mock_set_threshold.assert_called_with(1000, 15, 15)


if __name__ == "__main__":
    pytest.main([__file__])
