"""
Tests for cache memory pressure service.

This module tests the cache memory pressure handling functionality including
eviction strategies, pressure response coordination, and memory monitoring.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..utils.memory_utils import CacheMemoryStats, SystemMemoryPressure
from .cache_memory_pressure_service import (
    CacheEvictionConfig,
    CacheMemoryPressureService,
    CacheMemoryPressureStats,
    EvictionStrategy,
    MemoryPressureLevel,
    PressureResponse,
    PressureResponseLevel,
    get_cache_memory_pressure_service,
    initialize_cache_memory_pressure_service,
    shutdown_cache_memory_pressure_service,
)


class MockCacheService:
    """Mock cache service for testing."""

    def __init__(self, name: str, size: int = 1000):
        self.name = name
        self.size = size
        self.evicted_items = []
        self.evicted_size = 0.0

    async def get_cache_size(self) -> int:
        """Get current cache size."""
        return self.size

    async def evict_lru(self, target_eviction: int, batch_size: int) -> tuple[int, float]:
        """Mock LRU eviction."""
        evicted = min(target_eviction, self.size)
        self.evicted_items.extend([f"lru_item_{i}" for i in range(evicted)])
        self.evicted_size += evicted * 0.1  # 0.1 MB per item
        self.size -= evicted
        return evicted, evicted * 0.1

    async def evict_lfu(self, target_eviction: int, batch_size: int) -> tuple[int, float]:
        """Mock LFU eviction."""
        evicted = min(target_eviction, self.size)
        self.evicted_items.extend([f"lfu_item_{i}" for i in range(evicted)])
        self.evicted_size += evicted * 0.1
        self.size -= evicted
        return evicted, evicted * 0.1

    async def evict_expired(self, target_eviction: int, batch_size: int, ttl_threshold: float) -> tuple[int, float]:
        """Mock TTL-based eviction."""
        evicted = min(target_eviction, self.size // 2)  # Half are expired
        self.evicted_items.extend([f"expired_item_{i}" for i in range(evicted)])
        self.evicted_size += evicted * 0.1
        self.size -= evicted
        return evicted, evicted * 0.1

    async def evict_largest(self, target_eviction: int, batch_size: int, size_threshold: float) -> tuple[int, float]:
        """Mock size-based eviction."""
        evicted = min(target_eviction, self.size // 4)  # Quarter are large
        self.evicted_items.extend([f"large_item_{i}" for i in range(evicted)])
        self.evicted_size += evicted * 0.5  # Large items are 0.5 MB each
        self.size -= evicted
        return evicted, evicted * 0.5

    async def evict_random(self, target_eviction: int, batch_size: int) -> tuple[int, float]:
        """Mock random eviction."""
        evicted = min(target_eviction, self.size)
        self.evicted_items.extend([f"random_item_{i}" for i in range(evicted)])
        self.evicted_size += evicted * 0.1
        self.size -= evicted
        return evicted, evicted * 0.1

    async def evict_adaptive(self, target_eviction: int, batch_size: int, window_seconds: float) -> tuple[int, float]:
        """Mock adaptive eviction."""
        evicted = min(target_eviction, self.size)
        self.evicted_items.extend([f"adaptive_item_{i}" for i in range(evicted)])
        self.evicted_size += evicted * 0.1
        self.size -= evicted
        return evicted, evicted * 0.1


@pytest.fixture
def mock_cache_service():
    """Create a mock cache service."""
    return MockCacheService("test_cache", 1000)


@pytest.fixture
def pressure_service():
    """Create a cache memory pressure service."""
    return CacheMemoryPressureService()


@pytest.fixture
def eviction_config():
    """Create a test eviction configuration."""
    return CacheEvictionConfig(
        strategy=EvictionStrategy.LRU, batch_size=50, max_eviction_percentage=0.3, min_retention_count=100, ttl_threshold_seconds=300.0
    )


@pytest.mark.asyncio
async def test_service_initialization(pressure_service):
    """Test service initialization and shutdown."""
    # Test initialization
    await pressure_service.initialize()
    assert pressure_service._monitoring_task is not None
    assert not pressure_service._monitoring_task.done()

    # Test shutdown
    await pressure_service.shutdown()
    assert pressure_service._monitoring_task.done()


@pytest.mark.asyncio
async def test_cache_registration(pressure_service, mock_cache_service):
    """Test cache service registration and unregistration."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    assert "test_cache" in pressure_service._registered_caches
    assert pressure_service._registered_caches["test_cache"] == mock_cache_service
    assert "test_cache" in pressure_service._eviction_configs

    # Unregister cache
    pressure_service.unregister_cache("test_cache")

    assert "test_cache" not in pressure_service._registered_caches
    assert "test_cache" not in pressure_service._eviction_configs


@pytest.mark.asyncio
async def test_eviction_config_update(pressure_service, mock_cache_service, eviction_config):
    """Test eviction configuration updates."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Update config
    pressure_service.update_eviction_config("test_cache", eviction_config)

    assert pressure_service._eviction_configs["test_cache"] == eviction_config
    assert pressure_service._eviction_configs["test_cache"].strategy == EvictionStrategy.LRU


@pytest.mark.asyncio
async def test_pressure_response_update(pressure_service):
    """Test pressure response configuration updates."""
    new_response = PressureResponse(
        level=PressureResponseLevel.AGGRESSIVE,
        eviction_percentage=0.4,
        strategy_override=EvictionStrategy.SIZE,
        description="Custom aggressive response",
    )

    pressure_service.update_pressure_response(MemoryPressureLevel.HIGH, new_response)

    assert pressure_service._pressure_responses[MemoryPressureLevel.HIGH] == new_response


@pytest.mark.asyncio
async def test_lru_eviction(pressure_service, mock_cache_service):
    """Test LRU eviction strategy."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Create pressure response
    response = PressureResponse(level=PressureResponseLevel.MODERATE, eviction_percentage=0.1, strategy_override=EvictionStrategy.LRU)

    # Perform eviction
    evicted_count, evicted_size = await pressure_service._perform_eviction("test_cache", response)

    assert evicted_count == 100  # 10% of 1000 items
    assert evicted_size == 10.0  # 100 * 0.1 MB
    assert len(mock_cache_service.evicted_items) == 100
    assert all(item.startswith("lru_item_") for item in mock_cache_service.evicted_items)


@pytest.mark.asyncio
async def test_size_eviction(pressure_service, mock_cache_service):
    """Test size-based eviction strategy."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Create pressure response
    response = PressureResponse(level=PressureResponseLevel.MODERATE, eviction_percentage=0.1, strategy_override=EvictionStrategy.SIZE)

    # Perform eviction
    evicted_count, evicted_size = await pressure_service._perform_eviction("test_cache", response)

    assert evicted_count == 25  # 25% of target (100) since only 25% are large
    assert evicted_size == 12.5  # 25 * 0.5 MB
    assert len(mock_cache_service.evicted_items) == 25
    assert all(item.startswith("large_item_") for item in mock_cache_service.evicted_items)


@pytest.mark.asyncio
async def test_ttl_eviction(pressure_service, mock_cache_service):
    """Test TTL-based eviction strategy."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Create pressure response
    response = PressureResponse(level=PressureResponseLevel.MODERATE, eviction_percentage=0.1, strategy_override=EvictionStrategy.TTL)

    # Perform eviction
    evicted_count, evicted_size = await pressure_service._perform_eviction("test_cache", response)

    assert evicted_count == 50  # 50% of target (100) since only 50% are expired
    assert evicted_size == 5.0  # 50 * 0.1 MB
    assert len(mock_cache_service.evicted_items) == 50
    assert all(item.startswith("expired_item_") for item in mock_cache_service.evicted_items)


@pytest.mark.asyncio
async def test_adaptive_eviction(pressure_service, mock_cache_service):
    """Test adaptive eviction strategy."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Create pressure response
    response = PressureResponse(level=PressureResponseLevel.MODERATE, eviction_percentage=0.1, strategy_override=EvictionStrategy.ADAPTIVE)

    # Perform eviction
    evicted_count, evicted_size = await pressure_service._perform_eviction("test_cache", response)

    assert evicted_count == 100  # 10% of 1000 items
    assert evicted_size == 10.0  # 100 * 0.1 MB
    assert len(mock_cache_service.evicted_items) == 100
    assert all(item.startswith("adaptive_item_") for item in mock_cache_service.evicted_items)


@pytest.mark.asyncio
async def test_manual_eviction(pressure_service, mock_cache_service):
    """Test manual eviction trigger."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Perform manual eviction
    evicted_count, evicted_size = await pressure_service.manual_eviction("test_cache", EvictionStrategy.LRU, 50, "Test manual eviction")

    assert evicted_count == 50
    assert evicted_size == 5.0
    assert len(mock_cache_service.evicted_items) == 50


@pytest.mark.asyncio
async def test_pressure_handling_enablement(pressure_service, mock_cache_service):
    """Test enabling and disabling pressure handling."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Disable pressure handling
    pressure_service.disable_pressure_handling()
    assert not pressure_service._pressure_handling_enabled

    # Test that pressure handling is skipped when disabled
    with patch.object(pressure_service, "_perform_eviction") as mock_eviction:
        await pressure_service._handle_cache_pressure("test_cache", MemoryPressureLevel.HIGH, "Test pressure")
        mock_eviction.assert_not_called()

    # Enable pressure handling
    pressure_service.enable_pressure_handling()
    assert pressure_service._pressure_handling_enabled


@pytest.mark.asyncio
async def test_statistics_tracking(pressure_service, mock_cache_service):
    """Test pressure statistics tracking."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Perform eviction to generate statistics
    await pressure_service._handle_cache_pressure("test_cache", MemoryPressureLevel.HIGH, "Test statistics")

    stats = pressure_service.get_pressure_stats()
    assert stats.total_pressure_events == 1
    assert stats.total_evictions > 0
    assert stats.evicted_size_mb > 0
    assert PressureResponseLevel.MODERATE in stats.responses_by_level


@pytest.mark.asyncio
async def test_multiple_cache_handling(pressure_service):
    """Test handling multiple caches simultaneously."""
    # Create multiple mock caches
    cache1 = MockCacheService("cache1", 500)
    cache2 = MockCacheService("cache2", 800)

    # Register caches
    pressure_service.register_cache("cache1", cache1)
    pressure_service.register_cache("cache2", cache2)

    # Simulate system pressure
    system_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.HIGH,
        current_usage_mb=8000,
        available_mb=2000,
        total_mb=10000,
        cache_usage_mb=1000,
        cache_usage_percent=10.0,
        recommendation="Reduce cache sizes",
    )

    await pressure_service._handle_system_pressure(system_pressure)

    # Check that both caches were handled
    assert len(cache1.evicted_items) > 0
    assert len(cache2.evicted_items) > 0


@pytest.mark.asyncio
async def test_concurrent_pressure_handling(pressure_service, mock_cache_service):
    """Test concurrent pressure handling for the same cache."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Start multiple concurrent pressure handling tasks
    tasks = []
    for i in range(3):
        task = asyncio.create_task(
            pressure_service._handle_cache_pressure("test_cache", MemoryPressureLevel.MODERATE, f"Concurrent test {i}")
        )
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Only one should have actually performed eviction
    stats = pressure_service.get_pressure_stats()
    assert stats.total_pressure_events == 1


@pytest.mark.asyncio
async def test_eviction_limits(pressure_service, mock_cache_service, eviction_config):
    """Test eviction limits and constraints."""
    # Set strict limits
    eviction_config.max_eviction_percentage = 0.05  # Max 5%
    eviction_config.min_retention_count = 900  # Min 900 items

    # Register cache with config
    pressure_service.register_cache("test_cache", mock_cache_service, eviction_config)

    # Try to evict more than limit
    response = PressureResponse(
        level=PressureResponseLevel.CRITICAL,
        eviction_percentage=0.5,
        strategy_override=EvictionStrategy.LRU,  # Request 50% eviction
    )

    evicted_count, evicted_size = await pressure_service._perform_eviction("test_cache", response)

    # Should be limited by max_eviction_percentage and min_retention_count
    expected_max = min(int(1000 * 0.05), 1000 - 900)  # min(50, 100) = 50
    assert evicted_count == expected_max


@pytest.mark.asyncio
async def test_status_report(pressure_service, mock_cache_service):
    """Test status report generation."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Perform some operations
    await pressure_service._handle_cache_pressure("test_cache", MemoryPressureLevel.HIGH, "Test status")

    # Get status report
    report = pressure_service.get_status_report()

    assert report["enabled"] is True
    assert report["registered_caches"] == 1
    assert report["active_responses"] == 0
    assert "statistics" in report
    assert "configuration" in report
    assert report["statistics"]["total_pressure_events"] == 1


@pytest.mark.asyncio
async def test_stats_reset(pressure_service, mock_cache_service):
    """Test statistics reset functionality."""
    # Register cache and perform operations
    pressure_service.register_cache("test_cache", mock_cache_service)

    await pressure_service._handle_cache_pressure("test_cache", MemoryPressureLevel.HIGH, "Test reset")

    # Verify stats exist
    stats = pressure_service.get_pressure_stats()
    assert stats.total_pressure_events > 0

    # Reset stats
    pressure_service.reset_stats()

    # Verify stats are reset
    stats = pressure_service.get_pressure_stats()
    assert stats.total_pressure_events == 0
    assert stats.total_evictions == 0
    assert stats.evicted_size_mb == 0.0


@pytest.mark.asyncio
async def test_global_service_functions():
    """Test global service functions."""
    # Test initialization
    await initialize_cache_memory_pressure_service()
    service = get_cache_memory_pressure_service()
    assert service is not None

    # Test shutdown
    await shutdown_cache_memory_pressure_service()

    # Service should still be accessible but shut down
    service = get_cache_memory_pressure_service()
    assert service is not None


@pytest.mark.asyncio
async def test_error_handling_during_eviction(pressure_service, mock_cache_service):
    """Test error handling during eviction operations."""
    # Register cache
    pressure_service.register_cache("test_cache", mock_cache_service)

    # Mock eviction method to raise exception
    mock_cache_service.evict_lru = AsyncMock(side_effect=Exception("Eviction failed"))

    # Test that error is handled gracefully
    await pressure_service._handle_cache_pressure("test_cache", MemoryPressureLevel.HIGH, "Test error handling")

    # Service should continue functioning
    assert pressure_service._pressure_handling_enabled is True


@pytest.mark.asyncio
async def test_cache_without_eviction_methods(pressure_service):
    """Test handling of cache service without eviction methods."""
    # Create cache without eviction methods
    basic_cache = MagicMock()
    basic_cache.get_cache_size = AsyncMock(return_value=100)
    # No eviction methods defined

    pressure_service.register_cache("basic_cache", basic_cache)

    # Should not raise error, should return 0 evictions
    evicted_count, evicted_size = await pressure_service._perform_eviction(
        "basic_cache",
        PressureResponse(level=PressureResponseLevel.MODERATE, eviction_percentage=0.1, strategy_override=EvictionStrategy.LRU),
    )

    assert evicted_count == 0
    assert evicted_size == 0.0


@pytest.mark.asyncio
async def test_monitoring_loop_error_handling(pressure_service):
    """Test error handling in monitoring loop."""
    # Initialize service
    await pressure_service.initialize()

    # Mock get_system_memory_pressure to raise exception
    with patch("src.services.cache_memory_pressure_service.get_system_memory_pressure", side_effect=Exception("Monitoring error")):
        # Let monitoring loop run for a short time
        await asyncio.sleep(0.1)

        # Service should still be running
        assert not pressure_service._monitoring_task.done()

    # Cleanup
    await pressure_service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
