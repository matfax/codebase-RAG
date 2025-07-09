"""
Tests for adaptive cache sizing service.

This module tests the adaptive cache sizing functionality including
size calculations, resizing operations, and performance optimization.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..utils.memory_utils import CacheMemoryStats, MemoryPressureLevel, SystemMemoryPressure
from .adaptive_cache_sizing_service import (
    AdaptiveCacheSizingService,
    AdaptiveSizingStats,
    CacheSizingConfig,
    CacheUsageMetrics,
    SizingDecision,
    SizingStrategy,
    SizingTrigger,
    get_adaptive_sizing_service,
    initialize_adaptive_sizing_service,
    shutdown_adaptive_sizing_service,
)


class MockAdaptiveCacheService:
    """Mock cache service for adaptive sizing testing."""

    def __init__(self, name: str, current_size_mb: float = 100.0, max_size_mb: float = 1000.0):
        self.name = name
        self.current_size_mb = current_size_mb
        self.max_size_mb = max_size_mb
        self.resize_calls = []
        self.resize_success = True

    async def resize(self, target_size_mb: float) -> bool:
        """Mock resize operation."""
        self.resize_calls.append(target_size_mb)
        if self.resize_success:
            self.current_size_mb = target_size_mb
            return True
        return False

    async def set_max_size(self, max_size_mb: float) -> bool:
        """Mock set max size operation."""
        self.max_size_mb = max_size_mb
        return True

    async def get_stats(self) -> dict:
        """Mock get stats operation."""
        return {"current_size_mb": self.current_size_mb, "max_size_mb": self.max_size_mb, "hit_rate": 0.75, "eviction_rate": 0.05}


@pytest.fixture
def mock_cache_service():
    """Create a mock adaptive cache service."""
    return MockAdaptiveCacheService("test_cache", 100.0, 1000.0)


@pytest.fixture
def sizing_service():
    """Create an adaptive cache sizing service."""
    return AdaptiveCacheSizingService()


@pytest.fixture
def sizing_config():
    """Create a test sizing configuration."""
    return CacheSizingConfig(
        strategy=SizingStrategy.ADAPTIVE,
        min_size_mb=50.0,
        max_size_mb=500.0,
        target_memory_percentage=0.3,
        growth_factor=1.5,
        shrink_factor=0.8,
        resize_threshold_mb=25.0,
    )


@pytest.fixture
def cache_stats():
    """Create mock cache memory statistics."""
    return CacheMemoryStats(
        cache_name="test_cache",
        current_size_mb=100.0,
        peak_size_mb=150.0,
        total_allocated_mb=500.0,
        total_deallocated_mb=400.0,
        eviction_count=10,
        memory_efficiency=0.75,
        allocation_rate_mb_per_sec=2.0,
        deallocation_rate_mb_per_sec=1.5,
        fragmentation_ratio=0.1,
    )


@pytest.fixture
def system_pressure():
    """Create mock system memory pressure."""
    return SystemMemoryPressure(
        level=MemoryPressureLevel.LOW,
        current_usage_mb=2000.0,
        available_mb=6000.0,
        total_mb=8000.0,
        cache_usage_mb=500.0,
        cache_usage_percent=6.25,
        recommendation="Normal memory usage",
    )


@pytest.mark.asyncio
async def test_service_initialization(sizing_service):
    """Test service initialization and shutdown."""
    # Test initialization
    await sizing_service.initialize()
    assert sizing_service._monitoring_task is not None
    assert not sizing_service._monitoring_task.done()

    # Test shutdown
    await sizing_service.shutdown()
    assert sizing_service._monitoring_task.done()


@pytest.mark.asyncio
async def test_cache_registration(sizing_service, mock_cache_service, sizing_config):
    """Test cache registration and unregistration."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service, sizing_config)

    assert "test_cache" in sizing_service._registered_caches
    assert sizing_service._registered_caches["test_cache"] == mock_cache_service
    assert "test_cache" in sizing_service._sizing_configs
    assert sizing_service._sizing_configs["test_cache"] == sizing_config
    assert "test_cache" in sizing_service._usage_metrics

    # Unregister cache
    sizing_service.unregister_cache("test_cache")

    assert "test_cache" not in sizing_service._registered_caches
    assert "test_cache" not in sizing_service._sizing_configs
    assert "test_cache" not in sizing_service._usage_metrics


@pytest.mark.asyncio
async def test_sizing_config_update(sizing_service, mock_cache_service, sizing_config):
    """Test sizing configuration updates."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Update config
    sizing_service.update_sizing_config("test_cache", sizing_config)

    assert sizing_service._sizing_configs["test_cache"] == sizing_config
    assert sizing_service._sizing_configs["test_cache"].strategy == SizingStrategy.ADAPTIVE


@pytest.mark.asyncio
async def test_usage_metrics_update(sizing_service, cache_stats):
    """Test usage metrics updates from cache statistics."""
    metrics = CacheUsageMetrics()

    # Update metrics from cache stats
    metrics.update_from_cache_stats(cache_stats)

    assert metrics.memory_efficiency == 0.75
    assert metrics.growth_rate == 0.5  # 2.0 - 1.5
    assert metrics.fragmentation == 0.1
    assert metrics.last_updated > 0


@pytest.mark.asyncio
async def test_adaptive_size_calculation(sizing_service, mock_cache_service, cache_stats, system_pressure):
    """Test adaptive size calculation."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Calculate adaptive size
    adaptive_size = await sizing_service._calculate_adaptive_size("test_cache", cache_stats, system_pressure)

    assert adaptive_size >= sizing_service._sizing_configs["test_cache"].min_size_mb
    assert adaptive_size <= sizing_service._sizing_configs["test_cache"].max_size_mb
    assert adaptive_size > 0


@pytest.mark.asyncio
async def test_performance_based_size_calculation(sizing_service, mock_cache_service):
    """Test performance-based size calculation."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Create metrics with good performance
    metrics = CacheUsageMetrics()
    metrics.hit_rate = 0.95
    metrics.average_response_time = 5.0
    metrics.memory_efficiency = 0.85
    metrics.fragmentation = 0.05

    # Calculate performance-based size
    with patch.object(sizing_service, "get_cache_memory_stats") as mock_get_stats:
        mock_get_stats.return_value = CacheMemoryStats("test_cache", current_size_mb=100.0)

        performance_size = await sizing_service._calculate_performance_based_size("test_cache", metrics)

        # Should grow due to good performance
        assert performance_size > 100.0


@pytest.mark.asyncio
async def test_sizing_analysis_memory_pressure(sizing_service, mock_cache_service, cache_stats):
    """Test sizing analysis under memory pressure."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Create critical memory pressure
    critical_pressure = SystemMemoryPressure(
        level=MemoryPressureLevel.CRITICAL,
        current_usage_mb=7500.0,
        available_mb=500.0,
        total_mb=8000.0,
        cache_usage_mb=1000.0,
        cache_usage_percent=12.5,
        recommendation="Critical memory pressure",
    )

    # Analyze sizing need
    decision = await sizing_service._analyze_sizing_need("test_cache", cache_stats, critical_pressure)

    assert decision is not None
    assert decision.target_size_mb < decision.current_size_mb
    assert decision.trigger == SizingTrigger.MEMORY_PRESSURE
    assert decision.memory_pressure == MemoryPressureLevel.CRITICAL


@pytest.mark.asyncio
async def test_sizing_analysis_performance_trigger(sizing_service, mock_cache_service, cache_stats, system_pressure):
    """Test sizing analysis based on performance metrics."""
    # Register cache with performance strategy
    config = CacheSizingConfig(strategy=SizingStrategy.PERFORMANCE)
    sizing_service.register_cache("test_cache", mock_cache_service, config)

    # Set poor performance metrics
    metrics = sizing_service._usage_metrics["test_cache"]
    metrics.hit_rate = 0.3
    metrics.eviction_rate = 0.2

    # Analyze sizing need
    decision = await sizing_service._analyze_sizing_need("test_cache", cache_stats, system_pressure)

    assert decision is not None
    assert decision.target_size_mb > decision.current_size_mb
    assert decision.trigger == SizingTrigger.PERFORMANCE


@pytest.mark.asyncio
async def test_manual_resize(sizing_service, mock_cache_service):
    """Test manual resize operation."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Mock get_cache_memory_stats
    with patch("src.services.adaptive_cache_sizing_service.get_cache_memory_stats") as mock_get_stats:
        mock_get_stats.return_value = CacheMemoryStats("test_cache", current_size_mb=100.0)

        # Perform manual resize
        success = await sizing_service.manual_resize("test_cache", 200.0, "Test manual resize")

        assert success
        assert len(mock_cache_service.resize_calls) == 1
        assert mock_cache_service.resize_calls[0] == 200.0


@pytest.mark.asyncio
async def test_resize_operation_success(sizing_service, mock_cache_service, cache_stats, system_pressure):
    """Test successful resize operation."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Create sizing decision
    decision = SizingDecision(
        cache_name="test_cache",
        current_size_mb=100.0,
        target_size_mb=150.0,
        strategy=SizingStrategy.ADAPTIVE,
        trigger=SizingTrigger.PERFORMANCE,
        reasoning="Test resize",
        confidence=0.8,
        memory_pressure=MemoryPressureLevel.LOW,
        expected_impact="Improved performance",
    )

    # Perform resize
    await sizing_service._perform_resize(decision)

    # Verify resize was called
    assert len(mock_cache_service.resize_calls) == 1
    assert mock_cache_service.resize_calls[0] == 150.0

    # Verify statistics updated
    assert sizing_service._stats.total_resizes == 1
    assert sizing_service._stats.successful_resizes == 1
    assert sizing_service._stats.size_increases == 1

    # Verify decision stored in history
    assert len(sizing_service._sizing_history) == 1
    assert sizing_service._sizing_history[0] == decision


@pytest.mark.asyncio
async def test_resize_operation_failure(sizing_service, mock_cache_service):
    """Test failed resize operation."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Make resize fail
    mock_cache_service.resize_success = False

    # Create sizing decision
    decision = SizingDecision(
        cache_name="test_cache",
        current_size_mb=100.0,
        target_size_mb=150.0,
        strategy=SizingStrategy.ADAPTIVE,
        trigger=SizingTrigger.PERFORMANCE,
        reasoning="Test resize failure",
        confidence=0.8,
        memory_pressure=MemoryPressureLevel.LOW,
        expected_impact="Test failure",
    )

    # Perform resize
    await sizing_service._perform_resize(decision)

    # Verify statistics updated for failure
    assert sizing_service._stats.total_resizes == 1
    assert sizing_service._stats.successful_resizes == 0
    assert sizing_service._stats.failed_resizes == 1


@pytest.mark.asyncio
async def test_concurrent_resize_prevention(sizing_service, mock_cache_service):
    """Test prevention of concurrent resize operations."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Create two resize decisions
    decision1 = SizingDecision(
        cache_name="test_cache",
        current_size_mb=100.0,
        target_size_mb=150.0,
        strategy=SizingStrategy.ADAPTIVE,
        trigger=SizingTrigger.PERFORMANCE,
        reasoning="First resize",
        confidence=0.8,
        memory_pressure=MemoryPressureLevel.LOW,
        expected_impact="Test concurrent 1",
    )

    decision2 = SizingDecision(
        cache_name="test_cache",
        current_size_mb=100.0,
        target_size_mb=200.0,
        strategy=SizingStrategy.ADAPTIVE,
        trigger=SizingTrigger.PERFORMANCE,
        reasoning="Second resize",
        confidence=0.8,
        memory_pressure=MemoryPressureLevel.LOW,
        expected_impact="Test concurrent 2",
    )

    # Start concurrent resize operations
    tasks = [asyncio.create_task(sizing_service._perform_resize(decision1)), asyncio.create_task(sizing_service._perform_resize(decision2))]

    # Wait for completion
    await asyncio.gather(*tasks)

    # Only one resize should have been performed
    assert len(mock_cache_service.resize_calls) == 1
    assert sizing_service._stats.total_resizes == 1


@pytest.mark.asyncio
async def test_sizing_enablement(sizing_service, mock_cache_service):
    """Test enabling and disabling sizing."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Disable sizing
    sizing_service.disable_sizing()
    assert not sizing_service._sizing_enabled

    # Test that sizing is skipped when disabled
    with patch.object(sizing_service, "_analyze_sizing_need") as mock_analyze:
        await sizing_service._check_cache_sizing("test_cache")
        mock_analyze.assert_not_called()

    # Enable sizing
    sizing_service.enable_sizing()
    assert sizing_service._sizing_enabled


@pytest.mark.asyncio
async def test_sizing_constraints(sizing_service, mock_cache_service):
    """Test sizing constraints (min/max size limits)."""
    # Register cache with strict constraints
    config = CacheSizingConfig(min_size_mb=100.0, max_size_mb=200.0, resize_threshold_mb=10.0)
    sizing_service.register_cache("test_cache", mock_cache_service, config)

    # Test manual resize beyond max constraint
    with patch("src.services.adaptive_cache_sizing_service.get_cache_memory_stats") as mock_get_stats:
        mock_get_stats.return_value = CacheMemoryStats("test_cache", current_size_mb=150.0)

        success = await sizing_service.manual_resize("test_cache", 500.0, "Test constraint")

        # Should be constrained to max_size_mb
        assert success
        assert mock_cache_service.resize_calls[0] == 200.0


@pytest.mark.asyncio
async def test_optimal_size_prediction(sizing_service, mock_cache_service, cache_stats, system_pressure):
    """Test optimal size prediction."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Mock get_cache_memory_stats and get_system_memory_pressure
    with (
        patch("src.services.adaptive_cache_sizing_service.get_cache_memory_stats") as mock_get_stats,
        patch("src.services.adaptive_cache_sizing_service.get_system_memory_pressure") as mock_get_pressure,
    ):
        mock_get_stats.return_value = cache_stats
        mock_get_pressure.return_value = system_pressure

        # Predict optimal size
        optimal_size = sizing_service.predict_optimal_size("test_cache")

        assert optimal_size is not None
        assert optimal_size > 0
        assert optimal_size >= sizing_service._sizing_configs["test_cache"].min_size_mb
        assert optimal_size <= sizing_service._sizing_configs["test_cache"].max_size_mb


@pytest.mark.asyncio
async def test_sizing_history_tracking(sizing_service, mock_cache_service):
    """Test sizing history tracking and retrieval."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Perform multiple manual resizes
    with patch("src.services.adaptive_cache_sizing_service.get_cache_memory_stats") as mock_get_stats:
        mock_get_stats.return_value = CacheMemoryStats("test_cache", current_size_mb=100.0)

        await sizing_service.manual_resize("test_cache", 150.0, "First resize")
        await sizing_service.manual_resize("test_cache", 200.0, "Second resize")
        await sizing_service.manual_resize("test_cache", 120.0, "Third resize")

    # Check history
    history = sizing_service.get_sizing_history()
    assert len(history) == 3

    # Check cache-specific history
    cache_history = sizing_service.get_sizing_history("test_cache")
    assert len(cache_history) == 3

    # Check history limit
    limited_history = sizing_service.get_sizing_history("test_cache", limit=2)
    assert len(limited_history) == 2


@pytest.mark.asyncio
async def test_impact_prediction(sizing_service):
    """Test impact prediction for resizing."""
    metrics = CacheUsageMetrics()
    metrics.hit_rate = 0.75

    # Test size increase impact
    impact_increase = sizing_service._predict_impact(100.0, 150.0, metrics)
    assert "improvements" in impact_increase
    assert "hit rate" in impact_increase

    # Test size decrease impact
    impact_decrease = sizing_service._predict_impact(150.0, 100.0, metrics)
    assert "memory savings" in impact_decrease
    assert "50.0MB" in impact_decrease


@pytest.mark.asyncio
async def test_status_report(sizing_service, mock_cache_service):
    """Test status report generation."""
    # Register cache
    sizing_service.register_cache("test_cache", mock_cache_service)

    # Mock dependencies
    with (
        patch("src.services.adaptive_cache_sizing_service.get_cache_memory_stats") as mock_get_stats,
        patch("src.services.adaptive_cache_sizing_service.get_system_memory_pressure") as mock_get_pressure,
    ):
        mock_get_stats.return_value = CacheMemoryStats("test_cache", current_size_mb=100.0)
        mock_get_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=2000.0,
            available_mb=6000.0,
            total_mb=8000.0,
            cache_usage_mb=500.0,
            cache_usage_percent=6.25,
            recommendation="Normal",
        )

        # Get status report
        report = sizing_service.get_status_report()

        assert report["enabled"] is True
        assert report["registered_caches"] == 1
        assert report["active_resizes"] == 0
        assert "statistics" in report
        assert "cache_status" in report
        assert "system_memory" in report
        assert "test_cache" in report["cache_status"]


@pytest.mark.asyncio
async def test_statistics_reset(sizing_service, mock_cache_service):
    """Test statistics reset functionality."""
    # Register cache and perform operations
    sizing_service.register_cache("test_cache", mock_cache_service)

    with patch("src.services.adaptive_cache_sizing_service.get_cache_memory_stats") as mock_get_stats:
        mock_get_stats.return_value = CacheMemoryStats("test_cache", current_size_mb=100.0)

        await sizing_service.manual_resize("test_cache", 150.0, "Test resize")

    # Verify stats exist
    assert sizing_service._stats.total_resizes > 0
    assert len(sizing_service._sizing_history) > 0

    # Reset stats
    sizing_service.reset_stats()

    # Verify stats are reset
    assert sizing_service._stats.total_resizes == 0
    assert len(sizing_service._sizing_history) == 0


@pytest.mark.asyncio
async def test_cache_without_resize_methods(sizing_service):
    """Test handling of cache service without resize methods."""
    # Create cache without resize methods
    basic_cache = MagicMock()

    sizing_service.register_cache("basic_cache", basic_cache)

    # Should not raise error, should return False
    success = await sizing_service._resize_cache(basic_cache, 200.0)
    assert success is False


@pytest.mark.asyncio
async def test_global_service_functions():
    """Test global service functions."""
    # Test initialization
    await initialize_adaptive_sizing_service()
    service = get_adaptive_sizing_service()
    assert service is not None

    # Test shutdown
    await shutdown_adaptive_sizing_service()

    # Service should still be accessible but shut down
    service = get_adaptive_sizing_service()
    assert service is not None


@pytest.mark.asyncio
async def test_monitoring_loop_error_handling(sizing_service):
    """Test error handling in monitoring loop."""
    # Initialize service
    await sizing_service.initialize()

    # Mock get_system_memory_pressure to raise exception
    with patch("src.services.adaptive_cache_sizing_service.get_system_memory_pressure", side_effect=Exception("Monitoring error")):
        # Let monitoring loop run for a short time
        await asyncio.sleep(0.1)

        # Service should still be running
        assert not sizing_service._monitoring_task.done()

    # Cleanup
    await sizing_service.shutdown()


@pytest.mark.asyncio
async def test_resize_threshold_enforcement(sizing_service, mock_cache_service, cache_stats, system_pressure):
    """Test that resize threshold is enforced."""
    # Register cache with high threshold
    config = CacheSizingConfig(resize_threshold_mb=100.0)
    sizing_service.register_cache("test_cache", mock_cache_service, config)

    # Create decision with small change
    decision = await sizing_service._analyze_sizing_need("test_cache", cache_stats, system_pressure)

    # Should be None because change is below threshold
    assert decision is None or abs(decision.target_size_mb - decision.current_size_mb) >= 100.0


@pytest.mark.asyncio
async def test_fixed_strategy_no_resize(sizing_service, mock_cache_service, cache_stats, system_pressure):
    """Test that fixed strategy prevents automatic resizing."""
    # Register cache with fixed strategy
    config = CacheSizingConfig(strategy=SizingStrategy.FIXED)
    sizing_service.register_cache("test_cache", mock_cache_service, config)

    # Analyze sizing need
    decision = await sizing_service._analyze_sizing_need("test_cache", cache_stats, system_pressure)

    # Should return None for fixed strategy
    assert decision is None


if __name__ == "__main__":
    pytest.main([__file__])
