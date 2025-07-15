"""
Unit tests for the cache warmup service.

Tests cover memory-aware cache warmup strategies, execution, and coordination
with the memory management system.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..utils.memory_utils import MemoryPressureLevel, SystemMemoryPressure
from .cache_warmup_service import (
    AggressiveWarmupStrategy,
    BalancedWarmupStrategy,
    CacheWarmupService,
    ConservativeWarmupStrategy,
    WarmupItem,
    WarmupPhase,
    WarmupPlan,
    WarmupProgress,
    WarmupStrategy,
    execute_memory_aware_warmup,
    get_cache_warmup_service,
)


class TestWarmupItem:
    """Test WarmupItem functionality."""

    def test_warmup_item_creation(self):
        """Test creating a warmup item."""
        item = WarmupItem(
            cache_name="test_cache", item_key="test_key", item_data="test_data", priority=7, estimated_size_mb=1.5, usage_frequency=0.8
        )

        assert item.cache_name == "test_cache"
        assert item.item_key == "test_key"
        assert item.item_data == "test_data"
        assert item.priority == 7
        assert item.estimated_size_mb == 1.5
        assert item.usage_frequency == 0.8
        assert item.warmup_score > 0

    def test_warmup_score_calculation(self):
        """Test warmup score calculation."""
        # High priority, high frequency, low cost
        item1 = WarmupItem(
            cache_name="test", item_key="key1", item_data="data", priority=10, estimated_size_mb=1.0, usage_frequency=0.9, warmup_cost=0.1
        )

        # Low priority, low frequency, high cost
        item2 = WarmupItem(
            cache_name="test", item_key="key2", item_data="data", priority=2, estimated_size_mb=1.0, usage_frequency=0.1, warmup_cost=2.0
        )

        assert item1.warmup_score > item2.warmup_score


class TestWarmupStrategies:
    """Test warmup strategy implementations."""

    @pytest.fixture
    def mock_cache_services(self):
        """Mock cache services for testing."""
        cache_service = MagicMock()
        cache_service.get_warmup_candidates = AsyncMock(
            return_value=[
                WarmupItem("test_cache", "key1", "data1", priority=9, estimated_size_mb=2.0, usage_frequency=0.8),
                WarmupItem("test_cache", "key2", "data2", priority=6, estimated_size_mb=1.5, usage_frequency=0.5),
                WarmupItem("test_cache", "key3", "data3", priority=3, estimated_size_mb=0.5, usage_frequency=0.2),
            ]
        )
        return {"test_cache": cache_service}

    @pytest.fixture
    def mock_system_pressure(self):
        """Mock system memory pressure."""
        return SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=500.0,
            available_mb=1000.0,
            total_mb=2000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=5.0,
            recommendation="Normal usage",
        )

    @pytest.mark.asyncio
    async def test_aggressive_strategy(self, mock_cache_services, mock_system_pressure):
        """Test aggressive warmup strategy."""
        strategy = AggressiveWarmupStrategy()

        plan = await strategy.create_warmup_plan(mock_cache_services, 1000.0, mock_system_pressure, {})

        assert plan.strategy == WarmupStrategy.AGGRESSIVE
        assert plan.total_items == 3
        assert plan.memory_budget == 800.0  # 80% of 1000MB
        assert len(plan.phases) == 3

        # Check phase distribution
        critical_items = plan.get_phase_items(WarmupPhase.CRITICAL_PRELOAD)
        assert len(critical_items) == 1  # Only priority 9 item
        assert critical_items[0].priority >= 8

    @pytest.mark.asyncio
    async def test_balanced_strategy(self, mock_cache_services, mock_system_pressure):
        """Test balanced warmup strategy."""
        strategy = BalancedWarmupStrategy()

        plan = await strategy.create_warmup_plan(mock_cache_services, 1000.0, mock_system_pressure, {})

        assert plan.strategy == WarmupStrategy.BALANCED
        assert plan.memory_budget == 600.0  # 60% of 1000MB
        assert plan.total_items >= 2  # Should include high and medium priority items

    @pytest.mark.asyncio
    async def test_conservative_strategy(self, mock_cache_services, mock_system_pressure):
        """Test conservative warmup strategy."""
        strategy = ConservativeWarmupStrategy()

        plan = await strategy.create_warmup_plan(mock_cache_services, 1000.0, mock_system_pressure, {})

        assert plan.strategy == WarmupStrategy.CONSERVATIVE
        assert plan.memory_budget == 200.0  # 20% of 1000MB
        assert plan.total_items <= 1  # Should include only critical items

    @pytest.mark.asyncio
    async def test_strategy_memory_pressure_adaptation(self, mock_cache_services):
        """Test strategy adaptation to memory pressure."""
        strategy = BalancedWarmupStrategy()

        # Test with high memory pressure
        high_pressure = SystemMemoryPressure(
            level=MemoryPressureLevel.HIGH,
            current_usage_mb=800.0,
            available_mb=200.0,
            total_mb=1000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=10.0,
            recommendation="High pressure",
        )

        plan = await strategy.create_warmup_plan(mock_cache_services, 200.0, high_pressure, {})

        assert plan.memory_budget == 60.0  # 30% of 200MB due to high pressure

    @pytest.mark.asyncio
    async def test_should_continue_warmup(self):
        """Test warmup continuation logic."""
        strategy = BalancedWarmupStrategy()

        progress = WarmupProgress(total_items=100, completed_items=50)

        # Should continue with low pressure
        low_pressure = SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=500.0,
            available_mb=1000.0,
            total_mb=2000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=5.0,
            recommendation="Normal",
        )

        assert await strategy.should_continue_warmup(progress, low_pressure)

        # Should stop with critical pressure
        critical_pressure = SystemMemoryPressure(
            level=MemoryPressureLevel.CRITICAL,
            current_usage_mb=1800.0,
            available_mb=200.0,
            total_mb=2000.0,
            cache_usage_mb=300.0,
            cache_usage_percent=15.0,
            recommendation="Critical",
        )

        assert not await strategy.should_continue_warmup(progress, critical_pressure)


class TestCacheWarmupService:
    """Test cache warmup service."""

    @pytest.fixture
    def warmup_service(self):
        """Create warmup service for testing."""
        return CacheWarmupService()

    @pytest.fixture
    def mock_cache_service(self):
        """Mock cache service."""
        service = MagicMock()
        service.get_warmup_candidates = AsyncMock(
            return_value=[
                WarmupItem("test", "key1", "data1", priority=8, estimated_size_mb=1.0),
                WarmupItem("test", "key2", "data2", priority=5, estimated_size_mb=0.5),
            ]
        )
        service.warmup_item = AsyncMock(return_value=1.0)  # Returns memory usage
        return service

    @pytest.mark.asyncio
    async def test_register_cache_service(self, warmup_service, mock_cache_service):
        """Test registering cache service."""
        await warmup_service.register_cache_service("test_cache", mock_cache_service)

        assert "test_cache" in warmup_service.cache_services
        assert warmup_service.cache_services["test_cache"] == mock_cache_service

    @pytest.mark.asyncio
    async def test_unregister_cache_service(self, warmup_service, mock_cache_service):
        """Test unregistering cache service."""
        await warmup_service.register_cache_service("test_cache", mock_cache_service)
        await warmup_service.unregister_cache_service("test_cache")

        assert "test_cache" not in warmup_service.cache_services

    @pytest.mark.asyncio
    async def test_update_historical_data(self, warmup_service):
        """Test updating historical data."""
        historical_data = {"frequent_queries": ["query1", "query2"]}
        await warmup_service.update_historical_data("test_cache", historical_data)

        assert warmup_service.historical_data["test_cache"] == historical_data

    @pytest.mark.asyncio
    @patch("src.services.cache_warmup_service.get_system_memory_pressure")
    async def test_get_recommended_strategy(self, mock_pressure, warmup_service):
        """Test getting recommended strategy."""
        # Test with low pressure
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=500.0,
            available_mb=1000.0,
            total_mb=2000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=5.0,
            recommendation="Normal",
        )

        strategy = await warmup_service.get_recommended_strategy()
        assert strategy == WarmupStrategy.BALANCED

        # Test with critical pressure
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.CRITICAL,
            current_usage_mb=1800.0,
            available_mb=200.0,
            total_mb=2000.0,
            cache_usage_mb=300.0,
            cache_usage_percent=15.0,
            recommendation="Critical",
        )

        strategy = await warmup_service.get_recommended_strategy()
        assert strategy == WarmupStrategy.CONSERVATIVE

    @pytest.mark.asyncio
    @patch("src.services.cache_warmup_service.get_system_memory_pressure")
    async def test_create_warmup_plan(self, mock_pressure, warmup_service, mock_cache_service):
        """Test creating warmup plan."""
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=500.0,
            available_mb=1000.0,
            total_mb=2000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=5.0,
            recommendation="Normal",
        )

        await warmup_service.register_cache_service("test_cache", mock_cache_service)

        plan = await warmup_service.create_warmup_plan(WarmupStrategy.BALANCED)

        assert plan.strategy == WarmupStrategy.BALANCED
        assert plan.total_items >= 0
        assert plan.memory_budget > 0
        assert len(plan.phases) > 0

    @pytest.mark.asyncio
    async def test_execute_warmup_plan(self, warmup_service, mock_cache_service):
        """Test executing warmup plan."""
        await warmup_service.register_cache_service("test_cache", mock_cache_service)

        # Create a simple plan
        plan = WarmupPlan(
            strategy=WarmupStrategy.BALANCED, total_items=2, estimated_memory_mb=1.5, available_memory_mb=1000.0, memory_budget=100.0
        )

        # Add items to plan
        items = [
            WarmupItem("test_cache", "key1", "data1", priority=8, estimated_size_mb=1.0),
            WarmupItem("test_cache", "key2", "data2", priority=5, estimated_size_mb=0.5),
        ]
        plan.phases = [(WarmupPhase.CRITICAL_PRELOAD, items)]

        results = await warmup_service.execute_warmup_plan(plan)

        assert results["success"]
        assert results["strategy"] == WarmupStrategy.BALANCED.value
        assert results["total_items"] == 2
        assert results["phases_completed"] >= 1
        assert results["execution_time"] > 0

    @pytest.mark.asyncio
    async def test_warmup_status_inactive(self, warmup_service):
        """Test warmup status when inactive."""
        status = await warmup_service.get_warmup_status()

        assert not status["active"]
        assert status["last_warmup"] is None

    @pytest.mark.asyncio
    async def test_cancel_warmup_not_active(self, warmup_service):
        """Test cancelling warmup when not active."""
        result = await warmup_service.cancel_warmup()

        assert "error" in result
        assert "No warmup in progress" in result["error"]

    @pytest.mark.asyncio
    @patch("src.services.cache_warmup_service.get_system_memory_pressure")
    @patch("src.services.cache_warmup_service.get_total_cache_memory_usage")
    async def test_get_warmup_recommendations(self, mock_cache_usage, mock_pressure, warmup_service):
        """Test getting warmup recommendations."""
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.MODERATE,
            current_usage_mb=700.0,
            available_mb=500.0,
            total_mb=1000.0,
            cache_usage_mb=200.0,
            cache_usage_percent=20.0,
            recommendation="Moderate pressure",
        )
        mock_cache_usage.return_value = 200.0

        recommendations = await warmup_service.get_warmup_recommendations()

        assert recommendations["recommended_strategy"] == WarmupStrategy.BALANCED.value
        assert recommendations["system_pressure"] == MemoryPressureLevel.MODERATE.value
        assert recommendations["available_memory_mb"] == 500.0
        assert recommendations["cache_memory_usage_mb"] == 200.0
        assert len(recommendations["recommendations"]) > 0


class TestIntegrationFunctions:
    """Test integration functions."""

    @pytest.mark.asyncio
    async def test_get_cache_warmup_service_singleton(self):
        """Test that get_cache_warmup_service returns singleton."""
        service1 = await get_cache_warmup_service()
        service2 = await get_cache_warmup_service()

        assert service1 is service2

    @pytest.mark.asyncio
    @patch("src.services.cache_warmup_service.get_cache_warmup_service")
    async def test_execute_memory_aware_warmup(self, mock_get_service):
        """Test memory-aware warmup execution."""
        mock_service = MagicMock()
        mock_service.create_warmup_plan = AsyncMock(return_value=MagicMock(strategy=WarmupStrategy.BALANCED))
        mock_service.execute_warmup_plan = AsyncMock(return_value={"success": True})
        mock_get_service.return_value = mock_service

        result = await execute_memory_aware_warmup(WarmupStrategy.BALANCED, 500.0)

        assert result["success"]
        mock_service.create_warmup_plan.assert_called_once_with(WarmupStrategy.BALANCED, 500.0)
        mock_service.execute_warmup_plan.assert_called_once()


class TestWarmupProgress:
    """Test warmup progress tracking."""

    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = WarmupProgress(total_items=100, completed_items=25)

        assert progress.progress_percent == 25.0

        # Test with zero total items
        progress_zero = WarmupProgress(total_items=0, completed_items=0)
        assert progress_zero.progress_percent == 0.0

    def test_memory_usage_calculation(self):
        """Test memory usage percentage calculation."""
        progress = WarmupProgress(total_items=100, memory_used_mb=50.0, memory_budget_mb=200.0)

        assert progress.memory_usage_percent == 25.0

        # Test with zero budget
        progress_zero = WarmupProgress(total_items=100, memory_used_mb=50.0, memory_budget_mb=0.0)
        assert progress_zero.memory_usage_percent == 0.0


class TestWarmupPlan:
    """Test warmup plan functionality."""

    def test_get_phase_items(self):
        """Test getting items for specific phase."""
        items1 = [WarmupItem("cache1", "key1", "data1")]
        items2 = [WarmupItem("cache2", "key2", "data2")]

        plan = WarmupPlan(
            strategy=WarmupStrategy.BALANCED,
            total_items=2,
            estimated_memory_mb=2.0,
            available_memory_mb=1000.0,
            phases=[(WarmupPhase.CRITICAL_PRELOAD, items1), (WarmupPhase.STANDARD_PRELOAD, items2)],
        )

        critical_items = plan.get_phase_items(WarmupPhase.CRITICAL_PRELOAD)
        assert critical_items == items1

        standard_items = plan.get_phase_items(WarmupPhase.STANDARD_PRELOAD)
        assert standard_items == items2

        # Test non-existent phase
        background_items = plan.get_phase_items(WarmupPhase.BACKGROUND_PRELOAD)
        assert background_items == []


if __name__ == "__main__":
    pytest.main([__file__])
