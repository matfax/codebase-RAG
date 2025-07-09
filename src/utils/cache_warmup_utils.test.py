"""
Unit tests for cache warmup utilities.

Tests cover memory budget calculations, item prioritization, and
cache-specific warmup candidate generation.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..services.cache_warmup_service import WarmupItem, WarmupStrategy
from ..utils.memory_utils import MemoryPressureLevel, SystemMemoryPressure
from .cache_warmup_utils import (
    calculate_memory_budget,
    calculate_warmup_time_estimate,
    create_warmup_item,
    estimate_item_memory_usage,
    get_embedding_cache_warmup_candidates,
    get_file_cache_warmup_candidates,
    get_project_cache_warmup_candidates,
    get_search_cache_warmup_candidates,
    get_warmup_memory_recommendations,
    group_items_by_dependencies,
    prioritize_warmup_items,
    validate_warmup_environment,
)


class TestMemoryBudgetCalculation:
    """Test memory budget calculation functions."""

    def test_calculate_memory_budget_strategies(self):
        """Test memory budget calculation for different strategies."""
        available_memory = 1000.0
        pressure_level = MemoryPressureLevel.LOW

        # Test different strategies
        aggressive_budget = calculate_memory_budget(available_memory, WarmupStrategy.AGGRESSIVE, pressure_level)
        balanced_budget = calculate_memory_budget(available_memory, WarmupStrategy.BALANCED, pressure_level)
        conservative_budget = calculate_memory_budget(available_memory, WarmupStrategy.CONSERVATIVE, pressure_level)

        assert aggressive_budget > balanced_budget > conservative_budget
        assert aggressive_budget == 800.0  # 80% of 1000MB
        assert balanced_budget == 600.0  # 60% of 1000MB
        assert conservative_budget == 200.0  # 20% of 1000MB

    def test_calculate_memory_budget_pressure_adjustment(self):
        """Test memory budget adjustment based on pressure."""
        available_memory = 1000.0
        strategy = WarmupStrategy.BALANCED

        # Test different pressure levels
        low_budget = calculate_memory_budget(available_memory, strategy, MemoryPressureLevel.LOW)
        moderate_budget = calculate_memory_budget(available_memory, strategy, MemoryPressureLevel.MODERATE)
        high_budget = calculate_memory_budget(available_memory, strategy, MemoryPressureLevel.HIGH)
        critical_budget = calculate_memory_budget(available_memory, strategy, MemoryPressureLevel.CRITICAL)

        assert low_budget > moderate_budget > high_budget > critical_budget
        assert critical_budget == 0.0  # No warmup under critical pressure
        assert low_budget == 600.0  # 100% of base (60% of 1000MB)
        assert moderate_budget == 360.0  # 60% of base
        assert high_budget == 120.0  # 20% of base

    def test_calculate_memory_budget_minimum_and_maximum(self):
        """Test memory budget minimum and maximum limits."""
        # Test minimum budget
        small_memory = 50.0
        budget = calculate_memory_budget(small_memory, WarmupStrategy.BALANCED, MemoryPressureLevel.LOW)
        assert budget >= 10.0  # Minimum 10MB

        # Test maximum budget cap
        large_memory = 10000.0
        budget = calculate_memory_budget(large_memory, WarmupStrategy.AGGRESSIVE, MemoryPressureLevel.LOW)
        assert budget <= large_memory * 0.9  # Max 90% of available


class TestItemMemoryEstimation:
    """Test memory usage estimation for cache items."""

    def test_estimate_string_memory_usage(self):
        """Test memory estimation for string data."""
        cache_service = MagicMock()

        # Test small string
        small_string = "hello world"
        size = estimate_item_memory_usage("test_cache", "key1", small_string, cache_service)
        assert size > 0
        assert size < 0.001  # Less than 1KB

        # Test large string
        large_string = "x" * 1000000  # 1MB string
        size = estimate_item_memory_usage("test_cache", "key2", large_string, cache_service)
        assert size > 0.9  # Approximately 1MB

    def test_estimate_list_memory_usage(self):
        """Test memory estimation for list data."""
        cache_service = MagicMock()

        # Test small list
        small_list = [1, 2, 3, 4, 5]
        size = estimate_item_memory_usage("test_cache", "key1", small_list, cache_service)
        assert size > 0
        assert size < 0.1  # Small list

        # Test large list
        large_list = list(range(10000))
        size = estimate_item_memory_usage("test_cache", "key2", large_list, cache_service)
        assert size > 5.0  # Larger list

    def test_estimate_dict_memory_usage(self):
        """Test memory estimation for dictionary data."""
        cache_service = MagicMock()

        # Test small dict
        small_dict = {"key1": "value1", "key2": "value2"}
        size = estimate_item_memory_usage("test_cache", "key1", small_dict, cache_service)
        assert size > 0

        # Test large dict
        large_dict = {f"key{i}": f"value{i}" for i in range(1000)}
        size = estimate_item_memory_usage("test_cache", "key2", large_dict, cache_service)
        assert size > 0.01  # Larger dict

    def test_estimate_with_cache_service_method(self):
        """Test estimation when cache service provides estimate method."""
        cache_service = MagicMock()
        cache_service.estimate_item_size.return_value = 5.0

        size = estimate_item_memory_usage("test_cache", "key1", "data", cache_service)

        assert size == 5.0
        cache_service.estimate_item_size.assert_called_once_with("key1", "data")


class TestWarmupItemCreation:
    """Test warmup item creation."""

    def test_create_warmup_item_basic(self):
        """Test basic warmup item creation."""
        cache_service = MagicMock()

        item = create_warmup_item(
            cache_name="test_cache",
            item_key="test_key",
            item_data="test_data",
            cache_service=cache_service,
            priority=7,
            usage_frequency=0.5,
        )

        assert item.cache_name == "test_cache"
        assert item.item_key == "test_key"
        assert item.item_data == "test_data"
        assert item.priority == 7
        assert item.usage_frequency == 0.5
        assert item.estimated_size_mb > 0
        assert item.warmup_cost >= 0.1  # Base cost

    def test_create_warmup_item_with_dependencies(self):
        """Test warmup item creation with dependencies."""
        cache_service = MagicMock()
        dependencies = {"dep1", "dep2"}

        item = create_warmup_item(
            cache_name="test_cache", item_key="test_key", item_data="test_data", cache_service=cache_service, dependencies=dependencies
        )

        assert item.dependencies == dependencies

    def test_create_warmup_item_cost_calculation(self):
        """Test warmup cost calculation for different data types."""
        cache_service = MagicMock()

        # Test large string
        large_string = "x" * 20000
        item_string = create_warmup_item("test", "key1", large_string, cache_service)

        # Test large list
        large_list = list(range(2000))
        item_list = create_warmup_item("test", "key2", large_list, cache_service)

        # Test large dict
        large_dict = {f"key{i}": f"value{i}" for i in range(200)}
        item_dict = create_warmup_item("test", "key3", large_dict, cache_service)

        # Large items should have higher warmup cost
        assert item_string.warmup_cost > 0.5
        assert item_list.warmup_cost > 0.25
        assert item_dict.warmup_cost > 0.15


class TestCacheWarmupCandidates:
    """Test cache-specific warmup candidate generation."""

    @pytest.mark.asyncio
    async def test_get_embedding_cache_warmup_candidates(self):
        """Test getting embedding cache warmup candidates."""
        cache_service = MagicMock()

        historical_data = {
            "frequent_queries": [
                {"query": "test query 1", "frequency": 0.8, "last_used": time.time()},
                {"query": "test query 2", "frequency": 0.3, "last_used": time.time()},
                {"query": "test query 3", "frequency": 0.05, "last_used": time.time()},  # Below threshold
            ],
            "cached_models": [
                {"model": "model1", "usage_count": 50},
                {"model": "model2", "usage_count": 20},
                {"model": "model3", "usage_count": 3},  # Below threshold
            ],
        }

        candidates = await get_embedding_cache_warmup_candidates(cache_service, historical_data)

        # Should include frequent queries (>0.1 frequency) and models (>5 usage)
        assert len(candidates) >= 4

        # Check that low-frequency items are excluded
        candidate_keys = [item.item_key for item in candidates]
        assert "test query 3" not in candidate_keys
        assert "model:model3" not in candidate_keys

    @pytest.mark.asyncio
    async def test_get_search_cache_warmup_candidates(self):
        """Test getting search cache warmup candidates."""
        cache_service = MagicMock()

        historical_data = {
            "frequent_searches": [
                {
                    "query": "search query 1",
                    "parameters": {"n_results": 5, "search_mode": "hybrid"},
                    "frequency": 0.6,
                    "cached_results": ["result1", "result2"],
                },
                {"query": "search query 2", "parameters": {"n_results": 10}, "frequency": 0.1, "cached_results": ["result3"]},
                {"query": "search query 3", "parameters": {}, "frequency": 0.02, "cached_results": []},  # Below threshold
            ],
            "project_searches": [
                {"project": "project1", "search_count": 50},
                {"project": "project2", "search_count": 25},
                {"project": "project3", "search_count": 5},  # Below threshold
            ],
        }

        candidates = await get_search_cache_warmup_candidates(cache_service, historical_data)

        # Should include frequent searches (>0.05 frequency) and projects (>10 searches)
        assert len(candidates) >= 4

        # Check composite key generation
        candidate_keys = [item.item_key for item in candidates]
        assert any("search query 1" in key for key in candidate_keys)
        assert any("project:project1" in key for key in candidate_keys)

    @pytest.mark.asyncio
    async def test_get_file_cache_warmup_candidates(self):
        """Test getting file cache warmup candidates."""
        cache_service = MagicMock()

        historical_data = {
            "frequent_files": [
                {"path": "/path/to/file1.py", "parse_count": 20, "size": 50000, "language": "python"},
                {"path": "/path/to/file2.js", "parse_count": 8, "size": 200000, "language": "javascript"},
                {"path": "/path/to/file3.txt", "parse_count": 2, "size": 1000, "language": "text"},  # Below threshold
            ],
            "language_usage": {"python": {"count": 100}, "javascript": {"count": 50}, "go": {"count": 10}},  # Below threshold
        }

        candidates = await get_file_cache_warmup_candidates(cache_service, historical_data)

        # Should include frequent files (>3 parses) and languages (>20 usage)
        assert len(candidates) >= 4

        # Check priority adjustment for common languages and large files
        python_items = [item for item in candidates if "/path/to/file1.py" in item.item_key]
        assert len(python_items) > 0
        assert python_items[0].priority >= 6  # Should get priority boost

    @pytest.mark.asyncio
    async def test_get_project_cache_warmup_candidates(self):
        """Test getting project cache warmup candidates."""
        cache_service = MagicMock()

        historical_data = {
            "frequent_projects": [
                {"name": "project1", "access_count": 30, "info": {"type": "python", "files": 100}},
                {"name": "project2", "access_count": 12, "info": {"type": "javascript", "files": 50}},
                {"name": "project3", "access_count": 3, "info": {"type": "go", "files": 20}},  # Below threshold
            ],
            "project_stats": [
                {"project": "project1", "requests": 50},
                {"project": "project2", "requests": 25},
                {"project": "project4", "requests": 5},  # Below threshold
            ],
        }

        candidates = await get_project_cache_warmup_candidates(cache_service, historical_data)

        # Should include frequent projects (>5 access) and stats (>10 requests)
        assert len(candidates) >= 4

        # Check that low-access items are excluded
        candidate_keys = [item.item_key for item in candidates]
        assert "project3" not in candidate_keys
        assert "stats:project4" not in candidate_keys


class TestItemPrioritization:
    """Test warmup item prioritization."""

    def test_prioritize_warmup_items_strategy_filtering(self):
        """Test item filtering based on strategy."""
        items = [
            WarmupItem("cache1", "key1", "data1", priority=9, estimated_size_mb=1.0, usage_frequency=0.8),
            WarmupItem("cache1", "key2", "data2", priority=6, estimated_size_mb=1.0, usage_frequency=0.5),
            WarmupItem("cache1", "key3", "data3", priority=3, estimated_size_mb=1.0, usage_frequency=0.4),
            WarmupItem("cache1", "key4", "data4", priority=8, estimated_size_mb=1.0, usage_frequency=0.2),
        ]

        # Test priority-based strategy
        priority_items = prioritize_warmup_items(items, 10.0, WarmupStrategy.PRIORITY_BASED)
        assert len(priority_items) == 2  # Only priority >= 7
        assert all(item.priority >= 7 for item in priority_items)

        # Test conservative strategy
        conservative_items = prioritize_warmup_items(items, 10.0, WarmupStrategy.CONSERVATIVE)
        assert len(conservative_items) <= 2  # High priority and frequency
        assert all(item.priority >= 6 and item.usage_frequency >= 0.3 for item in conservative_items)

        # Test balanced strategy
        balanced_items = prioritize_warmup_items(items, 10.0, WarmupStrategy.BALANCED)
        assert len(balanced_items) >= 2  # Include medium priority items
        assert all(item.priority >= 4 and item.usage_frequency >= 0.1 for item in balanced_items)

    def test_prioritize_warmup_items_memory_budget(self):
        """Test item filtering based on memory budget."""
        items = [
            WarmupItem("cache1", "key1", "data1", priority=9, estimated_size_mb=5.0, usage_frequency=0.8),
            WarmupItem("cache1", "key2", "data2", priority=8, estimated_size_mb=3.0, usage_frequency=0.7),
            WarmupItem("cache1", "key3", "data3", priority=7, estimated_size_mb=4.0, usage_frequency=0.6),
        ]

        # Test with limited memory budget
        limited_items = prioritize_warmup_items(items, 8.0, WarmupStrategy.AGGRESSIVE)

        # Should select highest priority items that fit in budget
        assert len(limited_items) == 2  # First two items (5MB + 3MB = 8MB)
        assert sum(item.estimated_size_mb for item in limited_items) <= 8.0

    def test_prioritize_warmup_items_sorting(self):
        """Test that items are sorted by warmup score."""
        items = [
            WarmupItem("cache1", "key1", "data1", priority=5, estimated_size_mb=1.0, usage_frequency=0.3),
            WarmupItem("cache1", "key2", "data2", priority=9, estimated_size_mb=1.0, usage_frequency=0.9),
            WarmupItem("cache1", "key3", "data3", priority=7, estimated_size_mb=1.0, usage_frequency=0.6),
        ]

        prioritized = prioritize_warmup_items(items, 10.0, WarmupStrategy.AGGRESSIVE)

        # Should be sorted by warmup score (descending)
        scores = [item.warmup_score for item in prioritized]
        assert scores == sorted(scores, reverse=True)


class TestDependencyGrouping:
    """Test dependency grouping for warmup items."""

    def test_group_items_by_dependencies_simple(self):
        """Test simple dependency grouping."""
        items = [
            WarmupItem("cache1", "key1", "data1", dependencies=set()),
            WarmupItem("cache1", "key2", "data2", dependencies={"cache1:key1"}),
            WarmupItem("cache1", "key3", "data3", dependencies={"cache1:key2"}),
        ]

        phases = group_items_by_dependencies(items)

        # Should create phases in dependency order
        assert len(phases) >= 3

        # First phase should contain items with no dependencies
        first_phase = phases[0]
        assert len(first_phase) == 1
        assert first_phase[0].item_key == "key1"

    def test_group_items_by_dependencies_no_dependencies(self):
        """Test grouping when no dependencies exist."""
        items = [
            WarmupItem("cache1", "key1", "data1", dependencies=set()),
            WarmupItem("cache1", "key2", "data2", dependencies=set()),
            WarmupItem("cache1", "key3", "data3", dependencies=set()),
        ]

        phases = group_items_by_dependencies(items)

        # Should create single phase with all items
        assert len(phases) == 1
        assert len(phases[0]) == 3

    def test_group_items_by_dependencies_empty(self):
        """Test grouping with empty item list."""
        phases = group_items_by_dependencies([])
        assert phases == []


class TestWarmupValidation:
    """Test warmup environment validation."""

    @pytest.mark.asyncio
    @patch("src.utils.cache_warmup_utils.get_system_memory_pressure")
    @patch("src.utils.cache_warmup_utils.get_total_cache_memory_usage")
    @patch("src.utils.cache_warmup_utils.get_cache_memory_stats")
    async def test_validate_warmup_environment_suitable(self, mock_stats, mock_usage, mock_pressure):
        """Test validation when environment is suitable for warmup."""
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=500.0,
            available_mb=1000.0,
            total_mb=2000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=5.0,
            recommendation="Normal",
        )
        mock_usage.return_value = 100.0
        mock_stats.return_value = {}

        result = await validate_warmup_environment()

        assert result["suitable_for_warmup"]
        assert len(result["issues"]) == 0
        assert result["system_info"]["memory_pressure"] == "low"

    @pytest.mark.asyncio
    @patch("src.utils.cache_warmup_utils.get_system_memory_pressure")
    @patch("src.utils.cache_warmup_utils.get_total_cache_memory_usage")
    async def test_validate_warmup_environment_critical_pressure(self, mock_usage, mock_pressure):
        """Test validation with critical memory pressure."""
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.CRITICAL,
            current_usage_mb=1800.0,
            available_mb=200.0,
            total_mb=2000.0,
            cache_usage_mb=300.0,
            cache_usage_percent=15.0,
            recommendation="Critical",
        )
        mock_usage.return_value = 300.0

        result = await validate_warmup_environment()

        assert not result["suitable_for_warmup"]
        assert "Critical memory pressure detected" in result["issues"]
        assert "Wait for memory pressure to decrease" in result["recommendations"][0]

    @pytest.mark.asyncio
    @patch("src.utils.cache_warmup_utils.get_system_memory_pressure")
    @patch("src.utils.cache_warmup_utils.get_total_cache_memory_usage")
    async def test_validate_warmup_environment_insufficient_memory(self, mock_usage, mock_pressure):
        """Test validation with insufficient available memory."""
        mock_pressure.return_value = SystemMemoryPressure(
            level=MemoryPressureLevel.MODERATE,
            current_usage_mb=950.0,
            available_mb=50.0,  # Less than 100MB
            total_mb=1000.0,
            cache_usage_mb=100.0,
            cache_usage_percent=10.0,
            recommendation="Moderate",
        )
        mock_usage.return_value = 100.0

        result = await validate_warmup_environment()

        assert not result["suitable_for_warmup"]
        assert "Insufficient available memory" in result["issues"]


class TestTimeAndMemoryRecommendations:
    """Test time estimation and memory recommendations."""

    def test_calculate_warmup_time_estimate(self):
        """Test warmup time estimation."""
        items = [
            WarmupItem("cache1", "key1", "data1", estimated_size_mb=1.0, warmup_cost=0.5),
            WarmupItem("cache1", "key2", "data2", estimated_size_mb=2.0, warmup_cost=0.3),
            WarmupItem("cache1", "key3", "data3", estimated_size_mb=0.5, warmup_cost=0.1),
        ]

        time_estimate = calculate_warmup_time_estimate(items)

        assert time_estimate > 0
        assert time_estimate > 0.3  # Should be more than base time

        # Test with empty list
        empty_estimate = calculate_warmup_time_estimate([])
        assert empty_estimate == 0.0

    def test_get_warmup_memory_recommendations(self):
        """Test warmup memory recommendations."""
        recommendations = get_warmup_memory_recommendations(1000.0, 200.0)

        assert "memory_budget_options" in recommendations
        assert "recommended_strategy" in recommendations
        assert "warnings" in recommendations
        assert "optimizations" in recommendations

        # Check budget options
        budget_options = recommendations["memory_budget_options"]
        assert WarmupStrategy.CONSERVATIVE.value in budget_options
        assert WarmupStrategy.BALANCED.value in budget_options
        assert WarmupStrategy.AGGRESSIVE.value in budget_options

        # Conservative should have lowest budget
        conservative_budget = budget_options[WarmupStrategy.CONSERVATIVE.value]["budget_mb"]
        balanced_budget = budget_options[WarmupStrategy.BALANCED.value]["budget_mb"]
        aggressive_budget = budget_options[WarmupStrategy.AGGRESSIVE.value]["budget_mb"]

        assert conservative_budget < balanced_budget < aggressive_budget

    def test_get_warmup_memory_recommendations_high_usage(self):
        """Test recommendations with high current cache usage."""
        recommendations = get_warmup_memory_recommendations(1000.0, 800.0)  # 80% usage

        assert recommendations["recommended_strategy"] == WarmupStrategy.CONSERVATIVE.value
        assert "High current cache usage" in recommendations["warnings"][0]

    def test_get_warmup_memory_recommendations_optimizations(self):
        """Test optimization suggestions."""
        # Test with high cache usage
        recommendations = get_warmup_memory_recommendations(2000.0, 600.0)

        assert any("clearing unused caches" in opt for opt in recommendations["optimizations"])

        # Test with low available memory
        recommendations = get_warmup_memory_recommendations(800.0, 100.0)

        assert any("increasing system memory" in opt for opt in recommendations["optimizations"])


if __name__ == "__main__":
    pytest.main([__file__])
