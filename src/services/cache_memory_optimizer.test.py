"""
Comprehensive test suite for the cache memory optimizer.

Tests cover:
- Memory metrics analysis and recommendations
- Configuration optimization recommendations
- Performance pattern analysis
- Implementation planning and risk assessment
- Integration and end-to-end scenarios
"""

import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cache_memory_optimizer import (
    CacheConfiguration,
    CacheMemoryOptimizer,
    ImpactLevel,
    MemoryMetrics,
    OptimizationPlan,
    OptimizationRecommendation,
    OptimizationType,
    RecommendationPriority,
    generate_cache_optimization_plan,
    get_cache_optimization_summary,
    get_memory_optimizer,
    register_cache_configuration,
    register_cache_metrics,
)


class TestMemoryMetrics:
    """Test cases for MemoryMetrics data model."""

    def test_memory_metrics_creation(self):
        """Test creating memory metrics with all fields."""
        timestamp = datetime.now()
        metrics = MemoryMetrics(
            cache_name="test_cache",
            timestamp=timestamp,
            total_memory_mb=512.0,
            cache_memory_mb=256.0,
            entry_count=1000,
            hit_ratio=0.85,
            miss_ratio=0.15,
            eviction_rate=0.05,
            allocation_rate_mb_per_min=10.0,
            fragmentation_ratio=0.25,
            gc_frequency=5.0,
        )

        assert metrics.cache_name == "test_cache"
        assert metrics.timestamp == timestamp
        assert metrics.total_memory_mb == 512.0
        assert metrics.cache_memory_mb == 256.0
        assert metrics.entry_count == 1000
        assert metrics.hit_ratio == 0.85
        assert metrics.miss_ratio == 0.15
        assert metrics.eviction_rate == 0.05
        assert metrics.allocation_rate_mb_per_min == 10.0
        assert metrics.fragmentation_ratio == 0.25
        assert metrics.gc_frequency == 5.0


class TestCacheConfiguration:
    """Test cases for CacheConfiguration data model."""

    def test_cache_configuration_creation(self):
        """Test creating cache configuration with all fields."""
        config = CacheConfiguration(
            cache_name="test_cache",
            max_size_mb=1024.0,
            max_entries=10000,
            ttl_seconds=3600,
            eviction_policy="LRU",
            prefetch_enabled=True,
            compression_enabled=False,
            serialization_format="json",
            concurrency_level=4,
        )

        assert config.cache_name == "test_cache"
        assert config.max_size_mb == 1024.0
        assert config.max_entries == 10000
        assert config.ttl_seconds == 3600
        assert config.eviction_policy == "LRU"
        assert config.prefetch_enabled is True
        assert config.compression_enabled is False
        assert config.serialization_format == "json"
        assert config.concurrency_level == 4

    def test_cache_configuration_defaults(self):
        """Test cache configuration with default values."""
        config = CacheConfiguration(cache_name="test_cache")

        assert config.cache_name == "test_cache"
        assert config.max_size_mb is None
        assert config.max_entries is None
        assert config.ttl_seconds is None
        assert config.eviction_policy is None
        assert config.prefetch_enabled is False
        assert config.compression_enabled is False
        assert config.serialization_format == "pickle"
        assert config.concurrency_level == 1


class TestOptimizationRecommendation:
    """Test cases for OptimizationRecommendation data model."""

    def test_optimization_recommendation_creation(self):
        """Test creating optimization recommendation with all fields."""
        recommendation = OptimizationRecommendation(
            recommendation_id="test_rec_001",
            cache_name="test_cache",
            optimization_type=OptimizationType.MEMORY_REDUCTION,
            priority=RecommendationPriority.HIGH,
            impact_level=ImpactLevel.MAJOR,
            title="Test Optimization",
            description="Test optimization description",
            current_state="Current state description",
            recommended_change="Recommended change description",
            expected_benefits=["Benefit 1", "Benefit 2"],
            implementation_steps=["Step 1", "Step 2"],
            estimated_effort_hours=8.0,
            risk_assessment="Medium risk",
            success_metrics=["Metric 1", "Metric 2"],
            dependencies=["Dependency 1"],
            cost_benefit_ratio=3.5,
            metadata={"test": "value"},
        )

        assert recommendation.recommendation_id == "test_rec_001"
        assert recommendation.cache_name == "test_cache"
        assert recommendation.optimization_type == OptimizationType.MEMORY_REDUCTION
        assert recommendation.priority == RecommendationPriority.HIGH
        assert recommendation.impact_level == ImpactLevel.MAJOR
        assert recommendation.title == "Test Optimization"
        assert recommendation.description == "Test optimization description"
        assert recommendation.current_state == "Current state description"
        assert recommendation.recommended_change == "Recommended change description"
        assert recommendation.expected_benefits == ["Benefit 1", "Benefit 2"]
        assert recommendation.implementation_steps == ["Step 1", "Step 2"]
        assert recommendation.estimated_effort_hours == 8.0
        assert recommendation.risk_assessment == "Medium risk"
        assert recommendation.success_metrics == ["Metric 1", "Metric 2"]
        assert recommendation.dependencies == ["Dependency 1"]
        assert recommendation.cost_benefit_ratio == 3.5
        assert recommendation.metadata == {"test": "value"}


class TestCacheMemoryOptimizer:
    """Test cases for CacheMemoryOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a memory optimizer for testing."""
        return CacheMemoryOptimizer()

    @pytest.fixture
    def sample_metrics(self):
        """Create sample memory metrics for testing."""
        base_time = datetime.now()
        metrics = []

        for i in range(20):
            metrics.append(
                MemoryMetrics(
                    cache_name="test_cache",
                    timestamp=base_time + timedelta(minutes=i * 5),
                    total_memory_mb=200.0 + i * 10,  # Growing memory
                    cache_memory_mb=150.0 + i * 8,  # Growing cache memory
                    entry_count=1000 + i * 50,  # Growing entries
                    hit_ratio=0.8 - i * 0.01,  # Declining hit ratio
                    miss_ratio=0.2 + i * 0.01,  # Increasing miss ratio
                    eviction_rate=0.02 + i * 0.005,  # Increasing eviction
                    allocation_rate_mb_per_min=5.0 + i * 0.5,  # Increasing allocation
                    fragmentation_ratio=0.1 + i * 0.01,  # Increasing fragmentation
                    gc_frequency=3.0 + i * 0.5,  # Increasing GC frequency
                )
            )

        return metrics

    @pytest.fixture
    def sample_config(self):
        """Create sample cache configuration for testing."""
        return CacheConfiguration(
            cache_name="test_cache",
            max_size_mb=512.0,
            max_entries=5000,
            ttl_seconds=None,
            eviction_policy="LRU",
            prefetch_enabled=False,
            compression_enabled=False,
            serialization_format="pickle",
            concurrency_level=1,
        )

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer._cache_metrics == {}
        assert optimizer._cache_configs == {}
        assert optimizer._generated_plans == {}

    @pytest.mark.asyncio
    async def test_register_cache_metrics(self, optimizer, sample_metrics):
        """Test registering cache metrics."""
        for metrics in sample_metrics:
            await optimizer.register_cache_metrics(metrics)

        assert "test_cache" in optimizer._cache_metrics
        assert len(optimizer._cache_metrics["test_cache"]) == 20

    @pytest.mark.asyncio
    async def test_register_cache_config(self, optimizer, sample_config):
        """Test registering cache configuration."""
        await optimizer.register_cache_config(sample_config)

        assert "test_cache" in optimizer._cache_configs
        assert optimizer._cache_configs["test_cache"] == sample_config

    @pytest.mark.asyncio
    async def test_metrics_storage_limit(self, optimizer):
        """Test that metrics storage is limited to prevent unbounded growth."""
        cache_name = "test_cache"

        # Add more than the limit (1000) metrics
        for i in range(1200):
            metrics = MemoryMetrics(
                cache_name=cache_name,
                timestamp=datetime.now(),
                total_memory_mb=100.0,
                cache_memory_mb=80.0,
                entry_count=100,
            )
            await optimizer.register_cache_metrics(metrics)

        # Should only keep the most recent 1000
        assert len(optimizer._cache_metrics[cache_name]) == 1000

    @pytest.mark.asyncio
    async def test_generate_optimization_plan_no_metrics(self, optimizer):
        """Test generating optimization plan with no metrics."""
        plan = await optimizer.generate_optimization_plan("nonexistent_cache")
        assert plan is None

    @pytest.mark.asyncio
    async def test_generate_optimization_plan_basic(self, optimizer, sample_metrics, sample_config):
        """Test generating basic optimization plan."""
        # Register metrics and config
        for metrics in sample_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        # Generate plan
        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        assert plan.cache_name == "test_cache"
        assert isinstance(plan.generated_at, datetime)
        assert plan.current_metrics.cache_name == "test_cache"
        assert plan.current_config.cache_name == "test_cache"
        assert len(plan.recommendations) > 0
        assert plan.total_estimated_effort_hours > 0
        assert len(plan.implementation_phases) > 0

    @pytest.mark.asyncio
    async def test_memory_usage_analysis(self, optimizer, sample_config):
        """Test memory usage analysis and recommendations."""
        # Create metrics showing high memory usage with low hit ratio
        high_memory_metrics = [
            MemoryMetrics(
                cache_name="test_cache",
                timestamp=datetime.now(),
                total_memory_mb=800.0,  # High memory
                cache_memory_mb=600.0,  # High cache memory
                entry_count=5000,
                hit_ratio=0.6,  # Low hit ratio
                miss_ratio=0.4,
                eviction_rate=0.05,
                allocation_rate_mb_per_min=15.0,
                fragmentation_ratio=0.4,  # High fragmentation
                gc_frequency=8.0,
            )
            for _ in range(10)
        ]

        for metrics in high_memory_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        # Should have memory-related recommendations
        memory_recs = [r for r in plan.recommendations if r.optimization_type == OptimizationType.EVICTION_OPTIMIZATION]
        assert len(memory_recs) > 0

        fragmentation_recs = [r for r in plan.recommendations if "fragmentation" in r.title.lower()]
        assert len(fragmentation_recs) > 0

    @pytest.mark.asyncio
    async def test_performance_analysis(self, optimizer, sample_config):
        """Test performance analysis and recommendations."""
        # Create metrics showing poor performance
        poor_performance_metrics = [
            MemoryMetrics(
                cache_name="test_cache",
                timestamp=datetime.now(),
                total_memory_mb=300.0,
                cache_memory_mb=200.0,
                entry_count=1000,
                hit_ratio=0.5,  # Very low hit ratio
                miss_ratio=0.5,
                eviction_rate=0.15,  # High eviction rate
                allocation_rate_mb_per_min=10.0,
                fragmentation_ratio=0.2,
                gc_frequency=12.0,  # High GC frequency
            )
            for _ in range(10)
        ]

        for metrics in poor_performance_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        # Should have performance-related recommendations
        performance_recs = [r for r in plan.recommendations if r.optimization_type == OptimizationType.PERFORMANCE_IMPROVEMENT]
        assert len(performance_recs) > 0

        # Should have high priority recommendations for poor performance
        high_priority_recs = [r for r in plan.recommendations if r.priority == RecommendationPriority.HIGH]
        assert len(high_priority_recs) > 0

    @pytest.mark.asyncio
    async def test_configuration_analysis(self, optimizer, sample_metrics):
        """Test configuration analysis and recommendations."""
        # Create config with issues
        problematic_config = CacheConfiguration(
            cache_name="test_cache",
            max_size_mb=200.0,  # Small size for high usage
            max_entries=1000,
            ttl_seconds=None,  # No TTL
            eviction_policy="LRU",
            prefetch_enabled=False,
            compression_enabled=False,  # No compression for large cache
            serialization_format="pickle",
            concurrency_level=1,  # Single threaded
        )

        # Use metrics that would trigger config recommendations
        for metrics in sample_metrics:
            # Simulate near-capacity usage
            metrics.cache_memory_mb = 180.0  # 90% of max_size_mb
            await optimizer.register_cache_metrics(metrics)

        await optimizer.register_cache_config(problematic_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        # Should have configuration-related recommendations
        config_recs = [r for r in plan.recommendations if r.optimization_type == OptimizationType.CONFIGURATION_TUNING]
        assert len(config_recs) > 0

        # Should recommend capacity increase
        capacity_recs = [r for r in plan.recommendations if "capacity" in r.title.lower()]
        assert len(capacity_recs) > 0

    @pytest.mark.asyncio
    async def test_memory_growth_detection(self, optimizer, sample_config):
        """Test memory growth trend detection."""
        base_time = datetime.now()

        # Create metrics showing rapid memory growth
        growth_metrics = []
        for i in range(15):
            growth_metrics.append(
                MemoryMetrics(
                    cache_name="test_cache",
                    timestamp=base_time + timedelta(hours=i),
                    total_memory_mb=100.0 + i * 50,  # Rapid growth: 50MB per hour
                    cache_memory_mb=80.0 + i * 40,
                    entry_count=1000,
                    hit_ratio=0.8,
                    miss_ratio=0.2,
                    eviction_rate=0.05,
                    allocation_rate_mb_per_min=10.0,
                    fragmentation_ratio=0.1,
                    gc_frequency=5.0,
                )
            )

        for metrics in growth_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        # Should detect rapid growth and recommend action
        growth_recs = [r for r in plan.recommendations if "growth" in r.title.lower()]
        assert len(growth_recs) > 0

        # Growth-related recommendations should be high priority
        high_priority_growth = [r for r in growth_recs if r.priority == RecommendationPriority.HIGH]
        assert len(high_priority_growth) > 0

    @pytest.mark.asyncio
    async def test_compression_recommendation(self, optimizer, sample_config):
        """Test compression recommendation for large caches."""
        # Modify config to disable compression
        sample_config.compression_enabled = False

        # Create metrics showing large memory usage
        large_cache_metrics = [
            MemoryMetrics(
                cache_name="test_cache",
                timestamp=datetime.now(),
                total_memory_mb=600.0,
                cache_memory_mb=400.0,  # Large cache without compression
                entry_count=5000,
                hit_ratio=0.8,
                miss_ratio=0.2,
                eviction_rate=0.05,
                allocation_rate_mb_per_min=10.0,
                fragmentation_ratio=0.1,
                gc_frequency=5.0,
            )
            for _ in range(10)
        ]

        for metrics in large_cache_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        # Should recommend compression for large cache
        compression_recs = [r for r in plan.recommendations if "compression" in r.title.lower()]
        assert len(compression_recs) > 0

        compression_rec = compression_recs[0]
        assert compression_rec.optimization_type == OptimizationType.MEMORY_REDUCTION
        assert "memory usage" in compression_rec.description.lower()

    @pytest.mark.asyncio
    async def test_concurrency_optimization(self, optimizer, sample_config):
        """Test concurrency optimization recommendations."""
        # Config with single-threaded access for large cache
        sample_config.concurrency_level = 1

        # Create metrics for large cache
        large_cache_metrics = [
            MemoryMetrics(
                cache_name="test_cache",
                timestamp=datetime.now(),
                total_memory_mb=500.0,
                cache_memory_mb=300.0,  # Large cache
                entry_count=10000,
                hit_ratio=0.8,
                miss_ratio=0.2,
                eviction_rate=0.05,
                allocation_rate_mb_per_min=10.0,
                fragmentation_ratio=0.1,
                gc_frequency=5.0,
            )
            for _ in range(10)
        ]

        for metrics in large_cache_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        # Should recommend concurrency optimization
        concurrency_recs = [r for r in plan.recommendations if "concurrency" in r.title.lower()]
        assert len(concurrency_recs) > 0

        concurrency_rec = concurrency_recs[0]
        assert concurrency_rec.optimization_type == OptimizationType.PERFORMANCE_IMPROVEMENT

    @pytest.mark.asyncio
    async def test_implementation_phases(self, optimizer, sample_metrics, sample_config):
        """Test implementation phase creation."""
        for metrics in sample_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        assert len(plan.implementation_phases) > 0

        # Check phase structure
        phase = plan.implementation_phases[0]
        assert "phase" in phase
        assert "name" in phase
        assert "description" in phase
        assert "recommendations" in phase
        assert "estimated_effort_hours" in phase
        assert "expected_timeline_days" in phase

        # Verify phases are ordered by priority
        if len(plan.implementation_phases) > 1:
            assert plan.implementation_phases[0]["name"].startswith("Critical") or plan.implementation_phases[0]["name"].startswith("High")

    @pytest.mark.asyncio
    async def test_risk_assessment(self, optimizer, sample_metrics, sample_config):
        """Test risk assessment generation."""
        for metrics in sample_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        assert plan.risk_summary
        assert "Risk Assessment" in plan.risk_summary
        assert "risk" in plan.risk_summary.lower()

    @pytest.mark.asyncio
    async def test_recommendation_prioritization(self, optimizer, sample_config):
        """Test that recommendations are properly prioritized."""
        # Create metrics that would generate different priority recommendations
        mixed_metrics = [
            MemoryMetrics(
                cache_name="test_cache",
                timestamp=datetime.now(),
                total_memory_mb=800.0,  # High memory (high priority)
                cache_memory_mb=600.0,
                entry_count=5000,
                hit_ratio=0.5,  # Low hit ratio (high priority)
                miss_ratio=0.5,
                eviction_rate=0.15,  # High eviction (medium priority)
                allocation_rate_mb_per_min=15.0,
                fragmentation_ratio=0.2,  # Moderate fragmentation (medium priority)
                gc_frequency=8.0,
            )
            for _ in range(10)
        ]

        for metrics in mixed_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        assert len(plan.recommendations) > 0

        # Check that recommendations are sorted by priority
        priorities = [r.priority for r in plan.recommendations]
        priority_values = [{"critical": 4, "high": 3, "medium": 2, "low": 1}[p.value] for p in priorities]

        # Should be sorted in descending order (highest priority first)
        assert priority_values == sorted(priority_values, reverse=True)

    @pytest.mark.asyncio
    async def test_cost_benefit_analysis(self, optimizer, sample_metrics, sample_config):
        """Test cost-benefit analysis in recommendations."""
        for metrics in sample_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")

        assert plan is not None
        assert len(plan.recommendations) > 0

        # Check that recommendations have cost-benefit ratios
        for rec in plan.recommendations:
            assert rec.cost_benefit_ratio >= 0

        # High-impact, low-effort recommendations should have higher ratios
        high_impact_recs = [r for r in plan.recommendations if r.impact_level == ImpactLevel.MAJOR]
        if high_impact_recs:
            assert max(r.cost_benefit_ratio for r in high_impact_recs) > 1.0

    @pytest.mark.asyncio
    async def test_memory_growth_calculation(self, optimizer):
        """Test memory growth trend calculation."""
        base_time = datetime.now()

        # Create metrics with steady growth
        growth_metrics = []
        for i in range(10):
            growth_metrics.append(
                MemoryMetrics(
                    cache_name="test_cache",
                    timestamp=base_time + timedelta(hours=i),
                    total_memory_mb=100.0 + i * 10,  # 10MB per hour = 240MB per day
                    cache_memory_mb=80.0 + i * 8,
                    entry_count=1000,
                )
            )

        growth_rate = await optimizer._calculate_memory_growth_trend(growth_metrics)

        # Should detect approximately 240MB per day growth
        assert 200 <= growth_rate <= 280  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_expected_savings_calculation(self, optimizer):
        """Test expected memory savings calculation."""
        recommendations = [
            OptimizationRecommendation(
                recommendation_id="rec1",
                cache_name="test",
                optimization_type=OptimizationType.MEMORY_REDUCTION,
                priority=RecommendationPriority.HIGH,
                impact_level=ImpactLevel.MAJOR,
                title="Test",
                description="Test",
                current_state="Test",
                recommended_change="Test",
                expected_benefits=[],
                implementation_steps=[],
                estimated_effort_hours=4.0,
                risk_assessment="Low",
                success_metrics=[],
            ),
            OptimizationRecommendation(
                recommendation_id="rec2",
                cache_name="test",
                optimization_type=OptimizationType.MEMORY_REDUCTION,
                priority=RecommendationPriority.MEDIUM,
                impact_level=ImpactLevel.MODERATE,
                title="Test",
                description="Test",
                current_state="Test",
                recommended_change="Test",
                expected_benefits=[],
                implementation_steps=[],
                estimated_effort_hours=2.0,
                risk_assessment="Low",
                success_metrics=[],
            ),
        ]

        savings = await optimizer._calculate_expected_memory_savings(recommendations)

        # Should calculate savings based on impact levels
        assert savings > 0
        # Major + Moderate should give significant savings
        assert savings >= 150  # 100 (major) + 50 (moderate)

    def test_get_optimization_plan(self, optimizer):
        """Test getting optimization plan."""
        # Initially no plan
        plan = optimizer.get_optimization_plan("test_cache")
        assert plan is None

        # Add a dummy plan
        dummy_plan = OptimizationPlan(
            cache_name="test_cache",
            generated_at=datetime.now(),
            current_metrics=MemoryMetrics("test_cache", datetime.now(), 100.0, 80.0, 1000),
            current_config=CacheConfiguration("test_cache"),
            recommendations=[],
            total_estimated_effort_hours=0.0,
            expected_memory_savings_mb=0.0,
            expected_performance_improvement_percent=0.0,
        )

        optimizer._generated_plans["test_cache"] = dummy_plan

        retrieved_plan = optimizer.get_optimization_plan("test_cache")
        assert retrieved_plan == dummy_plan

    def test_get_all_optimization_plans(self, optimizer):
        """Test getting all optimization plans."""
        # Initially no plans
        plans = optimizer.get_all_optimization_plans()
        assert plans == {}

        # Add dummy plans
        plan1 = OptimizationPlan(
            cache_name="cache1",
            generated_at=datetime.now(),
            current_metrics=MemoryMetrics("cache1", datetime.now(), 100.0, 80.0, 1000),
            current_config=CacheConfiguration("cache1"),
            recommendations=[],
            total_estimated_effort_hours=0.0,
            expected_memory_savings_mb=0.0,
            expected_performance_improvement_percent=0.0,
        )

        plan2 = OptimizationPlan(
            cache_name="cache2",
            generated_at=datetime.now(),
            current_metrics=MemoryMetrics("cache2", datetime.now(), 200.0, 160.0, 2000),
            current_config=CacheConfiguration("cache2"),
            recommendations=[],
            total_estimated_effort_hours=0.0,
            expected_memory_savings_mb=0.0,
            expected_performance_improvement_percent=0.0,
        )

        optimizer._generated_plans["cache1"] = plan1
        optimizer._generated_plans["cache2"] = plan2

        all_plans = optimizer.get_all_optimization_plans()
        assert len(all_plans) == 2
        assert "cache1" in all_plans
        assert "cache2" in all_plans

    @pytest.mark.asyncio
    async def test_recommendation_summary_single_cache(self, optimizer, sample_metrics, sample_config):
        """Test getting recommendation summary for a single cache."""
        for metrics in sample_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(sample_config)

        plan = await optimizer.generate_optimization_plan("test_cache")
        assert plan is not None

        summary = await optimizer.get_recommendation_summary("test_cache")

        assert summary["status"] == "success"
        assert summary["cache_name"] == "test_cache"
        assert summary["recommendation_count"] > 0
        assert summary["total_effort_hours"] > 0
        assert "priorities" in summary
        assert "critical" in summary["priorities"]
        assert "high" in summary["priorities"]
        assert "medium" in summary["priorities"]
        assert "low" in summary["priorities"]

    @pytest.mark.asyncio
    async def test_recommendation_summary_nonexistent_cache(self, optimizer):
        """Test getting recommendation summary for nonexistent cache."""
        summary = await optimizer.get_recommendation_summary("nonexistent_cache")

        assert summary["status"] == "no_plan"
        assert summary["cache_name"] == "nonexistent_cache"

    @pytest.mark.asyncio
    async def test_recommendation_summary_all_caches(self, optimizer, sample_metrics, sample_config):
        """Test getting recommendation summary for all caches."""
        # Add data for multiple caches
        for i, cache_name in enumerate(["cache1", "cache2"]):
            config = CacheConfiguration(cache_name=cache_name)
            await optimizer.register_cache_config(config)

            for metrics in sample_metrics:
                metrics.cache_name = cache_name
                await optimizer.register_cache_metrics(metrics)

            await optimizer.generate_optimization_plan(cache_name)

        summary = await optimizer.get_recommendation_summary()

        assert summary["status"] == "success"
        assert summary["cache_count"] == 2
        assert summary["total_recommendations"] > 0
        assert summary["total_effort_hours"] > 0
        assert "caches" in summary
        assert len(summary["caches"]) == 2


class TestGlobalFunctions:
    """Test cases for global utility functions."""

    @pytest.mark.asyncio
    async def test_get_memory_optimizer_singleton(self):
        """Test that get_memory_optimizer returns the same instance."""
        optimizer1 = await get_memory_optimizer()
        optimizer2 = await get_memory_optimizer()

        assert optimizer1 is optimizer2

    @pytest.mark.asyncio
    async def test_register_cache_metrics_function(self):
        """Test global register_cache_metrics function."""
        metrics = MemoryMetrics(
            cache_name="test_cache",
            timestamp=datetime.now(),
            total_memory_mb=100.0,
            cache_memory_mb=80.0,
            entry_count=1000,
        )

        await register_cache_metrics("test_cache", metrics)

        optimizer = await get_memory_optimizer()
        assert "test_cache" in optimizer._cache_metrics
        assert len(optimizer._cache_metrics["test_cache"]) == 1

    @pytest.mark.asyncio
    async def test_register_cache_configuration_function(self):
        """Test global register_cache_configuration function."""
        config = CacheConfiguration(cache_name="test_cache", max_size_mb=512.0)

        await register_cache_configuration("test_cache", config)

        optimizer = await get_memory_optimizer()
        assert "test_cache" in optimizer._cache_configs
        assert optimizer._cache_configs["test_cache"] == config

    @pytest.mark.asyncio
    async def test_generate_cache_optimization_plan_function(self):
        """Test global generate_cache_optimization_plan function."""
        # Register some test data
        metrics = MemoryMetrics(
            cache_name="test_cache",
            timestamp=datetime.now(),
            total_memory_mb=100.0,
            cache_memory_mb=80.0,
            entry_count=1000,
        )
        config = CacheConfiguration(cache_name="test_cache")

        await register_cache_metrics("test_cache", metrics)
        await register_cache_configuration("test_cache", config)

        plan = await generate_cache_optimization_plan("test_cache")

        assert plan is not None
        assert plan.cache_name == "test_cache"

    @pytest.mark.asyncio
    async def test_get_cache_optimization_summary_function(self):
        """Test global get_cache_optimization_summary function."""
        # Register test data and generate plan
        metrics = MemoryMetrics(
            cache_name="test_cache",
            timestamp=datetime.now(),
            total_memory_mb=100.0,
            cache_memory_mb=80.0,
            entry_count=1000,
        )
        config = CacheConfiguration(cache_name="test_cache")

        await register_cache_metrics("test_cache", metrics)
        await register_cache_configuration("test_cache", config)
        await generate_cache_optimization_plan("test_cache")

        summary = await get_cache_optimization_summary("test_cache")

        assert summary["status"] == "success"
        assert summary["cache_name"] == "test_cache"


class TestEdgeCases:
    """Test cases for edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_empty_metrics_analysis(self):
        """Test analysis with empty metrics."""
        optimizer = CacheMemoryOptimizer()
        config = CacheConfiguration(cache_name="empty_cache")

        await optimizer.register_cache_config(config)

        # Should return None for empty metrics
        plan = await optimizer.generate_optimization_plan("empty_cache")
        assert plan is None

    @pytest.mark.asyncio
    async def test_single_metric_analysis(self):
        """Test analysis with only one metric."""
        optimizer = CacheMemoryOptimizer()

        metrics = MemoryMetrics(
            cache_name="single_metric_cache",
            timestamp=datetime.now(),
            total_memory_mb=100.0,
            cache_memory_mb=80.0,
            entry_count=1000,
        )

        await optimizer.register_cache_metrics(metrics)

        plan = await optimizer.generate_optimization_plan("single_metric_cache")

        # Should still generate a plan with inferred config
        assert plan is not None
        assert plan.cache_name == "single_metric_cache"

    @pytest.mark.asyncio
    async def test_zero_memory_metrics(self):
        """Test analysis with zero memory metrics."""
        optimizer = CacheMemoryOptimizer()

        zero_metrics = [
            MemoryMetrics(
                cache_name="zero_cache",
                timestamp=datetime.now(),
                total_memory_mb=0.0,
                cache_memory_mb=0.0,
                entry_count=0,
            )
            for _ in range(5)
        ]

        for metrics in zero_metrics:
            await optimizer.register_cache_metrics(metrics)

        plan = await optimizer.generate_optimization_plan("zero_cache")

        assert plan is not None
        # Should have minimal recommendations for zero usage
        assert len(plan.recommendations) == 0 or all(rec.priority == RecommendationPriority.LOW for rec in plan.recommendations)

    @pytest.mark.asyncio
    async def test_perfect_cache_metrics(self):
        """Test analysis with perfect cache metrics."""
        optimizer = CacheMemoryOptimizer()

        perfect_metrics = [
            MemoryMetrics(
                cache_name="perfect_cache",
                timestamp=datetime.now(),
                total_memory_mb=200.0,
                cache_memory_mb=150.0,
                entry_count=1000,
                hit_ratio=0.95,  # Excellent hit ratio
                miss_ratio=0.05,
                eviction_rate=0.01,  # Low eviction
                allocation_rate_mb_per_min=2.0,  # Low allocation
                fragmentation_ratio=0.05,  # Low fragmentation
                gc_frequency=1.0,  # Low GC frequency
            )
            for _ in range(10)
        ]

        config = CacheConfiguration(
            cache_name="perfect_cache",
            max_size_mb=512.0,  # Plenty of headroom
            max_entries=10000,
            ttl_seconds=3600,
            eviction_policy="LRU",
            compression_enabled=True,
            concurrency_level=4,
        )

        for metrics in perfect_metrics:
            await optimizer.register_cache_metrics(metrics)
        await optimizer.register_cache_config(config)

        plan = await optimizer.generate_optimization_plan("perfect_cache")

        assert plan is not None
        # Should have few or low-priority recommendations for perfect cache
        high_priority_recs = [
            r for r in plan.recommendations if r.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]
        ]
        assert len(high_priority_recs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
