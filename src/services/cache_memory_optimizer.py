"""
Cache memory optimization service for analyzing cache usage patterns and providing actionable optimization recommendations.

This module provides comprehensive optimization analysis including:
- Memory usage pattern analysis
- Cache configuration optimization
- Performance bottleneck identification
- Resource allocation recommendations
- Cost-benefit analysis of changes
"""

import asyncio
import logging
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of cache optimizations."""

    MEMORY_REDUCTION = "memory_reduction"  # Reduce memory usage
    PERFORMANCE_IMPROVEMENT = "performance_improvement"  # Improve cache performance
    CONFIGURATION_TUNING = "configuration_tuning"  # Optimize configuration
    EVICTION_OPTIMIZATION = "eviction_optimization"  # Optimize eviction policies
    RESOURCE_ALLOCATION = "resource_allocation"  # Optimize resource allocation
    ARCHITECTURE_IMPROVEMENT = "architecture_improvement"  # Architectural improvements


class RecommendationPriority(Enum):
    """Priority levels for optimization recommendations."""

    CRITICAL = "critical"  # Must implement immediately
    HIGH = "high"  # Should implement soon
    MEDIUM = "medium"  # Should consider implementing
    LOW = "low"  # Nice to have improvement


class ImpactLevel(Enum):
    """Expected impact levels of recommendations."""

    MAJOR = "major"  # Significant improvement expected
    MODERATE = "moderate"  # Moderate improvement expected
    MINOR = "minor"  # Small improvement expected


@dataclass
class MemoryMetrics:
    """Memory usage metrics for analysis."""

    cache_name: str
    timestamp: datetime
    total_memory_mb: float
    cache_memory_mb: float
    entry_count: int
    hit_ratio: float = 0.0
    miss_ratio: float = 0.0
    eviction_rate: float = 0.0
    allocation_rate_mb_per_min: float = 0.0
    fragmentation_ratio: float = 0.0
    gc_frequency: float = 0.0


@dataclass
class CacheConfiguration:
    """Cache configuration parameters."""

    cache_name: str
    max_size_mb: float | None = None
    max_entries: int | None = None
    ttl_seconds: int | None = None
    eviction_policy: str | None = None
    prefetch_enabled: bool = False
    compression_enabled: bool = False
    serialization_format: str = "pickle"
    concurrency_level: int = 1


@dataclass
class OptimizationRecommendation:
    """Represents a specific optimization recommendation."""

    recommendation_id: str
    cache_name: str
    optimization_type: OptimizationType
    priority: RecommendationPriority
    impact_level: ImpactLevel
    title: str
    description: str
    current_state: str
    recommended_change: str
    expected_benefits: list[str]
    implementation_steps: list[str]
    estimated_effort_hours: float
    risk_assessment: str
    success_metrics: list[str]
    dependencies: list[str] = field(default_factory=list)
    cost_benefit_ratio: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationPlan:
    """Complete optimization plan for a cache."""

    cache_name: str
    generated_at: datetime
    current_metrics: MemoryMetrics
    current_config: CacheConfiguration
    recommendations: list[OptimizationRecommendation]
    total_estimated_effort_hours: float
    expected_memory_savings_mb: float
    expected_performance_improvement_percent: float
    implementation_phases: list[dict[str, Any]] = field(default_factory=list)
    risk_summary: str = ""


class CacheMemoryOptimizer:
    """
    Advanced cache memory optimizer that analyzes usage patterns and provides actionable recommendations.

    Provides comprehensive optimization analysis including:
    - Memory usage pattern analysis
    - Configuration optimization recommendations
    - Performance bottleneck identification
    - Resource allocation optimization
    - Implementation planning
    """

    def __init__(self):
        """Initialize the cache memory optimizer."""
        self._cache_metrics: dict[str, list[MemoryMetrics]] = defaultdict(list)
        self._cache_configs: dict[str, CacheConfiguration] = {}
        self._generated_plans: dict[str, OptimizationPlan] = {}
        self._lock = asyncio.Lock()

        logger.info("Cache memory optimizer initialized")

    async def register_cache_metrics(self, metrics: MemoryMetrics) -> None:
        """Register memory metrics for a cache."""
        async with self._lock:
            self._cache_metrics[metrics.cache_name].append(metrics)

            # Keep only recent metrics (last 1000 entries)
            if len(self._cache_metrics[metrics.cache_name]) > 1000:
                self._cache_metrics[metrics.cache_name] = self._cache_metrics[metrics.cache_name][-1000:]

        logger.debug(f"Registered metrics for cache: {metrics.cache_name}")

    async def register_cache_config(self, config: CacheConfiguration) -> None:
        """Register cache configuration for analysis."""
        async with self._lock:
            self._cache_configs[config.cache_name] = config

        logger.debug(f"Registered configuration for cache: {config.cache_name}")

    async def generate_optimization_plan(self, cache_name: str) -> OptimizationPlan | None:
        """Generate a comprehensive optimization plan for a cache."""
        async with self._lock:
            metrics = self._cache_metrics.get(cache_name, [])
            config = self._cache_configs.get(cache_name)

        if not metrics:
            logger.warning(f"No metrics available for cache: {cache_name}")
            return None

        # Analyze current state
        current_metrics = await self._analyze_current_metrics(cache_name, metrics)
        current_config = config or await self._infer_configuration(cache_name, metrics)

        # Generate recommendations
        recommendations = await self._generate_recommendations(cache_name, current_metrics, current_config, metrics)

        # Calculate totals
        total_effort = sum(rec.estimated_effort_hours for rec in recommendations)
        expected_memory_savings = await self._calculate_expected_memory_savings(recommendations)
        expected_performance_improvement = await self._calculate_expected_performance_improvement(recommendations)

        # Create implementation phases
        implementation_phases = await self._create_implementation_phases(recommendations)

        # Generate risk summary
        risk_summary = await self._generate_risk_summary(recommendations)

        plan = OptimizationPlan(
            cache_name=cache_name,
            generated_at=datetime.now(),
            current_metrics=current_metrics,
            current_config=current_config,
            recommendations=recommendations,
            total_estimated_effort_hours=total_effort,
            expected_memory_savings_mb=expected_memory_savings,
            expected_performance_improvement_percent=expected_performance_improvement,
            implementation_phases=implementation_phases,
            risk_summary=risk_summary,
        )

        async with self._lock:
            self._generated_plans[cache_name] = plan

        logger.info(f"Generated optimization plan for cache: {cache_name} with {len(recommendations)} recommendations")
        return plan

    async def _analyze_current_metrics(self, cache_name: str, metrics: list[MemoryMetrics]) -> MemoryMetrics:
        """Analyze current cache metrics to determine baseline state."""
        if not metrics:
            return MemoryMetrics(cache_name=cache_name, timestamp=datetime.now(), total_memory_mb=0.0, cache_memory_mb=0.0, entry_count=0)

        # Use recent metrics for current state (last 10 entries)
        recent_metrics = metrics[-10:]

        return MemoryMetrics(
            cache_name=cache_name,
            timestamp=recent_metrics[-1].timestamp,
            total_memory_mb=statistics.mean(m.total_memory_mb for m in recent_metrics),
            cache_memory_mb=statistics.mean(m.cache_memory_mb for m in recent_metrics),
            entry_count=int(statistics.mean(m.entry_count for m in recent_metrics)),
            hit_ratio=statistics.mean(m.hit_ratio for m in recent_metrics if m.hit_ratio > 0),
            miss_ratio=statistics.mean(m.miss_ratio for m in recent_metrics if m.miss_ratio > 0),
            eviction_rate=statistics.mean(m.eviction_rate for m in recent_metrics if m.eviction_rate > 0),
            allocation_rate_mb_per_min=statistics.mean(
                m.allocation_rate_mb_per_min for m in recent_metrics if m.allocation_rate_mb_per_min > 0
            ),
            fragmentation_ratio=statistics.mean(m.fragmentation_ratio for m in recent_metrics if m.fragmentation_ratio > 0),
            gc_frequency=statistics.mean(m.gc_frequency for m in recent_metrics if m.gc_frequency > 0),
        )

    async def _infer_configuration(self, cache_name: str, metrics: list[MemoryMetrics]) -> CacheConfiguration:
        """Infer cache configuration from metrics when config not available."""
        if not metrics:
            return CacheConfiguration(cache_name=cache_name)

        recent_metrics = metrics[-20:]
        max_memory = max(m.cache_memory_mb for m in recent_metrics)
        max_entries = max(m.entry_count for m in recent_metrics)

        return CacheConfiguration(
            cache_name=cache_name,
            max_size_mb=max_memory * 1.2,  # Estimate with some buffer
            max_entries=max_entries * 2,  # Estimate with buffer
            eviction_policy="LRU",  # Default assumption
            serialization_format="pickle",
            concurrency_level=1,
        )

    async def _generate_recommendations(
        self, cache_name: str, current_metrics: MemoryMetrics, config: CacheConfiguration, historical_metrics: list[MemoryMetrics]
    ) -> list[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # Memory usage analysis
        memory_recommendations = await self._analyze_memory_usage(cache_name, current_metrics, config, historical_metrics)
        recommendations.extend(memory_recommendations)

        # Performance analysis
        performance_recommendations = await self._analyze_performance_patterns(cache_name, current_metrics, config, historical_metrics)
        recommendations.extend(performance_recommendations)

        # Configuration analysis
        config_recommendations = await self._analyze_configuration(cache_name, current_metrics, config)
        recommendations.extend(config_recommendations)

        # Eviction policy analysis
        eviction_recommendations = await self._analyze_eviction_patterns(cache_name, current_metrics, config, historical_metrics)
        recommendations.extend(eviction_recommendations)

        # Resource allocation analysis
        resource_recommendations = await self._analyze_resource_allocation(cache_name, current_metrics, config, historical_metrics)
        recommendations.extend(resource_recommendations)

        # Sort by priority and impact
        recommendations.sort(
            key=lambda r: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[r.priority.value],
                {"major": 3, "moderate": 2, "minor": 1}[r.impact_level.value],
            ),
            reverse=True,
        )

        return recommendations

    async def _analyze_memory_usage(
        self, cache_name: str, current_metrics: MemoryMetrics, config: CacheConfiguration, historical_metrics: list[MemoryMetrics]
    ) -> list[OptimizationRecommendation]:
        """Analyze memory usage patterns and generate recommendations."""
        recommendations = []

        # High memory usage analysis
        if current_metrics.cache_memory_mb > 500:  # > 500MB
            if current_metrics.hit_ratio < 0.8:  # Low hit ratio with high memory
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"{cache_name}_memory_hit_ratio_{int(time.time())}",
                        cache_name=cache_name,
                        optimization_type=OptimizationType.EVICTION_OPTIMIZATION,
                        priority=RecommendationPriority.HIGH,
                        impact_level=ImpactLevel.MAJOR,
                        title="Optimize Cache Hit Ratio to Reduce Memory Waste",
                        description="High memory usage with low hit ratio indicates inefficient caching strategy",
                        current_state=f"Memory: {current_metrics.cache_memory_mb:.1f}MB, Hit ratio: {current_metrics.hit_ratio:.2%}",
                        recommended_change="Implement smarter eviction policy and cache key optimization",
                        expected_benefits=[
                            "Reduce memory usage by 30-50%",
                            "Improve cache efficiency",
                            "Better resource utilization",
                        ],
                        implementation_steps=[
                            "Analyze cache key patterns and access frequency",
                            "Implement LFU or adaptive eviction policy",
                            "Add cache key namespacing and categorization",
                            "Monitor hit ratio improvements",
                        ],
                        estimated_effort_hours=8.0,
                        risk_assessment="Low risk - gradual implementation possible",
                        success_metrics=["Hit ratio > 85%", "Memory reduction > 30%", "Response time improvement"],
                        cost_benefit_ratio=4.0,
                    )
                )

        # Memory fragmentation analysis
        if current_metrics.fragmentation_ratio > 0.3:  # > 30% fragmentation
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_fragmentation_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.MEMORY_REDUCTION,
                    priority=RecommendationPriority.MEDIUM,
                    impact_level=ImpactLevel.MODERATE,
                    title="Reduce Memory Fragmentation",
                    description="High memory fragmentation reduces effective cache capacity",
                    current_state=f"Fragmentation ratio: {current_metrics.fragmentation_ratio:.1%}",
                    recommended_change="Implement memory pool allocation and object size optimization",
                    expected_benefits=[
                        "Reduce effective memory usage by 20-30%",
                        "Improve memory allocation efficiency",
                        "Reduce garbage collection pressure",
                    ],
                    implementation_steps=[
                        "Implement object size analysis",
                        "Add memory pool for common object sizes",
                        "Optimize data serialization formats",
                        "Add periodic defragmentation",
                    ],
                    estimated_effort_hours=12.0,
                    risk_assessment="Medium risk - requires careful testing",
                    success_metrics=["Fragmentation ratio < 15%", "Memory efficiency improvement", "GC frequency reduction"],
                    cost_benefit_ratio=2.5,
                )
            )

        # Memory growth analysis
        if len(historical_metrics) >= 10:
            recent_growth = await self._calculate_memory_growth_trend(historical_metrics[-10:])
            if recent_growth > 10:  # > 10MB growth per day
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"{cache_name}_growth_trend_{int(time.time())}",
                        cache_name=cache_name,
                        optimization_type=OptimizationType.MEMORY_REDUCTION,
                        priority=RecommendationPriority.HIGH,
                        impact_level=ImpactLevel.MAJOR,
                        title="Address Rapid Memory Growth",
                        description="Cache memory is growing rapidly, indicating potential leaks or inefficient policies",
                        current_state=f"Memory growth: {recent_growth:.1f}MB/day",
                        recommended_change="Implement aggressive eviction and memory leak detection",
                        expected_benefits=[
                            "Stabilize memory usage",
                            "Prevent memory exhaustion",
                            "Improve system stability",
                        ],
                        implementation_steps=[
                            "Enable memory leak detection",
                            "Implement time-based eviction",
                            "Add memory usage alerts",
                            "Review cache entry lifecycle",
                        ],
                        estimated_effort_hours=6.0,
                        risk_assessment="Low risk - preventive measure",
                        success_metrics=["Memory growth < 5MB/day", "Stable memory usage", "No memory alerts"],
                        cost_benefit_ratio=5.0,
                    )
                )

        return recommendations

    async def _analyze_performance_patterns(
        self, cache_name: str, current_metrics: MemoryMetrics, config: CacheConfiguration, historical_metrics: list[MemoryMetrics]
    ) -> list[OptimizationRecommendation]:
        """Analyze performance patterns and generate recommendations."""
        recommendations = []

        # Low hit ratio optimization
        if current_metrics.hit_ratio < 0.7:  # < 70% hit ratio
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_hit_ratio_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.PERFORMANCE_IMPROVEMENT,
                    priority=RecommendationPriority.HIGH,
                    impact_level=ImpactLevel.MAJOR,
                    title="Improve Cache Hit Ratio",
                    description="Low hit ratio indicates poor cache effectiveness",
                    current_state=f"Hit ratio: {current_metrics.hit_ratio:.1%}",
                    recommended_change="Optimize cache strategy and key patterns",
                    expected_benefits=[
                        "Improve application response time by 40-60%",
                        "Reduce backend load",
                        "Better user experience",
                    ],
                    implementation_steps=[
                        "Analyze cache key patterns and access frequency",
                        "Implement predictive caching",
                        "Optimize cache warm-up strategies",
                        "Add cache key prefix optimization",
                    ],
                    estimated_effort_hours=10.0,
                    risk_assessment="Low risk - performance improvement",
                    success_metrics=["Hit ratio > 85%", "Response time improvement > 40%", "Reduced backend calls"],
                    cost_benefit_ratio=6.0,
                )
            )

        # High eviction rate analysis
        if current_metrics.eviction_rate > 0.1:  # > 10% eviction rate
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_eviction_rate_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.CONFIGURATION_TUNING,
                    priority=RecommendationPriority.MEDIUM,
                    impact_level=ImpactLevel.MODERATE,
                    title="Reduce High Eviction Rate",
                    description="High eviction rate indicates insufficient cache size or poor eviction policy",
                    current_state=f"Eviction rate: {current_metrics.eviction_rate:.1%}",
                    recommended_change="Increase cache size or optimize eviction policy",
                    expected_benefits=[
                        "Improve cache efficiency",
                        "Reduce cache misses",
                        "Better memory utilization",
                    ],
                    implementation_steps=[
                        "Analyze eviction patterns and timing",
                        "Consider increasing cache size if memory allows",
                        "Implement smarter eviction algorithms",
                        "Add eviction monitoring and alerts",
                    ],
                    estimated_effort_hours=4.0,
                    risk_assessment="Low risk - configuration change",
                    success_metrics=["Eviction rate < 5%", "Improved hit ratio", "Stable cache size"],
                    cost_benefit_ratio=3.0,
                )
            )

        # GC frequency analysis
        if current_metrics.gc_frequency > 10:  # > 10 GCs per minute
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_gc_frequency_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.PERFORMANCE_IMPROVEMENT,
                    priority=RecommendationPriority.MEDIUM,
                    impact_level=ImpactLevel.MODERATE,
                    title="Reduce Garbage Collection Pressure",
                    description="High GC frequency indicates memory allocation inefficiency",
                    current_state=f"GC frequency: {current_metrics.gc_frequency:.1f}/min",
                    recommended_change="Optimize object lifecycle and memory allocation patterns",
                    expected_benefits=[
                        "Reduce GC pause times",
                        "Improve application responsiveness",
                        "Better memory efficiency",
                    ],
                    implementation_steps=[
                        "Implement object pooling for frequently allocated objects",
                        "Optimize data structure choices",
                        "Add memory-efficient serialization",
                        "Monitor GC metrics and tune",
                    ],
                    estimated_effort_hours=8.0,
                    risk_assessment="Medium risk - requires performance testing",
                    success_metrics=["GC frequency < 5/min", "Reduced GC pause times", "Memory allocation efficiency"],
                    cost_benefit_ratio=2.8,
                )
            )

        return recommendations

    async def _analyze_configuration(
        self, cache_name: str, current_metrics: MemoryMetrics, config: CacheConfiguration
    ) -> list[OptimizationRecommendation]:
        """Analyze cache configuration and generate recommendations."""
        recommendations = []

        # Cache size optimization
        if config.max_size_mb and current_metrics.cache_memory_mb / config.max_size_mb > 0.9:  # > 90% capacity
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_capacity_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.CONFIGURATION_TUNING,
                    priority=RecommendationPriority.HIGH,
                    impact_level=ImpactLevel.MODERATE,
                    title="Increase Cache Capacity",
                    description="Cache is near maximum capacity, limiting effectiveness",
                    current_state=f"Usage: {current_metrics.cache_memory_mb:.1f}MB / {config.max_size_mb:.1f}MB "
                    f"({current_metrics.cache_memory_mb/config.max_size_mb:.1%})",
                    recommended_change=f"Increase cache size to {config.max_size_mb * 1.5:.0f}MB",
                    expected_benefits=[
                        "Reduce eviction pressure",
                        "Improve hit ratio",
                        "Better cache performance",
                    ],
                    implementation_steps=[
                        "Monitor system memory availability",
                        "Gradually increase cache size",
                        "Monitor performance impact",
                        "Adjust if needed based on metrics",
                    ],
                    estimated_effort_hours=2.0,
                    risk_assessment="Low risk - gradual increase",
                    success_metrics=["Cache utilization < 80%", "Improved hit ratio", "Reduced evictions"],
                    cost_benefit_ratio=4.0,
                )
            )

        # Compression recommendation
        if not config.compression_enabled and current_metrics.cache_memory_mb > 100:  # > 100MB without compression
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_compression_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.MEMORY_REDUCTION,
                    priority=RecommendationPriority.MEDIUM,
                    impact_level=ImpactLevel.MODERATE,
                    title="Enable Data Compression",
                    description="Large cache without compression wastes memory",
                    current_state=f"Compression: disabled, Memory: {current_metrics.cache_memory_mb:.1f}MB",
                    recommended_change="Enable compression for cache data",
                    expected_benefits=[
                        "Reduce memory usage by 40-70%",
                        "Increase effective cache capacity",
                        "Better resource utilization",
                    ],
                    implementation_steps=[
                        "Implement compression for cache values",
                        "Choose appropriate compression algorithm",
                        "Monitor compression ratio and performance",
                        "Adjust compression settings if needed",
                    ],
                    estimated_effort_hours=6.0,
                    risk_assessment="Medium risk - CPU/memory tradeoff",
                    success_metrics=["Memory reduction > 40%", "Compression ratio > 2:1", "Acceptable performance"],
                    cost_benefit_ratio=3.5,
                )
            )

        # TTL optimization
        if not config.ttl_seconds and current_metrics.eviction_rate < 0.05:  # No TTL with low eviction
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_ttl_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.CONFIGURATION_TUNING,
                    priority=RecommendationPriority.LOW,
                    impact_level=ImpactLevel.MINOR,
                    title="Implement Time-Based Expiration",
                    description="No TTL may lead to stale data accumulation",
                    current_state="TTL: not configured",
                    recommended_change="Implement appropriate TTL based on data freshness requirements",
                    expected_benefits=[
                        "Ensure data freshness",
                        "Prevent stale data issues",
                        "Automatic cache cleanup",
                    ],
                    implementation_steps=[
                        "Analyze data freshness requirements",
                        "Implement configurable TTL",
                        "Monitor cache effectiveness with TTL",
                        "Fine-tune TTL values based on usage",
                    ],
                    estimated_effort_hours=4.0,
                    risk_assessment="Low risk - configurable feature",
                    success_metrics=["Appropriate data freshness", "Automatic cleanup", "No stale data issues"],
                    cost_benefit_ratio=2.0,
                )
            )

        return recommendations

    async def _analyze_eviction_patterns(
        self, cache_name: str, current_metrics: MemoryMetrics, config: CacheConfiguration, historical_metrics: list[MemoryMetrics]
    ) -> list[OptimizationRecommendation]:
        """Analyze eviction patterns and generate recommendations."""
        recommendations = []

        # Eviction policy optimization
        if config.eviction_policy == "LRU" and current_metrics.hit_ratio < 0.75:
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_eviction_policy_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.EVICTION_OPTIMIZATION,
                    priority=RecommendationPriority.MEDIUM,
                    impact_level=ImpactLevel.MODERATE,
                    title="Optimize Eviction Policy",
                    description="LRU policy may not be optimal for current access patterns",
                    current_state=f"Policy: {config.eviction_policy}, Hit ratio: {current_metrics.hit_ratio:.1%}",
                    recommended_change="Implement adaptive or LFU eviction policy",
                    expected_benefits=[
                        "Improve cache hit ratio by 15-25%",
                        "Better match access patterns",
                        "Reduced cache misses",
                    ],
                    implementation_steps=[
                        "Analyze access pattern distribution",
                        "Implement LFU or adaptive eviction",
                        "A/B test different policies",
                        "Monitor hit ratio improvements",
                    ],
                    estimated_effort_hours=8.0,
                    risk_assessment="Medium risk - requires access pattern analysis",
                    success_metrics=["Hit ratio improvement > 15%", "Better access pattern matching", "Reduced misses"],
                    cost_benefit_ratio=3.2,
                )
            )

        return recommendations

    async def _analyze_resource_allocation(
        self, cache_name: str, current_metrics: MemoryMetrics, config: CacheConfiguration, historical_metrics: list[MemoryMetrics]
    ) -> list[OptimizationRecommendation]:
        """Analyze resource allocation and generate recommendations."""
        recommendations = []

        # Concurrency optimization
        if config.concurrency_level == 1 and current_metrics.cache_memory_mb > 200:  # Large cache, single thread
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id=f"{cache_name}_concurrency_{int(time.time())}",
                    cache_name=cache_name,
                    optimization_type=OptimizationType.PERFORMANCE_IMPROVEMENT,
                    priority=RecommendationPriority.MEDIUM,
                    impact_level=ImpactLevel.MODERATE,
                    title="Optimize Concurrency Level",
                    description="Large cache with single-threaded access may create bottlenecks",
                    current_state=f"Concurrency: {config.concurrency_level}, Memory: {current_metrics.cache_memory_mb:.1f}MB",
                    recommended_change="Increase concurrency level for better throughput",
                    expected_benefits=[
                        "Improve concurrent access performance",
                        "Reduce lock contention",
                        "Better resource utilization",
                    ],
                    implementation_steps=[
                        "Analyze current access patterns and contention",
                        "Implement higher concurrency level",
                        "Monitor performance and lock contention",
                        "Fine-tune based on workload",
                    ],
                    estimated_effort_hours=6.0,
                    risk_assessment="Medium risk - concurrent access complexity",
                    success_metrics=["Improved throughput", "Reduced lock contention", "Better response times"],
                    cost_benefit_ratio=2.8,
                )
            )

        return recommendations

    async def _calculate_memory_growth_trend(self, metrics: list[MemoryMetrics]) -> float:
        """Calculate memory growth trend in MB per day."""
        if len(metrics) < 2:
            return 0.0

        # Calculate linear regression for memory growth
        x_values = [(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics]
        y_values = [m.cache_memory_mb for m in metrics]

        if not x_values or max(x_values) == 0:
            return 0.0

        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Convert slope to MB per day
        return slope * 86400  # seconds per day

    async def _calculate_expected_memory_savings(self, recommendations: list[OptimizationRecommendation]) -> float:
        """Calculate expected memory savings from recommendations."""
        total_savings = 0.0

        for rec in recommendations:
            if rec.optimization_type in [OptimizationType.MEMORY_REDUCTION, OptimizationType.CONFIGURATION_TUNING]:
                # Estimate savings based on impact level and type
                if rec.impact_level == ImpactLevel.MAJOR:
                    total_savings += 100.0  # Assume 100MB savings for major improvements
                elif rec.impact_level == ImpactLevel.MODERATE:
                    total_savings += 50.0  # Assume 50MB savings for moderate improvements
                else:
                    total_savings += 20.0  # Assume 20MB savings for minor improvements

        return total_savings

    async def _calculate_expected_performance_improvement(self, recommendations: list[OptimizationRecommendation]) -> float:
        """Calculate expected performance improvement percentage."""
        total_improvement = 0.0

        for rec in recommendations:
            if rec.optimization_type in [OptimizationType.PERFORMANCE_IMPROVEMENT, OptimizationType.EVICTION_OPTIMIZATION]:
                # Estimate improvement based on impact level
                if rec.impact_level == ImpactLevel.MAJOR:
                    total_improvement += 30.0  # 30% improvement for major changes
                elif rec.impact_level == ImpactLevel.MODERATE:
                    total_improvement += 15.0  # 15% improvement for moderate changes
                else:
                    total_improvement += 5.0  # 5% improvement for minor changes

        # Cap at 100% improvement
        return min(total_improvement, 100.0)

    async def _create_implementation_phases(self, recommendations: list[OptimizationRecommendation]) -> list[dict[str, Any]]:
        """Create implementation phases for recommendations."""
        phases = []

        # Phase 1: Critical and high priority items
        critical_high = [r for r in recommendations if r.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]]
        if critical_high:
            phases.append(
                {
                    "phase": 1,
                    "name": "Critical and High Priority Optimizations",
                    "description": "Address urgent performance and memory issues",
                    "recommendations": [r.recommendation_id for r in critical_high],
                    "estimated_effort_hours": sum(r.estimated_effort_hours for r in critical_high),
                    "expected_timeline_days": math.ceil(sum(r.estimated_effort_hours for r in critical_high) / 8),
                    "dependencies": [],
                }
            )

        # Phase 2: Medium priority items
        medium = [r for r in recommendations if r.priority == RecommendationPriority.MEDIUM]
        if medium:
            phases.append(
                {
                    "phase": 2,
                    "name": "Medium Priority Optimizations",
                    "description": "Performance and efficiency improvements",
                    "recommendations": [r.recommendation_id for r in medium],
                    "estimated_effort_hours": sum(r.estimated_effort_hours for r in medium),
                    "expected_timeline_days": math.ceil(sum(r.estimated_effort_hours for r in medium) / 8),
                    "dependencies": ["Phase 1 completion"],
                }
            )

        # Phase 3: Low priority items
        low = [r for r in recommendations if r.priority == RecommendationPriority.LOW]
        if low:
            phases.append(
                {
                    "phase": 3,
                    "name": "Low Priority Optimizations",
                    "description": "Nice-to-have improvements and fine-tuning",
                    "recommendations": [r.recommendation_id for r in low],
                    "estimated_effort_hours": sum(r.estimated_effort_hours for r in low),
                    "expected_timeline_days": math.ceil(sum(r.estimated_effort_hours for r in low) / 8),
                    "dependencies": ["Phase 2 completion"],
                }
            )

        return phases

    async def _generate_risk_summary(self, recommendations: list[OptimizationRecommendation]) -> str:
        """Generate a risk summary for the optimization plan."""
        high_risk_count = len([r for r in recommendations if "High risk" in r.risk_assessment])
        medium_risk_count = len([r for r in recommendations if "Medium risk" in r.risk_assessment])
        low_risk_count = len([r for r in recommendations if "Low risk" in r.risk_assessment])

        risk_summary = (
            f"Risk Assessment: {high_risk_count} high-risk, {medium_risk_count} medium-risk, {low_risk_count} low-risk recommendations. "
        )

        if high_risk_count > 0:
            risk_summary += "High-risk changes require thorough testing and gradual rollout. "

        if medium_risk_count > 0:
            risk_summary += "Medium-risk changes should be implemented in controlled environments first. "

        risk_summary += "All changes should be monitored closely for performance and stability impact."

        return risk_summary

    def get_optimization_plan(self, cache_name: str) -> OptimizationPlan | None:
        """Get the current optimization plan for a cache."""
        return self._generated_plans.get(cache_name)

    def get_all_optimization_plans(self) -> dict[str, OptimizationPlan]:
        """Get all generated optimization plans."""
        return dict(self._generated_plans)

    async def get_recommendation_summary(self, cache_name: str | None = None) -> dict[str, Any]:
        """Get a summary of recommendations across caches."""
        if cache_name:
            plan = self._generated_plans.get(cache_name)
            if not plan:
                return {"status": "no_plan", "cache_name": cache_name}

            return {
                "status": "success",
                "cache_name": cache_name,
                "recommendation_count": len(plan.recommendations),
                "total_effort_hours": plan.total_estimated_effort_hours,
                "expected_memory_savings_mb": plan.expected_memory_savings_mb,
                "expected_performance_improvement_percent": plan.expected_performance_improvement_percent,
                "implementation_phases": len(plan.implementation_phases),
                "priorities": {
                    "critical": len([r for r in plan.recommendations if r.priority == RecommendationPriority.CRITICAL]),
                    "high": len([r for r in plan.recommendations if r.priority == RecommendationPriority.HIGH]),
                    "medium": len([r for r in plan.recommendations if r.priority == RecommendationPriority.MEDIUM]),
                    "low": len([r for r in plan.recommendations if r.priority == RecommendationPriority.LOW]),
                },
            }
        else:
            # Summary across all caches
            all_plans = list(self._generated_plans.values())
            if not all_plans:
                return {"status": "no_plans"}

            total_recommendations = sum(len(plan.recommendations) for plan in all_plans)
            total_effort = sum(plan.total_estimated_effort_hours for plan in all_plans)
            total_memory_savings = sum(plan.expected_memory_savings_mb for plan in all_plans)
            avg_performance_improvement = statistics.mean([plan.expected_performance_improvement_percent for plan in all_plans])

            return {
                "status": "success",
                "cache_count": len(all_plans),
                "total_recommendations": total_recommendations,
                "total_effort_hours": total_effort,
                "total_expected_memory_savings_mb": total_memory_savings,
                "average_performance_improvement_percent": avg_performance_improvement,
                "caches": list(self._generated_plans.keys()),
            }


# Global optimizer instance
_global_optimizer: CacheMemoryOptimizer | None = None


async def get_memory_optimizer() -> CacheMemoryOptimizer:
    """Get or create the global memory optimizer instance."""
    global _global_optimizer

    if _global_optimizer is None:
        _global_optimizer = CacheMemoryOptimizer()

    return _global_optimizer


async def register_cache_metrics(cache_name: str, metrics: MemoryMetrics) -> None:
    """Register cache metrics for optimization analysis."""
    optimizer = await get_memory_optimizer()
    await optimizer.register_cache_metrics(metrics)


async def register_cache_configuration(cache_name: str, config: CacheConfiguration) -> None:
    """Register cache configuration for optimization analysis."""
    optimizer = await get_memory_optimizer()
    await optimizer.register_cache_config(config)


async def generate_cache_optimization_plan(cache_name: str) -> OptimizationPlan | None:
    """Generate an optimization plan for a specific cache."""
    optimizer = await get_memory_optimizer()
    return await optimizer.generate_optimization_plan(cache_name)


async def get_cache_optimization_summary(cache_name: str | None = None) -> dict[str, Any]:
    """Get optimization summary for a cache or all caches."""
    optimizer = await get_memory_optimizer()
    return await optimizer.get_recommendation_summary(cache_name)
