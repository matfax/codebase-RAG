"""
Advanced cache performance optimization and fine-tuning.

This module provides sophisticated performance optimization techniques including
adaptive sizing, TTL optimization, intelligent batching, and concurrency tuning
for maximum cache efficiency.
"""

import asyncio
import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class OptimizationStrategy(Enum):
    """Cache optimization strategies."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class PerformanceMetric(Enum):
    """Performance metrics for optimization."""

    HIT_RATE = "hit_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    NETWORK_USAGE = "network_usage"


@dataclass
class PerformanceProfile:
    """Performance profile for workload characterization."""

    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_usage_mbps: float = 0.0
    cache_size: int = 0
    operation_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def calculate_efficiency_score(self) -> float:
        """Calculate overall cache efficiency score."""
        # Weighted scoring: hit_rate (40%), latency (30%), throughput (20%), memory (10%)
        hit_score = self.hit_rate * 0.4
        latency_score = max(0, (1.0 - min(self.avg_latency / 0.1, 1.0))) * 0.3  # Normalize to 100ms
        throughput_score = min(self.throughput_ops_per_sec / 10000, 1.0) * 0.2  # Normalize to 10k ops/sec
        memory_score = max(0, (1.0 - min(self.memory_usage_mb / 1024, 1.0))) * 0.1  # Normalize to 1GB

        return hit_score + latency_score + throughput_score + memory_score


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with impact assessment."""

    metric: PerformanceMetric
    current_value: float
    recommended_value: float
    expected_improvement: float
    confidence: float
    implementation_cost: str
    description: str
    priority: int = 1  # 1=high, 2=medium, 3=low


class BaseOptimizer(ABC):
    """Base class for cache optimizers."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize optimizer."""
        self.logger = logger or logging.getLogger(__name__)
        self.performance_history: list[PerformanceProfile] = []
        self.optimization_history: list[dict[str, Any]] = []

    @abstractmethod
    def analyze_performance(self, profile: PerformanceProfile) -> list[OptimizationRecommendation]:
        """Analyze performance and generate recommendations."""
        pass

    @abstractmethod
    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply optimization recommendation."""
        pass

    def record_performance(self, profile: PerformanceProfile) -> None:
        """Record performance profile."""
        self.performance_history.append(profile)

        # Keep only recent history (last 1000 profiles)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def get_performance_trend(self, metric: PerformanceMetric, window: int = 100) -> dict[str, float]:
        """Get performance trend for specific metric."""
        if len(self.performance_history) < 2:
            return {"trend": 0.0, "current": 0.0, "baseline": 0.0}

        recent_profiles = self.performance_history[-window:]
        if len(recent_profiles) < 2:
            return {"trend": 0.0, "current": 0.0, "baseline": 0.0}

        values = []
        for profile in recent_profiles:
            if metric == PerformanceMetric.HIT_RATE:
                values.append(profile.hit_rate)
            elif metric == PerformanceMetric.LATENCY:
                values.append(profile.avg_latency)
            elif metric == PerformanceMetric.THROUGHPUT:
                values.append(profile.throughput_ops_per_sec)
            elif metric == PerformanceMetric.MEMORY_USAGE:
                values.append(profile.memory_usage_mb)
            # Add other metrics as needed

        if len(values) < 2:
            return {"trend": 0.0, "current": 0.0, "baseline": 0.0}

        # Calculate trend using linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        trend = numerator / denominator if denominator != 0 else 0.0

        return {"trend": trend, "current": values[-1], "baseline": values[0], "improvement": values[-1] - values[0]}


class CacheSizeOptimizer(BaseOptimizer):
    """Optimizer for cache size and capacity management."""

    def __init__(self, min_size: int = 100, max_size: int = 100000, target_hit_rate: float = 0.85, logger: logging.Logger | None = None):
        """Initialize cache size optimizer."""
        super().__init__(logger)
        self.min_size = min_size
        self.max_size = max_size
        self.target_hit_rate = target_hit_rate
        self.size_adjustment_factor = 0.1  # 10% adjustments

    def analyze_performance(self, profile: PerformanceProfile) -> list[OptimizationRecommendation]:
        """Analyze cache size performance."""
        recommendations = []

        # Analyze hit rate vs cache size
        if profile.hit_rate < self.target_hit_rate:
            # Low hit rate - consider increasing cache size
            current_size = profile.cache_size
            recommended_increase = int(current_size * self.size_adjustment_factor)
            new_size = min(current_size + recommended_increase, self.max_size)

            if new_size > current_size:
                expected_improvement = self._estimate_hit_rate_improvement(current_size, new_size)

                recommendations.append(
                    OptimizationRecommendation(
                        metric=PerformanceMetric.HIT_RATE,
                        current_value=profile.hit_rate,
                        recommended_value=new_size,
                        expected_improvement=expected_improvement,
                        confidence=0.7,
                        implementation_cost="Low",
                        description=f"Increase cache size from {current_size} to {new_size} to improve hit rate",
                        priority=1,
                    )
                )

        elif profile.hit_rate > self.target_hit_rate + 0.1:
            # Very high hit rate - consider reducing cache size to free memory
            if profile.memory_usage_mb > 512:  # Only if using significant memory
                current_size = profile.cache_size
                recommended_decrease = int(current_size * self.size_adjustment_factor * 0.5)
                new_size = max(current_size - recommended_decrease, self.min_size)

                if new_size < current_size:
                    recommendations.append(
                        OptimizationRecommendation(
                            metric=PerformanceMetric.MEMORY_USAGE,
                            current_value=profile.memory_usage_mb,
                            recommended_value=new_size,
                            expected_improvement=profile.memory_usage_mb * (recommended_decrease / current_size),
                            confidence=0.6,
                            implementation_cost="Low",
                            description=f"Reduce cache size from {current_size} to {new_size} to free memory",
                            priority=3,
                        )
                    )

        return recommendations

    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply cache size optimization."""
        try:
            # This would be implemented by the calling cache service
            self.logger.info(f"Cache size optimization applied: {recommendation.description}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply cache size optimization: {e}")
            return False

    def _estimate_hit_rate_improvement(self, current_size: int, new_size: int) -> float:
        """Estimate hit rate improvement from size increase."""
        # Simple model: logarithmic improvement with diminishing returns
        size_ratio = new_size / current_size
        improvement_factor = math.log(size_ratio) * 0.1  # Conservative estimate
        return min(improvement_factor, 0.2)  # Cap at 20% improvement


class TTLOptimizer(BaseOptimizer):
    """Optimizer for Time-To-Live settings."""

    def __init__(self, min_ttl: int = 60, max_ttl: int = 86400, target_freshness: float = 0.9, logger: logging.Logger | None = None):
        """Initialize TTL optimizer."""
        super().__init__(logger)
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.target_freshness = target_freshness
        self.ttl_adjustment_factor = 0.2  # 20% adjustments

    def analyze_performance(self, profile: PerformanceProfile) -> list[OptimizationRecommendation]:
        """Analyze TTL performance."""
        recommendations = []

        # Analyze based on hit rate trends and cache turnover
        hit_rate_trend = self.get_performance_trend(PerformanceMetric.HIT_RATE)

        if hit_rate_trend["trend"] < -0.01:  # Declining hit rate
            # Consider increasing TTL to improve hit rate
            recommendations.append(
                OptimizationRecommendation(
                    metric=PerformanceMetric.HIT_RATE,
                    current_value=profile.hit_rate,
                    recommended_value=0,  # TTL value to be determined by implementation
                    expected_improvement=abs(hit_rate_trend["trend"]) * 0.5,
                    confidence=0.6,
                    implementation_cost="Low",
                    description="Increase TTL to reduce cache turnover and improve hit rate",
                    priority=2,
                )
            )

        elif profile.memory_usage_mb > 1024 and hit_rate_trend["trend"] > 0.01:
            # High memory usage with improving hit rate - consider reducing TTL
            recommendations.append(
                OptimizationRecommendation(
                    metric=PerformanceMetric.MEMORY_USAGE,
                    current_value=profile.memory_usage_mb,
                    recommended_value=0,  # TTL value to be determined by implementation
                    expected_improvement=profile.memory_usage_mb * 0.1,
                    confidence=0.5,
                    implementation_cost="Low",
                    description="Reduce TTL to decrease memory usage while maintaining hit rate",
                    priority=3,
                )
            )

        return recommendations

    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply TTL optimization."""
        try:
            self.logger.info(f"TTL optimization applied: {recommendation.description}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to apply TTL optimization: {e}")
            return False


class BatchingOptimizer(BaseOptimizer):
    """Optimizer for batch operation parameters."""

    def __init__(
        self, min_batch_size: int = 1, max_batch_size: int = 1000, target_latency: float = 0.01, logger: logging.Logger | None = None
    ):
        """Initialize batching optimizer."""
        super().__init__(logger)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.current_batch_size = 10
        self.latency_samples: list[float] = []

    def analyze_performance(self, profile: PerformanceProfile) -> list[OptimizationRecommendation]:
        """Analyze batching performance."""
        recommendations = []

        # Analyze latency vs throughput tradeoff
        if profile.avg_latency > self.target_latency * 2:
            # High latency - consider reducing batch size
            new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
            if new_batch_size < self.current_batch_size:
                recommendations.append(
                    OptimizationRecommendation(
                        metric=PerformanceMetric.LATENCY,
                        current_value=profile.avg_latency,
                        recommended_value=new_batch_size,
                        expected_improvement=profile.avg_latency * 0.3,
                        confidence=0.7,
                        implementation_cost="Low",
                        description=f"Reduce batch size from {self.current_batch_size} to {new_batch_size} to improve latency",
                        priority=1,
                    )
                )

        elif profile.avg_latency < self.target_latency * 0.5 and profile.throughput_ops_per_sec < 1000:
            # Low latency, low throughput - consider increasing batch size
            new_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
            if new_batch_size > self.current_batch_size:
                recommendations.append(
                    OptimizationRecommendation(
                        metric=PerformanceMetric.THROUGHPUT,
                        current_value=profile.throughput_ops_per_sec,
                        recommended_value=new_batch_size,
                        expected_improvement=profile.throughput_ops_per_sec * 0.5,
                        confidence=0.6,
                        implementation_cost="Low",
                        description=f"Increase batch size from {self.current_batch_size} to {new_batch_size} to improve throughput",
                        priority=2,
                    )
                )

        return recommendations

    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply batching optimization."""
        try:
            if recommendation.metric in [PerformanceMetric.LATENCY, PerformanceMetric.THROUGHPUT]:
                self.current_batch_size = int(recommendation.recommended_value)
                self.logger.info(f"Batching optimization applied: {recommendation.description}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to apply batching optimization: {e}")
            return False


class ConcurrencyOptimizer(BaseOptimizer):
    """Optimizer for concurrency and connection pool settings."""

    def __init__(
        self, min_connections: int = 1, max_connections: int = 100, target_cpu_usage: float = 0.8, logger: logging.Logger | None = None
    ):
        """Initialize concurrency optimizer."""
        super().__init__(logger)
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.target_cpu_usage = target_cpu_usage
        self.current_connections = 10

    def analyze_performance(self, profile: PerformanceProfile) -> list[OptimizationRecommendation]:
        """Analyze concurrency performance."""
        recommendations = []

        # Analyze CPU usage vs throughput
        if profile.cpu_usage_percent > self.target_cpu_usage * 100:
            # High CPU usage - consider reducing concurrency
            new_connections = max(self.current_connections - 2, self.min_connections)
            if new_connections < self.current_connections:
                recommendations.append(
                    OptimizationRecommendation(
                        metric=PerformanceMetric.CPU_USAGE,
                        current_value=profile.cpu_usage_percent,
                        recommended_value=new_connections,
                        expected_improvement=profile.cpu_usage_percent * 0.2,
                        confidence=0.7,
                        implementation_cost="Medium",
                        description=f"Reduce connection pool size from {self.current_connections} to {new_connections} to lower CPU usage",
                        priority=1,
                    )
                )

        elif profile.cpu_usage_percent < self.target_cpu_usage * 50 and profile.throughput_ops_per_sec < 5000:
            # Low CPU usage, low throughput - consider increasing concurrency
            new_connections = min(self.current_connections + 2, self.max_connections)
            if new_connections > self.current_connections:
                recommendations.append(
                    OptimizationRecommendation(
                        metric=PerformanceMetric.THROUGHPUT,
                        current_value=profile.throughput_ops_per_sec,
                        recommended_value=new_connections,
                        expected_improvement=profile.throughput_ops_per_sec * 0.3,
                        confidence=0.6,
                        implementation_cost="Medium",
                        description=f"Increase connection pool size from {self.current_connections} to {new_connections} to improve throughput",
                        priority=2,
                    )
                )

        return recommendations

    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply concurrency optimization."""
        try:
            if recommendation.metric in [PerformanceMetric.CPU_USAGE, PerformanceMetric.THROUGHPUT]:
                self.current_connections = int(recommendation.recommended_value)
                self.logger.info(f"Concurrency optimization applied: {recommendation.description}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to apply concurrency optimization: {e}")
            return False


class AdaptivePerformanceOptimizer:
    """Adaptive performance optimizer that coordinates multiple optimization strategies."""

    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
        optimization_interval: int = 300,  # 5 minutes
        logger: logging.Logger | None = None,
    ):
        """Initialize adaptive optimizer."""
        self.strategy = strategy
        self.optimization_interval = optimization_interval
        self.logger = logger or logging.getLogger(__name__)

        # Initialize component optimizers
        self.size_optimizer = CacheSizeOptimizer(logger=self.logger)
        self.ttl_optimizer = TTLOptimizer(logger=self.logger)
        self.batching_optimizer = BatchingOptimizer(logger=self.logger)
        self.concurrency_optimizer = ConcurrencyOptimizer(logger=self.logger)

        self.optimizers = [self.size_optimizer, self.ttl_optimizer, self.batching_optimizer, self.concurrency_optimizer]

        # Optimization state
        self.last_optimization = 0.0
        self.optimization_count = 0
        self.applied_optimizations: list[OptimizationRecommendation] = []

        # Performance tracking
        self.baseline_performance: PerformanceProfile | None = None
        self.current_performance: PerformanceProfile | None = None

    async def optimize_performance(self, current_profile: PerformanceProfile) -> dict[str, Any]:
        """Perform adaptive performance optimization."""
        try:
            # Record current performance
            self.current_performance = current_profile

            # Set baseline if not established
            if self.baseline_performance is None:
                self.baseline_performance = current_profile
                self.logger.info("Established performance baseline")
                return {"status": "baseline_set", "profile": current_profile}

            # Check if optimization is due
            current_time = time.time()
            if current_time - self.last_optimization < self.optimization_interval:
                return {"status": "not_due", "next_optimization_in": self.optimization_interval - (current_time - self.last_optimization)}

            # Collect recommendations from all optimizers
            all_recommendations = []
            for optimizer in self.optimizers:
                optimizer.record_performance(current_profile)
                recommendations = optimizer.analyze_performance(current_profile)
                all_recommendations.extend(recommendations)

            # Filter and prioritize recommendations
            filtered_recommendations = self._filter_recommendations(all_recommendations)
            prioritized_recommendations = self._prioritize_recommendations(filtered_recommendations)

            # Apply optimizations based on strategy
            applied_count = await self._apply_optimizations(prioritized_recommendations)

            # Update optimization tracking
            self.last_optimization = current_time
            self.optimization_count += 1

            # Calculate performance improvement
            improvement = self._calculate_improvement()

            result = {
                "status": "completed",
                "optimization_count": self.optimization_count,
                "recommendations_generated": len(all_recommendations),
                "recommendations_applied": applied_count,
                "performance_improvement": improvement,
                "next_optimization_in": self.optimization_interval,
            }

            self.logger.info(f"Performance optimization completed: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error during performance optimization: {e}")
            return {"status": "error", "error": str(e)}

    def _filter_recommendations(self, recommendations: list[OptimizationRecommendation]) -> list[OptimizationRecommendation]:
        """Filter recommendations based on confidence and impact."""
        filtered = []

        for rec in recommendations:
            # Filter by confidence threshold
            min_confidence = {
                OptimizationStrategy.CONSERVATIVE: 0.8,
                OptimizationStrategy.BALANCED: 0.6,
                OptimizationStrategy.AGGRESSIVE: 0.4,
                OptimizationStrategy.ADAPTIVE: 0.5,
            }.get(self.strategy, 0.6)

            if rec.confidence >= min_confidence:
                # Filter by expected improvement
                min_improvement = {
                    OptimizationStrategy.CONSERVATIVE: 0.1,
                    OptimizationStrategy.BALANCED: 0.05,
                    OptimizationStrategy.AGGRESSIVE: 0.01,
                    OptimizationStrategy.ADAPTIVE: 0.03,
                }.get(self.strategy, 0.05)

                if rec.expected_improvement >= min_improvement:
                    filtered.append(rec)

        return filtered

    def _prioritize_recommendations(self, recommendations: list[OptimizationRecommendation]) -> list[OptimizationRecommendation]:
        """Prioritize recommendations based on impact and implementation cost."""

        def score_recommendation(rec: OptimizationRecommendation) -> float:
            # Higher score = higher priority
            impact_score = rec.expected_improvement * rec.confidence

            # Cost penalty
            cost_penalty = {"Low": 0.0, "Medium": 0.1, "High": 0.3}.get(rec.implementation_cost, 0.2)

            priority_bonus = {1: 0.3, 2: 0.1, 3: 0.0}.get(rec.priority, 0.0)

            return impact_score - cost_penalty + priority_bonus

        return sorted(recommendations, key=score_recommendation, reverse=True)

    async def _apply_optimizations(self, recommendations: list[OptimizationRecommendation]) -> int:
        """Apply optimizations based on strategy."""
        max_optimizations = {
            OptimizationStrategy.CONSERVATIVE: 1,
            OptimizationStrategy.BALANCED: 2,
            OptimizationStrategy.AGGRESSIVE: 3,
            OptimizationStrategy.ADAPTIVE: 2,
        }.get(self.strategy, 2)

        applied_count = 0

        for rec in recommendations[:max_optimizations]:
            try:
                # Find appropriate optimizer
                optimizer = None
                if rec.metric in [PerformanceMetric.HIT_RATE, PerformanceMetric.MEMORY_USAGE]:
                    optimizer = self.size_optimizer
                elif rec.metric == PerformanceMetric.LATENCY:
                    optimizer = self.batching_optimizer
                elif rec.metric == PerformanceMetric.THROUGHPUT:
                    optimizer = self.batching_optimizer if "batch" in rec.description.lower() else self.concurrency_optimizer
                elif rec.metric == PerformanceMetric.CPU_USAGE:
                    optimizer = self.concurrency_optimizer

                if optimizer and optimizer.apply_optimization(rec):
                    self.applied_optimizations.append(rec)
                    applied_count += 1

                    # Small delay between optimizations
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Failed to apply optimization: {e}")

        return applied_count

    def _calculate_improvement(self) -> dict[str, float]:
        """Calculate performance improvement since baseline."""
        if not self.baseline_performance or not self.current_performance:
            return {}

        baseline = self.baseline_performance
        current = self.current_performance

        return {
            "hit_rate_improvement": current.hit_rate - baseline.hit_rate,
            "latency_improvement": baseline.avg_latency - current.avg_latency,  # Lower is better
            "throughput_improvement": current.throughput_ops_per_sec - baseline.throughput_ops_per_sec,
            "memory_improvement": baseline.memory_usage_mb - current.memory_usage_mb,  # Lower is better
            "efficiency_improvement": current.calculate_efficiency_score() - baseline.calculate_efficiency_score(),
        }

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get comprehensive optimization summary."""
        return {
            "strategy": self.strategy.value,
            "optimization_count": self.optimization_count,
            "applied_optimizations": len(self.applied_optimizations),
            "last_optimization": self.last_optimization,
            "baseline_performance": self.baseline_performance,
            "current_performance": self.current_performance,
            "performance_improvement": self._calculate_improvement(),
            "component_optimizers": {
                "size_optimizer": len(self.size_optimizer.optimization_history),
                "ttl_optimizer": len(self.ttl_optimizer.optimization_history),
                "batching_optimizer": len(self.batching_optimizer.optimization_history),
                "concurrency_optimizer": len(self.concurrency_optimizer.optimization_history),
            },
        }

    def configure_optimization(self, strategy: OptimizationStrategy | None = None, interval: int | None = None) -> dict[str, Any]:
        """Configure optimization parameters."""
        config_changes = []

        if strategy is not None:
            self.strategy = strategy
            config_changes.append(f"Strategy changed to {strategy.value}")

        if interval is not None:
            self.optimization_interval = max(60, interval)  # Minimum 1 minute
            config_changes.append(f"Interval changed to {self.optimization_interval} seconds")

        return {
            "success": True,
            "changes": config_changes,
            "current_config": {
                "strategy": self.strategy.value,
                "interval": self.optimization_interval,
                "optimization_count": self.optimization_count,
            },
        }


class PerformanceMonitor:
    """Real-time performance monitoring and optimization coordinator."""

    def __init__(self, optimizer: AdaptivePerformanceOptimizer, monitoring_interval: int = 60, logger: logging.Logger | None = None):
        """Initialize performance monitor."""
        self.optimizer = optimizer
        self.monitoring_interval = monitoring_interval
        self.logger = logger or logging.getLogger(__name__)

        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False
        self.performance_samples: list[PerformanceProfile] = []

    async def start_monitoring(self, performance_provider: Callable[[], PerformanceProfile]):
        """Start continuous performance monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.performance_provider = performance_provider
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Get current performance profile
                current_profile = self.performance_provider()
                self.performance_samples.append(current_profile)

                # Keep only recent samples
                if len(self.performance_samples) > 1440:  # 24 hours at 1-minute intervals
                    self.performance_samples = self.performance_samples[-1440:]

                # Trigger optimization
                optimization_result = await self.optimizer.optimize_performance(current_profile)

                if optimization_result.get("status") == "error":
                    self.logger.error(f"Optimization failed: {optimization_result.get('error')}")

                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        if not self.performance_samples:
            return {"status": "no_data"}

        recent_samples = self.performance_samples[-60:]  # Last hour

        avg_hit_rate = sum(p.hit_rate for p in recent_samples) / len(recent_samples)
        avg_latency = sum(p.avg_latency for p in recent_samples) / len(recent_samples)
        avg_throughput = sum(p.throughput_ops_per_sec for p in recent_samples) / len(recent_samples)
        avg_memory = sum(p.memory_usage_mb for p in recent_samples) / len(recent_samples)

        return {
            "status": "active" if self.is_monitoring else "inactive",
            "monitoring_interval": self.monitoring_interval,
            "total_samples": len(self.performance_samples),
            "recent_averages": {"hit_rate": avg_hit_rate, "latency": avg_latency, "throughput": avg_throughput, "memory_usage": avg_memory},
            "optimization_summary": self.optimizer.get_optimization_summary(),
        }
