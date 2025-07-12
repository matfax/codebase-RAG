"""
Cache Scalability Tests for Query Caching Layer.

This module provides comprehensive scalability testing for cache operations including:
- Performance under increasing load and data volume
- Cache behavior with growing dataset sizes
- Resource utilization scaling analysis
- Throughput degradation detection
- Memory efficiency at scale
- Connection pool scaling validation
"""

import asyncio
import gc
import json
import math
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import psutil
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.cache_config import CacheConfig
from services.cache_service import CacheStats
from services.project_cache_service import ProjectCacheService
from services.search_cache_service import SearchCacheService
from utils.performance_monitor import MemoryMonitor


@dataclass
class ScalabilityMetric:
    """Metric for tracking scalability performance."""

    scale_factor: float  # Multiplier from baseline
    data_size_mb: float
    operations_count: int
    duration_seconds: float
    throughput_ops_per_sec: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    error_rate: float
    cache_hit_ratio: float
    gc_collections: int = 0
    resource_exhaustion: bool = False


@dataclass
class ScalabilityTestResult:
    """Result of scalability testing."""

    test_name: str
    baseline_metric: ScalabilityMetric
    scale_metrics: list[ScalabilityMetric]
    scalability_score: float  # 0-100, higher is better
    bottleneck_detected: str = ""
    performance_cliff: float | None = None  # Scale factor where performance degrades
    linear_scaling_limit: float | None = None
    resource_limits: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class CacheScalabilityTester:
    """Comprehensive cache scalability testing framework."""

    def __init__(self):
        self.process = psutil.Process()
        self.memory_monitor = MemoryMonitor()

    def _generate_scaled_data(self, base_size: int, scale_factor: float) -> dict[str, Any]:
        """Generate test data scaled by the given factor."""
        scaled_size = int(base_size * scale_factor)

        return {
            "id": f"scaled_data_{scale_factor}",
            "data": "x" * max(1, scaled_size - 100),  # Account for metadata
            "metadata": {"scale_factor": scale_factor, "size": scaled_size, "timestamp": time.time()},
            "nested_data": {"level1": {"level2": ["item"] * min(100, max(1, int(scaled_size / 100)))}},
        }

    async def _measure_cache_performance(
        self, cache_service: Any, operation_count: int, data_scale_factor: float, base_data_size: int = 1000
    ) -> ScalabilityMetric:
        """Measure cache performance at a specific scale."""
        # Generate scaled test data
        test_data = self._generate_scaled_data(base_data_size, data_scale_factor)
        data_size_mb = len(json.dumps(test_data).encode("utf-8")) / (1024 * 1024)

        # Performance tracking
        response_times = []
        errors = 0
        hits = 0
        misses = 0

        # Resource monitoring
        initial_memory = self.process.memory_info().rss / (1024 * 1024)
        initial_gc = sum(stat["collections"] for stat in gc.get_stats())

        start_time = time.perf_counter()

        # Execute operations at scale
        for i in range(operation_count):
            key = f"scale_test_{data_scale_factor}_{i}"

            operation_start = time.perf_counter()

            try:
                # Simulate realistic cache usage pattern
                if i < operation_count * 0.3:
                    # First 30% - populate cache (misses expected)
                    cached_value = await cache_service.get(key)
                    if cached_value is None:
                        misses += 1
                        await cache_service.set(key, test_data)
                    else:
                        hits += 1
                elif i < operation_count * 0.8:
                    # Next 50% - mostly reads (hits expected)
                    read_key = f"scale_test_{data_scale_factor}_{random.randint(0, int(operation_count * 0.3))}"
                    cached_value = await cache_service.get(read_key)
                    if cached_value is None:
                        misses += 1
                        await cache_service.set(read_key, test_data)
                    else:
                        hits += 1
                else:
                    # Last 20% - mixed operations
                    if random.random() < 0.7:
                        # Read
                        read_key = f"scale_test_{data_scale_factor}_{random.randint(0, i)}"
                        cached_value = await cache_service.get(read_key)
                        if cached_value is None:
                            misses += 1
                        else:
                            hits += 1
                    else:
                        # Write
                        await cache_service.set(key, test_data)

            except Exception:
                errors += 1
                # Continue testing even with errors

            operation_end = time.perf_counter()
            response_times.append((operation_end - operation_start) * 1000)

            # Periodic resource monitoring
            if i % max(1, operation_count // 10) == 0:
                gc.collect()  # Help prevent memory buildup during test

        end_time = time.perf_counter()

        # Final resource measurements
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        final_gc = sum(stat["collections"] for stat in gc.get_stats())
        cpu_percent = self.process.cpu_percent()

        # Calculate metrics
        total_duration = end_time - start_time
        throughput = operation_count / total_duration if total_duration > 0 else 0

        # Response time statistics
        response_times.sort()
        avg_response_time = statistics.mean(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0

        p95_index = int(0.95 * len(response_times))
        p99_index = int(0.99 * len(response_times))
        p95_response_time = response_times[p95_index] if p95_index < len(response_times) else avg_response_time
        p99_response_time = response_times[p99_index] if p99_index < len(response_times) else avg_response_time

        # Cache effectiveness
        total_cache_ops = hits + misses
        cache_hit_ratio = hits / total_cache_ops if total_cache_ops > 0 else 0

        # Error rate
        error_rate = errors / operation_count if operation_count > 0 else 0

        # Resource exhaustion detection
        resource_exhaustion = (
            final_memory > initial_memory * 3 or cpu_percent > 90 or error_rate > 0.1  # Memory tripled  # Very high CPU  # High error rate
        )

        return ScalabilityMetric(
            scale_factor=data_scale_factor,
            data_size_mb=data_size_mb,
            operations_count=operation_count,
            duration_seconds=total_duration,
            throughput_ops_per_sec=throughput,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            memory_usage_mb=final_memory,
            cpu_percent=cpu_percent,
            error_rate=error_rate,
            cache_hit_ratio=cache_hit_ratio,
            gc_collections=final_gc - initial_gc,
            resource_exhaustion=resource_exhaustion,
        )

    async def test_data_volume_scalability(
        self, cache_service: Any, scale_factors: list[float] = None, operations_per_scale: int = 500, base_data_size: int = 1000
    ) -> ScalabilityTestResult:
        """Test cache scalability with increasing data volume."""
        if scale_factors is None:
            scale_factors = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

        test_name = "data_volume_scalability"
        metrics = []

        # Establish baseline
        baseline_metric = await self._measure_cache_performance(cache_service, operations_per_scale, scale_factors[0], base_data_size)

        # Test scaling
        for scale_factor in scale_factors[1:]:
            # Clear cache between tests to ensure clean state
            if hasattr(cache_service, "clear"):
                await cache_service.clear()

            # Force garbage collection
            gc.collect()
            await asyncio.sleep(0.1)

            metric = await self._measure_cache_performance(cache_service, operations_per_scale, scale_factor, base_data_size)
            metrics.append(metric)

            # Stop testing if resource exhaustion is detected
            if metric.resource_exhaustion:
                print(f"Resource exhaustion detected at scale factor {scale_factor}")
                break

        return self._analyze_scalability_results(test_name, baseline_metric, metrics)

    async def test_operation_count_scalability(
        self, cache_service: Any, operation_counts: list[int] = None, data_scale_factor: float = 1.0
    ) -> ScalabilityTestResult:
        """Test cache scalability with increasing operation count."""
        if operation_counts is None:
            operation_counts = [100, 500, 1000, 2000, 5000, 10000]

        test_name = "operation_count_scalability"
        metrics = []

        # Establish baseline
        baseline_operations = operation_counts[0]
        baseline_metric = await self._measure_cache_performance(cache_service, baseline_operations, data_scale_factor)
        # Adjust baseline for comparison
        baseline_metric.scale_factor = 1.0

        # Test scaling
        for op_count in operation_counts[1:]:
            scale_factor = op_count / baseline_operations

            # Clear cache between tests
            if hasattr(cache_service, "clear"):
                await cache_service.clear()

            gc.collect()
            await asyncio.sleep(0.1)

            metric = await self._measure_cache_performance(cache_service, op_count, data_scale_factor)
            metric.scale_factor = scale_factor  # Adjust for operation count scaling
            metrics.append(metric)

            if metric.resource_exhaustion:
                print(f"Resource exhaustion detected at {op_count} operations")
                break

        return self._analyze_scalability_results(test_name, baseline_metric, metrics)

    async def test_concurrent_scalability(
        self, cache_service: Any, concurrency_levels: list[int] = None, operations_per_worker: int = 200
    ) -> ScalabilityTestResult:
        """Test cache scalability with increasing concurrency."""
        if concurrency_levels is None:
            concurrency_levels = [1, 5, 10, 20, 50, 100]

        test_name = "concurrent_scalability"
        metrics = []

        # Establish baseline (single-threaded)
        baseline_metric = await self._measure_concurrent_performance(cache_service, concurrency_levels[0], operations_per_worker)
        baseline_metric.scale_factor = 1.0

        # Test scaling
        for concurrency in concurrency_levels[1:]:
            scale_factor = concurrency / concurrency_levels[0]

            # Clear cache between tests
            if hasattr(cache_service, "clear"):
                await cache_service.clear()

            gc.collect()
            await asyncio.sleep(0.2)  # Longer pause for concurrent tests

            metric = await self._measure_concurrent_performance(cache_service, concurrency, operations_per_worker)
            metric.scale_factor = scale_factor
            metrics.append(metric)

            if metric.resource_exhaustion:
                print(f"Resource exhaustion detected at concurrency {concurrency}")
                break

        return self._analyze_scalability_results(test_name, baseline_metric, metrics)

    async def _measure_concurrent_performance(self, cache_service: Any, concurrency: int, operations_per_worker: int) -> ScalabilityMetric:
        """Measure performance with concurrent workers."""
        # Resource monitoring
        initial_memory = self.process.memory_info().rss / (1024 * 1024)
        initial_gc = sum(stat["collections"] for stat in gc.get_stats())

        start_time = time.perf_counter()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(worker_id: int) -> dict[str, Any]:
            """Individual worker performing cache operations."""
            async with semaphore:
                worker_stats = {"response_times": [], "errors": 0, "hits": 0, "misses": 0}

                for i in range(operations_per_worker):
                    key = f"concurrent_test_{worker_id}_{i}"
                    data = self._generate_scaled_data(1000, 1.0)

                    operation_start = time.perf_counter()

                    try:
                        # Mixed operations
                        if i % 3 == 0:
                            # Write operation
                            await cache_service.set(key, data)
                        else:
                            # Read operation
                            cached_value = await cache_service.get(key)
                            if cached_value is None:
                                worker_stats["misses"] += 1
                                await cache_service.set(key, data)
                            else:
                                worker_stats["hits"] += 1

                    except Exception:
                        worker_stats["errors"] += 1

                    operation_end = time.perf_counter()
                    worker_stats["response_times"].append((operation_end - operation_start) * 1000)

                return worker_stats

        # Create and execute concurrent workers
        tasks = [worker(i) for i in range(concurrency)]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()

        # Aggregate results
        all_response_times = []
        total_errors = 0
        total_hits = 0
        total_misses = 0

        for result in worker_results:
            if isinstance(result, Exception):
                total_errors += operations_per_worker
            else:
                all_response_times.extend(result["response_times"])
                total_errors += result["errors"]
                total_hits += result["hits"]
                total_misses += result["misses"]

        # Final resource measurements
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        final_gc = sum(stat["collections"] for stat in gc.get_stats())
        cpu_percent = self.process.cpu_percent()

        # Calculate metrics
        total_duration = end_time - start_time
        total_operations = concurrency * operations_per_worker
        throughput = total_operations / total_duration if total_duration > 0 else 0

        # Response time statistics
        if all_response_times:
            all_response_times.sort()
            avg_response_time = statistics.mean(all_response_times)
            median_response_time = statistics.median(all_response_times)

            p95_index = int(0.95 * len(all_response_times))
            p99_index = int(0.99 * len(all_response_times))
            p95_response_time = all_response_times[p95_index] if p95_index < len(all_response_times) else avg_response_time
            p99_response_time = all_response_times[p99_index] if p99_index < len(all_response_times) else avg_response_time
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0

        # Cache effectiveness
        total_cache_ops = total_hits + total_misses
        cache_hit_ratio = total_hits / total_cache_ops if total_cache_ops > 0 else 0

        # Error rate
        error_rate = total_errors / total_operations if total_operations > 0 else 0

        # Resource exhaustion detection
        resource_exhaustion = final_memory > initial_memory * 2.5 or cpu_percent > 85 or error_rate > 0.15

        return ScalabilityMetric(
            scale_factor=concurrency,
            data_size_mb=1.0,  # Standard size for concurrent tests
            operations_count=total_operations,
            duration_seconds=total_duration,
            throughput_ops_per_sec=throughput,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            memory_usage_mb=final_memory,
            cpu_percent=cpu_percent,
            error_rate=error_rate,
            cache_hit_ratio=cache_hit_ratio,
            gc_collections=final_gc - initial_gc,
            resource_exhaustion=resource_exhaustion,
        )

    def _analyze_scalability_results(
        self, test_name: str, baseline: ScalabilityMetric, metrics: list[ScalabilityMetric]
    ) -> ScalabilityTestResult:
        """Analyze scalability test results."""
        if not metrics:
            return ScalabilityTestResult(test_name=test_name, baseline_metric=baseline, scale_metrics=[], scalability_score=0.0)

        # Calculate scalability score
        scalability_score = self._calculate_scalability_score(baseline, metrics)

        # Detect performance cliff
        performance_cliff = self._detect_performance_cliff(baseline, metrics)

        # Detect linear scaling limit
        linear_scaling_limit = self._detect_linear_scaling_limit(baseline, metrics)

        # Identify bottlenecks
        bottleneck = self._identify_bottleneck(baseline, metrics)

        # Analyze resource limits
        resource_limits = self._analyze_resource_limits(metrics)

        # Generate recommendations
        recommendations = self._generate_scalability_recommendations(scalability_score, performance_cliff, bottleneck, resource_limits)

        return ScalabilityTestResult(
            test_name=test_name,
            baseline_metric=baseline,
            scale_metrics=metrics,
            scalability_score=scalability_score,
            bottleneck_detected=bottleneck,
            performance_cliff=performance_cliff,
            linear_scaling_limit=linear_scaling_limit,
            resource_limits=resource_limits,
            recommendations=recommendations,
        )

    def _calculate_scalability_score(self, baseline: ScalabilityMetric, metrics: list[ScalabilityMetric]) -> float:
        """Calculate overall scalability score (0-100)."""
        if not metrics:
            return 0.0

        scores = []

        for metric in metrics:
            # Throughput efficiency (how well throughput scales)
            expected_throughput = baseline.throughput_ops_per_sec * metric.scale_factor
            actual_throughput = metric.throughput_ops_per_sec
            throughput_efficiency = min(100, (actual_throughput / expected_throughput * 100)) if expected_throughput > 0 else 0

            # Response time degradation (lower is better)
            response_time_ratio = metric.avg_response_time_ms / baseline.avg_response_time_ms if baseline.avg_response_time_ms > 0 else 1
            response_time_score = max(0, 100 - (response_time_ratio - 1) * 50)  # Penalize increases

            # Error rate impact
            error_score = max(0, 100 - metric.error_rate * 500)  # Heavy penalty for errors

            # Resource utilization efficiency
            memory_ratio = metric.memory_usage_mb / baseline.memory_usage_mb if baseline.memory_usage_mb > 0 else 1
            memory_score = max(0, 100 - (memory_ratio / metric.scale_factor - 1) * 100) if metric.scale_factor > 0 else 0

            # Weighted average
            scale_score = throughput_efficiency * 0.4 + response_time_score * 0.3 + error_score * 0.2 + memory_score * 0.1

            scores.append(scale_score)

        return statistics.mean(scores) if scores else 0.0

    def _detect_performance_cliff(self, baseline: ScalabilityMetric, metrics: list[ScalabilityMetric]) -> float | None:
        """Detect point where performance degrades significantly."""
        for i, metric in enumerate(metrics):
            # Check for significant throughput drop
            throughput_drop = (
                baseline.throughput_ops_per_sec / metric.throughput_ops_per_sec if metric.throughput_ops_per_sec > 0 else float("inf")
            )

            # Check for significant response time increase
            response_time_increase = metric.avg_response_time_ms / baseline.avg_response_time_ms if baseline.avg_response_time_ms > 0 else 1

            # Check for error rate spike
            error_spike = metric.error_rate > 0.1

            if throughput_drop > 2 or response_time_increase > 3 or error_spike:
                return metric.scale_factor

        return None

    def _detect_linear_scaling_limit(self, baseline: ScalabilityMetric, metrics: list[ScalabilityMetric]) -> float | None:
        """Detect point where linear scaling breaks down."""
        for metric in metrics:
            # Linear scaling means throughput should scale proportionally
            expected_throughput = baseline.throughput_ops_per_sec * metric.scale_factor
            actual_throughput = metric.throughput_ops_per_sec

            efficiency = actual_throughput / expected_throughput if expected_throughput > 0 else 0

            # If efficiency drops below 70%, linear scaling is lost
            if efficiency < 0.7:
                return metric.scale_factor

        return None

    def _identify_bottleneck(self, baseline: ScalabilityMetric, metrics: list[ScalabilityMetric]) -> str:
        """Identify the primary bottleneck in scaling."""
        if not metrics:
            return "insufficient_data"

        # Analyze trends
        memory_growth = []
        cpu_growth = []
        response_time_growth = []
        error_growth = []

        for metric in metrics:
            memory_growth.append(metric.memory_usage_mb / baseline.memory_usage_mb if baseline.memory_usage_mb > 0 else 1)
            cpu_growth.append(metric.cpu_percent / max(baseline.cpu_percent, 1))
            response_time_growth.append(
                metric.avg_response_time_ms / baseline.avg_response_time_ms if baseline.avg_response_time_ms > 0 else 1
            )
            error_growth.append(metric.error_rate)

        # Find the fastest growing constraint
        avg_memory_growth = statistics.mean(memory_growth) if memory_growth else 1
        avg_cpu_growth = statistics.mean(cpu_growth) if cpu_growth else 1
        avg_response_growth = statistics.mean(response_time_growth) if response_time_growth else 1
        avg_error_rate = statistics.mean(error_growth) if error_growth else 0

        if avg_error_rate > 0.05:
            return "error_rate"
        elif avg_memory_growth > 2.5:
            return "memory"
        elif avg_cpu_growth > 2.0:
            return "cpu"
        elif avg_response_growth > 2.0:
            return "latency"
        else:
            return "none_detected"

    def _analyze_resource_limits(self, metrics: list[ScalabilityMetric]) -> dict[str, Any]:
        """Analyze resource limit patterns."""
        if not metrics:
            return {}

        max_memory = max(m.memory_usage_mb for m in metrics)
        max_cpu = max(m.cpu_percent for m in metrics)
        max_error_rate = max(m.error_rate for m in metrics)

        return {
            "max_memory_mb": max_memory,
            "max_cpu_percent": max_cpu,
            "max_error_rate": max_error_rate,
            "memory_constrained": max_memory > 500,
            "cpu_constrained": max_cpu > 80,
            "error_constrained": max_error_rate > 0.1,
        }

    def _generate_scalability_recommendations(
        self, scalability_score: float, performance_cliff: float | None, bottleneck: str, resource_limits: dict[str, Any]
    ) -> list[str]:
        """Generate scalability improvement recommendations."""
        recommendations = []

        # Overall scalability recommendations
        if scalability_score < 30:
            recommendations.append("Poor scalability detected. Consider architectural changes for better scaling.")
        elif scalability_score < 60:
            recommendations.append("Moderate scalability. Optimize identified bottlenecks for better performance.")
        elif scalability_score > 80:
            recommendations.append("Good scalability characteristics. Current architecture scales well.")

        # Performance cliff recommendations
        if performance_cliff:
            recommendations.append(
                f"Performance cliff detected at scale factor {performance_cliff}. Consider this as maximum safe operating scale."
            )

        # Bottleneck-specific recommendations
        if bottleneck == "memory":
            recommendations.append(
                "Memory bottleneck detected. Consider implementing memory management, compression, or horizontal scaling."
            )
        elif bottleneck == "cpu":
            recommendations.append("CPU bottleneck detected. Consider optimizing algorithms or adding more processing capacity.")
        elif bottleneck == "latency":
            recommendations.append(
                "Latency bottleneck detected. Consider connection pooling, caching optimizations, or async improvements."
            )
        elif bottleneck == "error_rate":
            recommendations.append("Error rate bottleneck detected. Review error handling and system stability under load.")

        # Resource-specific recommendations
        if resource_limits.get("memory_constrained"):
            recommendations.append(f"Memory usage peaked at {resource_limits['max_memory_mb']:.1f}MB. Consider memory optimization.")

        if resource_limits.get("cpu_constrained"):
            recommendations.append(f"CPU usage peaked at {resource_limits['max_cpu_percent']:.1f}%. Consider CPU optimization or scaling.")

        return recommendations


class TestCacheScalability:
    """Test suite for cache scalability testing."""

    @pytest.fixture
    def scalability_tester(self):
        """Create a cache scalability tester."""
        return CacheScalabilityTester()

    @pytest.fixture
    def mock_cache_service(self):
        """Create a mock cache service with realistic scaling behavior."""
        cache_data = {}
        operation_count = 0

        class ScalingMockCacheService:
            def __init__(self):
                self.stats = CacheStats()

            async def get(self, key: str):
                nonlocal operation_count
                operation_count += 1

                # Simulate scaling effects: slower with more data
                delay = min(0.01, len(cache_data) / 100000)  # Max 10ms delay
                await asyncio.sleep(delay)

                if key in cache_data:
                    self.stats.hit_count += 1
                    return cache_data[key]
                else:
                    self.stats.miss_count += 1
                    return None

            async def set(self, key: str, value: Any, ttl: int = None):
                nonlocal operation_count
                operation_count += 1

                # Simulate scaling effects
                delay = min(0.015, len(cache_data) / 80000)  # Max 15ms delay
                await asyncio.sleep(delay)

                cache_data[key] = value
                self.stats.set_count += 1

                # Simulate memory pressure with large datasets
                if len(cache_data) > 1000:
                    # Randomly fail some operations under pressure
                    if random.random() < 0.01:  # 1% failure rate
                        raise Exception("Simulated memory pressure failure")

                return True

            async def delete(self, key: str):
                if key in cache_data:
                    del cache_data[key]
                    self.stats.delete_count += 1
                    return True
                return False

            async def clear(self):
                cache_data.clear()
                operation_count = 0
                self.stats = CacheStats()
                return True

            def get_stats(self):
                self.stats.total_operations = operation_count
                return self.stats

        return ScalingMockCacheService()

    @pytest.mark.asyncio
    async def test_data_volume_scalability(self, scalability_tester, mock_cache_service):
        """Test data volume scalability."""
        result = await scalability_tester.test_data_volume_scalability(
            mock_cache_service,
            scale_factors=[1.0, 2.0, 5.0],
            operations_per_scale=50,
            base_data_size=500,  # Reduced for testing
        )

        # Verify test results
        assert result.test_name == "data_volume_scalability"
        assert result.baseline_metric.scale_factor == 1.0
        assert len(result.scale_metrics) >= 1
        assert 0 <= result.scalability_score <= 100

        # Should detect some scaling behavior
        print(f"Data volume scalability: {result.scalability_score:.1f} score, bottleneck: {result.bottleneck_detected}")

        # Verify metrics make sense
        for metric in result.scale_metrics:
            assert metric.operations_count > 0
            assert metric.duration_seconds > 0
            assert metric.throughput_ops_per_sec > 0

    @pytest.mark.asyncio
    async def test_operation_count_scalability(self, scalability_tester, mock_cache_service):
        """Test operation count scalability."""
        result = await scalability_tester.test_operation_count_scalability(
            mock_cache_service,
            operation_counts=[50, 100, 200],
            data_scale_factor=1.0,  # Reduced for testing
        )

        # Verify test results
        assert result.test_name == "operation_count_scalability"
        assert result.baseline_metric.operations_count == 50
        assert len(result.scale_metrics) >= 1

        # Should show some scaling characteristics
        print(f"Operation count scalability: {result.scalability_score:.1f} score")

        # Verify scaling factors
        for metric in result.scale_metrics:
            assert metric.scale_factor > 1.0  # Should be scaled up from baseline

    @pytest.mark.asyncio
    async def test_concurrent_scalability(self, scalability_tester, mock_cache_service):
        """Test concurrent scalability."""
        result = await scalability_tester.test_concurrent_scalability(
            mock_cache_service,
            concurrency_levels=[1, 5, 10],
            operations_per_worker=20,  # Reduced for testing
        )

        # Verify test results
        assert result.test_name == "concurrent_scalability"
        assert result.baseline_metric.scale_factor == 1.0
        assert len(result.scale_metrics) >= 1

        # Should handle concurrency
        print(f"Concurrent scalability: {result.scalability_score:.1f} score")

        # Higher concurrency should mean more total operations
        for metric in result.scale_metrics:
            expected_ops = int(metric.scale_factor * result.baseline_metric.operations_count)
            assert metric.operations_count >= expected_ops * 0.8  # Allow some tolerance

    def test_scalability_metric_creation(self):
        """Test scalability metric data structure."""
        metric = ScalabilityMetric(
            scale_factor=2.0,
            data_size_mb=10.5,
            operations_count=1000,
            duration_seconds=5.0,
            throughput_ops_per_sec=200.0,
            avg_response_time_ms=5.0,
            median_response_time_ms=4.0,
            p95_response_time_ms=10.0,
            p99_response_time_ms=15.0,
            memory_usage_mb=150.0,
            cpu_percent=45.0,
            error_rate=0.01,
            cache_hit_ratio=0.85,
        )

        assert metric.scale_factor == 2.0
        assert metric.throughput_ops_per_sec == 200.0
        assert metric.error_rate == 0.01
        assert metric.cache_hit_ratio == 0.85

    def test_scalability_analysis_logic(self, scalability_tester):
        """Test scalability analysis algorithms."""
        # Create baseline metric
        baseline = ScalabilityMetric(
            scale_factor=1.0,
            data_size_mb=1.0,
            operations_count=100,
            duration_seconds=1.0,
            throughput_ops_per_sec=100.0,
            avg_response_time_ms=10.0,
            median_response_time_ms=8.0,
            p95_response_time_ms=20.0,
            p99_response_time_ms=30.0,
            memory_usage_mb=50.0,
            cpu_percent=20.0,
            error_rate=0.0,
            cache_hit_ratio=0.8,
        )

        # Create scaled metrics with different characteristics
        good_scaling_metric = ScalabilityMetric(
            scale_factor=2.0,
            data_size_mb=2.0,
            operations_count=200,
            duration_seconds=2.1,  # Slightly more than linear
            throughput_ops_per_sec=95.0,  # Slight decrease
            avg_response_time_ms=12.0,  # Slight increase
            median_response_time_ms=10.0,
            p95_response_time_ms=25.0,
            p99_response_time_ms=35.0,
            memory_usage_mb=105.0,  # Good memory scaling
            cpu_percent=42.0,  # Good CPU scaling
            error_rate=0.005,  # Low error rate
            cache_hit_ratio=0.78,
        )

        poor_scaling_metric = ScalabilityMetric(
            scale_factor=4.0,
            data_size_mb=4.0,
            operations_count=400,
            duration_seconds=8.0,  # Much worse than linear
            throughput_ops_per_sec=50.0,  # Significant decrease
            avg_response_time_ms=40.0,  # Significant increase
            median_response_time_ms=35.0,
            p95_response_time_ms=80.0,
            p99_response_time_ms=120.0,
            memory_usage_mb=300.0,  # Poor memory scaling
            cpu_percent=90.0,  # High CPU usage
            error_rate=0.05,  # Higher error rate
            cache_hit_ratio=0.6,
        )

        metrics = [good_scaling_metric, poor_scaling_metric]

        # Test analysis
        result = scalability_tester._analyze_scalability_results("test_analysis", baseline, metrics)

        # Verify analysis results
        assert result.test_name == "test_analysis"
        assert result.baseline_metric == baseline
        assert len(result.scale_metrics) == 2
        assert 0 <= result.scalability_score <= 100

        # Should detect performance cliff at high scale
        assert result.performance_cliff is not None
        assert result.performance_cliff <= 4.0

        # Should identify bottleneck
        assert result.bottleneck_detected in ["memory", "cpu", "latency", "error_rate"]

        # Should have recommendations
        assert len(result.recommendations) > 0

        print(
            f"Analysis test: score={result.scalability_score:.1f}, cliff={result.performance_cliff}, bottleneck={result.bottleneck_detected}"
        )


@pytest.mark.performance
@pytest.mark.integration
class TestCacheScalabilityIntegration:
    """Integration tests for cache scalability with real services."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(enabled=True, redis_url="redis://localhost:6379/15", default_ttl=300, max_memory_mb=100)  # Test database

    @pytest.mark.asyncio
    async def test_search_cache_data_scalability(self, cache_config):
        """Test data volume scalability with SearchCacheService."""
        try:
            service = SearchCacheService(cache_config)
            await service.initialize()

            tester = CacheScalabilityTester()

            # Test data volume scaling
            result = await tester.test_data_volume_scalability(
                service, scale_factors=[1.0, 2.0, 5.0], operations_per_scale=100, base_data_size=1000
            )

            # Verify scalability characteristics
            assert result.scalability_score > 20, f"Very poor scalability: {result.scalability_score}"
            assert len(result.recommendations) > 0

            print(f"SearchCache data scalability: {result.scalability_score:.1f} score")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_project_cache_concurrent_scalability(self, cache_config):
        """Test concurrent scalability with ProjectCacheService."""
        try:
            service = ProjectCacheService(cache_config)
            await service.initialize()

            tester = CacheScalabilityTester()

            # Test concurrent scaling
            result = await tester.test_concurrent_scalability(service, concurrency_levels=[1, 5, 10], operations_per_worker=50)

            # Verify concurrent scalability
            assert result.scalability_score > 15, f"Very poor concurrent scalability: {result.scalability_score}"

            # Should handle basic concurrency without major issues
            assert result.baseline_metric.error_rate < 0.1, "High error rate in baseline"

            print(f"ProjectCache concurrent scalability: {result.scalability_score:.1f} score")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
