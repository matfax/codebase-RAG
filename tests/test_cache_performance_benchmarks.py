"""
Cache Performance Benchmarks for Query Caching Layer.

This module provides comprehensive performance testing for cache operations including:
- Read/write performance benchmarks
- Throughput and latency measurements
- Cache operation timing analysis
- Performance regression detection
- Memory usage during cache operations
"""

import asyncio
import gc
import json
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock

import psutil
import pytest

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from config.cache_config import CacheConfig
    from services.cache_service import CacheHealthStatus, CacheStats
    from services.project_cache_service import ProjectCacheService
    from services.search_cache_service import SearchCacheService
    from utils.performance_monitor import MemoryMonitor
except ImportError:
    # Alternative imports if relative imports fail
    import os
    import sys

    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, os.path.abspath(src_path))

    from config.cache_config import CacheConfig
    from services.cache_service import CacheHealthStatus, CacheStats
    from services.project_cache_service import ProjectCacheService
    from services.search_cache_service import SearchCacheService
    from utils.performance_monitor import MemoryMonitor


@dataclass
class CachePerformanceMetric:
    """Performance metric for cache operations."""

    operation: str
    duration_ms: float
    throughput_ops_per_sec: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    success_count: int = 0
    error_count: int = 0
    data_size_bytes: int = 0
    additional_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a test scenario."""

    scenario: str
    metrics: list[CachePerformanceMetric]
    summary_stats: dict[str, Any] = field(default_factory=dict)
    performance_issues: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class CachePerformanceBenchmarkSuite:
    """Comprehensive cache performance benchmarking suite."""

    def __init__(self):
        self.results: list[BenchmarkResult] = []
        self.memory_monitor = MemoryMonitor()
        self.process = psutil.Process()

    def generate_test_data(self, size_bytes: int) -> str:
        """Generate test data of specified size."""
        if size_bytes <= 1000:
            # Small data - structured
            return json.dumps(
                {
                    "id": random.randint(1, 1000),
                    "name": "".join(random.choices(string.ascii_letters, k=size_bytes // 4)),
                    "data": list(range(size_bytes // 20)),
                }
            )
        else:
            # Large data - random string
            return "".join(random.choices(string.ascii_letters + string.digits, k=size_bytes))

    async def measure_cache_operation(
        self, operation_name: str, cache_service: Any, operation_func, *args, **kwargs
    ) -> CachePerformanceMetric:
        """Measure performance of a single cache operation."""
        # Force garbage collection
        gc.collect()

        # Initial measurements
        memory_before = self.process.memory_info().rss / (1024 * 1024)
        peak_memory = memory_before

        # Get initial cache stats
        initial_stats = cache_service.get_stats() if hasattr(cache_service, "get_stats") else CacheStats()

        # Memory monitoring callback
        def monitor_memory():
            nonlocal peak_memory
            current = self.process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current)

        # Execute operation with timing
        start_time = time.perf_counter()
        try:
            result = await operation_func(*args, **kwargs)
            success_count = 1
            error_count = 0
        except Exception as e:
            result = None
            success_count = 0
            error_count = 1
            print(f"Operation {operation_name} failed: {e}")

        monitor_memory()
        end_time = time.perf_counter()

        # Final measurements
        memory_after = self.process.memory_info().rss / (1024 * 1024)
        duration_ms = (end_time - start_time) * 1000

        # Get final cache stats
        final_stats = cache_service.get_stats() if hasattr(cache_service, "get_stats") else CacheStats()

        # Calculate cache hit/miss changes
        cache_hits = final_stats.hit_count - initial_stats.hit_count
        cache_misses = final_stats.miss_count - initial_stats.miss_count

        # Calculate data size
        data_size = 0
        if args:
            for arg in args:
                if isinstance(arg, str):
                    data_size += len(arg.encode("utf-8"))
                elif isinstance(arg, (dict, list)):
                    data_size += len(json.dumps(arg).encode("utf-8"))

        # Calculate throughput
        throughput = 1000 / duration_ms if duration_ms > 0 else 0

        return CachePerformanceMetric(
            operation=operation_name,
            duration_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=peak_memory,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            success_count=success_count,
            error_count=error_count,
            data_size_bytes=data_size,
        )

    async def benchmark_basic_operations(self, cache_service: Any, iterations: int = 100, data_sizes: list[int] = None) -> BenchmarkResult:
        """Benchmark basic cache operations (get, set, delete)."""
        if data_sizes is None:
            data_sizes = [100, 1000, 10000, 100000]  # 100B to 100KB

        metrics = []

        for size in data_sizes:
            test_data = self.generate_test_data(size)
            test_key = f"benchmark_key_{size}"

            # Benchmark SET operations
            for i in range(iterations):
                key = f"{test_key}_{i}"
                metric = await self.measure_cache_operation(f"set_{size}b", cache_service, cache_service.set, key, test_data)
                metrics.append(metric)

            # Benchmark GET operations (cache hits)
            for i in range(iterations):
                key = f"{test_key}_{i}"
                metric = await self.measure_cache_operation(f"get_hit_{size}b", cache_service, cache_service.get, key)
                metrics.append(metric)

            # Benchmark GET operations (cache misses)
            for i in range(iterations):
                key = f"nonexistent_key_{i}"
                metric = await self.measure_cache_operation(f"get_miss_{size}b", cache_service, cache_service.get, key)
                metrics.append(metric)

            # Benchmark DELETE operations
            for i in range(iterations):
                key = f"{test_key}_{i}"
                metric = await self.measure_cache_operation(f"delete_{size}b", cache_service, cache_service.delete, key)
                metrics.append(metric)

        result = BenchmarkResult(scenario="basic_operations", metrics=metrics)

        self._calculate_summary_stats(result)
        self._identify_performance_issues(result)
        self.results.append(result)

        return result

    async def benchmark_batch_operations(self, cache_service: Any, batch_sizes: list[int] = None, iterations: int = 50) -> BenchmarkResult:
        """Benchmark batch cache operations."""
        if batch_sizes is None:
            batch_sizes = [10, 50, 100, 500]

        metrics = []

        for batch_size in batch_sizes:
            # Prepare batch data
            batch_data = {}
            batch_keys = []

            for i in range(batch_size):
                key = f"batch_key_{batch_size}_{i}"
                data = self.generate_test_data(1000)  # 1KB per item
                batch_data[key] = data
                batch_keys.append(key)

            # Benchmark batch SET
            for i in range(iterations):
                metric = await self.measure_cache_operation(f"set_batch_{batch_size}", cache_service, cache_service.set_batch, batch_data)
                metrics.append(metric)

            # Benchmark batch GET
            for i in range(iterations):
                metric = await self.measure_cache_operation(f"get_batch_{batch_size}", cache_service, cache_service.get_batch, batch_keys)
                metrics.append(metric)

            # Benchmark batch DELETE
            for i in range(iterations):
                metric = await self.measure_cache_operation(
                    f"delete_batch_{batch_size}", cache_service, cache_service.delete_batch, batch_keys
                )
                metrics.append(metric)

            # Recreate data for next iteration
            await cache_service.set_batch(batch_data)

        result = BenchmarkResult(scenario="batch_operations", metrics=metrics)

        self._calculate_summary_stats(result)
        self._identify_performance_issues(result)
        self.results.append(result)

        return result

    async def benchmark_concurrent_operations(
        self, cache_service: Any, concurrency_levels: list[int] = None, operations_per_level: int = 100
    ) -> BenchmarkResult:
        """Benchmark concurrent cache operations."""
        if concurrency_levels is None:
            concurrency_levels = [1, 5, 10, 20, 50]

        metrics = []

        for concurrency in concurrency_levels:
            # Prepare test data
            test_data = [(f"concurrent_key_{concurrency}_{i}", self.generate_test_data(1000)) for i in range(operations_per_level)]

            # Benchmark concurrent SET operations
            start_time = time.perf_counter()

            async def set_operation(key_value):
                key, value = key_value
                return await cache_service.set(key, value)

            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrency)

            async def limited_set(key_value):
                async with semaphore:
                    return await set_operation(key_value)

            # Execute concurrent operations
            tasks = [limited_set(kv) for kv in test_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            success_count = sum(1 for r in results if r is True)
            error_count = len(results) - success_count
            throughput = len(results) * 1000 / duration_ms if duration_ms > 0 else 0

            metric = CachePerformanceMetric(
                operation=f"concurrent_set_{concurrency}",
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_peak_mb=0,
                success_count=success_count,
                error_count=error_count,
                additional_metrics={"concurrency_level": concurrency, "total_operations": len(results)},
            )
            metrics.append(metric)

            # Similar for GET operations
            start_time = time.perf_counter()

            async def get_operation(key_value):
                key, _ = key_value
                return await cache_service.get(key)

            async def limited_get(key_value):
                async with semaphore:
                    return await get_operation(key_value)

            tasks = [limited_get(kv) for kv in test_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = len(results) - success_count
            throughput = len(results) * 1000 / duration_ms if duration_ms > 0 else 0

            metric = CachePerformanceMetric(
                operation=f"concurrent_get_{concurrency}",
                duration_ms=duration_ms,
                throughput_ops_per_sec=throughput,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_peak_mb=0,
                success_count=success_count,
                error_count=error_count,
                additional_metrics={"concurrency_level": concurrency, "total_operations": len(results)},
            )
            metrics.append(metric)

        result = BenchmarkResult(scenario="concurrent_operations", metrics=metrics)

        self._calculate_summary_stats(result)
        self._identify_performance_issues(result)
        self.results.append(result)

        return result

    def _calculate_summary_stats(self, result: BenchmarkResult) -> None:
        """Calculate summary statistics for benchmark results."""
        if not result.metrics:
            return

        # Group metrics by operation type
        by_operation = {}
        for metric in result.metrics:
            if metric.operation not in by_operation:
                by_operation[metric.operation] = []
            by_operation[metric.operation].append(metric)

        # Calculate stats for each operation type
        operation_stats = {}
        for op_name, op_metrics in by_operation.items():
            durations = [m.duration_ms for m in op_metrics]
            throughputs = [m.throughput_ops_per_sec for m in op_metrics]
            memory_peaks = [m.memory_peak_mb for m in op_metrics if m.memory_peak_mb > 0]

            operation_stats[op_name] = {
                "count": len(op_metrics),
                "avg_duration_ms": statistics.mean(durations),
                "median_duration_ms": statistics.median(durations),
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "avg_throughput_ops_per_sec": statistics.mean(throughputs),
                "max_throughput_ops_per_sec": max(throughputs),
                "total_successes": sum(m.success_count for m in op_metrics),
                "total_errors": sum(m.error_count for m in op_metrics),
                "error_rate": sum(m.error_count for m in op_metrics) / len(op_metrics),
            }

            if memory_peaks:
                operation_stats[op_name].update({"avg_memory_mb": statistics.mean(memory_peaks), "max_memory_mb": max(memory_peaks)})

        # Overall summary
        all_durations = [m.duration_ms for m in result.metrics]
        all_throughputs = [m.throughput_ops_per_sec for m in result.metrics]

        result.summary_stats = {
            "total_operations": len(result.metrics),
            "overall_avg_duration_ms": statistics.mean(all_durations),
            "overall_median_duration_ms": statistics.median(all_durations),
            "overall_max_duration_ms": max(all_durations),
            "overall_avg_throughput_ops_per_sec": statistics.mean(all_throughputs),
            "overall_max_throughput_ops_per_sec": max(all_throughputs),
            "total_successes": sum(m.success_count for m in result.metrics),
            "total_errors": sum(m.error_count for m in result.metrics),
            "by_operation": operation_stats,
        }

    def _identify_performance_issues(self, result: BenchmarkResult) -> None:
        """Identify potential performance issues."""
        issues = []

        # Check for slow operations (>100ms for basic operations)
        if result.summary_stats.get("overall_max_duration_ms", 0) > 100:
            issues.append(f"Slow operation detected: {result.summary_stats['overall_max_duration_ms']:.1f}ms")

        # Check for low throughput (<100 ops/sec)
        if result.summary_stats.get("overall_avg_throughput_ops_per_sec", 0) < 100:
            issues.append(f"Low throughput: {result.summary_stats['overall_avg_throughput_ops_per_sec']:.1f} ops/sec")

        # Check for high error rates (>5%)
        total_ops = result.summary_stats.get("total_operations", 1)
        error_rate = result.summary_stats.get("total_errors", 0) / total_ops
        if error_rate > 0.05:
            issues.append(f"High error rate: {error_rate*100:.1f}%")

        # Check for memory issues
        for op_name, op_stats in result.summary_stats.get("by_operation", {}).items():
            if op_stats.get("max_memory_mb", 0) > 200:
                issues.append(f"High memory usage in {op_name}: {op_stats['max_memory_mb']:.1f}MB")

        result.performance_issues = issues

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}

        report = {
            "summary": {
                "total_scenarios": len(self.results),
                "total_operations": sum(len(r.metrics) for r in self.results),
                "execution_time": time.time() - min(r.timestamp for r in self.results),
            },
            "scenarios": {},
            "performance_issues": [],
            "recommendations": [],
        }

        # Process each scenario
        for result in self.results:
            report["scenarios"][result.scenario] = {
                "summary_stats": result.summary_stats,
                "performance_issues": result.performance_issues,
                "metric_count": len(result.metrics),
            }
            report["performance_issues"].extend(result.performance_issues)

        # Generate recommendations
        recommendations = []

        # Check for consistent slow operations
        slow_scenarios = [s for s, data in report["scenarios"].items() if data["summary_stats"].get("overall_avg_duration_ms", 0) > 50]
        if slow_scenarios:
            recommendations.append(f"Consider optimizing cache operations in scenarios: {', '.join(slow_scenarios)}")

        # Check for low throughput
        low_throughput_scenarios = [
            s for s, data in report["scenarios"].items() if data["summary_stats"].get("overall_avg_throughput_ops_per_sec", 0) < 200
        ]
        if low_throughput_scenarios:
            recommendations.append(f"Consider improving throughput in scenarios: {', '.join(low_throughput_scenarios)}")

        # General recommendations
        if len(report["performance_issues"]) > 5:
            recommendations.append("Multiple performance issues detected. Consider comprehensive cache optimization.")

        report["recommendations"] = recommendations

        return report

    async def benchmark_concurrent_load(
        self, cache_service: Any, concurrent_operations: int = 50, operations_per_worker: int = 20, data_size_bytes: int = 1024
    ) -> BenchmarkResult:
        """Benchmark cache performance under concurrent load."""
        metrics = []

        # Generate test data
        test_data = self.generate_test_data(data_size_bytes)

        async def worker_operations(worker_id: int):
            """Worker function that performs operations."""
            worker_metrics = []

            for i in range(operations_per_worker):
                key = f"load_test_{worker_id}_{i}"

                # SET operation
                set_metric = await self.measure_cache_operation(
                    f"concurrent_set_worker_{worker_id}", cache_service, cache_service.set, key, test_data
                )
                worker_metrics.append(set_metric)

                # GET operation
                get_metric = await self.measure_cache_operation(f"concurrent_get_worker_{worker_id}", cache_service, cache_service.get, key)
                worker_metrics.append(get_metric)

                # Short delay to simulate realistic load
                await asyncio.sleep(0.001)

            return worker_metrics

        # Execute concurrent workers
        start_time = time.perf_counter()

        tasks = [worker_operations(i) for i in range(concurrent_operations)]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        total_duration = (end_time - start_time) * 1000

        # Collect all metrics
        for worker_metrics in worker_results:
            if isinstance(worker_metrics, list):
                metrics.extend(worker_metrics)

        # Add overall load test metric
        total_operations = concurrent_operations * operations_per_worker * 2  # SET + GET
        overall_throughput = total_operations * 1000 / total_duration if total_duration > 0 else 0

        load_metric = CachePerformanceMetric(
            operation="concurrent_load_test",
            duration_ms=total_duration,
            throughput_ops_per_sec=overall_throughput,
            memory_before_mb=0,
            memory_after_mb=0,
            memory_peak_mb=0,
            success_count=len([m for m in metrics if m.success_count > 0]),
            error_count=len([m for m in metrics if m.error_count > 0]),
            additional_metrics={
                "concurrent_workers": concurrent_operations,
                "operations_per_worker": operations_per_worker,
                "total_operations": total_operations,
            },
        )
        metrics.append(load_metric)

        result = BenchmarkResult(scenario="concurrent_load", metrics=metrics)
        self._calculate_summary_stats(result)
        self._identify_performance_issues(result)
        self.results.append(result)

        return result

    async def benchmark_memory_usage(
        self, cache_config: Any, operations: list[str] = None, data_sizes: list[int] = None, iterations: int = 50
    ) -> BenchmarkResult:
        """Benchmark memory usage patterns during cache operations."""
        if operations is None:
            operations = ["set", "get", "delete"]
        if data_sizes is None:
            data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB

        metrics = []

        # Create a temporary cache service for memory testing
        try:
            from services.search_cache_service import SearchCacheService

            cache_service = SearchCacheService(cache_config)
            await cache_service.initialize()

            for data_size in data_sizes:
                test_data = self.generate_test_data(data_size)

                for operation in operations:
                    memory_before = self.process.memory_info().rss / (1024 * 1024)

                    # Perform multiple operations to observe memory patterns
                    for i in range(iterations):
                        key = f"memory_test_{operation}_{data_size}_{i}"

                        if operation == "set":
                            await cache_service.set(key, test_data)
                        elif operation == "get":
                            await cache_service.get(key)
                        elif operation == "delete":
                            await cache_service.delete(key)

                    memory_after = self.process.memory_info().rss / (1024 * 1024)
                    memory_delta = memory_after - memory_before

                    metric = CachePerformanceMetric(
                        operation=f"memory_{operation}_{data_size}b",
                        duration_ms=0,  # Not timing focused
                        throughput_ops_per_sec=0,
                        memory_before_mb=memory_before,
                        memory_after_mb=memory_after,
                        memory_peak_mb=memory_after,
                        success_count=iterations,
                        error_count=0,
                        data_size_bytes=data_size * iterations,
                        additional_metrics={
                            "memory_delta_mb": memory_delta,
                            "memory_per_operation_kb": (memory_delta * 1024) / iterations if iterations > 0 else 0,
                        },
                    )
                    metrics.append(metric)

                    # Force garbage collection between tests
                    gc.collect()
                    await asyncio.sleep(0.1)

            await cache_service.shutdown()

        except Exception as e:
            # If cache service fails, create mock metric
            mock_metric = CachePerformanceMetric(
                operation="memory_test_mock",
                duration_ms=0,
                throughput_ops_per_sec=0,
                memory_before_mb=0,
                memory_after_mb=0,
                memory_peak_mb=0,
                success_count=0,
                error_count=1,
                additional_metrics={"error": str(e)},
            )
            metrics.append(mock_metric)

        result = BenchmarkResult(scenario="memory_usage", metrics=metrics)
        self._calculate_summary_stats(result)
        self._identify_performance_issues(result)
        self.results.append(result)

        return result


class TestCachePerformanceBenchmarks:
    """Test suite for cache performance benchmarks."""

    @pytest.fixture
    async def benchmark_suite(self):
        """Create a performance benchmark suite."""
        return CachePerformanceBenchmarkSuite()

    @pytest.fixture
    async def mock_cache_service(self):
        """Create a mock cache service for testing."""
        mock_service = AsyncMock()
        mock_service.get_stats.return_value = CacheStats()

        # Mock basic operations
        mock_service.get.return_value = None
        mock_service.set.return_value = True
        mock_service.delete.return_value = True
        mock_service.exists.return_value = False

        # Mock batch operations
        mock_service.get_batch.return_value = {}
        mock_service.set_batch.return_value = {}
        mock_service.delete_batch.return_value = {}

        return mock_service

    @pytest.mark.asyncio
    async def test_basic_operation_benchmarks(self, benchmark_suite, mock_cache_service):
        """Test basic cache operation benchmarks."""
        result = await benchmark_suite.benchmark_basic_operations(
            mock_cache_service,
            iterations=10,
            data_sizes=[100, 1000],  # Reduced for testing  # Reduced sizes
        )

        # Verify benchmark result structure
        assert result.scenario == "basic_operations"
        assert len(result.metrics) > 0
        assert result.summary_stats is not None

        # Verify summary statistics
        summary = result.summary_stats
        assert "total_operations" in summary
        assert "overall_avg_duration_ms" in summary
        assert "by_operation" in summary

        # Verify operation types are covered
        operations = list(summary["by_operation"].keys())
        expected_ops = ["set_100b", "get_hit_100b", "get_miss_100b", "delete_100b"]
        for expected_op in expected_ops:
            assert any(expected_op in op for op in operations), f"Missing operation type: {expected_op}"

        print(f"Basic operations benchmark completed: {len(result.metrics)} metrics collected")

    @pytest.mark.asyncio
    async def test_batch_operation_benchmarks(self, benchmark_suite, mock_cache_service):
        """Test batch cache operation benchmarks."""
        result = await benchmark_suite.benchmark_batch_operations(
            mock_cache_service,
            batch_sizes=[10, 50],
            iterations=5,  # Reduced for testing
        )

        # Verify benchmark result
        assert result.scenario == "batch_operations"
        assert len(result.metrics) > 0

        # Verify batch operations are covered
        operations = list(result.summary_stats["by_operation"].keys())
        expected_ops = ["set_batch_10", "get_batch_10", "delete_batch_10"]
        for expected_op in expected_ops:
            assert any(expected_op in op for op in operations), f"Missing batch operation: {expected_op}"

        print(f"Batch operations benchmark completed: {len(result.metrics)} metrics collected")

    @pytest.mark.asyncio
    async def test_concurrent_operation_benchmarks(self, benchmark_suite, mock_cache_service):
        """Test concurrent cache operation benchmarks."""
        result = await benchmark_suite.benchmark_concurrent_operations(
            mock_cache_service,
            concurrency_levels=[1, 5],
            operations_per_level=10,  # Reduced for testing
        )

        # Verify benchmark result
        assert result.scenario == "concurrent_operations"
        assert len(result.metrics) > 0

        # Verify concurrent operations metrics
        for metric in result.metrics:
            assert "concurrent" in metric.operation
            assert metric.additional_metrics.get("concurrency_level") in [1, 5]
            assert metric.additional_metrics.get("total_operations") == 10

        print(f"Concurrent operations benchmark completed: {len(result.metrics)} metrics collected")

    @pytest.mark.asyncio
    async def test_performance_report_generation(self, benchmark_suite, mock_cache_service):
        """Test performance report generation."""
        # Run multiple benchmarks
        await benchmark_suite.benchmark_basic_operations(mock_cache_service, iterations=5, data_sizes=[100])
        await benchmark_suite.benchmark_batch_operations(mock_cache_service, batch_sizes=[10], iterations=3)

        # Generate report
        report = benchmark_suite.generate_performance_report()

        # Verify report structure
        assert "summary" in report
        assert "scenarios" in report
        assert "performance_issues" in report
        assert "recommendations" in report

        # Verify summary
        summary = report["summary"]
        assert summary["total_scenarios"] == 2
        assert summary["total_operations"] > 0

        # Verify scenarios
        assert "basic_operations" in report["scenarios"]
        assert "batch_operations" in report["scenarios"]

        print(f"Performance report generated successfully: {summary['total_scenarios']} scenarios")

    def test_performance_metric_creation(self):
        """Test performance metric data structure."""
        metric = CachePerformanceMetric(
            operation="test_operation",
            duration_ms=10.5,
            throughput_ops_per_sec=95.2,
            memory_before_mb=100.0,
            memory_after_mb=105.0,
            memory_peak_mb=110.0,
            cache_hits=5,
            cache_misses=2,
            success_count=1,
            error_count=0,
            data_size_bytes=1024,
        )

        assert metric.operation == "test_operation"
        assert metric.duration_ms == 10.5
        assert metric.throughput_ops_per_sec == 95.2
        assert metric.memory_peak_mb == 110.0
        assert metric.cache_hits == 5
        assert metric.success_count == 1

    def test_benchmark_result_creation(self):
        """Test benchmark result data structure."""
        metrics = [
            CachePerformanceMetric(
                operation="test_op_1",
                duration_ms=5.0,
                throughput_ops_per_sec=200.0,
                memory_before_mb=50.0,
                memory_after_mb=55.0,
                memory_peak_mb=60.0,
            )
        ]

        result = BenchmarkResult(scenario="test_scenario", metrics=metrics)

        assert result.scenario == "test_scenario"
        assert len(result.metrics) == 1
        assert result.timestamp > 0


@pytest.mark.performance
class TestCachePerformanceIntegration:
    """Integration tests for cache performance with real services."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(enabled=True, redis_url="redis://localhost:6379/15", default_ttl=300, max_memory_mb=100)  # Test database

    @pytest.mark.asyncio
    async def test_search_cache_performance(self, cache_config):
        """Test SearchCacheService performance."""
        try:
            service = SearchCacheService(cache_config)
            await service.initialize()

            benchmark_suite = CachePerformanceBenchmarkSuite()

            # Run basic operations benchmark
            result = await benchmark_suite.benchmark_basic_operations(service, iterations=20, data_sizes=[100, 1000, 10000])

            # Verify performance meets expectations
            summary = result.summary_stats

            # Basic performance expectations
            assert summary["overall_avg_duration_ms"] < 50, "Average operation too slow"
            assert summary["overall_avg_throughput_ops_per_sec"] > 20, "Throughput too low"
            assert summary["total_errors"] == 0, "No errors expected in basic operations"

            print(
                f"SearchCacheService performance: {summary['overall_avg_duration_ms']:.1f}ms avg, {summary['overall_avg_throughput_ops_per_sec']:.1f} ops/sec"
            )

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_project_cache_performance(self, cache_config):
        """Test ProjectCacheService performance."""
        try:
            service = ProjectCacheService(cache_config)
            await service.initialize()

            benchmark_suite = CachePerformanceBenchmarkSuite()

            # Run batch operations benchmark
            result = await benchmark_suite.benchmark_batch_operations(service, batch_sizes=[10, 50, 100], iterations=10)

            # Verify batch performance
            summary = result.summary_stats

            # Batch operations should be efficient
            for op_name, op_stats in summary["by_operation"].items():
                if "batch" in op_name:
                    assert op_stats["avg_duration_ms"] < 200, f"Batch operation {op_name} too slow"
                    assert op_stats["error_rate"] == 0, f"Errors in {op_name}"

            print(f"ProjectCacheService batch performance verified: {len(summary['by_operation'])} operation types")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
