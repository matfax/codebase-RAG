"""
Performance Benchmarking Framework - Wave 5.0 Implementation.

Provides comprehensive performance benchmarking capabilities including:
- Automated performance regression testing
- Response time benchmarking and validation
- Memory usage testing and optimization validation
- Throughput and scalability testing
- Comparative performance analysis
- Historical performance tracking
- Load testing and stress testing
"""

import asyncio
import gc
import json
import logging
import math
import statistics
import time
import tracemalloc
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil

from src.services.performance_monitor import get_performance_monitor
from src.utils.performance_monitor import get_cache_performance_monitor


class BenchmarkType(Enum):
    """Types of performance benchmarks."""

    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_PERFORMANCE = "cache_performance"
    LOAD_TEST = "load_test"
    STRESS_TEST = "stress_test"
    REGRESSION_TEST = "regression_test"
    CUSTOM = "custom"


class BenchmarkStatus(Enum):
    """Benchmark execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkMetric:
    """Individual benchmark metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"name": self.name, "value": self.value, "unit": self.unit, "timestamp": self.timestamp, "metadata": self.metadata}


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""

    benchmark_id: str
    benchmark_type: BenchmarkType
    name: str
    description: str

    # Execution details
    start_time: float
    end_time: float | None = None
    duration_seconds: float | None = None
    status: BenchmarkStatus = BenchmarkStatus.PENDING

    # Results
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    summary_stats: dict[str, float] = field(default_factory=dict)

    # Comparison results
    baseline_comparison: dict[str, Any] | None = None
    regression_detected: bool = False
    performance_change_percent: float | None = None

    # Error handling
    error_message: str | None = None
    error_details: str | None = None

    # Metadata
    test_parameters: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, Any] = field(default_factory=dict)
    environment_info: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, value: float, unit: str, metadata: dict[str, Any] | None = None):
        """Add a metric to the benchmark result."""
        metric = BenchmarkMetric(name=name, value=value, unit=unit, timestamp=time.time(), metadata=metadata or {})
        self.metrics.append(metric)

    def calculate_summary_stats(self):
        """Calculate summary statistics from metrics."""
        if not self.metrics:
            return

        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in self.metrics:
            metrics_by_name[metric.name].append(metric.value)

        # Calculate stats for each metric
        for name, values in metrics_by_name.items():
            if values:
                self.summary_stats[f"{name}_mean"] = statistics.mean(values)
                self.summary_stats[f"{name}_min"] = min(values)
                self.summary_stats[f"{name}_max"] = max(values)
                self.summary_stats[f"{name}_count"] = len(values)

                if len(values) > 1:
                    self.summary_stats[f"{name}_stdev"] = statistics.stdev(values)
                    self.summary_stats[f"{name}_median"] = statistics.median(values)

    def complete(self, status: BenchmarkStatus = BenchmarkStatus.COMPLETED):
        """Mark the benchmark as complete."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        self.status = status
        self.calculate_summary_stats()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_type": self.benchmark_type.value,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary_stats": self.summary_stats,
            "baseline_comparison": self.baseline_comparison,
            "regression_detected": self.regression_detected,
            "performance_change_percent": self.performance_change_percent,
            "error_message": self.error_message,
            "test_parameters": self.test_parameters,
            "system_info": self.system_info,
            "environment_info": self.environment_info,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # Execution settings
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    timeout_seconds: float = 300.0  # 5 minutes

    # Regression testing
    enable_regression_detection: bool = True
    regression_threshold_percent: float = 10.0  # 10% degradation threshold
    baseline_comparison_enabled: bool = True

    # System monitoring
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_cache: bool = True

    # Data collection
    collect_detailed_metrics: bool = True
    save_raw_data: bool = True

    # Concurrency settings
    max_concurrent_benchmarks: int = 3
    parallel_execution_enabled: bool = False

    # Output settings
    output_directory: str | None = None
    save_results: bool = True
    generate_reports: bool = True


class BenchmarkSuite:
    """Collection of related benchmarks."""

    def __init__(self, suite_id: str, name: str, description: str = ""):
        self.suite_id = suite_id
        self.name = name
        self.description = description
        self.benchmarks: list[Callable] = []
        self.results: list[BenchmarkResult] = []
        self.suite_start_time: float | None = None
        self.suite_end_time: float | None = None

    def add_benchmark(self, benchmark_func: Callable):
        """Add a benchmark function to the suite."""
        self.benchmarks.append(benchmark_func)

    def get_summary(self) -> dict[str, Any]:
        """Get suite execution summary."""
        total_benchmarks = len(self.results)
        completed = len([r for r in self.results if r.status == BenchmarkStatus.COMPLETED])
        failed = len([r for r in self.results if r.status == BenchmarkStatus.FAILED])
        regressions = len([r for r in self.results if r.regression_detected])

        total_duration = 0.0
        if self.suite_start_time and self.suite_end_time:
            total_duration = self.suite_end_time - self.suite_start_time

        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "total_benchmarks": total_benchmarks,
            "completed": completed,
            "failed": failed,
            "regressions_detected": regressions,
            "total_duration_seconds": total_duration,
            "start_time": self.suite_start_time,
            "end_time": self.suite_end_time,
        }


class PerformanceBenchmarkRunner:
    """
    Comprehensive performance benchmarking framework.

    Provides automated performance testing, regression detection, and
    comparative analysis capabilities for the RAG system.
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """
        Initialize the benchmark runner.

        Args:
            config: Benchmark configuration settings
        """
        self.config = config or BenchmarkConfig()
        self.logger = logging.getLogger(__name__)

        # Performance monitoring integration
        self.performance_monitor = get_performance_monitor()
        self.cache_monitor = get_cache_performance_monitor()

        # Benchmark state
        self._benchmark_history: deque = deque(maxlen=1000)
        self._benchmark_suites: dict[str, BenchmarkSuite] = {}
        self._baseline_results: dict[str, BenchmarkResult] = {}
        self._running_benchmarks: dict[str, BenchmarkResult] = {}

        # Execution control
        self._execution_semaphore = asyncio.Semaphore(self.config.max_concurrent_benchmarks)
        self._cancelled_benchmarks: set = set()

        # Built-in benchmarks
        self._builtin_benchmarks = {}
        self._register_builtin_benchmarks()

        self.logger.info("PerformanceBenchmarkRunner initialized")

    def _register_builtin_benchmarks(self):
        """Register built-in benchmark functions."""
        self._builtin_benchmarks = {
            "response_time": self._benchmark_response_time,
            "memory_usage": self._benchmark_memory_usage,
            "cache_performance": self._benchmark_cache_performance,
            "throughput": self._benchmark_throughput,
            "concurrent_load": self._benchmark_concurrent_load,
            "stress_test": self._benchmark_stress_test,
        }

    async def run_benchmark(
        self,
        benchmark_name: str,
        benchmark_func: Callable | None = None,
        parameters: dict[str, Any] | None = None,
        compare_to_baseline: bool = True,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            benchmark_name: Name of the benchmark
            benchmark_func: Custom benchmark function (optional)
            parameters: Benchmark parameters
            compare_to_baseline: Whether to compare against baseline

        Returns:
            BenchmarkResult with execution results
        """
        benchmark_id = f"bench_{benchmark_name}_{int(time.time())}"
        parameters = parameters or {}

        # Create benchmark result
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.CUSTOM if benchmark_func else BenchmarkType.RESPONSE_TIME,
            name=benchmark_name,
            description=f"Performance benchmark: {benchmark_name}",
            start_time=time.time(),
            test_parameters=parameters.copy(),
        )

        try:
            async with self._execution_semaphore:
                if benchmark_id in self._cancelled_benchmarks:
                    result.status = BenchmarkStatus.CANCELLED
                    return result

                self._running_benchmarks[benchmark_id] = result
                result.status = BenchmarkStatus.RUNNING

                # Capture system info
                result.system_info = await self._capture_system_info()

                # Use provided function or built-in
                if benchmark_func:
                    await self._execute_custom_benchmark(result, benchmark_func, parameters)
                elif benchmark_name in self._builtin_benchmarks:
                    await self._builtin_benchmarks[benchmark_name](result, parameters)
                else:
                    raise ValueError(f"Unknown benchmark: {benchmark_name}")

                # Compare to baseline if requested
                if compare_to_baseline and benchmark_name in self._baseline_results:
                    await self._compare_to_baseline(result, self._baseline_results[benchmark_name])

                result.complete(BenchmarkStatus.COMPLETED)
                self.logger.info(f"Benchmark {benchmark_name} completed successfully")

        except asyncio.CancelledError:
            result.status = BenchmarkStatus.CANCELLED
            result.error_message = "Benchmark was cancelled"

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_message = str(e)
            result.error_details = traceback.format_exc()
            self.logger.error(f"Benchmark {benchmark_name} failed: {e}")

        finally:
            result.complete(result.status)
            self._running_benchmarks.pop(benchmark_id, None)
            self._benchmark_history.append(result)

        return result

    async def run_benchmark_suite(self, suite: BenchmarkSuite) -> dict[str, Any]:
        """
        Run a complete benchmark suite.

        Args:
            suite: BenchmarkSuite to execute

        Returns:
            Suite execution summary
        """
        self.logger.info(f"Starting benchmark suite: {suite.name}")

        suite.suite_start_time = time.time()
        suite.results.clear()

        try:
            if self.config.parallel_execution_enabled:
                # Run benchmarks in parallel
                tasks = []
                for benchmark_func in suite.benchmarks:
                    task = asyncio.create_task(self._execute_suite_benchmark(benchmark_func))
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)
                suite.results.extend([r for r in results if isinstance(r, BenchmarkResult)])

            else:
                # Run benchmarks sequentially
                for benchmark_func in suite.benchmarks:
                    result = await self._execute_suite_benchmark(benchmark_func)
                    suite.results.append(result)

            suite.suite_end_time = time.time()
            self._benchmark_suites[suite.suite_id] = suite

            summary = suite.get_summary()
            self.logger.info(f"Benchmark suite {suite.name} completed: {summary}")

            return summary

        except Exception as e:
            self.logger.error(f"Error running benchmark suite {suite.name}: {e}")
            suite.suite_end_time = time.time()
            return {"error": str(e), "suite_id": suite.suite_id}

    async def _execute_suite_benchmark(self, benchmark_func: Callable) -> BenchmarkResult:
        """Execute a benchmark function from a suite."""
        try:
            # Extract benchmark name from function
            benchmark_name = getattr(benchmark_func, "__name__", "unknown_benchmark")

            if asyncio.iscoroutinefunction(benchmark_func):
                return await benchmark_func()
            else:
                return benchmark_func()

        except Exception as e:
            # Create error result
            return BenchmarkResult(
                benchmark_id=f"error_{int(time.time())}",
                benchmark_type=BenchmarkType.CUSTOM,
                name=getattr(benchmark_func, "__name__", "unknown"),
                description="Failed benchmark execution",
                start_time=time.time(),
                status=BenchmarkStatus.FAILED,
                error_message=str(e),
            )

    async def _execute_custom_benchmark(self, result: BenchmarkResult, benchmark_func: Callable, parameters: dict[str, Any]):
        """Execute a custom benchmark function."""
        try:
            # Setup memory tracking if enabled
            if self.config.monitor_memory:
                tracemalloc.start()
                initial_memory = self._get_memory_usage()

            # Warmup iterations
            for i in range(self.config.warmup_iterations):
                if asyncio.iscoroutinefunction(benchmark_func):
                    await benchmark_func(**parameters)
                else:
                    benchmark_func(**parameters)

            # Measurement iterations
            iteration_times = []
            for i in range(self.config.measurement_iterations):
                start_time = time.time()

                if asyncio.iscoroutinefunction(benchmark_func):
                    await benchmark_func(**parameters)
                else:
                    benchmark_func(**parameters)

                end_time = time.time()
                iteration_time = (end_time - start_time) * 1000  # Convert to milliseconds
                iteration_times.append(iteration_time)

                result.add_metric(name="iteration_time", value=iteration_time, unit="ms", metadata={"iteration": i + 1})

            # Calculate timing statistics
            result.add_metric("average_time", statistics.mean(iteration_times), "ms")
            result.add_metric("min_time", min(iteration_times), "ms")
            result.add_metric("max_time", max(iteration_times), "ms")
            result.add_metric("median_time", statistics.median(iteration_times), "ms")

            if len(iteration_times) > 1:
                result.add_metric("time_stdev", statistics.stdev(iteration_times), "ms")

            # Memory tracking
            if self.config.monitor_memory:
                final_memory = self._get_memory_usage()
                memory_delta = final_memory - initial_memory
                result.add_metric("memory_delta", memory_delta, "MB")
                result.add_metric("final_memory", final_memory, "MB")
                tracemalloc.stop()

        except Exception as e:
            raise RuntimeError(f"Custom benchmark execution failed: {e}")

    async def _benchmark_response_time(self, result: BenchmarkResult, parameters: dict[str, Any]):
        """Built-in response time benchmark."""
        result.benchmark_type = BenchmarkType.RESPONSE_TIME

        # Get target function to benchmark
        target_func = parameters.get("target_function")
        if not target_func:
            raise ValueError("target_function parameter required for response time benchmark")

        func_args = parameters.get("function_args", [])
        func_kwargs = parameters.get("function_kwargs", {})

        # Warmup
        for _ in range(self.config.warmup_iterations):
            if asyncio.iscoroutinefunction(target_func):
                await target_func(*func_args, **func_kwargs)
            else:
                target_func(*func_args, **func_kwargs)

        # Measurement
        response_times = []
        for i in range(self.config.measurement_iterations):
            start_time = time.time()

            if asyncio.iscoroutinefunction(target_func):
                await target_func(*func_args, **func_kwargs)
            else:
                target_func(*func_args, **func_kwargs)

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)

            result.add_metric("response_time", response_time, "ms", {"iteration": i + 1})

        # Calculate statistics
        result.add_metric("avg_response_time", statistics.mean(response_times), "ms")
        result.add_metric("min_response_time", min(response_times), "ms")
        result.add_metric("max_response_time", max(response_times), "ms")
        result.add_metric("p95_response_time", self._calculate_percentile(response_times, 95), "ms")
        result.add_metric("p99_response_time", self._calculate_percentile(response_times, 99), "ms")

        # Check if response time meets requirements
        target_response_time = parameters.get("target_response_time_ms", 15000)  # 15 seconds default
        avg_response_time = statistics.mean(response_times)

        if avg_response_time > target_response_time:
            result.add_metric("performance_target_met", 0, "boolean")
            self.logger.warning(f"Response time target not met: {avg_response_time:.2f}ms > {target_response_time}ms")
        else:
            result.add_metric("performance_target_met", 1, "boolean")

    async def _benchmark_memory_usage(self, result: BenchmarkResult, parameters: dict[str, Any]):
        """Built-in memory usage benchmark."""
        result.benchmark_type = BenchmarkType.MEMORY_USAGE

        target_func = parameters.get("target_function")
        if not target_func:
            raise ValueError("target_function parameter required for memory usage benchmark")

        func_args = parameters.get("function_args", [])
        func_kwargs = parameters.get("function_kwargs", {})

        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean up before measurement
        initial_memory = self._get_memory_usage()

        memory_measurements = []

        for i in range(self.config.measurement_iterations):
            gc.collect()
            pre_memory = self._get_memory_usage()

            if asyncio.iscoroutinefunction(target_func):
                await target_func(*func_args, **func_kwargs)
            else:
                target_func(*func_args, **func_kwargs)

            post_memory = self._get_memory_usage()
            memory_delta = post_memory - pre_memory
            memory_measurements.append(memory_delta)

            result.add_metric("memory_delta", memory_delta, "MB", {"iteration": i + 1})
            result.add_metric("total_memory", post_memory, "MB", {"iteration": i + 1})

        # Calculate memory statistics
        result.add_metric("avg_memory_delta", statistics.mean(memory_measurements), "MB")
        result.add_metric("max_memory_delta", max(memory_measurements), "MB")
        result.add_metric("total_memory_increase", self._get_memory_usage() - initial_memory, "MB")

        # Check memory target
        memory_target = parameters.get("target_memory_mb", 2048)  # 2GB default
        current_memory = self._get_memory_usage()

        if current_memory > memory_target:
            result.add_metric("memory_target_met", 0, "boolean")
        else:
            result.add_metric("memory_target_met", 1, "boolean")

        tracemalloc.stop()

    async def _benchmark_cache_performance(self, result: BenchmarkResult, parameters: dict[str, Any]):
        """Built-in cache performance benchmark."""
        result.benchmark_type = BenchmarkType.CACHE_PERFORMANCE

        # Get initial cache metrics
        initial_metrics = self.cache_monitor.get_aggregated_metrics()
        initial_hit_rate = initial_metrics.get("summary", {}).get("overall_hit_rate", 0) if initial_metrics else 0

        target_func = parameters.get("target_function")
        if not target_func:
            raise ValueError("target_function parameter required for cache performance benchmark")

        func_args = parameters.get("function_args", [])
        func_kwargs = parameters.get("function_kwargs", {})

        # Execute function multiple times to generate cache activity
        for i in range(self.config.measurement_iterations * 2):  # More iterations for cache testing
            if asyncio.iscoroutinefunction(target_func):
                await target_func(*func_args, **func_kwargs)
            else:
                target_func(*func_args, **func_kwargs)

        # Get final cache metrics
        final_metrics = self.cache_monitor.get_aggregated_metrics()

        if final_metrics and "summary" in final_metrics:
            summary = final_metrics["summary"]

            result.add_metric("cache_hit_rate", summary.get("overall_hit_rate", 0) * 100, "%")
            result.add_metric("total_operations", summary.get("total_operations", 0), "count")
            result.add_metric("cache_size", summary.get("total_size_mb", 0), "MB")
            result.add_metric("avg_response_time", summary.get("average_response_time_ms", 0), "ms")
            result.add_metric("error_rate", summary.get("overall_error_rate", 0) * 100, "%")

            # Check cache performance targets
            hit_rate = summary.get("overall_hit_rate", 0) * 100
            target_hit_rate = parameters.get("target_hit_rate_percent", 80)

            if hit_rate >= target_hit_rate:
                result.add_metric("cache_target_met", 1, "boolean")
            else:
                result.add_metric("cache_target_met", 0, "boolean")

    async def _benchmark_throughput(self, result: BenchmarkResult, parameters: dict[str, Any]):
        """Built-in throughput benchmark."""
        result.benchmark_type = BenchmarkType.THROUGHPUT

        target_func = parameters.get("target_function")
        if not target_func:
            raise ValueError("target_function parameter required for throughput benchmark")

        func_args = parameters.get("function_args", [])
        func_kwargs = parameters.get("function_kwargs", {})
        duration_seconds = parameters.get("duration_seconds", 60)  # 1 minute default

        start_time = time.time()
        end_time = start_time + duration_seconds
        operation_count = 0
        operation_times = []

        while time.time() < end_time:
            op_start = time.time()

            if asyncio.iscoroutinefunction(target_func):
                await target_func(*func_args, **func_kwargs)
            else:
                target_func(*func_args, **func_kwargs)

            op_end = time.time()
            operation_times.append((op_end - op_start) * 1000)  # Convert to ms
            operation_count += 1

        actual_duration = time.time() - start_time
        throughput = operation_count / actual_duration

        result.add_metric("throughput", throughput, "ops/sec")
        result.add_metric("total_operations", operation_count, "count")
        result.add_metric("test_duration", actual_duration, "seconds")
        result.add_metric("avg_operation_time", statistics.mean(operation_times), "ms")

        # Check throughput target
        target_throughput = parameters.get("target_throughput_ops_per_sec", 10)
        if throughput >= target_throughput:
            result.add_metric("throughput_target_met", 1, "boolean")
        else:
            result.add_metric("throughput_target_met", 0, "boolean")

    async def _benchmark_concurrent_load(self, result: BenchmarkResult, parameters: dict[str, Any]):
        """Built-in concurrent load benchmark."""
        result.benchmark_type = BenchmarkType.LOAD_TEST

        target_func = parameters.get("target_function")
        if not target_func:
            raise ValueError("target_function parameter required for concurrent load benchmark")

        func_args = parameters.get("function_args", [])
        func_kwargs = parameters.get("function_kwargs", {})
        concurrent_users = parameters.get("concurrent_users", 5)
        operations_per_user = parameters.get("operations_per_user", 10)

        async def user_simulation(user_id: int):
            """Simulate a single user's operations."""
            user_times = []
            for op in range(operations_per_user):
                start_time = time.time()

                try:
                    if asyncio.iscoroutinefunction(target_func):
                        await target_func(*func_args, **func_kwargs)
                    else:
                        target_func(*func_args, **func_kwargs)

                    end_time = time.time()
                    operation_time = (end_time - start_time) * 1000
                    user_times.append(operation_time)

                except Exception:
                    # Record error but continue
                    user_times.append(float("inf"))  # Mark as failed

            return user_id, user_times

        # Run concurrent user simulations
        start_time = time.time()
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time

        # Aggregate results
        all_times = []
        failed_operations = 0

        for user_id, user_times in user_results:
            for op_time in user_times:
                if op_time == float("inf"):
                    failed_operations += 1
                else:
                    all_times.append(op_time)

        total_operations = concurrent_users * operations_per_user
        successful_operations = len(all_times)

        result.add_metric("concurrent_users", concurrent_users, "count")
        result.add_metric("total_operations", total_operations, "count")
        result.add_metric("successful_operations", successful_operations, "count")
        result.add_metric("failed_operations", failed_operations, "count")
        result.add_metric("success_rate", (successful_operations / total_operations) * 100, "%")
        result.add_metric("total_duration", total_duration, "seconds")

        if all_times:
            result.add_metric("avg_response_time", statistics.mean(all_times), "ms")
            result.add_metric("p95_response_time", self._calculate_percentile(all_times, 95), "ms")
            result.add_metric("p99_response_time", self._calculate_percentile(all_times, 99), "ms")
            result.add_metric("throughput", successful_operations / total_duration, "ops/sec")

    async def _benchmark_stress_test(self, result: BenchmarkResult, parameters: dict[str, Any]):
        """Built-in stress test benchmark."""
        result.benchmark_type = BenchmarkType.STRESS_TEST

        target_func = parameters.get("target_function")
        if not target_func:
            raise ValueError("target_function parameter required for stress test")

        func_args = parameters.get("function_args", [])
        func_kwargs = parameters.get("function_kwargs", {})
        stress_duration = parameters.get("stress_duration_seconds", 300)  # 5 minutes
        max_concurrent = parameters.get("max_concurrent_operations", 20)

        # Gradually increase load
        start_time = time.time()
        end_time = start_time + stress_duration
        current_concurrent = 1

        async def stress_operation():
            """Single stress test operation."""
            op_start = time.time()
            try:
                if asyncio.iscoroutinefunction(target_func):
                    await target_func(*func_args, **func_kwargs)
                else:
                    target_func(*func_args, **func_kwargs)

                return time.time() - op_start, True
            except Exception:
                return time.time() - op_start, False

        operation_times = []
        success_count = 0
        failure_count = 0

        while time.time() < end_time:
            # Create batch of concurrent operations
            tasks = [stress_operation() for _ in range(current_concurrent)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result_item in results:
                if isinstance(result_item, tuple):
                    op_time, success = result_item
                    operation_times.append(op_time * 1000)  # Convert to ms
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                else:
                    failure_count += 1

            # Gradually increase load
            if current_concurrent < max_concurrent:
                current_concurrent = min(current_concurrent + 1, max_concurrent)

            # Brief pause between batches
            await asyncio.sleep(0.1)

        total_operations = success_count + failure_count

        result.add_metric("max_concurrent_reached", current_concurrent, "count")
        result.add_metric("total_operations", total_operations, "count")
        result.add_metric("successful_operations", success_count, "count")
        result.add_metric("failed_operations", failure_count, "count")
        result.add_metric("failure_rate", (failure_count / total_operations) * 100 if total_operations > 0 else 0, "%")

        if operation_times:
            result.add_metric("avg_response_time", statistics.mean(operation_times), "ms")
            result.add_metric("max_response_time", max(operation_times), "ms")
            result.add_metric("p99_response_time", self._calculate_percentile(operation_times, 99), "ms")

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Calculate the specified percentile of a list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index

            if upper_index >= len(sorted_values):
                return sorted_values[lower_index]

            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    async def _capture_system_info(self) -> dict[str, Any]:
        """Capture current system information."""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()

            return {
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "cpu_count": cpu_count,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "timestamp": time.time(),
            }
        except Exception as e:
            self.logger.error(f"Error capturing system info: {e}")
            return {}

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    async def _compare_to_baseline(self, result: BenchmarkResult, baseline: BenchmarkResult):
        """Compare benchmark result to baseline."""
        try:
            if not baseline.summary_stats:
                return

            comparison = {}
            regression_detected = False

            # Compare key metrics
            key_metrics = ["avg_response_time", "avg_memory_delta", "cache_hit_rate", "throughput"]

            for metric in key_metrics:
                current_key = f"{metric.replace('avg_', '')}_mean"
                baseline_key = f"{metric.replace('avg_', '')}_mean"

                current_value = result.summary_stats.get(current_key)
                baseline_value = baseline.summary_stats.get(baseline_key)

                if current_value is not None and baseline_value is not None and baseline_value != 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    comparison[metric] = {"current": current_value, "baseline": baseline_value, "change_percent": change_percent}

                    # Check for regression (performance degradation)
                    if metric in ["avg_response_time", "avg_memory_delta"]:
                        # Higher is worse for these metrics
                        if change_percent > self.config.regression_threshold_percent:
                            regression_detected = True
                    elif metric in ["cache_hit_rate", "throughput"]:
                        # Lower is worse for these metrics
                        if change_percent < -self.config.regression_threshold_percent:
                            regression_detected = True

            result.baseline_comparison = comparison
            result.regression_detected = regression_detected

            # Calculate overall performance change
            if comparison:
                changes = [abs(c["change_percent"]) for c in comparison.values()]
                result.performance_change_percent = statistics.mean(changes)

        except Exception as e:
            self.logger.error(f"Error comparing to baseline: {e}")

    def set_baseline(self, benchmark_name: str, result: BenchmarkResult):
        """Set a benchmark result as the baseline for comparison."""
        self._baseline_results[benchmark_name] = result
        self.logger.info(f"Set baseline for benchmark: {benchmark_name}")

    def get_baseline(self, benchmark_name: str) -> BenchmarkResult | None:
        """Get the baseline result for a benchmark."""
        return self._baseline_results.get(benchmark_name)

    def cancel_benchmark(self, benchmark_id: str):
        """Cancel a running benchmark."""
        self._cancelled_benchmarks.add(benchmark_id)
        if benchmark_id in self._running_benchmarks:
            self._running_benchmarks[benchmark_id].status = BenchmarkStatus.CANCELLED

    def get_benchmark_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get benchmark execution history."""
        cutoff_time = time.time() - (hours * 3600)

        return [result.to_dict() for result in self._benchmark_history if result.start_time > cutoff_time]

    def get_running_benchmarks(self) -> dict[str, dict[str, Any]]:
        """Get currently running benchmarks."""
        return {
            benchmark_id: {
                "name": result.name,
                "status": result.status.value,
                "start_time": result.start_time,
                "duration_so_far": time.time() - result.start_time,
            }
            for benchmark_id, result in self._running_benchmarks.items()
        }

    def create_benchmark_suite(self, suite_id: str, name: str, description: str = "") -> BenchmarkSuite:
        """Create a new benchmark suite."""
        suite = BenchmarkSuite(suite_id, name, description)
        return suite

    async def generate_benchmark_report(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Generate a comprehensive benchmark report."""
        try:
            if not results:
                return {"error": "No benchmark results provided"}

            # Overall statistics
            total_benchmarks = len(results)
            completed = len([r for r in results if r.status == BenchmarkStatus.COMPLETED])
            failed = len([r for r in results if r.status == BenchmarkStatus.FAILED])
            regressions = len([r for r in results if r.regression_detected])

            # Performance statistics
            response_times = []
            memory_usage = []

            for result in results:
                if "response_time_mean" in result.summary_stats:
                    response_times.append(result.summary_stats["response_time_mean"])
                if "memory_delta_mean" in result.summary_stats:
                    memory_usage.append(result.summary_stats["memory_delta_mean"])

            report = {
                "report_timestamp": time.time(),
                "summary": {
                    "total_benchmarks": total_benchmarks,
                    "completed": completed,
                    "failed": failed,
                    "success_rate": (completed / total_benchmarks) * 100 if total_benchmarks > 0 else 0,
                    "regressions_detected": regressions,
                },
                "performance_overview": {
                    "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
                    "avg_memory_usage_mb": statistics.mean(memory_usage) if memory_usage else 0,
                    "total_test_duration": sum(r.duration_seconds or 0 for r in results),
                },
                "benchmark_results": [result.to_dict() for result in results],
                "recommendations": self._generate_performance_recommendations(results),
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating benchmark report: {e}")
            return {"error": str(e)}

    def _generate_performance_recommendations(self, results: list[BenchmarkResult]) -> list[dict[str, Any]]:
        """Generate performance recommendations based on benchmark results."""
        recommendations = []

        try:
            # Analyze response times
            slow_benchmarks = [r for r in results if r.summary_stats.get("response_time_mean", 0) > 15000]  # > 15 seconds

            if slow_benchmarks:
                recommendations.append(
                    {
                        "type": "response_time",
                        "priority": "high",
                        "message": f"{len(slow_benchmarks)} benchmarks show slow response times",
                        "details": "Consider optimizing database queries, adding caching, or improving algorithms",
                        "affected_benchmarks": [r.name for r in slow_benchmarks],
                    }
                )

            # Analyze memory usage
            memory_heavy_benchmarks = [r for r in results if r.summary_stats.get("memory_delta_mean", 0) > 500]  # > 500MB

            if memory_heavy_benchmarks:
                recommendations.append(
                    {
                        "type": "memory_usage",
                        "priority": "medium",
                        "message": f"{len(memory_heavy_benchmarks)} benchmarks show high memory usage",
                        "details": "Consider implementing memory pooling, optimizing data structures, or adding garbage collection",
                        "affected_benchmarks": [r.name for r in memory_heavy_benchmarks],
                    }
                )

            # Analyze regressions
            regressed_benchmarks = [r for r in results if r.regression_detected]

            if regressed_benchmarks:
                recommendations.append(
                    {
                        "type": "regression",
                        "priority": "critical",
                        "message": f"{len(regressed_benchmarks)} benchmarks show performance regressions",
                        "details": "Investigate recent changes that may have caused performance degradation",
                        "affected_benchmarks": [r.name for r in regressed_benchmarks],
                    }
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []

    async def save_benchmark_results(self, results: list[BenchmarkResult], file_path: str):
        """Save benchmark results to file."""
        try:
            report = await self.generate_benchmark_report(results)

            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Benchmark results saved to {file_path}")

        except Exception as e:
            self.logger.error(f"Error saving benchmark results: {e}")

    async def shutdown(self):
        """Shutdown the benchmark runner."""
        self.logger.info("Shutting down PerformanceBenchmarkRunner")

        # Cancel running benchmarks
        for benchmark_id in list(self._running_benchmarks.keys()):
            self.cancel_benchmark(benchmark_id)

        # Wait for benchmarks to complete or timeout
        timeout = 30.0  # 30 seconds timeout
        start_time = time.time()

        while self._running_benchmarks and (time.time() - start_time) < timeout:
            await asyncio.sleep(1.0)

        # Clear state
        self._benchmark_history.clear()
        self._benchmark_suites.clear()
        self._baseline_results.clear()
        self._running_benchmarks.clear()

        self.logger.info("PerformanceBenchmarkRunner shutdown complete")


# Global benchmark runner instance
_benchmark_runner: PerformanceBenchmarkRunner | None = None


def get_benchmark_runner() -> PerformanceBenchmarkRunner:
    """Get the global benchmark runner instance."""
    global _benchmark_runner
    if _benchmark_runner is None:
        _benchmark_runner = PerformanceBenchmarkRunner()
    return _benchmark_runner
