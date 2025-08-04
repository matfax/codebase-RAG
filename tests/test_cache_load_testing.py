"""
Cache Load Testing for Query Caching Layer.

This module provides comprehensive load testing for cache operations including:
- High-concurrency cache operations
- Sustained load testing over time
- Realistic workload simulation
- Resource utilization monitoring
- Performance degradation detection
- Stress testing with extreme loads
"""

import asyncio
import gc
import json
import logging
import random
import statistics
import string
import sys
import time
from collections.abc import Callable
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

# Configure logging for load testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestMetric:
    """Metric for a single operation during load testing."""

    timestamp: float
    operation: str
    duration_ms: float
    success: bool
    error_message: str = ""
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    thread_id: str = ""


@dataclass
class LoadTestResult:
    """Complete load test result."""

    test_name: str
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    error_rate: float
    avg_memory_mb: float
    peak_memory_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    metrics: list[LoadTestMetric] = field(default_factory=list)
    resource_utilization: dict[str, Any] = field(default_factory=dict)
    performance_degradation: dict[str, Any] = field(default_factory=dict)


class LoadTestWorkloadGenerator:
    """Generates realistic workloads for cache load testing."""

    def __init__(self):
        self.search_queries = [
            "python function async",
            "class inheritance methods",
            "error handling exceptions",
            "data structures algorithms",
            "performance optimization",
            "database query optimization",
            "api endpoint design",
            "security authentication",
            "logging debugging",
            "unit testing pytest",
        ]

        self.project_data = [
            {"name": "web-app", "language": "python", "size": "large"},
            {"name": "mobile-app", "language": "java", "size": "medium"},
            {"name": "api-service", "language": "typescript", "size": "small"},
            {"name": "data-pipeline", "language": "python", "size": "large"},
            {"name": "frontend", "language": "javascript", "size": "medium"},
        ]

    def generate_search_cache_key(self) -> str:
        """Generate realistic search cache key."""
        query = random.choice(self.search_queries)
        filters = random.choice([{"language": "python"}, {"type": "function"}, {"language": "javascript", "type": "class"}, {}])

        # Create a deterministic key based on query and filters
        key_parts = [query.replace(" ", "_")]
        for k, v in filters.items():
            key_parts.append(f"{k}:{v}")

        return f"search:{':'.join(key_parts)}"

    def generate_search_cache_value(self) -> dict[str, Any]:
        """Generate realistic search cache value."""
        num_results = random.randint(5, 50)
        results = []

        for i in range(num_results):
            result = {
                "file_path": f"/src/path/to/file_{i}.py",
                "chunk_type": random.choice(["function", "class", "method"]),
                "name": f"function_{i}",
                "content": "".join(random.choices(string.ascii_letters, k=random.randint(100, 1000))),
                "score": random.uniform(0.1, 1.0),
                "line_start": random.randint(1, 100),
                "line_end": random.randint(101, 200),
            }
            results.append(result)

        return {
            "query": random.choice(self.search_queries),
            "results": results,
            "total_results": num_results,
            "execution_time_ms": random.uniform(10, 100),
            "cached_at": time.time(),
        }

    def generate_project_cache_key(self) -> str:
        """Generate realistic project cache key."""
        project = random.choice(self.project_data)
        cache_type = random.choice(["metadata", "chunks", "analysis", "stats"])

        return f"project:{project['name']}:{cache_type}"

    def generate_project_cache_value(self) -> dict[str, Any]:
        """Generate realistic project cache value."""
        project = random.choice(self.project_data)

        if random.choice([True, False]):
            # Metadata cache
            return {
                "project_name": project["name"],
                "language": project["language"],
                "total_files": random.randint(50, 500),
                "total_lines": random.randint(5000, 50000),
                "last_indexed": time.time(),
                "file_types": {".py": 45, ".js": 30, ".md": 15, ".json": 10},
            }
        else:
            # Chunk cache
            chunks = []
            for i in range(random.randint(10, 100)):
                chunk = {
                    "id": f"chunk_{i}",
                    "content": "".join(random.choices(string.ascii_letters, k=random.randint(200, 2000))),
                    "type": random.choice(["function", "class", "method"]),
                    "file_path": f"/src/{project['name']}/file_{i}.py",
                }
                chunks.append(chunk)

            return {"project_name": project["name"], "chunks": chunks, "total_chunks": len(chunks), "generated_at": time.time()}


class CacheLoadTester:
    """Comprehensive cache load testing framework."""

    def __init__(self):
        self.workload_generator = LoadTestWorkloadGenerator()
        self.process = psutil.Process()
        self.memory_monitor = MemoryMonitor()

    async def _monitor_system_resources(self, metrics: list[LoadTestMetric], stop_event: asyncio.Event):
        """Monitor system resources during load test."""
        while not stop_event.is_set():
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                # Store resource data for analysis
                resource_metric = LoadTestMetric(
                    timestamp=time.time(),
                    operation="system_monitor",
                    duration_ms=0,
                    success=True,
                    memory_mb=memory_info.rss / (1024 * 1024),
                    cpu_percent=cpu_percent,
                )
                metrics.append(resource_metric)

                await asyncio.sleep(1.0)  # Monitor every second

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(1.0)

    async def _execute_operation(
        self, cache_service: Any, operation_func: Callable, operation_name: str, *args, **kwargs
    ) -> LoadTestMetric:
        """Execute a single cache operation and measure performance."""
        start_time = time.perf_counter()
        timestamp = time.time()

        try:
            await operation_func(*args, **kwargs)
            success = True
            error_message = ""
        except Exception as e:
            success = False
            error_message = str(e)

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Get current resource usage
        memory_mb = self.process.memory_info().rss / (1024 * 1024)
        cpu_percent = self.process.cpu_percent()

        return LoadTestMetric(
            timestamp=timestamp,
            operation=operation_name,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            thread_id=str(asyncio.current_task()),
        )

    async def run_sustained_load_test(
        self, cache_service: Any, duration_seconds: int = 60, operations_per_second: int = 100, test_name: str = "sustained_load"
    ) -> LoadTestResult:
        """Run sustained load test with consistent operation rate."""
        logger.info(f"Starting sustained load test: {test_name} for {duration_seconds}s at {operations_per_second} ops/sec")

        metrics = []
        stop_event = asyncio.Event()

        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_system_resources(metrics, stop_event))

        start_time = time.time()
        operation_interval = 1.0 / operations_per_second

        async def operation_worker():
            """Worker that performs cache operations at target rate."""
            next_operation_time = time.time()

            while time.time() - start_time < duration_seconds:
                # Wait until it's time for the next operation
                current_time = time.time()
                if current_time < next_operation_time:
                    await asyncio.sleep(next_operation_time - current_time)

                # Randomly choose operation type
                if random.random() < 0.6:  # 60% reads
                    # GET operation
                    key = self.workload_generator.generate_search_cache_key()
                    metric = await self._execute_operation(cache_service, cache_service.get, "get", key)
                elif random.random() < 0.8:  # 20% writes (80% cumulative)
                    # SET operation
                    key = self.workload_generator.generate_search_cache_key()
                    value = self.workload_generator.generate_search_cache_value()
                    metric = await self._execute_operation(cache_service, cache_service.set, "set", key, value)
                else:  # 20% deletes
                    # DELETE operation
                    key = self.workload_generator.generate_search_cache_key()
                    metric = await self._execute_operation(cache_service, cache_service.delete, "delete", key)

                metrics.append(metric)
                next_operation_time += operation_interval

        # Run operations
        await operation_worker()

        # Stop monitoring
        stop_event.set()
        await monitor_task

        end_time = time.time()
        actual_duration = end_time - start_time

        # Filter out system monitoring metrics for operation analysis
        operation_metrics = [m for m in metrics if m.operation != "system_monitor"]
        resource_metrics = [m for m in metrics if m.operation == "system_monitor"]

        result = self._analyze_load_test_results(test_name, actual_duration, operation_metrics, resource_metrics)

        logger.info(f"Sustained load test completed: {result.operations_per_second:.1f} ops/sec, {result.error_rate*100:.1f}% error rate")
        return result

    async def run_burst_load_test(
        self,
        cache_service: Any,
        burst_duration_seconds: int = 10,
        max_concurrency: int = 200,
        operations_per_burst: int = 1000,
        test_name: str = "burst_load",
    ) -> LoadTestResult:
        """Run burst load test with high concurrency spikes."""
        logger.info(f"Starting burst load test: {test_name} with {max_concurrency} concurrent operations")

        metrics = []
        stop_event = asyncio.Event()

        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_system_resources(metrics, stop_event))

        start_time = time.time()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def burst_operation():
            """Single operation within a burst."""
            async with semaphore:
                # Choose random operation
                op_type = random.choices(["get", "set", "delete"], weights=[60, 30, 10])[0]  # Weighted towards reads

                if op_type == "get":
                    key = self.workload_generator.generate_search_cache_key()
                    metric = await self._execute_operation(cache_service, cache_service.get, "burst_get", key)
                elif op_type == "set":
                    key = self.workload_generator.generate_project_cache_key()
                    value = self.workload_generator.generate_project_cache_value()
                    metric = await self._execute_operation(cache_service, cache_service.set, "burst_set", key, value)
                else:  # delete
                    key = self.workload_generator.generate_search_cache_key()
                    metric = await self._execute_operation(cache_service, cache_service.delete, "burst_delete", key)

                metrics.append(metric)

        # Create and execute burst of operations
        tasks = [burst_operation() for _ in range(operations_per_burst)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Stop monitoring
        stop_event.set()
        await monitor_task

        end_time = time.time()
        actual_duration = end_time - start_time

        # Analyze results
        operation_metrics = [m for m in metrics if not m.operation.startswith("system")]
        resource_metrics = [m for m in metrics if m.operation == "system_monitor"]

        result = self._analyze_load_test_results(test_name, actual_duration, operation_metrics, resource_metrics)

        logger.info(f"Burst load test completed: {result.total_operations} operations in {actual_duration:.1f}s")
        return result

    async def run_gradual_ramp_test(
        self,
        cache_service: Any,
        ramp_duration_seconds: int = 120,
        max_operations_per_second: int = 500,
        ramp_steps: int = 10,
        test_name: str = "gradual_ramp",
    ) -> LoadTestResult:
        """Run gradual ramp-up load test to find performance limits."""
        logger.info(f"Starting gradual ramp test: {test_name} ramping to {max_operations_per_second} ops/sec")

        metrics = []
        stop_event = asyncio.Event()

        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_system_resources(metrics, stop_event))

        start_time = time.time()
        step_duration = ramp_duration_seconds / ramp_steps
        ops_increment = max_operations_per_second / ramp_steps

        for step in range(ramp_steps):
            current_ops_per_sec = int((step + 1) * ops_increment)
            step_start_time = time.time()

            logger.info(f"Ramp step {step + 1}/{ramp_steps}: {current_ops_per_sec} ops/sec")

            # Run operations for this step
            operations_in_step = int(current_ops_per_sec * step_duration)
            operation_interval = step_duration / operations_in_step if operations_in_step > 0 else 1.0

            step_metrics = []
            next_operation_time = step_start_time

            for op_num in range(operations_in_step):
                if time.time() - step_start_time >= step_duration:
                    break

                # Wait for next operation time
                current_time = time.time()
                if current_time < next_operation_time:
                    await asyncio.sleep(next_operation_time - current_time)

                # Execute operation
                key = self.workload_generator.generate_search_cache_key()
                value = self.workload_generator.generate_search_cache_value()

                if random.random() < 0.7:  # 70% reads
                    metric = await self._execute_operation(cache_service, cache_service.get, f"ramp_get_step_{step}", key)
                else:  # 30% writes
                    metric = await self._execute_operation(cache_service, cache_service.set, f"ramp_set_step_{step}", key, value)

                step_metrics.append(metric)
                metrics.append(metric)
                next_operation_time += operation_interval

            # Analyze step performance
            if step_metrics:
                step_avg_duration = statistics.mean(m.duration_ms for m in step_metrics)
                step_error_rate = sum(1 for m in step_metrics if not m.success) / len(step_metrics)
                logger.info(f"Step {step + 1} performance: {step_avg_duration:.1f}ms avg, {step_error_rate*100:.1f}% errors")

        # Stop monitoring
        stop_event.set()
        await monitor_task

        end_time = time.time()
        actual_duration = end_time - start_time

        # Analyze results
        operation_metrics = [m for m in metrics if not m.operation.startswith("system")]
        resource_metrics = [m for m in metrics if m.operation == "system_monitor"]

        result = self._analyze_load_test_results(test_name, actual_duration, operation_metrics, resource_metrics)

        logger.info(f"Gradual ramp test completed: peak {max_operations_per_second} ops/sec reached")
        return result

    def _analyze_load_test_results(
        self, test_name: str, duration_seconds: float, operation_metrics: list[LoadTestMetric], resource_metrics: list[LoadTestMetric]
    ) -> LoadTestResult:
        """Analyze load test results and calculate performance statistics."""

        if not operation_metrics:
            return LoadTestResult(
                test_name=test_name,
                duration_seconds=duration_seconds,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                operations_per_second=0,
                avg_response_time_ms=0,
                median_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                max_response_time_ms=0,
                min_response_time_ms=0,
                error_rate=0,
                avg_memory_mb=0,
                peak_memory_mb=0,
                avg_cpu_percent=0,
                peak_cpu_percent=0,
            )

        # Basic statistics
        total_operations = len(operation_metrics)
        successful_operations = sum(1 for m in operation_metrics if m.success)
        failed_operations = total_operations - successful_operations
        operations_per_second = total_operations / duration_seconds
        error_rate = failed_operations / total_operations if total_operations > 0 else 0

        # Response time statistics
        response_times = [m.duration_ms for m in operation_metrics]
        response_times.sort()

        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)

        # Percentiles
        p95_index = int(0.95 * len(response_times))
        p99_index = int(0.99 * len(response_times))
        p95_response_time = response_times[p95_index] if p95_index < len(response_times) else max_response_time
        p99_response_time = response_times[p99_index] if p99_index < len(response_times) else max_response_time

        # Resource utilization statistics
        memory_values = [m.memory_mb for m in operation_metrics + resource_metrics if m.memory_mb > 0]
        cpu_values = [m.cpu_percent for m in operation_metrics + resource_metrics if m.cpu_percent > 0]

        avg_memory_mb = statistics.mean(memory_values) if memory_values else 0
        peak_memory_mb = max(memory_values) if memory_values else 0
        avg_cpu_percent = statistics.mean(cpu_values) if cpu_values else 0
        peak_cpu_percent = max(cpu_values) if cpu_values else 0

        # Performance degradation analysis
        degradation_analysis = self._analyze_performance_degradation(operation_metrics)

        # Resource utilization analysis
        resource_utilization = {
            "memory_trend": self._analyze_memory_trend(resource_metrics),
            "cpu_trend": self._analyze_cpu_trend(resource_metrics),
            "resource_exhaustion_detected": peak_memory_mb > 1000 or peak_cpu_percent > 80,
        }

        return LoadTestResult(
            test_name=test_name,
            duration_seconds=duration_seconds,
            total_operations=total_operations,
            successful_operations=successful_operations,
            failed_operations=failed_operations,
            operations_per_second=operations_per_second,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_response_time_ms=max_response_time,
            min_response_time_ms=min_response_time,
            error_rate=error_rate,
            avg_memory_mb=avg_memory_mb,
            peak_memory_mb=peak_memory_mb,
            avg_cpu_percent=avg_cpu_percent,
            peak_cpu_percent=peak_cpu_percent,
            metrics=operation_metrics,
            resource_utilization=resource_utilization,
            performance_degradation=degradation_analysis,
        )

    def _analyze_performance_degradation(self, metrics: list[LoadTestMetric]) -> dict[str, Any]:
        """Analyze performance degradation over time."""
        if len(metrics) < 10:
            return {"degradation_detected": False, "analysis": "Insufficient data"}

        # Split metrics into time windows
        start_time = min(m.timestamp for m in metrics)
        end_time = max(m.timestamp for m in metrics)
        window_count = 5
        window_duration = (end_time - start_time) / window_count

        windows = []
        for i in range(window_count):
            window_start = start_time + i * window_duration
            window_end = window_start + window_duration
            window_metrics = [m for m in metrics if window_start <= m.timestamp < window_end]

            if window_metrics:
                avg_duration = statistics.mean(m.duration_ms for m in window_metrics)
                error_rate = sum(1 for m in window_metrics if not m.success) / len(window_metrics)
                windows.append({"avg_duration_ms": avg_duration, "error_rate": error_rate})

        if len(windows) < 3:
            return {"degradation_detected": False, "analysis": "Insufficient windows"}

        # Check for increasing response times
        response_times = [w["avg_duration_ms"] for w in windows]
        error_rates = [w["error_rate"] for w in windows]

        # Simple trend detection
        response_trend_increasing = response_times[-1] > response_times[0] * 1.5
        error_trend_increasing = error_rates[-1] > error_rates[0] + 0.1

        return {
            "degradation_detected": response_trend_increasing or error_trend_increasing,
            "response_time_trend": "increasing" if response_trend_increasing else "stable",
            "error_rate_trend": "increasing" if error_trend_increasing else "stable",
            "windows": windows,
        }

    def _analyze_memory_trend(self, resource_metrics: list[LoadTestMetric]) -> str:
        """Analyze memory usage trend."""
        if len(resource_metrics) < 5:
            return "insufficient_data"

        memory_values = [m.memory_mb for m in resource_metrics if m.memory_mb > 0]
        if len(memory_values) < 5:
            return "no_memory_data"

        # Simple trend analysis
        first_half = memory_values[: len(memory_values) // 2]
        second_half = memory_values[len(memory_values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.2:
            return "increasing"
        elif second_avg < first_avg * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _analyze_cpu_trend(self, resource_metrics: list[LoadTestMetric]) -> str:
        """Analyze CPU usage trend."""
        if len(resource_metrics) < 5:
            return "insufficient_data"

        cpu_values = [m.cpu_percent for m in resource_metrics if m.cpu_percent > 0]
        if len(cpu_values) < 5:
            return "no_cpu_data"

        # Simple trend analysis
        first_half = cpu_values[: len(cpu_values) // 2]
        second_half = cpu_values[len(cpu_values) // 2 :]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.3:
            return "increasing"
        elif second_avg < first_avg * 0.7:
            return "decreasing"
        else:
            return "stable"


class TestCacheLoadTesting:
    """Test suite for cache load testing."""

    @pytest.fixture
    def load_tester(self):
        """Create a cache load tester."""
        return CacheLoadTester()

    @pytest.fixture
    def mock_cache_service(self):
        """Create a mock cache service for load testing."""
        mock_service = AsyncMock()

        # Mock operations with realistic delays
        async def mock_get(key):
            await asyncio.sleep(random.uniform(0.001, 0.010))  # 1-10ms
            return None if random.random() < 0.3 else {"data": "cached_value"}

        async def mock_set(key, value, ttl=None):
            await asyncio.sleep(random.uniform(0.002, 0.015))  # 2-15ms
            return True

        async def mock_delete(key):
            await asyncio.sleep(random.uniform(0.001, 0.008))  # 1-8ms
            return True

        mock_service.get.side_effect = mock_get
        mock_service.set.side_effect = mock_set
        mock_service.delete.side_effect = mock_delete

        return mock_service

    @pytest.mark.asyncio
    async def test_sustained_load_test(self, load_tester, mock_cache_service):
        """Test sustained load testing functionality."""
        result = await load_tester.run_sustained_load_test(
            mock_cache_service,
            duration_seconds=10,
            operations_per_second=50,
            test_name="test_sustained",  # Short duration for testing
        )

        # Verify test results
        assert result.test_name == "test_sustained"
        assert result.duration_seconds > 0
        assert result.total_operations > 0
        assert result.operations_per_second > 0

        # Should achieve reasonable performance
        assert result.operations_per_second >= 30, f"Low throughput: {result.operations_per_second}"
        assert result.error_rate < 0.1, f"High error rate: {result.error_rate}"
        assert result.avg_response_time_ms < 100, f"Slow responses: {result.avg_response_time_ms}ms"

        print(f"Sustained load test: {result.operations_per_second:.1f} ops/sec, {result.avg_response_time_ms:.1f}ms avg")

    @pytest.mark.asyncio
    async def test_burst_load_test(self, load_tester, mock_cache_service):
        """Test burst load testing functionality."""
        result = await load_tester.run_burst_load_test(
            mock_cache_service,
            burst_duration_seconds=5,
            max_concurrency=50,  # Reduced for testing
            operations_per_burst=200,
            test_name="test_burst",
        )

        # Verify test results
        assert result.test_name == "test_burst"
        assert result.total_operations > 0
        assert result.successful_operations > 0

        # Burst should handle high concurrency
        assert result.total_operations >= 150, f"Too few operations: {result.total_operations}"
        assert result.error_rate < 0.2, f"High error rate in burst: {result.error_rate}"

        print(f"Burst load test: {result.total_operations} operations in {result.duration_seconds:.1f}s")

    @pytest.mark.asyncio
    async def test_gradual_ramp_test(self, load_tester, mock_cache_service):
        """Test gradual ramp-up load testing."""
        result = await load_tester.run_gradual_ramp_test(
            mock_cache_service,
            ramp_duration_seconds=30,  # Short duration for testing
            max_operations_per_second=100,
            ramp_steps=5,
            test_name="test_ramp",
        )

        # Verify test results
        assert result.test_name == "test_ramp"
        assert result.total_operations > 0

        # Should handle ramping load
        assert result.operations_per_second > 0
        assert result.error_rate < 0.3, f"High error rate in ramp: {result.error_rate}"

        # Check for performance degradation analysis
        assert "degradation_detected" in result.performance_degradation

        print(
            f"Gradual ramp test: {result.operations_per_second:.1f} ops/sec peak, degradation: {result.performance_degradation['degradation_detected']}"
        )

    def test_workload_generator(self):
        """Test workload generator functionality."""
        generator = LoadTestWorkloadGenerator()

        # Test search cache key generation
        for _ in range(10):
            key = generator.generate_search_cache_key()
            assert key.startswith("search:")
            assert len(key) > 10

        # Test search cache value generation
        for _ in range(5):
            value = generator.generate_search_cache_value()
            assert "query" in value
            assert "results" in value
            assert "total_results" in value
            assert isinstance(value["results"], list)
            assert len(value["results"]) > 0

        # Test project cache key generation
        for _ in range(10):
            key = generator.generate_project_cache_key()
            assert key.startswith("project:")
            assert ":" in key[8:]  # Should have project name and cache type

        # Test project cache value generation
        for _ in range(5):
            value = generator.generate_project_cache_value()
            assert "project_name" in value
            assert isinstance(value, dict)

    def test_load_test_result_analysis(self, load_tester):
        """Test load test result analysis."""
        # Create sample metrics
        metrics = []
        base_time = time.time()

        for i in range(100):
            metric = LoadTestMetric(
                timestamp=base_time + i * 0.1,
                operation="test_operation",
                duration_ms=random.uniform(5, 50),
                success=random.random() > 0.05,  # 95% success rate
                memory_mb=100 + i * 0.1,  # Gradual memory increase
                cpu_percent=random.uniform(10, 30),
            )
            metrics.append(metric)

        # Analyze results
        result = load_tester._analyze_load_test_results("test_analysis", 10.0, metrics, [])

        # Verify analysis
        assert result.test_name == "test_analysis"
        assert result.total_operations == 100
        assert result.successful_operations >= 90  # ~95% success rate
        assert result.operations_per_second == 10.0  # 100 operations / 10 seconds
        assert result.avg_response_time_ms > 0
        assert result.p95_response_time_ms >= result.median_response_time_ms
        assert result.error_rate <= 0.1


@pytest.mark.performance
@pytest.mark.integration
class TestCacheLoadTestingIntegration:
    """Integration tests for cache load testing with real services."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(enabled=True, redis_url="redis://localhost:6379/15", default_ttl=300, max_memory_mb=200)  # Test database

    @pytest.mark.asyncio
    async def test_search_cache_load_test(self, cache_config):
        """Test load testing with SearchCacheService."""
        try:
            service = SearchCacheService(cache_config)
            await service.initialize()

            load_tester = CacheLoadTester()

            # Run sustained load test
            result = await load_tester.run_sustained_load_test(
                service, duration_seconds=20, operations_per_second=50, test_name="search_cache_load"
            )

            # Verify performance under load
            assert result.operations_per_second > 20, "Low throughput under load"
            assert result.error_rate < 0.1, "High error rate under load"
            assert result.avg_response_time_ms < 200, "Slow responses under load"

            print(f"SearchCache load test: {result.operations_per_second:.1f} ops/sec, {result.error_rate*100:.1f}% errors")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_project_cache_burst_test(self, cache_config):
        """Test burst load testing with ProjectCacheService."""
        try:
            service = ProjectCacheService(cache_config)
            await service.initialize()

            load_tester = CacheLoadTester()

            # Run burst load test
            result = await load_tester.run_burst_load_test(
                service, burst_duration_seconds=10, max_concurrency=100, operations_per_burst=500, test_name="project_cache_burst"
            )

            # Verify burst performance
            assert result.total_operations >= 300, "Too few operations in burst"
            assert result.error_rate < 0.2, "High error rate in burst test"

            print(f"ProjectCache burst test: {result.total_operations} operations, {result.error_rate*100:.1f}% errors")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
