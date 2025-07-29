#!/usr/bin/env python3
"""
Wave 8.0 Task 8.7: Stress Testing Framework

This module tests system limits and breaking points by applying extreme loads,
resource constraints, and edge conditions to identify failure modes,
bottlenecks, and system boundaries.
"""

import asyncio
import concurrent.futures
import gc
import json
import logging
import os
import random
import signal
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.embedding_service import EmbeddingService
from services.graph_rag_service import GraphRAGService
from services.hybrid_search_service import HybridSearchService


@dataclass
class StressTestConfig:
    """Configuration for stress testing scenarios"""

    test_name: str
    duration_seconds: int
    concurrent_users: int
    queries_per_user: int
    query_complexity: str  # "simple", "medium", "complex", "mixed"
    memory_pressure: bool
    cpu_pressure: bool
    disk_pressure: bool
    network_simulation: bool
    failure_injection: bool


@dataclass
class SystemMetrics:
    """System metrics during stress testing"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    open_files: int
    active_threads: int
    response_time_ms: float
    error_count: int
    success_count: int


@dataclass
class StressTestResult:
    """Result of stress testing"""

    config: StressTestConfig
    start_time: str
    end_time: str
    duration_seconds: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    error_rate_percent: float
    peak_metrics: SystemMetrics
    average_metrics: dict[str, float]
    breaking_point: dict[str, Any] | None
    failure_modes: list[str]
    recovery_time_seconds: float
    recommendations: list[str]
    metrics_timeline: list[SystemMetrics]


class StressTester:
    """Comprehensive stress testing framework"""

    def __init__(self):
        self.results: list[StressTestResult] = []
        self.logger = self._setup_logging()
        self.stop_flag = threading.Event()
        self.metrics_collection_interval = 1.0  # seconds

        # Stress test configurations
        self.stress_configs = self._define_stress_configs()

        # Failure thresholds
        self.failure_thresholds = {
            "cpu_percent": 95.0,
            "memory_percent": 90.0,
            "error_rate_percent": 50.0,
            "response_time_ms": 30000.0,  # 30 seconds
            "open_files": 1000,
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for stress tests"""
        logger = logging.getLogger("stress_tester")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _define_stress_configs(self) -> list[StressTestConfig]:
        """Define stress testing configurations"""
        return [
            # Gradual load increase
            StressTestConfig("gradual_load_test", 120, 10, 50, "mixed", False, False, False, False, False),
            # High concurrency stress
            StressTestConfig("high_concurrency_stress", 180, 25, 100, "medium", False, False, False, False, False),
            # Memory pressure test
            StressTestConfig("memory_pressure_test", 150, 15, 75, "complex", True, False, False, False, False),
            # CPU stress test
            StressTestConfig("cpu_stress_test", 120, 20, 60, "complex", False, True, False, False, False),
            # Combined resource stress
            StressTestConfig("combined_resource_stress", 200, 30, 80, "mixed", True, True, True, False, False),
            # Failure injection test
            StressTestConfig("failure_injection_test", 100, 12, 40, "medium", False, False, False, False, True),
            # Extreme load test (breaking point)
            StressTestConfig("extreme_load_test", 300, 50, 150, "complex", True, True, True, True, True),
        ]

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            process = psutil.Process()

            # CPU and Memory
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            memory_mb = memory_info.rss / 1024 / 1024

            # Disk I/O
            try:
                disk_io = process.io_counters()
                disk_read_mb = disk_io.read_bytes / 1024 / 1024
                disk_write_mb = disk_io.write_bytes / 1024 / 1024
            except:
                disk_read_mb = disk_write_mb = 0.0

            # Network (system-wide)
            try:
                net_io = psutil.net_io_counters()
                net_sent = net_io.bytes_sent
                net_recv = net_io.bytes_recv
            except:
                net_sent = net_recv = 0.0

            # Process info
            open_files = len(process.open_files())
            active_threads = process.num_threads()

            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_bytes_sent=net_sent,
                network_bytes_recv=net_recv,
                open_files=open_files,
                active_threads=active_threads,
                response_time_ms=0.0,  # Will be updated during tests
                error_count=0,  # Will be updated during tests
                success_count=0,  # Will be updated during tests
            )

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_bytes_sent=0.0,
                network_bytes_recv=0.0,
                open_files=0,
                active_threads=0,
                response_time_ms=0.0,
                error_count=0,
                success_count=0,
            )

    async def _metrics_collector(self, metrics_list: list[SystemMetrics]):
        """Background metrics collection"""
        while not self.stop_flag.is_set():
            try:
                metrics = self._collect_system_metrics()
                metrics_list.append(metrics)
                await asyncio.sleep(self.metrics_collection_interval)
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")

    def _apply_memory_pressure(self):
        """Apply memory pressure to stress test memory management"""
        try:
            # Allocate large chunks of memory
            memory_chunks = []
            chunk_size = 10 * 1024 * 1024  # 10MB chunks

            for i in range(50):  # 500MB total
                chunk = bytearray(chunk_size)
                memory_chunks.append(chunk)
                time.sleep(0.1)  # Gradual allocation

            # Hold memory for a while
            time.sleep(30)

            # Clean up
            del memory_chunks
            gc.collect()

        except Exception as e:
            self.logger.error(f"Memory pressure error: {e}")

    def _apply_cpu_pressure(self):
        """Apply CPU pressure to stress test CPU usage"""
        try:
            # CPU-intensive calculations
            def cpu_intensive_task():
                end_time = time.time() + 60  # Run for 60 seconds
                while time.time() < end_time:
                    # Fibonacci calculation (CPU intensive)
                    for n in range(35):
                        self._fibonacci(n)

            # Start multiple CPU-intensive threads
            threads = []
            for i in range(psutil.cpu_count()):
                thread = threading.Thread(target=cpu_intensive_task)
                thread.start()
                threads.append(thread)

            # Wait for completion
            for thread in threads:
                thread.join()

        except Exception as e:
            self.logger.error(f"CPU pressure error: {e}")

    def _fibonacci(self, n: int) -> int:
        """CPU-intensive Fibonacci calculation"""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)

    def _apply_disk_pressure(self):
        """Apply disk I/O pressure"""
        try:
            # Create temporary files and perform I/O operations
            temp_dir = tempfile.mkdtemp()

            for i in range(10):
                file_path = os.path.join(temp_dir, f"stress_file_{i}.dat")

                # Write large file
                with open(file_path, "wb") as f:
                    data = os.urandom(1024 * 1024)  # 1MB of random data
                    for _ in range(50):  # 50MB total per file
                        f.write(data)

                # Read file back
                with open(file_path, "rb") as f:
                    while f.read(1024 * 1024):  # Read in 1MB chunks
                        pass

            # Clean up
            import shutil

            shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error(f"Disk pressure error: {e}")

    async def _simulate_user_load(self, user_id: int, config: StressTestConfig, results: dict[str, Any]):
        """Simulate user load for stress testing"""
        user_results = {"queries": 0, "successes": 0, "errors": 0, "response_times": []}

        try:
            for query_num in range(config.queries_per_user):
                if self.stop_flag.is_set():
                    break

                # Generate query based on complexity
                query = self._generate_stress_query(config.query_complexity)

                start_time = time.time()
                success = False

                try:
                    # Execute query with potential failure injection
                    if config.failure_injection and random.random() < 0.1:  # 10% failure rate
                        raise Exception("Simulated failure")

                    result = await self._execute_stress_query(query, config.query_complexity)
                    success = True
                    user_results["successes"] += 1

                except Exception as e:
                    user_results["errors"] += 1
                    if "Simulated failure" not in str(e):
                        self.logger.warning(f"User {user_id} query failed: {e}")

                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # ms
                user_results["response_times"].append(response_time)
                user_results["queries"] += 1

                # Variable delay between queries
                delay = random.uniform(0.1, 2.0) if config.query_complexity == "simple" else random.uniform(0.5, 3.0)
                await asyncio.sleep(delay)

        except Exception as e:
            self.logger.error(f"User {user_id} simulation error: {e}")

        # Store results
        with threading.Lock():
            for key, value in user_results.items():
                if key in results:
                    if isinstance(value, list):
                        results[key].extend(value)
                    else:
                        results[key] += value
                else:
                    results[key] = value if isinstance(value, list) else [value]

    def _generate_stress_query(self, complexity: str) -> str:
        """Generate queries for stress testing"""
        queries = {
            "simple": ["def function", "class MyClass", "import os", "return value", "if condition", "for loop", "try except", "async def"],
            "medium": [
                "find database connection functions",
                "search for error handling patterns",
                "locate authentication methods",
                "find API endpoint implementations",
                "search for logging utilities",
            ],
            "complex": [
                "trace function calls from login to database storage",
                "analyze dependency chain for user authentication system",
                "find similar implementation patterns across all modules",
                "map complete data flow through the indexing pipeline",
                "identify all architectural patterns in the codebase",
            ],
            "mixed": [],  # Will randomly select from all categories
        }

        if complexity == "mixed":
            all_queries = []
            for query_list in queries.values():
                all_queries.extend(query_list)
            return random.choice(all_queries)
        else:
            return random.choice(queries.get(complexity, queries["simple"]))

    async def _execute_stress_query(self, query: str, complexity: str) -> dict[str, Any]:
        """Execute query with stress testing considerations"""
        # Simulate different execution times based on complexity
        if complexity == "simple":
            await asyncio.sleep(random.uniform(0.01, 0.1))
        elif complexity == "medium":
            await asyncio.sleep(random.uniform(0.1, 0.5))
        elif complexity == "complex":
            await asyncio.sleep(random.uniform(0.5, 2.0))
        else:  # mixed
            await asyncio.sleep(random.uniform(0.01, 2.0))

        # Simulate resource usage
        if random.random() < 0.1:  # 10% chance of memory allocation
            temp_data = [random.random() for _ in range(10000)]

        return {"query": query, "results": random.randint(1, 50), "complexity": complexity}

    def _detect_breaking_point(self, metrics_list: list[SystemMetrics]) -> dict[str, Any] | None:
        """Detect system breaking point from metrics"""
        breaking_point = None

        for i, metrics in enumerate(metrics_list):
            failure_conditions = []

            if metrics.cpu_percent > self.failure_thresholds["cpu_percent"]:
                failure_conditions.append(f"CPU usage: {metrics.cpu_percent:.1f}%")

            if metrics.memory_percent > self.failure_thresholds["memory_percent"]:
                failure_conditions.append(f"Memory usage: {metrics.memory_percent:.1f}%")

            if metrics.response_time_ms > self.failure_thresholds["response_time_ms"]:
                failure_conditions.append(f"Response time: {metrics.response_time_ms:.1f}ms")

            if metrics.open_files > self.failure_thresholds["open_files"]:
                failure_conditions.append(f"Open files: {metrics.open_files}")

            if failure_conditions:
                breaking_point = {
                    "timestamp": metrics.timestamp,
                    "time_to_failure_seconds": i * self.metrics_collection_interval,
                    "failure_conditions": failure_conditions,
                    "metrics_at_failure": asdict(metrics),
                }
                break

        return breaking_point

    def _identify_failure_modes(self, metrics_list: list[SystemMetrics], user_results: dict[str, Any]) -> list[str]:
        """Identify failure modes from test results"""
        failure_modes = []

        # High error rate
        if user_results.get("errors", 0) > 0:
            error_rate = (user_results["errors"] / max(1, user_results["queries"])) * 100
            if error_rate > 10:
                failure_modes.append(f"High error rate: {error_rate:.1f}%")

        # Memory exhaustion
        if metrics_list:
            max_memory = max(m.memory_percent for m in metrics_list)
            if max_memory > 85:
                failure_modes.append(f"Memory exhaustion: {max_memory:.1f}%")

        # CPU saturation
        if metrics_list:
            avg_cpu = statistics.mean(m.cpu_percent for m in metrics_list)
            if avg_cpu > 80:
                failure_modes.append(f"CPU saturation: {avg_cpu:.1f}%")

        # Response time degradation
        if user_results.get("response_times"):
            p95_response_time = statistics.quantiles(user_results["response_times"], n=20)[18]
            if p95_response_time > 10000:  # 10 seconds
                failure_modes.append(f"Response time degradation: P95 = {p95_response_time:.1f}ms")

        # Resource leaks
        if metrics_list and len(metrics_list) > 10:
            initial_files = metrics_list[0].open_files
            final_files = metrics_list[-1].open_files
            if final_files > initial_files + 100:
                failure_modes.append(f"File descriptor leak: {final_files - initial_files} files")

        return failure_modes

    def _calculate_recovery_time(self, metrics_list: list[SystemMetrics]) -> float:
        """Calculate system recovery time after stress"""
        if len(metrics_list) < 10:
            return 0.0

        # Find when metrics return to normal after peak stress
        # This is a simplified calculation
        peak_cpu = max(m.cpu_percent for m in metrics_list)
        peak_memory = max(m.memory_percent for m in metrics_list)

        # Look for recovery (metrics drop to < 50% of peak)
        recovery_threshold_cpu = peak_cpu * 0.5
        recovery_threshold_memory = peak_memory * 0.5

        recovery_index = None
        for i in range(len(metrics_list) - 1, -1, -1):
            metrics = metrics_list[i]
            if metrics.cpu_percent < recovery_threshold_cpu and metrics.memory_percent < recovery_threshold_memory:
                recovery_index = i
                break

        if recovery_index and recovery_index < len(metrics_list) - 1:
            recovery_time = (len(metrics_list) - 1 - recovery_index) * self.metrics_collection_interval
            return recovery_time
        else:
            return 0.0  # No recovery detected

    async def run_stress_test(self, config: StressTestConfig) -> StressTestResult:
        """Run a single stress test"""
        self.logger.info(f"Starting stress test: {config.test_name}")

        start_time = datetime.now()
        self.stop_flag.clear()

        # Initialize metrics collection
        metrics_list = []
        metrics_task = asyncio.create_task(self._metrics_collector(metrics_list))

        # Initialize user results
        user_results = {"queries": 0, "successes": 0, "errors": 0, "response_times": []}

        try:
            # Apply stress conditions
            stress_tasks = []

            if config.memory_pressure:
                stress_tasks.append(asyncio.create_task(asyncio.to_thread(self._apply_memory_pressure)))

            if config.cpu_pressure:
                stress_tasks.append(asyncio.create_task(asyncio.to_thread(self._apply_cpu_pressure)))

            if config.disk_pressure:
                stress_tasks.append(asyncio.create_task(asyncio.to_thread(self._apply_disk_pressure)))

            # Create user load
            user_tasks = []
            for user_id in range(config.concurrent_users):
                task = asyncio.create_task(self._simulate_user_load(user_id, config, user_results))
                user_tasks.append(task)

            # Run test for specified duration
            await asyncio.sleep(config.duration_seconds)

            # Stop all tasks
            self.stop_flag.set()

            # Wait for user tasks to complete
            await asyncio.gather(*user_tasks, return_exceptions=True)

            # Wait for stress tasks to complete
            if stress_tasks:
                await asyncio.gather(*stress_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Stress test error: {e}")

        finally:
            # Stop metrics collection
            self.stop_flag.set()
            metrics_task.cancel()

        end_time = datetime.now()
        actual_duration = (end_time - start_time).total_seconds()

        # Calculate results
        total_queries = user_results.get("queries", 0)
        successful_queries = user_results.get("successes", 0)
        failed_queries = user_results.get("errors", 0)
        error_rate = (failed_queries / max(1, total_queries)) * 100

        # Find peak metrics
        if metrics_list:
            peak_metrics = max(metrics_list, key=lambda m: m.cpu_percent + m.memory_percent)

            # Calculate average metrics
            avg_metrics = {
                "cpu_percent": statistics.mean(m.cpu_percent for m in metrics_list),
                "memory_percent": statistics.mean(m.memory_percent for m in metrics_list),
                "memory_mb": statistics.mean(m.memory_mb for m in metrics_list),
                "open_files": statistics.mean(m.open_files for m in metrics_list),
                "active_threads": statistics.mean(m.active_threads for m in metrics_list),
            }
        else:
            peak_metrics = self._collect_system_metrics()
            avg_metrics = {}

        # Update peak metrics with response time info
        if user_results.get("response_times"):
            peak_metrics.response_time_ms = max(user_results["response_times"])

        peak_metrics.error_count = failed_queries
        peak_metrics.success_count = successful_queries

        # Analyze results
        breaking_point = self._detect_breaking_point(metrics_list)
        failure_modes = self._identify_failure_modes(metrics_list, user_results)
        recovery_time = self._calculate_recovery_time(metrics_list)

        # Generate recommendations
        recommendations = self._generate_stress_recommendations(config, failure_modes, breaking_point, avg_metrics)

        result = StressTestResult(
            config=config,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=actual_duration,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            error_rate_percent=error_rate,
            peak_metrics=peak_metrics,
            average_metrics=avg_metrics,
            breaking_point=breaking_point,
            failure_modes=failure_modes,
            recovery_time_seconds=recovery_time,
            recommendations=recommendations,
            metrics_timeline=metrics_list,
        )

        self.results.append(result)
        self.logger.info(f"Completed stress test: {config.test_name}")

        return result

    def _generate_stress_recommendations(
        self, config: StressTestConfig, failure_modes: list[str], breaking_point: dict[str, Any] | None, avg_metrics: dict[str, float]
    ) -> list[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []

        if breaking_point:
            recommendations.append(f"System breaking point reached after {breaking_point['time_to_failure_seconds']:.1f}s")

        for failure_mode in failure_modes:
            if "error rate" in failure_mode.lower():
                recommendations.append("Improve error handling and retry mechanisms")
            elif "memory" in failure_mode.lower():
                recommendations.append("Optimize memory usage and implement garbage collection")
            elif "cpu" in failure_mode.lower():
                recommendations.append("Optimize CPU-intensive operations and add caching")
            elif "response time" in failure_mode.lower():
                recommendations.append("Implement query optimization and result caching")
            elif "leak" in failure_mode.lower():
                recommendations.append("Fix resource leaks and improve cleanup")

        if avg_metrics.get("cpu_percent", 0) > 60:
            recommendations.append("Consider horizontal scaling for CPU-intensive workloads")

        if avg_metrics.get("memory_percent", 0) > 70:
            recommendations.append("Increase memory allocation or optimize memory usage")

        if not failure_modes and not breaking_point:
            recommendations.append("System handled stress well - consider testing higher loads")

        return recommendations

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all stress tests"""
        self.logger.info(f"Starting {len(self.stress_configs)} stress tests...")

        for config in self.stress_configs:
            try:
                await self.run_stress_test(config)
                self.logger.info(f"Completed stress test: {config.test_name}")

                # Recovery period between tests
                await asyncio.sleep(10)
                gc.collect()

            except Exception as e:
                self.logger.error(f"Failed stress test {config.test_name}: {e}")

        # Generate summary
        summary = self._generate_summary()
        self.logger.info("All stress tests completed")

        return summary

    def _generate_summary(self) -> dict[str, Any]:
        """Generate comprehensive stress test summary"""
        total_tests = len(self.results)
        tests_with_failures = len([r for r in self.results if r.failure_modes])
        tests_with_breaking_points = len([r for r in self.results if r.breaking_point])

        # Aggregate statistics
        if self.results:
            avg_error_rate = statistics.mean([r.error_rate_percent for r in self.results])
            max_concurrent_users = max([r.config.concurrent_users for r in self.results])
            avg_recovery_time = statistics.mean([r.recovery_time_seconds for r in self.results])

            # Peak resource usage across all tests
            peak_cpu = max([r.peak_metrics.cpu_percent for r in self.results])
            peak_memory = max([r.peak_metrics.memory_percent for r in self.results])
            peak_response_time = max([r.peak_metrics.response_time_ms for r in self.results])
        else:
            avg_error_rate = 0
            max_concurrent_users = 0
            avg_recovery_time = 0
            peak_cpu = peak_memory = peak_response_time = 0

        # Collect all failure modes
        all_failure_modes = []
        for result in self.results:
            all_failure_modes.extend(result.failure_modes)

        failure_mode_frequency = {}
        for mode in all_failure_modes:
            failure_mode_frequency[mode] = failure_mode_frequency.get(mode, 0) + 1

        summary = {
            "total_tests": total_tests,
            "tests_with_failures": tests_with_failures,
            "tests_with_breaking_points": tests_with_breaking_points,
            "system_stability_score": ((total_tests - tests_with_failures) / max(1, total_tests)) * 100,
            "stress_statistics": {
                "average_error_rate_percent": avg_error_rate,
                "max_concurrent_users_tested": max_concurrent_users,
                "average_recovery_time_seconds": avg_recovery_time,
                "peak_cpu_percent": peak_cpu,
                "peak_memory_percent": peak_memory,
                "peak_response_time_ms": peak_response_time,
            },
            "common_failure_modes": failure_mode_frequency,
            "test_results": [asdict(result) for result in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def generate_report(self, output_file: str = "stress_test_report.json"):
        """Generate detailed stress test report"""
        summary = self._generate_summary()

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Generated stress test report: {output_file}")
        return summary

    def print_summary(self):
        """Print human-readable stress test summary"""
        summary = self._generate_summary()

        print("\n=== Stress Test Summary ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Tests with Failures: {summary['tests_with_failures']}")
        print(f"Tests with Breaking Points: {summary['tests_with_breaking_points']}")
        print(f"System Stability Score: {summary['system_stability_score']:.1f}%")

        stats = summary["stress_statistics"]
        print("\nStress Statistics:")
        print(f"  Average Error Rate: {stats['average_error_rate_percent']:.2f}%")
        print(f"  Max Concurrent Users: {stats['max_concurrent_users_tested']}")
        print(f"  Average Recovery Time: {stats['average_recovery_time_seconds']:.1f}s")
        print(f"  Peak CPU Usage: {stats['peak_cpu_percent']:.1f}%")
        print(f"  Peak Memory Usage: {stats['peak_memory_percent']:.1f}%")
        print(f"  Peak Response Time: {stats['peak_response_time_ms']:.1f}ms")

        if summary["common_failure_modes"]:
            print("\nCommon Failure Modes:")
            for mode, frequency in summary["common_failure_modes"].items():
                print(f"  {mode}: {frequency} occurrences")


async def main():
    """Main function to run stress tests"""
    tester = StressTester()

    # Run tests
    print("Running stress tests...")
    summary = await tester.run_all_tests()

    # Generate report
    tester.generate_report("wave_8_stress_test_report.json")

    # Print summary
    tester.print_summary()

    return summary


if __name__ == "__main__":
    asyncio.run(main())
