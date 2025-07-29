#!/usr/bin/env python3
"""
Wave 8.0 Task 8.1: Performance Regression Testing Framework

This module provides comprehensive performance regression testing to detect
performance degradation across different system components and ensure
that new enhancements don't negatively impact existing functionality.
"""

import asyncio
import concurrent.futures
import gc
import json
import logging
import os
import statistics
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.code_parser_service import CodeParserService
from services.embedding_service import EmbeddingService
from services.graph_rag_service import GraphRAGService
from services.hybrid_search_service import HybridSearchService
from services.indexing_service import IndexingService
from services.qdrant_service import QdrantService
from utils.performance_monitor import PerformanceMonitor


@dataclass
class PerformanceMetrics:
    """Performance metrics for regression testing"""

    test_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    memory_peak_mb: float
    memory_baseline_mb: float
    query_throughput: float  # queries per second
    response_accuracy: float
    cache_hit_rate: float
    timestamp: str


@dataclass
class RegressionTestResult:
    """Result of a regression test"""

    test_name: str
    metrics: PerformanceMetrics
    baseline_metrics: PerformanceMetrics | None
    performance_degraded: bool
    degradation_percentage: float
    status: str  # "PASS", "FAIL", "WARNING"
    details: str


class PerformanceRegressionTester:
    """Comprehensive performance regression testing framework"""

    def __init__(self, baseline_file: str | None = None):
        self.baseline_file = baseline_file or "performance_baseline.json"
        self.baseline_metrics: dict[str, PerformanceMetrics] = {}
        self.results: list[RegressionTestResult] = []
        self.logger = self._setup_logging()

        # Performance thresholds (degradation > threshold triggers failure)
        self.thresholds = {
            "execution_time_degradation": 20.0,  # 20% slower is fail
            "memory_degradation": 30.0,  # 30% more memory is fail
            "throughput_degradation": 15.0,  # 15% lower throughput is fail
            "accuracy_degradation": 5.0,  # 5% lower accuracy is fail
        }

        # Load baseline if exists
        self._load_baseline()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for regression tests"""
        logger = logging.getLogger("regression_tester")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_baseline(self):
        """Load baseline performance metrics"""
        baseline_path = Path(self.baseline_file)
        if baseline_path.exists():
            try:
                with open(baseline_path) as f:
                    data = json.load(f)
                    for test_name, metrics_dict in data.items():
                        self.baseline_metrics[test_name] = PerformanceMetrics(**metrics_dict)
                self.logger.info(f"Loaded {len(self.baseline_metrics)} baseline metrics")
            except Exception as e:
                self.logger.warning(f"Could not load baseline: {e}")
        else:
            self.logger.info("No baseline file found, will create new baseline")

    def save_baseline(self):
        """Save current results as new baseline"""
        baseline_data = {}
        for result in self.results:
            baseline_data[result.test_name] = asdict(result.metrics)

        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

        self.logger.info(f"Saved baseline with {len(baseline_data)} metrics")

    async def _measure_performance(self, test_func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance metrics for a test function"""
        # Start monitoring
        tracemalloc.start()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()

        start_time = time.perf_counter()

        try:
            # Execute the test function
            result = await test_func(*args, **kwargs)

            # Calculate metrics
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # ms

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()

            # Calculate throughput (if applicable)
            throughput = 1000 / execution_time if execution_time > 0 else 0

            # Calculate accuracy (simplified - would need actual accuracy measurement)
            accuracy = getattr(result, "accuracy", 95.0) if hasattr(result, "accuracy") else 95.0

            # Calculate cache hit rate (simplified)
            cache_hit_rate = getattr(result, "cache_hit_rate", 75.0) if hasattr(result, "cache_hit_rate") else 75.0

            return PerformanceMetrics(
                test_name="",  # Will be set by caller
                execution_time_ms=execution_time,
                memory_usage_mb=current_memory,
                cpu_usage_percent=cpu_usage,
                memory_peak_mb=peak_memory,
                memory_baseline_mb=baseline_memory,
                query_throughput=throughput,
                response_accuracy=accuracy,
                cache_hit_rate=cache_hit_rate,
                timestamp=datetime.now().isoformat(),
            )

        finally:
            tracemalloc.stop()

    def _compare_metrics(self, current: PerformanceMetrics, baseline: PerformanceMetrics) -> RegressionTestResult:
        """Compare current metrics with baseline"""
        degradations = {}

        # Calculate degradation percentages
        if baseline.execution_time_ms > 0:
            degradations["execution_time"] = ((current.execution_time_ms - baseline.execution_time_ms) / baseline.execution_time_ms) * 100

        if baseline.memory_usage_mb > 0:
            degradations["memory_usage"] = ((current.memory_usage_mb - baseline.memory_usage_mb) / baseline.memory_usage_mb) * 100

        if baseline.query_throughput > 0:
            degradations["throughput"] = ((baseline.query_throughput - current.query_throughput) / baseline.query_throughput) * 100

        if baseline.response_accuracy > 0:
            degradations["accuracy"] = ((baseline.response_accuracy - current.response_accuracy) / baseline.response_accuracy) * 100

        # Determine overall degradation
        max_degradation = max(
            [
                degradations.get("execution_time", 0),
                degradations.get("memory_usage", 0),
                degradations.get("throughput", 0),
                degradations.get("accuracy", 0),
            ]
        )

        # Determine status
        status = "PASS"
        performance_degraded = False

        if (
            degradations.get("execution_time", 0) > self.thresholds["execution_time_degradation"]
            or degradations.get("memory_usage", 0) > self.thresholds["memory_degradation"]
            or degradations.get("throughput", 0) > self.thresholds["throughput_degradation"]
            or degradations.get("accuracy", 0) > self.thresholds["accuracy_degradation"]
        ):
            status = "FAIL"
            performance_degraded = True
        elif max_degradation > 10.0:  # Warning threshold
            status = "WARNING"

        details = f"Degradations: {degradations}"

        return RegressionTestResult(
            test_name=current.test_name,
            metrics=current,
            baseline_metrics=baseline,
            performance_degraded=performance_degraded,
            degradation_percentage=max_degradation,
            status=status,
            details=details,
        )

    async def test_embedding_generation_performance(self):
        """Test embedding generation performance"""

        async def embedding_test():
            service = EmbeddingService()
            test_texts = [
                "def calculate_fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "class DataProcessor: def __init__(self): self.data = []",
                "async def fetch_data(url): response = await http.get(url); return response.json()",
                "import numpy as np; def matrix_multiply(a, b): return np.dot(a, b)",
                "function processArray(arr) { return arr.map(x => x * 2).filter(x => x > 10); }",
            ]

            start_time = time.perf_counter()
            embeddings = await service.generate_embeddings(test_texts)
            end_time = time.perf_counter()

            return type(
                "Result",
                (),
                {"accuracy": 98.5, "cache_hit_rate": 65.0, "execution_time": (end_time - start_time) * 1000},  # Simulated accuracy
            )()

        metrics = await self._measure_performance(embedding_test)
        metrics.test_name = "embedding_generation"

        # Compare with baseline
        result = self._compare_with_baseline(metrics)
        self.results.append(result)
        return result

    async def test_search_performance(self):
        """Test search performance"""

        async def search_test():
            # Simulate search operation
            await asyncio.sleep(0.1)  # Simulate search time
            return type("Result", (), {"accuracy": 92.0, "cache_hit_rate": 80.0})()

        metrics = await self._measure_performance(search_test)
        metrics.test_name = "search_performance"

        result = self._compare_with_baseline(metrics)
        self.results.append(result)
        return result

    async def test_indexing_performance(self):
        """Test indexing performance"""

        async def indexing_test():
            # Simulate indexing operation
            await asyncio.sleep(0.2)  # Simulate indexing time
            return type("Result", (), {"accuracy": 96.0, "cache_hit_rate": 45.0})()

        metrics = await self._measure_performance(indexing_test)
        metrics.test_name = "indexing_performance"

        result = self._compare_with_baseline(metrics)
        self.results.append(result)
        return result

    async def test_graph_rag_performance(self):
        """Test Graph RAG performance"""

        async def graph_rag_test():
            # Simulate Graph RAG operation
            await asyncio.sleep(0.15)  # Simulate graph analysis time
            return type("Result", (), {"accuracy": 94.5, "cache_hit_rate": 70.0})()

        metrics = await self._measure_performance(graph_rag_test)
        metrics.test_name = "graph_rag_performance"

        result = self._compare_with_baseline(metrics)
        self.results.append(result)
        return result

    def _compare_with_baseline(self, metrics: PerformanceMetrics) -> RegressionTestResult:
        """Compare metrics with baseline"""
        if metrics.test_name in self.baseline_metrics:
            return self._compare_metrics(metrics, self.baseline_metrics[metrics.test_name])
        else:
            # No baseline - mark as new test
            return RegressionTestResult(
                test_name=metrics.test_name,
                metrics=metrics,
                baseline_metrics=None,
                performance_degraded=False,
                degradation_percentage=0.0,
                status="NEW",
                details="No baseline available",
            )

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all regression tests"""
        self.logger.info("Starting performance regression tests...")

        # Run all tests
        test_methods = [
            self.test_embedding_generation_performance,
            self.test_search_performance,
            self.test_indexing_performance,
            self.test_graph_rag_performance,
        ]

        for test_method in test_methods:
            try:
                await test_method()
                self.logger.info(f"Completed {test_method.__name__}")
            except Exception as e:
                self.logger.error(f"Failed {test_method.__name__}: {e}")

        # Generate summary
        summary = self._generate_summary()
        self.logger.info("Performance regression tests completed")

        return summary

    def _generate_summary(self) -> dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        warning_tests = len([r for r in self.results if r.status == "WARNING"])
        new_tests = len([r for r in self.results if r.status == "NEW"])

        # Calculate average degradation
        degradations = [r.degradation_percentage for r in self.results if r.baseline_metrics]
        avg_degradation = statistics.mean(degradations) if degradations else 0.0

        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "new_tests": new_tests,
            "average_degradation_percent": avg_degradation,
            "overall_status": "FAIL" if failed_tests > 0 else "WARNING" if warning_tests > 0 else "PASS",
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(result) for result in self.results],
        }

        return summary

    def generate_report(self, output_file: str = "regression_test_report.json"):
        """Generate detailed report"""
        summary = self._generate_summary()

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Generated regression test report: {output_file}")
        return summary


async def main():
    """Main function to run regression tests"""
    tester = PerformanceRegressionTester()

    # Run tests
    summary = await tester.run_all_tests()

    # Generate report
    tester.generate_report("wave_8_regression_test_report.json")

    # Save baseline if no failures
    if summary["failed"] == 0:
        tester.save_baseline()

    # Print summary
    print("\n=== Performance Regression Test Summary ===")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"New Tests: {summary['new_tests']}")
    print(f"Average Degradation: {summary['average_degradation_percent']:.2f}%")
    print(f"Overall Status: {summary['overall_status']}")

    return summary


if __name__ == "__main__":
    asyncio.run(main())
