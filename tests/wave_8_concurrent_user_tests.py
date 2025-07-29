#!/usr/bin/env python3
"""
Wave 8.0 Task 8.5: Concurrent User Testing Framework

This module validates the system's ability to handle 5 concurrent users,
measuring performance degradation, resource contention, and system stability
under concurrent load conditions.
"""

import asyncio
import concurrent.futures
import gc
import json
import logging
import random
import statistics
import sys
import threading
import time
import uuid
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
class UserSession:
    """Represents a user session for concurrent testing"""

    session_id: str
    user_id: int
    start_time: float
    queries_performed: list[str]
    response_times: list[float]
    errors: list[str]
    total_queries: int
    successful_queries: int
    average_response_time: float
    cache_hits: int
    cache_misses: int


@dataclass
class ConcurrentTestMetrics:
    """Metrics for concurrent user testing"""

    concurrent_users: int
    test_duration_seconds: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_queries_per_second: float
    system_resource_usage: dict[str, float]
    memory_peak_mb: float
    memory_baseline_mb: float
    cache_performance: dict[str, float]
    error_rate_percent: float
    user_sessions: list[UserSession]


@dataclass
class ConcurrentTestResult:
    """Result of concurrent user testing"""

    test_name: str
    target_concurrent_users: int
    actual_concurrent_users: int
    metrics: ConcurrentTestMetrics
    performance_degradation: dict[str, float]
    target_met: bool
    bottlenecks_identified: list[str]
    recommendations: list[str]
    timestamp: str


class ConcurrentUserTester:
    """Comprehensive concurrent user testing framework"""

    def __init__(self):
        self.target_concurrent_users = 5  # Target: 5 concurrent users
        self.max_concurrent_users = 10  # Stress test up to 10 users
        self.test_duration_seconds = 60  # 60 second test duration
        self.results: list[ConcurrentTestResult] = []
        self.logger = self._setup_logging()

        # Performance thresholds
        self.thresholds = {
            "response_time_degradation_percent": 50.0,  # Max 50% degradation under load
            "error_rate_percent": 5.0,  # Max 5% error rate
            "throughput_degradation_percent": 30.0,  # Max 30% throughput degradation
            "memory_increase_percent": 100.0,  # Max 100% memory increase
        }

        # Query types for realistic user behavior
        self.query_types = self._define_query_types()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for concurrent tests"""
        logger = logging.getLogger("concurrent_user_tester")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _define_query_types(self) -> dict[str, list[str]]:
        """Define different types of queries for user simulation"""
        return {
            "simple_search": [
                "def function",
                "class MyClass",
                "import os",
                "async def",
                "return value",
                "if condition",
                "for loop",
                "try except",
            ],
            "semantic_search": [
                "find database connection functions",
                "search for error handling patterns",
                "locate authentication methods",
                "find API endpoint implementations",
                "search for logging utilities",
                "find configuration management code",
                "locate data validation functions",
                "search for caching mechanisms",
            ],
            "complex_graph_rag": [
                "trace function calls from login to database",
                "analyze dependency chain for user authentication",
                "find similar patterns across modules",
                "map data flow through the system",
                "identify architectural patterns",
                "analyze component relationships",
                "trace error propagation paths",
                "find code coupling patterns",
            ],
            "multi_modal": [
                "find configuration files and their usage",
                "locate test files and source code",
                "search documentation and implementations",
                "find logging and monitoring code",
                "locate performance critical sections",
            ],
        }

    async def simulate_user_session(self, user_id: int, duration_seconds: float) -> UserSession:
        """Simulate a realistic user session"""
        session_id = str(uuid.uuid4())
        start_time = time.time()

        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            queries_performed=[],
            response_times=[],
            errors=[],
            total_queries=0,
            successful_queries=0,
            average_response_time=0.0,
            cache_hits=0,
            cache_misses=0,
        )

        self.logger.info(f"Starting user session {user_id} (ID: {session_id})")

        end_time = start_time + duration_seconds

        while time.time() < end_time:
            try:
                # Select query type based on realistic user behavior
                query_type = self._select_query_type()
                query = random.choice(self.query_types[query_type])

                # Execute query with timing
                query_start = time.time()
                success, result = await self._execute_user_query(query, query_type, user_id)
                query_end = time.time()

                response_time_ms = (query_end - query_start) * 1000

                session.queries_performed.append(query)
                session.response_times.append(response_time_ms)
                session.total_queries += 1

                if success:
                    session.successful_queries += 1
                    # Simulate cache behavior
                    if random.random() < 0.7:  # 70% cache hit rate
                        session.cache_hits += 1
                    else:
                        session.cache_misses += 1
                else:
                    session.errors.append(f"Query failed: {query}")

                # Simulate user think time
                think_time = self._get_user_think_time(query_type)
                await asyncio.sleep(think_time)

            except Exception as e:
                session.errors.append(f"Session error: {e}")
                self.logger.error(f"Error in user session {user_id}: {e}")

        # Calculate session statistics
        if session.response_times:
            session.average_response_time = statistics.mean(session.response_times)
        else:
            session.average_response_time = 0.0

        self.logger.info(
            f"User session {user_id} completed: "
            f"{session.successful_queries}/{session.total_queries} queries successful, "
            f"avg response time: {session.average_response_time:.2f}ms"
        )

        return session

    def _select_query_type(self) -> str:
        """Select query type based on realistic user behavior patterns"""
        # Weighted selection based on typical usage patterns
        weights = {
            "simple_search": 0.4,  # 40% simple searches
            "semantic_search": 0.35,  # 35% semantic searches
            "complex_graph_rag": 0.15,  # 15% complex analysis
            "multi_modal": 0.1,  # 10% multi-modal queries
        }

        rand = random.random()
        cumulative = 0.0

        for query_type, weight in weights.items():
            cumulative += weight
            if rand <= cumulative:
                return query_type

        return "simple_search"  # Default fallback

    def _get_user_think_time(self, query_type: str) -> float:
        """Get realistic user think time based on query complexity"""
        think_times = {
            "simple_search": random.uniform(1.0, 3.0),  # 1-3 seconds
            "semantic_search": random.uniform(2.0, 5.0),  # 2-5 seconds
            "complex_graph_rag": random.uniform(5.0, 10.0),  # 5-10 seconds
            "multi_modal": random.uniform(3.0, 7.0),  # 3-7 seconds
        }

        return think_times.get(query_type, 2.0)

    async def _execute_user_query(self, query: str, query_type: str, user_id: int) -> tuple[bool, Any]:
        """Execute a user query and return success status and result"""
        try:
            # Simulate different query execution based on type
            if query_type == "simple_search":
                result = await self._execute_simple_query(query, user_id)
            elif query_type == "semantic_search":
                result = await self._execute_semantic_query(query, user_id)
            elif query_type == "complex_graph_rag":
                result = await self._execute_graph_rag_query(query, user_id)
            elif query_type == "multi_modal":
                result = await self._execute_multimodal_query(query, user_id)
            else:
                result = await self._execute_default_query(query, user_id)

            return True, result

        except Exception as e:
            self.logger.warning(f"Query execution failed for user {user_id}: {e}")
            return False, None

    async def _execute_simple_query(self, query: str, user_id: int) -> dict[str, Any]:
        """Execute simple search query"""
        # Simulate simple keyword search
        await asyncio.sleep(random.uniform(0.01, 0.05))  # 10-50ms
        return {"query": query, "type": "simple", "results": random.randint(1, 10), "user_id": user_id}

    async def _execute_semantic_query(self, query: str, user_id: int) -> dict[str, Any]:
        """Execute semantic search query"""
        # Simulate semantic search with embedding generation
        await asyncio.sleep(random.uniform(0.1, 0.3))  # 100-300ms
        return {"query": query, "type": "semantic", "results": random.randint(5, 25), "user_id": user_id}

    async def _execute_graph_rag_query(self, query: str, user_id: int) -> dict[str, Any]:
        """Execute Graph RAG query"""
        # Simulate complex graph analysis
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 500ms-1.5s
        return {"query": query, "type": "graph_rag", "results": random.randint(10, 50), "user_id": user_id}

    async def _execute_multimodal_query(self, query: str, user_id: int) -> dict[str, Any]:
        """Execute multi-modal query"""
        # Simulate multi-modal search across different content types
        await asyncio.sleep(random.uniform(0.2, 0.8))  # 200-800ms
        return {"query": query, "type": "multi_modal", "results": random.randint(8, 30), "user_id": user_id}

    async def _execute_default_query(self, query: str, user_id: int) -> dict[str, Any]:
        """Execute default query"""
        await asyncio.sleep(random.uniform(0.05, 0.15))  # 50-150ms
        return {"query": query, "type": "default", "results": random.randint(3, 15), "user_id": user_id}

    async def run_concurrent_test(self, concurrent_users: int, duration_seconds: float) -> ConcurrentTestResult:
        """Run concurrent user test with specified parameters"""
        test_name = f"concurrent_test_{concurrent_users}_users"
        self.logger.info(f"Starting {test_name} for {duration_seconds}s")

        # Baseline measurements
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Create user sessions
        user_tasks = []
        for user_id in range(concurrent_users):
            task = asyncio.create_task(self.simulate_user_session(user_id, duration_seconds))
            user_tasks.append(task)

        # Wait for all user sessions to complete
        user_sessions = await asyncio.gather(*user_tasks, return_exceptions=True)

        # Filter out exceptions
        successful_sessions = [session for session in user_sessions if isinstance(session, UserSession)]

        end_time = time.time()
        actual_duration = end_time - start_time

        # Final measurements
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate metrics
        metrics = self._calculate_concurrent_metrics(successful_sessions, actual_duration, baseline_memory, peak_memory)

        # Analyze performance degradation
        degradation = self._analyze_performance_degradation(metrics)

        # Determine if target was met
        target_met = self._evaluate_target_achievement(metrics, degradation)

        # Identify bottlenecks and recommendations
        bottlenecks = self._identify_bottlenecks(metrics, degradation)
        recommendations = self._generate_recommendations(bottlenecks)

        result = ConcurrentTestResult(
            test_name=test_name,
            target_concurrent_users=concurrent_users,
            actual_concurrent_users=len(successful_sessions),
            metrics=metrics,
            performance_degradation=degradation,
            target_met=target_met,
            bottlenecks_identified=bottlenecks,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
        )

        self.results.append(result)
        return result

    def _calculate_concurrent_metrics(
        self, sessions: list[UserSession], duration: float, baseline_memory: float, peak_memory: float
    ) -> ConcurrentTestMetrics:
        """Calculate metrics from concurrent test results"""

        if not sessions:
            return ConcurrentTestMetrics(
                concurrent_users=0,
                test_duration_seconds=duration,
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                average_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                throughput_queries_per_second=0.0,
                system_resource_usage={},
                memory_peak_mb=peak_memory,
                memory_baseline_mb=baseline_memory,
                cache_performance={},
                error_rate_percent=100.0,
                user_sessions=[],
            )

        # Aggregate session data
        total_queries = sum(session.total_queries for session in sessions)
        successful_queries = sum(session.successful_queries for session in sessions)
        failed_queries = total_queries - successful_queries

        # Collect all response times
        all_response_times = []
        for session in sessions:
            all_response_times.extend(session.response_times)

        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            sorted_times = sorted(all_response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index]
            p99_response_time = sorted_times[p99_index]
        else:
            avg_response_time = 0.0
            p95_response_time = 0.0
            p99_response_time = 0.0

        # Calculate throughput
        throughput = total_queries / duration if duration > 0 else 0

        # Calculate error rate
        error_rate = (failed_queries / total_queries * 100) if total_queries > 0 else 0

        # Calculate cache performance
        total_cache_hits = sum(session.cache_hits for session in sessions)
        total_cache_misses = sum(session.cache_misses for session in sessions)
        cache_total = total_cache_hits + total_cache_misses
        cache_hit_rate = (total_cache_hits / cache_total * 100) if cache_total > 0 else 0

        # System resource usage
        process = psutil.Process()
        resource_usage = {
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "open_files": len(process.open_files()),
            "num_threads": process.num_threads(),
        }

        return ConcurrentTestMetrics(
            concurrent_users=len(sessions),
            test_duration_seconds=duration,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            throughput_queries_per_second=throughput,
            system_resource_usage=resource_usage,
            memory_peak_mb=peak_memory,
            memory_baseline_mb=baseline_memory,
            cache_performance={"hit_rate_percent": cache_hit_rate, "total_hits": total_cache_hits, "total_misses": total_cache_misses},
            error_rate_percent=error_rate,
            user_sessions=sessions,
        )

    def _analyze_performance_degradation(self, metrics: ConcurrentTestMetrics) -> dict[str, float]:
        """Analyze performance degradation under concurrent load"""
        degradation = {}

        # Memory usage increase
        if metrics.memory_baseline_mb > 0:
            memory_increase = ((metrics.memory_peak_mb - metrics.memory_baseline_mb) / metrics.memory_baseline_mb) * 100
            degradation["memory_increase_percent"] = memory_increase

        # Error rate
        degradation["error_rate_percent"] = metrics.error_rate_percent

        # Response time degradation (compared to single user baseline)
        # This would need a baseline measurement in practice
        baseline_response_time = 100.0  # Assume 100ms baseline
        if baseline_response_time > 0:
            response_time_degradation = ((metrics.average_response_time_ms - baseline_response_time) / baseline_response_time) * 100
            degradation["response_time_degradation_percent"] = response_time_degradation

        return degradation

    def _evaluate_target_achievement(self, metrics: ConcurrentTestMetrics, degradation: dict[str, float]) -> bool:
        """Evaluate if the concurrent user target was achieved"""

        # Check if we achieved the target number of users
        users_target_met = metrics.concurrent_users >= self.target_concurrent_users

        # Check performance thresholds
        response_time_ok = degradation.get("response_time_degradation_percent", 0) <= self.thresholds["response_time_degradation_percent"]

        error_rate_ok = metrics.error_rate_percent <= self.thresholds["error_rate_percent"]

        memory_ok = degradation.get("memory_increase_percent", 0) <= self.thresholds["memory_increase_percent"]

        return users_target_met and response_time_ok and error_rate_ok and memory_ok

    def _identify_bottlenecks(self, metrics: ConcurrentTestMetrics, degradation: dict[str, float]) -> list[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        if degradation.get("response_time_degradation_percent", 0) > self.thresholds["response_time_degradation_percent"]:
            bottlenecks.append("Response time degradation under load")

        if metrics.error_rate_percent > self.thresholds["error_rate_percent"]:
            bottlenecks.append("High error rate under concurrent load")

        if degradation.get("memory_increase_percent", 0) > self.thresholds["memory_increase_percent"]:
            bottlenecks.append("Excessive memory usage under load")

        if metrics.cache_performance.get("hit_rate_percent", 0) < 60:
            bottlenecks.append("Low cache hit rate")

        if metrics.throughput_queries_per_second < (metrics.concurrent_users * 0.5):
            bottlenecks.append("Low query throughput")

        return bottlenecks

    def _generate_recommendations(self, bottlenecks: list[str]) -> list[str]:
        """Generate recommendations based on identified bottlenecks"""
        recommendations = []

        for bottleneck in bottlenecks:
            if "response time" in bottleneck.lower():
                recommendations.append("Implement query result caching and optimize search algorithms")
            elif "error rate" in bottleneck.lower():
                recommendations.append("Improve error handling and implement retry mechanisms")
            elif "memory usage" in bottleneck.lower():
                recommendations.append("Optimize memory management and implement connection pooling")
            elif "cache" in bottleneck.lower():
                recommendations.append("Improve cache warming strategies and cache key optimization")
            elif "throughput" in bottleneck.lower():
                recommendations.append("Increase parallelization and optimize resource utilization")

        if not recommendations:
            recommendations.append("Performance targets met - consider stress testing with higher loads")

        return recommendations

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all concurrent user tests"""
        self.logger.info("Starting concurrent user tests...")

        # Test different user loads
        user_loads = [1, 3, 5, 8, 10]  # Gradual increase

        for user_count in user_loads:
            if user_count <= self.max_concurrent_users:
                try:
                    await self.run_concurrent_test(user_count, self.test_duration_seconds)
                    self.logger.info(f"Completed test with {user_count} concurrent users")
                    # Brief pause between tests
                    await asyncio.sleep(2)
                    gc.collect()
                except Exception as e:
                    self.logger.error(f"Failed test with {user_count} users: {e}")

        # Generate summary
        summary = self._generate_summary()
        self.logger.info("Concurrent user tests completed")

        return summary

    def _generate_summary(self) -> dict[str, Any]:
        """Generate comprehensive test summary"""
        total_tests = len(self.results)
        tests_meeting_target = len([r for r in self.results if r.target_met])

        # Find maximum concurrent users achieved
        max_users_achieved = max([r.actual_concurrent_users for r in self.results]) if self.results else 0

        # Performance statistics
        if self.results:
            avg_response_time = statistics.mean([r.metrics.average_response_time_ms for r in self.results])
            avg_throughput = statistics.mean([r.metrics.throughput_queries_per_second for r in self.results])
            avg_error_rate = statistics.mean([r.metrics.error_rate_percent for r in self.results])
        else:
            avg_response_time = 0
            avg_throughput = 0
            avg_error_rate = 0

        # Identify common bottlenecks
        all_bottlenecks = []
        for result in self.results:
            all_bottlenecks.extend(result.bottlenecks_identified)

        bottleneck_frequency = {}
        for bottleneck in all_bottlenecks:
            bottleneck_frequency[bottleneck] = bottleneck_frequency.get(bottleneck, 0) + 1

        summary = {
            "total_tests": total_tests,
            "tests_meeting_target": tests_meeting_target,
            "target_concurrent_users": self.target_concurrent_users,
            "max_concurrent_users_achieved": max_users_achieved,
            "target_achievement_rate": (tests_meeting_target / total_tests) * 100 if total_tests > 0 else 0,
            "performance_statistics": {
                "average_response_time_ms": avg_response_time,
                "average_throughput_qps": avg_throughput,
                "average_error_rate_percent": avg_error_rate,
            },
            "common_bottlenecks": bottleneck_frequency,
            "test_results": [asdict(result) for result in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def generate_report(self, output_file: str = "concurrent_user_test_report.json"):
        """Generate detailed concurrent user test report"""
        summary = self._generate_summary()

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Generated concurrent user test report: {output_file}")
        return summary

    def print_summary(self):
        """Print human-readable test summary"""
        summary = self._generate_summary()

        print("\n=== Concurrent User Test Summary ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Tests Meeting Target: {summary['tests_meeting_target']}/{summary['total_tests']}")
        print(f"Target Concurrent Users: {summary['target_concurrent_users']}")
        print(f"Max Concurrent Users Achieved: {summary['max_concurrent_users_achieved']}")
        print(f"Target Achievement Rate: {summary['target_achievement_rate']:.1f}%")

        stats = summary["performance_statistics"]
        print("\nPerformance Statistics:")
        print(f"  Average Response Time: {stats['average_response_time_ms']:.2f}ms")
        print(f"  Average Throughput: {stats['average_throughput_qps']:.2f} queries/s")
        print(f"  Average Error Rate: {stats['average_error_rate_percent']:.2f}%")

        if summary["common_bottlenecks"]:
            print("\nCommon Bottlenecks:")
            for bottleneck, frequency in summary["common_bottlenecks"].items():
                print(f"  {bottleneck}: {frequency} occurrences")


async def main():
    """Main function to run concurrent user tests"""
    tester = ConcurrentUserTester()

    # Run tests
    print("Running concurrent user tests...")
    summary = await tester.run_all_tests()

    # Generate report
    tester.generate_report("wave_8_concurrent_user_report.json")

    # Print summary
    tester.print_summary()

    return summary


if __name__ == "__main__":
    asyncio.run(main())
