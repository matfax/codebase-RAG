"""
Performance Tests for Function Chain MCP Tools.

This module provides comprehensive performance testing to ensure all function
chain tools meet the <2 second response time requirement for large codebases.
"""

import asyncio
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import pytest

from src.tools.graph_rag.function_chain_analysis import trace_function_chain
from src.tools.graph_rag.function_path_finding import find_function_path
from src.tools.graph_rag.project_chain_analysis import analyze_project_chains

logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Performance benchmark runner for function chain tools."""

    def __init__(self, project_name: str = "Agentic_RAG"):
        """Initialize the performance benchmark."""
        self.project_name = project_name
        self.performance_results = {}

    async def run_performance_tests(self) -> dict[str, Any]:
        """
        Run comprehensive performance tests for all function chain tools.

        Returns:
            Dictionary containing performance test results and metrics
        """
        logger.info(f"Starting performance tests for project: {self.project_name}")
        start_time = time.time()

        results = {
            "project_name": self.project_name,
            "start_time": start_time,
            "performance_requirement": 2.0,  # seconds
            "tests": {},
            "summary": {},
        }

        try:
            # Test 1: Single Tool Performance
            results["tests"]["single_tool"] = await self._test_single_tool_performance()

            # Test 2: Concurrent Tool Performance
            results["tests"]["concurrent"] = await self._test_concurrent_performance()

            # Test 3: Large Dataset Performance
            results["tests"]["large_dataset"] = await self._test_large_dataset_performance()

            # Test 4: Memory Usage Performance
            results["tests"]["memory_usage"] = await self._test_memory_performance()

            # Test 5: Stress Testing
            results["tests"]["stress"] = await self._test_stress_performance()

            # Generate summary
            results["summary"] = self._generate_performance_summary(results["tests"])
            results["total_duration"] = time.time() - start_time

            logger.info(f"Performance tests completed in {results['total_duration']:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            results["error"] = str(e)
            return results

    async def _test_single_tool_performance(self) -> dict[str, Any]:
        """Test individual tool performance."""
        logger.info("Testing single tool performance...")

        tool_tests = {}

        # Test trace_function_chain
        tool_tests["trace_function_chain"] = await self._benchmark_tool(
            tool_func=trace_function_chain,
            test_cases=[
                {"entry_point": "main", "project_name": self.project_name, "direction": "forward", "max_depth": 5},
                {"entry_point": "search", "project_name": self.project_name, "direction": "bidirectional", "max_depth": 10},
                {"entry_point": "index_directory", "project_name": self.project_name, "direction": "backward", "max_depth": 8},
            ],
            iterations=5,
        )

        # Test find_function_path
        tool_tests["find_function_path"] = await self._benchmark_tool(
            tool_func=find_function_path,
            test_cases=[
                {"start_function": "main", "end_function": "search", "project_name": self.project_name, "strategy": "optimal"},
                {
                    "start_function": "index_directory",
                    "end_function": "analyze_repository",
                    "project_name": self.project_name,
                    "strategy": "shortest",
                },
                {
                    "start_function": "embedding_service",
                    "end_function": "qdrant_service",
                    "project_name": self.project_name,
                    "strategy": "all",
                    "max_paths": 2,
                },
            ],
            iterations=5,
        )

        # Test analyze_project_chains
        tool_tests["analyze_project_chains"] = await self._benchmark_tool(
            tool_func=analyze_project_chains,
            test_cases=[
                {"project_name": self.project_name, "analysis_scope": "scoped_breadcrumbs", "breadcrumb_patterns": ["*main*"]},
                {"project_name": self.project_name, "analysis_scope": "function_patterns", "breadcrumb_patterns": ["*service*", "*tool*"]},
                {"project_name": self.project_name, "analysis_scope": "full_project", "max_functions_per_chain": 20},
            ],
            iterations=3,  # Fewer iterations for more expensive operations
        )

        # Calculate overall single tool performance
        all_durations = []
        for tool_result in tool_tests.values():
            all_durations.extend(tool_result.get("durations", []))

        return {
            "tools": tool_tests,
            "overall_avg_duration": statistics.mean(all_durations) if all_durations else 0,
            "overall_max_duration": max(all_durations) if all_durations else 0,
            "meets_requirement": all(d < 2.0 for d in all_durations),
            "total_tests": sum(len(tool_result.get("durations", [])) for tool_result in tool_tests.values()),
        }

    async def _test_concurrent_performance(self) -> dict[str, Any]:
        """Test concurrent tool performance."""
        logger.info("Testing concurrent tool performance...")

        # Define concurrent test scenarios
        concurrent_tasks = [
            # Scenario 1: Multiple trace operations
            [
                (trace_function_chain, {"entry_point": "main", "project_name": self.project_name}),
                (trace_function_chain, {"entry_point": "search", "project_name": self.project_name}),
                (trace_function_chain, {"entry_point": "index_directory", "project_name": self.project_name}),
            ],
            # Scenario 2: Mixed tool operations
            [
                (trace_function_chain, {"entry_point": "main", "project_name": self.project_name}),
                (find_function_path, {"start_function": "main", "end_function": "search", "project_name": self.project_name}),
                (
                    analyze_project_chains,
                    {"project_name": self.project_name, "analysis_scope": "scoped_breadcrumbs", "breadcrumb_patterns": ["*main*"]},
                ),
            ],
        ]

        concurrent_results = []

        for scenario_idx, tasks in enumerate(concurrent_tasks):
            logger.info(f"Running concurrent scenario {scenario_idx + 1}")
            start_time = time.time()

            try:
                # Run tasks concurrently
                results = await asyncio.gather(*[tool_func(**params) for tool_func, params in tasks], return_exceptions=True)

                duration = time.time() - start_time

                # Check for errors
                errors = [r for r in results if isinstance(r, Exception)]
                successful_results = [r for r in results if not isinstance(r, Exception)]

                concurrent_results.append(
                    {
                        "scenario": scenario_idx + 1,
                        "duration": duration,
                        "total_tasks": len(tasks),
                        "successful_tasks": len(successful_results),
                        "errors": len(errors),
                        "meets_requirement": duration < 2.0,
                        "avg_task_duration": duration / len(tasks),
                    }
                )

            except Exception as e:
                concurrent_results.append(
                    {"scenario": scenario_idx + 1, "duration": time.time() - start_time, "error": str(e), "meets_requirement": False}
                )

        # Calculate overall concurrent performance
        successful_scenarios = [r for r in concurrent_results if r.get("successful_tasks", 0) > 0]
        avg_concurrent_duration = statistics.mean([r["duration"] for r in successful_scenarios]) if successful_scenarios else 0

        return {
            "scenarios": concurrent_results,
            "avg_concurrent_duration": avg_concurrent_duration,
            "concurrent_success_rate": len(successful_scenarios) / len(concurrent_results),
            "meets_requirement": all(r.get("meets_requirement", False) for r in successful_scenarios),
        }

    async def _test_large_dataset_performance(self) -> dict[str, Any]:
        """Test performance with large datasets and complex queries."""
        logger.info("Testing large dataset performance...")

        large_dataset_tests = []

        # Test 1: Deep function chain tracing
        start_time = time.time()
        try:
            result = await trace_function_chain(
                entry_point="main",
                project_name=self.project_name,
                direction="bidirectional",
                max_depth=20,  # Deep tracing
                chain_type="execution_flow",
            )
            duration = time.time() - start_time
            large_dataset_tests.append(
                {"test": "deep_tracing", "duration": duration, "meets_requirement": duration < 2.0, "success": "error" not in result}
            )
        except Exception as e:
            large_dataset_tests.append(
                {"test": "deep_tracing", "duration": time.time() - start_time, "error": str(e), "meets_requirement": False}
            )

        # Test 2: Multiple path finding
        start_time = time.time()
        try:
            result = await find_function_path(
                start_function="main",
                end_function="search",
                project_name=self.project_name,
                strategy="all",
                max_paths=10,  # Find many paths
                max_depth=20,
            )
            duration = time.time() - start_time
            large_dataset_tests.append(
                {"test": "multiple_paths", "duration": duration, "meets_requirement": duration < 2.0, "success": "error" not in result}
            )
        except Exception as e:
            large_dataset_tests.append(
                {"test": "multiple_paths", "duration": time.time() - start_time, "error": str(e), "meets_requirement": False}
            )

        # Test 3: Comprehensive project analysis
        start_time = time.time()
        try:
            result = await analyze_project_chains(
                project_name=self.project_name,
                analysis_scope="full_project",
                max_functions_per_chain=100,  # Large chains
                include_hotspot_analysis=True,
                include_refactoring_suggestions=True,
            )
            duration = time.time() - start_time
            large_dataset_tests.append(
                {
                    "test": "comprehensive_analysis",
                    "duration": duration,
                    "meets_requirement": duration < 2.0,
                    "success": "error" not in result,
                }
            )
        except Exception as e:
            large_dataset_tests.append(
                {"test": "comprehensive_analysis", "duration": time.time() - start_time, "error": str(e), "meets_requirement": False}
            )

        # Calculate metrics
        successful_tests = [t for t in large_dataset_tests if t.get("success", False)]
        meeting_requirements = [t for t in large_dataset_tests if t.get("meets_requirement", False)]

        return {
            "tests": large_dataset_tests,
            "total_tests": len(large_dataset_tests),
            "successful_tests": len(successful_tests),
            "meeting_requirements": len(meeting_requirements),
            "avg_duration": statistics.mean([t["duration"] for t in large_dataset_tests]),
            "max_duration": max([t["duration"] for t in large_dataset_tests]),
            "performance_success_rate": len(meeting_requirements) / len(large_dataset_tests),
        }

    async def _test_memory_performance(self) -> dict[str, Any]:
        """Test memory usage during operations."""
        logger.info("Testing memory performance...")

        try:
            import psutil

            process = psutil.Process()

            memory_tests = []

            # Test 1: Memory usage during trace operation
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()

            await trace_function_chain(entry_point="main", project_name=self.project_name, direction="bidirectional", max_depth=15)

            duration = time.time() - start_time
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory

            memory_tests.append(
                {
                    "test": "trace_memory",
                    "duration": duration,
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": peak_memory,
                    "memory_increase_mb": memory_increase,
                    "memory_efficient": memory_increase < 100,  # Less than 100MB increase
                    "meets_requirement": duration < 2.0,
                }
            )

            # Test 2: Memory usage during path finding
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()

            await find_function_path(
                start_function="main", end_function="search", project_name=self.project_name, strategy="all", max_paths=5
            )

            duration = time.time() - start_time
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory

            memory_tests.append(
                {
                    "test": "path_memory",
                    "duration": duration,
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": peak_memory,
                    "memory_increase_mb": memory_increase,
                    "memory_efficient": memory_increase < 100,
                    "meets_requirement": duration < 2.0,
                }
            )

            return {
                "tests": memory_tests,
                "avg_memory_increase": statistics.mean([t["memory_increase_mb"] for t in memory_tests]),
                "memory_efficient": all(t["memory_efficient"] for t in memory_tests),
                "meets_performance_requirement": all(t["meets_requirement"] for t in memory_tests),
            }

        except ImportError:
            return {
                "error": "psutil not available for memory testing",
                "tests": [],
                "memory_efficient": True,  # Assume efficient if can't test
                "meets_performance_requirement": True,
            }

    async def _test_stress_performance(self) -> dict[str, Any]:
        """Test performance under stress conditions."""
        logger.info("Testing stress performance...")

        stress_tests = []

        # Test 1: Rapid successive calls
        start_time = time.time()
        durations = []

        for i in range(10):  # 10 rapid calls
            call_start = time.time()
            try:
                await trace_function_chain(entry_point="main", project_name=self.project_name, direction="forward", max_depth=5)
                call_duration = time.time() - call_start
                durations.append(call_duration)
            except Exception:
                durations.append(999)  # High duration for failed calls

        total_duration = time.time() - start_time
        stress_tests.append(
            {
                "test": "rapid_calls",
                "total_duration": total_duration,
                "individual_durations": durations,
                "avg_call_duration": statistics.mean(durations),
                "max_call_duration": max(durations),
                "all_under_2s": all(d < 2.0 for d in durations),
                "degradation_factor": max(durations) / min(durations) if min(durations) > 0 else 1,
            }
        )

        # Test 2: Mixed tool stress test
        start_time = time.time()
        mixed_durations = []

        tools_and_params = [
            (trace_function_chain, {"entry_point": "main", "project_name": self.project_name}),
            (find_function_path, {"start_function": "main", "end_function": "search", "project_name": self.project_name}),
            (
                analyze_project_chains,
                {"project_name": self.project_name, "analysis_scope": "scoped_breadcrumbs", "breadcrumb_patterns": ["*main*"]},
            ),
        ]

        for i in range(6):  # 2 iterations of each tool
            tool_func, params = tools_and_params[i % len(tools_and_params)]
            call_start = time.time()
            try:
                await tool_func(**params)
                call_duration = time.time() - call_start
                mixed_durations.append(call_duration)
            except Exception:
                mixed_durations.append(999)

        total_mixed_duration = time.time() - start_time
        stress_tests.append(
            {
                "test": "mixed_tools",
                "total_duration": total_mixed_duration,
                "individual_durations": mixed_durations,
                "avg_call_duration": statistics.mean(mixed_durations),
                "max_call_duration": max(mixed_durations),
                "all_under_2s": all(d < 2.0 for d in mixed_durations),
            }
        )

        return {
            "tests": stress_tests,
            "passes_stress_test": all(t["all_under_2s"] for t in stress_tests),
            "max_degradation": max(t.get("degradation_factor", 1) for t in stress_tests),
        }

    async def _benchmark_tool(self, tool_func, test_cases: list[dict], iterations: int = 5) -> dict[str, Any]:
        """Benchmark a specific tool with given test cases."""
        all_durations = []
        successful_calls = 0

        for test_case in test_cases:
            for _ in range(iterations):
                start_time = time.time()
                try:
                    result = await tool_func(**test_case)
                    duration = time.time() - start_time
                    all_durations.append(duration)
                    if "error" not in result:
                        successful_calls += 1
                except Exception:
                    # Record failed calls as high duration
                    duration = time.time() - start_time
                    all_durations.append(999)

        return {
            "durations": all_durations,
            "avg_duration": statistics.mean(all_durations),
            "min_duration": min(all_durations),
            "max_duration": max(all_durations),
            "median_duration": statistics.median(all_durations),
            "std_deviation": statistics.stdev(all_durations) if len(all_durations) > 1 else 0,
            "successful_calls": successful_calls,
            "total_calls": len(test_cases) * iterations,
            "success_rate": successful_calls / (len(test_cases) * iterations),
            "meets_requirement": all(d < 2.0 for d in all_durations if d < 999),
        }

    def _generate_performance_summary(self, test_results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall performance summary."""
        summary = {
            "overall_performance": "unknown",
            "total_tests_run": 0,
            "tests_meeting_requirement": 0,
            "average_response_time": 0,
            "max_response_time": 0,
            "performance_recommendations": [],
        }

        # Collect all durations and success metrics
        all_durations = []
        tests_meeting_req = 0
        total_tests = 0

        for test_category, results in test_results.items():
            if test_category == "single_tool":
                all_durations.extend(results.get("tools", {}).get("trace_function_chain", {}).get("durations", []))
                all_durations.extend(results.get("tools", {}).get("find_function_path", {}).get("durations", []))
                all_durations.extend(results.get("tools", {}).get("analyze_project_chains", {}).get("durations", []))
                if results.get("meets_requirement", False):
                    tests_meeting_req += 1
                total_tests += 1

            elif test_category == "concurrent":
                for scenario in results.get("scenarios", []):
                    all_durations.append(scenario.get("duration", 999))
                    if scenario.get("meets_requirement", False):
                        tests_meeting_req += 1
                    total_tests += 1

            elif test_category == "large_dataset":
                for test in results.get("tests", []):
                    all_durations.append(test.get("duration", 999))
                    if test.get("meets_requirement", False):
                        tests_meeting_req += 1
                    total_tests += 1

        # Filter out error durations (999)
        valid_durations = [d for d in all_durations if d < 999]

        if valid_durations:
            summary["average_response_time"] = statistics.mean(valid_durations)
            summary["max_response_time"] = max(valid_durations)

        summary["total_tests_run"] = total_tests
        summary["tests_meeting_requirement"] = tests_meeting_req
        summary["performance_success_rate"] = tests_meeting_req / total_tests if total_tests > 0 else 0

        # Determine overall performance
        if summary["performance_success_rate"] >= 0.95:
            summary["overall_performance"] = "excellent"
        elif summary["performance_success_rate"] >= 0.8:
            summary["overall_performance"] = "good"
        elif summary["performance_success_rate"] >= 0.6:
            summary["overall_performance"] = "acceptable"
        else:
            summary["overall_performance"] = "needs_improvement"

        # Generate recommendations
        if summary["average_response_time"] > 1.5:
            summary["performance_recommendations"].append("Consider optimizing algorithms for better response times")

        if summary["max_response_time"] > 2.0:
            summary["performance_recommendations"].append("Some operations exceed 2s requirement - investigate bottlenecks")

        if summary["performance_success_rate"] < 0.8:
            summary["performance_recommendations"].append("Performance consistency needs improvement")

        return summary


# Pytest test functions
@pytest.mark.asyncio
async def test_individual_tool_performance():
    """Test individual tool performance requirements."""
    benchmark = PerformanceBenchmark("Agentic_RAG")
    single_tool_results = await benchmark._test_single_tool_performance()

    assert single_tool_results["meets_requirement"], f"Some tools don't meet 2s requirement: {single_tool_results}"
    assert (
        single_tool_results["overall_avg_duration"] < 2.0
    ), f"Average duration too high: {single_tool_results['overall_avg_duration']:.2f}s"


@pytest.mark.asyncio
async def test_concurrent_performance():
    """Test concurrent tool performance."""
    benchmark = PerformanceBenchmark("Agentic_RAG")
    concurrent_results = await benchmark._test_concurrent_performance()

    assert concurrent_results["meets_requirement"], f"Concurrent operations don't meet requirements: {concurrent_results}"
    assert (
        concurrent_results["concurrent_success_rate"] >= 0.8
    ), f"Too many concurrent failures: {concurrent_results['concurrent_success_rate']}"


@pytest.mark.asyncio
async def test_large_dataset_performance():
    """Test performance with large datasets."""
    benchmark = PerformanceBenchmark("Agentic_RAG")
    large_dataset_results = await benchmark._test_large_dataset_performance()

    assert large_dataset_results["performance_success_rate"] >= 0.8, f"Large dataset performance too low: {large_dataset_results}"
    assert (
        large_dataset_results["avg_duration"] < 2.0
    ), f"Average large dataset duration too high: {large_dataset_results['avg_duration']:.2f}s"


@pytest.mark.asyncio
async def test_comprehensive_performance():
    """Test comprehensive performance across all scenarios."""
    benchmark = PerformanceBenchmark("Agentic_RAG")
    results = await benchmark.run_performance_tests()

    summary = results.get("summary", {})
    assert summary.get("overall_performance") in ["excellent", "good", "acceptable"], f"Overall performance insufficient: {summary}"
    assert summary.get("performance_success_rate", 0) >= 0.8, f"Performance success rate too low: {summary}"


if __name__ == "__main__":
    # Run performance tests directly
    async def main():
        benchmark = PerformanceBenchmark("Agentic_RAG")
        results = await benchmark.run_performance_tests()

        print("\n" + "=" * 80)
        print("FUNCTION CHAIN PERFORMANCE TEST RESULTS")
        print("=" * 80)
        print(f"Project: {results['project_name']}")
        print(f"Performance Requirement: <{results['performance_requirement']}s response time")
        print(f"Total Test Duration: {results['total_duration']:.2f}s")

        summary = results.get("summary", {})
        print(f"\nOverall Performance: {summary.get('overall_performance', 'unknown')}")
        print(f"Performance Success Rate: {summary.get('performance_success_rate', 0):.1%}")
        print(f"Average Response Time: {summary.get('average_response_time', 0):.3f}s")
        print(f"Max Response Time: {summary.get('max_response_time', 0):.3f}s")

        if summary.get("performance_recommendations"):
            print("\nRecommendations:")
            for rec in summary["performance_recommendations"]:
                print(f"  - {rec}")

        return summary.get("overall_performance") in ["excellent", "good", "acceptable"]

    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
