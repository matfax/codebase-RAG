#!/usr/bin/env python3
"""
Graph RAG Performance Test Runner

This script runs comprehensive performance tests for the Graph RAG function
chain analysis and path finding tools to ensure they meet performance
requirements in large codebases.
"""

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pytest


class PerformanceTestRunner:
    """Runner for Graph RAG performance tests."""

    def __init__(self):
        self.test_results = {}
        self.performance_benchmarks = {}

    def run_all_tests(self) -> dict[str, Any]:
        """Run all performance tests and collect results."""
        print("Starting Graph RAG Performance Tests...")
        print("=" * 60)

        # Run pytest with performance test file
        test_file = "src/tools/graph_rag/function_path_finding_performance.test.py"

        if not os.path.exists(test_file):
            print(f"Error: Test file {test_file} not found!")
            return {"error": "Test file not found"}

        print(f"Running performance tests from: {test_file}")

        # Run tests with pytest
        pytest_args = [test_file, "-v", "--tb=short", "--capture=no", "--durations=10"]

        start_time = time.time()
        result = pytest.main(pytest_args)
        total_time = time.time() - start_time

        print(f"\nTotal test execution time: {total_time:.2f} seconds")

        # Collect results
        self.test_results = {
            "total_execution_time": total_time,
            "pytest_exit_code": result,
            "test_status": "PASSED" if result == 0 else "FAILED",
            "timestamp": time.time(),
        }

        return self.test_results

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "test_summary": self.test_results,
            "performance_requirements": {
                "small_codebase": {
                    "functions": "< 1000",
                    "max_response_time": "1 second",
                    "max_memory_usage": "< 100MB",
                    "concurrent_requests": "10 requests/second",
                },
                "medium_codebase": {
                    "functions": "1000-10000",
                    "max_response_time": "3 seconds",
                    "max_memory_usage": "< 500MB",
                    "concurrent_requests": "5 requests/second",
                },
                "large_codebase": {
                    "functions": "> 10000",
                    "max_response_time": "5 seconds",
                    "max_memory_usage": "< 1GB",
                    "concurrent_requests": "2 requests/second",
                },
            },
            "test_categories": {
                "function_chain_analysis": {
                    "small_codebase_performance": "Tests with < 1000 functions",
                    "medium_codebase_performance": "Tests with 1000-10000 functions",
                    "large_codebase_performance": "Tests with > 10000 functions",
                    "concurrent_requests": "Tests with multiple simultaneous requests",
                    "memory_usage": "Tests for memory efficiency",
                },
                "function_path_finding": {
                    "small_codebase_performance": "Tests with small path sets",
                    "large_path_set_performance": "Tests with large path sets",
                    "strategy_performance": "Tests for different strategies",
                    "stress_testing": "Tests with many consecutive requests",
                },
                "scalability_benchmarks": {
                    "chain_analysis_scalability": "Scalability tests for chain analysis",
                    "path_finding_scalability": "Scalability tests for path finding",
                },
            },
            "performance_metrics": {
                "response_time": "Time to complete requests",
                "throughput": "Requests per second",
                "memory_usage": "Memory consumption during processing",
                "concurrent_capacity": "Maximum concurrent requests",
                "scalability_factor": "Performance scaling with data size",
            },
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> list[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []

        if self.test_results.get("test_status") == "PASSED":
            recommendations.extend(
                [
                    "✅ All performance tests passed",
                    "✅ Tools meet performance requirements for large codebases",
                    "✅ Concurrent request handling is efficient",
                    "✅ Memory usage is within acceptable limits",
                    "✅ Scalability benchmarks show good performance characteristics",
                ]
            )
        else:
            recommendations.extend(
                [
                    "❌ Some performance tests failed",
                    "⚠️ Review failed tests and optimize performance",
                    "⚠️ Consider caching strategies for better performance",
                    "⚠️ Implement request throttling for large codebases",
                    "⚠️ Monitor memory usage in production",
                ]
            )

        # General recommendations
        recommendations.extend(
            [
                "📊 Monitor performance metrics in production",
                "🔄 Run performance tests regularly during development",
                "📈 Set up performance monitoring and alerting",
                "🛠️ Consider performance optimizations for specific use cases",
                "📋 Document performance characteristics for users",
            ]
        )

        return recommendations

    def save_report(self, filename: str = "graph_rag_performance_report.json"):
        """Save performance report to file."""
        report = self.generate_performance_report()

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nPerformance report saved to: {filename}")
        return filename

    def print_summary(self):
        """Print a summary of performance test results."""
        print("\n" + "=" * 60)
        print("GRAPH RAG PERFORMANCE TEST SUMMARY")
        print("=" * 60)

        if self.test_results.get("test_status") == "PASSED":
            print("✅ STATUS: ALL TESTS PASSED")
        else:
            print("❌ STATUS: SOME TESTS FAILED")

        print(f"⏱️  Total execution time: {self.test_results.get('total_execution_time', 0):.2f} seconds")
        print(f"🔧 Pytest exit code: {self.test_results.get('pytest_exit_code', 'unknown')}")

        print("\n📋 PERFORMANCE REQUIREMENTS:")
        print("   • Small codebase (< 1000 functions): < 1 second response time")
        print("   • Medium codebase (1000-10000 functions): < 3 seconds response time")
        print("   • Large codebase (> 10000 functions): < 5 seconds response time")
        print("   • Concurrent requests: Support multiple simultaneous requests")
        print("   • Memory usage: Efficient memory management")

        print("\n🎯 TEST CATEGORIES:")
        print("   • Function Chain Analysis Performance")
        print("   • Function Path Finding Performance")
        print("   • Scalability Benchmarks")
        print("   • Concurrent Request Handling")
        print("   • Memory Usage Testing")

        recommendations = self._generate_recommendations()
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")

        print("\n" + "=" * 60)


def main():
    """Main function to run performance tests."""
    runner = PerformanceTestRunner()

    try:
        # Run all performance tests
        results = runner.run_all_tests()

        # Print summary
        runner.print_summary()

        # Save detailed report
        report_file = runner.save_report()

        # Exit with appropriate code
        exit_code = results.get("pytest_exit_code", 1)

        if exit_code == 0:
            print(f"\n🎉 All performance tests passed! Report saved to {report_file}")
        else:
            print(f"\n⚠️  Some performance tests failed. Check {report_file} for details.")

        sys.exit(exit_code)

    except Exception as e:
        print(f"Error running performance tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
