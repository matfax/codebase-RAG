"""
User Acceptance Testing for Function Chain MCP Tools.

This module validates natural language input conversion accuracy with a target
of >90% success rate for converting user queries to proper function breadcrumbs
and understanding user intent.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.tools.graph_rag.function_chain_analysis import trace_function_chain
from src.tools.graph_rag.function_path_finding import find_function_path
from src.tools.graph_rag.project_chain_analysis import analyze_project_chains

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case for user acceptance testing."""

    input_query: str
    expected_breadcrumb: str
    tool_type: str  # "trace", "path", "analysis"
    expected_success: bool = True
    description: str = ""


@dataclass
class TestResult:
    """Result of a user acceptance test."""

    test_case: TestCase
    actual_breadcrumb: str
    resolved_successfully: bool
    tool_executed_successfully: bool
    response_time: float
    accuracy_score: float  # 0.0-1.0


class UserAcceptanceTester:
    """User acceptance tester for Function Chain tools."""

    def __init__(self, project_name: str = "Agentic_RAG"):
        """Initialize the user acceptance tester."""
        self.project_name = project_name
        self.breadcrumb_resolver = BreadcrumbResolver()
        self.test_cases = self._define_test_cases()

    def _define_test_cases(self) -> list[TestCase]:
        """Define comprehensive test cases for user acceptance testing."""
        return [
            # Function Chain Tracing Test Cases
            TestCase(
                input_query="main function",
                expected_breadcrumb="main",
                tool_type="trace",
                description="Simple main function identification",
            ),
            TestCase(
                input_query="user authentication function",
                expected_breadcrumb="*auth*",
                tool_type="trace",
                description="Natural language function description",
            ),
            TestCase(
                input_query="database connection setup",
                expected_breadcrumb="*db*",
                tool_type="trace",
                description="Database-related function identification",
            ),
            TestCase(
                input_query="search functionality",
                expected_breadcrumb="*search*",
                tool_type="trace",
                description="Search-related function identification",
            ),
            TestCase(
                input_query="the function that handles file indexing",
                expected_breadcrumb="*index*",
                tool_type="trace",
                description="Descriptive function identification",
            ),
            TestCase(
                input_query="cache service implementation",
                expected_breadcrumb="*cache*service*",
                tool_type="trace",
                description="Service-level function identification",
            ),
            TestCase(
                input_query="error handling function",
                expected_breadcrumb="*error*",
                tool_type="trace",
                description="Error handling function identification",
            ),
            TestCase(
                input_query="API endpoint handler",
                expected_breadcrumb="*api*",
                tool_type="trace",
                description="API-related function identification",
            ),
            # Function Path Finding Test Cases
            TestCase(input_query="main", expected_breadcrumb="main", tool_type="path", description="Simple function name for path finding"),
            TestCase(
                input_query="search function",
                expected_breadcrumb="*search*",
                tool_type="path",
                description="Natural language to breadcrumb for path finding",
            ),
            TestCase(
                input_query="the indexing service",
                expected_breadcrumb="*index*service*",
                tool_type="path",
                description="Service identification for path finding",
            ),
            TestCase(
                input_query="database query function",
                expected_breadcrumb="*db*query*",
                tool_type="path",
                description="Database function identification",
            ),
            TestCase(
                input_query="file processing handler",
                expected_breadcrumb="*file*process*",
                tool_type="path",
                description="File processing function identification",
            ),
            TestCase(
                input_query="authentication validator",
                expected_breadcrumb="*auth*valid*",
                tool_type="path",
                description="Authentication validation function",
            ),
            # Project Analysis Test Cases
            TestCase(
                input_query="service functions",
                expected_breadcrumb="*service*",
                tool_type="analysis",
                description="Service pattern analysis",
            ),
            TestCase(
                input_query="all authentication related code",
                expected_breadcrumb="*auth*",
                tool_type="analysis",
                description="Authentication pattern analysis",
            ),
            TestCase(
                input_query="database related functions",
                expected_breadcrumb="*db*",
                tool_type="analysis",
                description="Database pattern analysis",
            ),
            TestCase(
                input_query="API handlers and endpoints",
                expected_breadcrumb="*api*",
                tool_type="analysis",
                description="API pattern analysis",
            ),
            TestCase(
                input_query="caching mechanisms", expected_breadcrumb="*cache*", tool_type="analysis", description="Cache pattern analysis"
            ),
            # Edge Cases and Complex Queries
            TestCase(
                input_query="the main entry point function",
                expected_breadcrumb="main",
                tool_type="trace",
                description="Verbose main function description",
            ),
            TestCase(
                input_query="async function that processes data",
                expected_breadcrumb="*async*process*",
                tool_type="trace",
                description="Async function with description",
            ),
            TestCase(
                input_query="Redis cache service",
                expected_breadcrumb="*redis*cache*",
                tool_type="trace",
                description="Specific technology mention",
            ),
            TestCase(
                input_query="method that validates user input",
                expected_breadcrumb="*valid*user*input*",
                tool_type="trace",
                description="Method with specific purpose",
            ),
            # Negative Test Cases (should handle gracefully)
            TestCase(
                input_query="nonexistent_function_12345",
                expected_breadcrumb="nonexistent_function_12345",
                tool_type="trace",
                expected_success=False,
                description="Non-existent function handling",
            ),
            TestCase(input_query="", expected_breadcrumb="", tool_type="trace", expected_success=False, description="Empty input handling"),
            TestCase(
                input_query="@#$%^&*()",
                expected_breadcrumb="@#$%^&*()",
                tool_type="trace",
                expected_success=False,
                description="Invalid characters handling",
            ),
        ]

    async def run_user_acceptance_tests(self) -> dict[str, Any]:
        """
        Run comprehensive user acceptance tests.

        Returns:
            Dictionary containing test results and accuracy metrics
        """
        logger.info(f"Starting user acceptance tests for project: {self.project_name}")
        start_time = time.time()

        results = {
            "project_name": self.project_name,
            "start_time": start_time,
            "target_accuracy": 0.90,  # 90% target
            "test_results": [],
            "summary": {},
        }

        # Run all test cases
        for test_case in self.test_cases:
            test_result = await self._run_single_test(test_case)
            results["test_results"].append(test_result)

        # Calculate summary statistics
        results["summary"] = self._calculate_summary(results["test_results"])
        results["total_duration"] = time.time() - start_time

        logger.info(f"User acceptance tests completed in {results['total_duration']:.2f}s")
        logger.info(f"Overall accuracy: {results['summary']['overall_accuracy']:.1%}")

        return results

    async def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single user acceptance test case."""
        logger.debug(f"Testing: {test_case.input_query}")
        start_time = time.time()

        try:
            # Test breadcrumb resolution
            resolved_breadcrumb = await self._resolve_breadcrumb(test_case.input_query)
            resolution_success = self._evaluate_breadcrumb_resolution(test_case.expected_breadcrumb, resolved_breadcrumb)

            # Test tool execution with resolved breadcrumb
            tool_success = await self._test_tool_execution(test_case, resolved_breadcrumb)

            response_time = time.time() - start_time

            # Calculate accuracy score
            accuracy_score = self._calculate_accuracy_score(test_case, resolved_breadcrumb, resolution_success, tool_success)

            return TestResult(
                test_case=test_case,
                actual_breadcrumb=resolved_breadcrumb,
                resolved_successfully=resolution_success,
                tool_executed_successfully=tool_success,
                response_time=response_time,
                accuracy_score=accuracy_score,
            )

        except Exception as e:
            logger.warning(f"Test failed for '{test_case.input_query}': {e}")
            return TestResult(
                test_case=test_case,
                actual_breadcrumb="",
                resolved_successfully=False,
                tool_executed_successfully=False,
                response_time=time.time() - start_time,
                accuracy_score=0.0,
            )

    async def _resolve_breadcrumb(self, query: str) -> str:
        """Resolve natural language query to breadcrumb."""
        try:
            # Use breadcrumb resolver to convert natural language
            resolved = await self.breadcrumb_resolver.resolve_breadcrumb(query, self.project_name)
            return resolved.get("breadcrumb", query)
        except Exception:
            # Fallback to original query if resolution fails
            return query

    def _evaluate_breadcrumb_resolution(self, expected: str, actual: str) -> bool:
        """Evaluate if breadcrumb resolution was successful."""
        if not expected or not actual:
            return expected == actual

        # Handle wildcard patterns in expected breadcrumbs
        if "*" in expected:
            pattern_parts = expected.lower().split("*")
            actual_lower = actual.lower()

            # Check if all non-wildcard parts are present in order
            position = 0
            for part in pattern_parts:
                if part:  # Skip empty parts from consecutive *
                    found_pos = actual_lower.find(part, position)
                    if found_pos == -1:
                        return False
                    position = found_pos + len(part)
            return True
        else:
            # Exact match or substring match
            return expected.lower() in actual.lower() or actual.lower() in expected.lower()

    async def _test_tool_execution(self, test_case: TestCase, breadcrumb: str) -> bool:
        """Test if the tool executes successfully with the resolved breadcrumb."""
        try:
            if test_case.tool_type == "trace":
                result = await trace_function_chain(
                    entry_point=breadcrumb, project_name=self.project_name, direction="forward", max_depth=5
                )
                return "error" not in result or not test_case.expected_success

            elif test_case.tool_type == "path":
                # For path finding, use breadcrumb as start and search for common endpoints
                result = await find_function_path(
                    start_function=breadcrumb,
                    end_function="search",  # Common endpoint
                    project_name=self.project_name,
                    strategy="optimal",
                    max_paths=1,
                )
                return "error" not in result or not test_case.expected_success

            elif test_case.tool_type == "analysis":
                result = await analyze_project_chains(
                    project_name=self.project_name,
                    analysis_scope="scoped_breadcrumbs",
                    breadcrumb_patterns=[breadcrumb] if breadcrumb else ["*main*"],
                    max_functions_per_chain=10,
                )
                return "error" not in result or not test_case.expected_success

            return False

        except Exception as e:
            logger.debug(f"Tool execution failed for {breadcrumb}: {e}")
            return not test_case.expected_success  # If we expect failure, this is success

    def _calculate_accuracy_score(
        self, test_case: TestCase, resolved_breadcrumb: str, resolution_success: bool, tool_success: bool
    ) -> float:
        """Calculate accuracy score for a test case."""
        score = 0.0

        # Breadcrumb resolution score (50% weight)
        if resolution_success:
            score += 0.5
        elif test_case.expected_success:  # Partial credit for attempting resolution
            if resolved_breadcrumb and resolved_breadcrumb != test_case.input_query:
                score += 0.25  # Attempted resolution but not quite right

        # Tool execution score (50% weight)
        if tool_success:
            score += 0.5

        # Bonus for handling negative cases correctly
        if not test_case.expected_success and not (resolution_success and tool_success):
            score = 1.0  # Perfect score for correctly handling expected failures

        return score

    def _calculate_summary(self, test_results: list[TestResult]) -> dict[str, Any]:
        """Calculate summary statistics from test results."""
        if not test_results:
            return {}

        total_tests = len(test_results)
        successful_resolutions = sum(1 for r in test_results if r.resolved_successfully)
        successful_executions = sum(1 for r in test_results if r.tool_executed_successfully)

        # Calculate accuracy by tool type
        tool_type_stats = {}
        for tool_type in ["trace", "path", "analysis"]:
            tool_results = [r for r in test_results if r.test_case.tool_type == tool_type]
            if tool_results:
                avg_accuracy = sum(r.accuracy_score for r in tool_results) / len(tool_results)
                tool_type_stats[tool_type] = {
                    "count": len(tool_results),
                    "avg_accuracy": avg_accuracy,
                    "resolution_success_rate": sum(1 for r in tool_results if r.resolved_successfully) / len(tool_results),
                    "execution_success_rate": sum(1 for r in tool_results if r.tool_executed_successfully) / len(tool_results),
                }

        # Overall accuracy calculation
        overall_accuracy = sum(r.accuracy_score for r in test_results) / total_tests

        # Response time statistics
        response_times = [r.response_time for r in test_results]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Identify problematic test cases
        failed_tests = [r for r in test_results if r.accuracy_score < 0.5]

        return {
            "total_tests": total_tests,
            "overall_accuracy": overall_accuracy,
            "meets_target": overall_accuracy >= 0.90,
            "resolution_success_rate": successful_resolutions / total_tests,
            "execution_success_rate": successful_executions / total_tests,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "tool_type_stats": tool_type_stats,
            "failed_tests_count": len(failed_tests),
            "failed_tests": [
                {
                    "query": r.test_case.input_query,
                    "expected": r.test_case.expected_breadcrumb,
                    "actual": r.actual_breadcrumb,
                    "accuracy": r.accuracy_score,
                }
                for r in failed_tests
            ],
        }

    async def generate_accuracy_report(self) -> str:
        """Generate a human-readable accuracy report."""
        results = await self.run_user_acceptance_tests()
        summary = results["summary"]

        report_lines = [
            "=" * 80,
            "USER ACCEPTANCE TEST REPORT",
            "=" * 80,
            f"Project: {results['project_name']}",
            f"Test Duration: {results['total_duration']:.2f}s",
            f"Target Accuracy: {results['target_accuracy']:.1%}",
            "",
            "OVERALL RESULTS:",
            f"  Total Tests: {summary['total_tests']}",
            f"  Overall Accuracy: {summary['overall_accuracy']:.1%}",
            f"  Meets Target: {'✅ YES' if summary['meets_target'] else '❌ NO'}",
            f"  Resolution Success Rate: {summary['resolution_success_rate']:.1%}",
            f"  Tool Execution Success Rate: {summary['execution_success_rate']:.1%}",
            f"  Average Response Time: {summary['avg_response_time']:.3f}s",
            "",
            "ACCURACY BY TOOL TYPE:",
        ]

        for tool_type, stats in summary["tool_type_stats"].items():
            report_lines.extend(
                [
                    f"  {tool_type.upper()}:",
                    f"    Tests: {stats['count']}",
                    f"    Accuracy: {stats['avg_accuracy']:.1%}",
                    f"    Resolution Rate: {stats['resolution_success_rate']:.1%}",
                    f"    Execution Rate: {stats['execution_success_rate']:.1%}",
                ]
            )

        if summary["failed_tests"]:
            report_lines.extend(
                [
                    "",
                    "FAILED TESTS:",
                ]
            )
            for failed_test in summary["failed_tests"][:5]:  # Show first 5 failures
                report_lines.extend(
                    [
                        f"  Query: '{failed_test['query']}'",
                        f"  Expected: {failed_test['expected']}",
                        f"  Actual: {failed_test['actual']}",
                        f"  Accuracy: {failed_test['accuracy']:.1%}",
                        "",
                    ]
                )

        report_lines.extend(["=" * 80, f"RECOMMENDATION: {'PASSED' if summary['meets_target'] else 'NEEDS IMPROVEMENT'}", "=" * 80])

        return "\n".join(report_lines)


# Test functions for pytest
async def test_user_acceptance_accuracy():
    """Test that user acceptance meets 90% accuracy target."""
    tester = UserAcceptanceTester("Agentic_RAG")
    results = await tester.run_user_acceptance_tests()

    summary = results["summary"]
    assert summary["overall_accuracy"] >= 0.90, f"Accuracy {summary['overall_accuracy']:.1%} below 90% target"
    assert summary["meets_target"], "User acceptance tests do not meet target accuracy"


async def test_natural_language_resolution():
    """Test natural language to breadcrumb resolution accuracy."""
    tester = UserAcceptanceTester("Agentic_RAG")

    # Test specific resolution cases
    test_cases = [
        ("main function", "main"),
        ("search functionality", "*search*"),
        ("database connection", "*db*"),
        ("authentication handler", "*auth*"),
    ]

    resolution_successes = 0
    for query, expected in test_cases:
        resolved = await tester._resolve_breadcrumb(query)
        success = tester._evaluate_breadcrumb_resolution(expected, resolved)
        if success:
            resolution_successes += 1

    accuracy = resolution_successes / len(test_cases)
    assert accuracy >= 0.80, f"Natural language resolution accuracy {accuracy:.1%} below 80% minimum"


async def test_tool_execution_reliability():
    """Test that tools execute reliably with various inputs."""
    tester = UserAcceptanceTester("Agentic_RAG")

    # Test tool execution with various breadcrumbs
    test_breadcrumbs = ["main", "*search*", "*index*", "*service*"]

    execution_successes = 0
    total_executions = 0

    for breadcrumb in test_breadcrumbs:
        # Test each tool type
        for tool_type in ["trace", "path", "analysis"]:
            test_case = TestCase(input_query=breadcrumb, expected_breadcrumb=breadcrumb, tool_type=tool_type)
            success = await tester._test_tool_execution(test_case, breadcrumb)
            if success:
                execution_successes += 1
            total_executions += 1

    reliability = execution_successes / total_executions
    assert reliability >= 0.75, f"Tool execution reliability {reliability:.1%} below 75% minimum"


if __name__ == "__main__":
    # Run user acceptance tests directly
    async def main():
        tester = UserAcceptanceTester("Agentic_RAG")
        report = await tester.generate_accuracy_report()
        print(report)

        # Return success/failure for exit code
        results = await tester.run_user_acceptance_tests()
        return results["summary"]["meets_target"]

    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
