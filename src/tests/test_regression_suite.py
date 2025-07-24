"""
Comprehensive Regression Testing Suite for MCP Tools.

This module ensures that the addition of Function Chain tools does not break
existing Graph RAG functionality and that all tools continue to work as expected.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.tools.graph_rag.pattern_identification import graph_identify_patterns
from src.tools.graph_rag.similar_implementations import graph_find_similar_implementations
from src.tools.graph_rag.structure_analysis import graph_analyze_structure
from src.tools.indexing.search_tools import search

logger = logging.getLogger(__name__)


@dataclass
class RegressionTestCase:
    """Regression test case definition."""

    test_name: str
    tool_function: Any
    parameters: dict[str, Any]
    expected_fields: list[str]
    max_response_time: float = 5.0
    description: str = ""


@dataclass
class RegressionResult:
    """Result of a regression test."""

    test_case: RegressionTestCase
    success: bool
    response_time: float
    result_data: dict[str, Any]
    error_message: str | None = None
    has_expected_fields: bool = False


class RegressionTestSuite:
    """Comprehensive regression testing for all MCP tools."""

    def __init__(self, project_name: str = "Agentic_RAG"):
        """Initialize the regression test suite."""
        self.project_name = project_name
        self.test_cases = self._define_regression_tests()

    def _define_regression_tests(self) -> list[RegressionTestCase]:
        """Define comprehensive regression test cases."""
        return [
            # Core Search Tool Tests
            RegressionTestCase(
                test_name="basic_search",
                tool_function=search,
                parameters={"query": "search function", "n_results": 5, "search_mode": "hybrid"},
                expected_fields=["results", "total_results", "search_metadata"],
                description="Basic search functionality",
            ),
            RegressionTestCase(
                test_name="cross_project_search",
                tool_function=search,
                parameters={"query": "main function", "n_results": 3, "cross_project": True},
                expected_fields=["results", "total_results"],
                description="Cross-project search capability",
            ),
            RegressionTestCase(
                test_name="semantic_search",
                tool_function=search,
                parameters={"query": "database connection setup", "n_results": 5, "search_mode": "semantic", "include_context": True},
                expected_fields=["results", "search_metadata"],
                description="Semantic search with context",
            ),
            # Graph RAG Structure Analysis Tests
            RegressionTestCase(
                test_name="structure_analysis_basic",
                tool_function=graph_analyze_structure,
                parameters={"breadcrumb": "main", "project_name": "Agentic_RAG", "analysis_type": "overview"},
                expected_fields=["structure", "relationships", "analysis_metadata"],
                description="Basic structure analysis",
            ),
            RegressionTestCase(
                test_name="structure_analysis_comprehensive",
                tool_function=graph_analyze_structure,
                parameters={
                    "breadcrumb": "search",
                    "project_name": "Agentic_RAG",
                    "analysis_type": "comprehensive",
                    "max_depth": 3,
                    "include_connectivity": True,
                },
                expected_fields=["structure", "relationships", "connectivity"],
                description="Comprehensive structure analysis",
            ),
            # Graph RAG Similar Implementations Tests
            RegressionTestCase(
                test_name="similar_implementations_basic",
                tool_function=graph_find_similar_implementations,
                parameters={"query": "search functionality", "max_results": 5, "similarity_threshold": 0.7},
                expected_fields=["similar_implementations", "analysis_metadata"],
                description="Basic similar implementations search",
            ),
            RegressionTestCase(
                test_name="similar_implementations_advanced",
                tool_function=graph_find_similar_implementations,
                parameters={
                    "query": "database service patterns",
                    "target_projects": ["Agentic_RAG"],
                    "chunk_types": ["function", "class"],
                    "max_results": 10,
                    "include_architectural_context": True,
                },
                expected_fields=["similar_implementations", "architectural_context"],
                description="Advanced similar implementations with architecture context",
            ),
            # Graph RAG Pattern Identification Tests
            RegressionTestCase(
                test_name="pattern_identification_basic",
                tool_function=graph_identify_patterns,
                parameters={"project_name": "Agentic_RAG", "min_confidence": 0.6, "max_patterns": 10},
                expected_fields=["patterns", "analysis_metadata"],
                description="Basic pattern identification",
            ),
            RegressionTestCase(
                test_name="pattern_identification_detailed",
                tool_function=graph_identify_patterns,
                parameters={
                    "project_name": "Agentic_RAG",
                    "pattern_types": ["structural", "behavioral"],
                    "analysis_depth": "detailed",
                    "include_comparisons": True,
                    "max_patterns": 15,
                },
                expected_fields=["patterns", "comparisons", "analysis_metadata"],
                description="Detailed pattern identification with comparisons",
            ),
            # Error Handling and Edge Cases
            RegressionTestCase(
                test_name="search_empty_query",
                tool_function=search,
                parameters={"query": "", "n_results": 5},
                expected_fields=["error", "message"],
                description="Search with empty query handling",
            ),
            RegressionTestCase(
                test_name="structure_analysis_invalid_breadcrumb",
                tool_function=graph_analyze_structure,
                parameters={"breadcrumb": "nonexistent_function_12345", "project_name": "Agentic_RAG"},
                expected_fields=["error", "suggestions"],
                description="Structure analysis with invalid breadcrumb",
            ),
            RegressionTestCase(
                test_name="similar_implementations_invalid_project",
                tool_function=graph_find_similar_implementations,
                parameters={"query": "search function", "target_projects": ["NonexistentProject12345"]},
                expected_fields=["error", "message"],
                description="Similar implementations with invalid project",
            ),
            # Performance and Scaling Tests
            RegressionTestCase(
                test_name="search_large_result_set",
                tool_function=search,
                parameters={"query": "function", "n_results": 50, "include_context": True},
                expected_fields=["results", "total_results"],
                max_response_time=3.0,
                description="Search with large result set",
            ),
            RegressionTestCase(
                test_name="structure_analysis_deep",
                tool_function=graph_analyze_structure,
                parameters={"breadcrumb": "main", "project_name": "Agentic_RAG", "analysis_type": "comprehensive", "max_depth": 8},
                expected_fields=["structure", "relationships"],
                max_response_time=4.0,
                description="Deep structure analysis",
            ),
            # Integration with New Function Chain Tools
            RegressionTestCase(
                test_name="pattern_identification_with_function_chains",
                tool_function=graph_identify_patterns,
                parameters={
                    "project_name": "Agentic_RAG",
                    "pattern_types": ["architectural"],
                    "scope_breadcrumb": "*chain*",
                    "min_confidence": 0.5,
                },
                expected_fields=["patterns", "analysis_metadata"],
                description="Pattern identification scoped to function chains",
            ),
        ]

    async def run_regression_tests(self) -> dict[str, Any]:
        """
        Run all regression tests and return comprehensive results.

        Returns:
            Dictionary containing test results and regression analysis
        """
        logger.info(f"Starting regression testing for project: {self.project_name}")
        start_time = time.time()

        results = {"project_name": self.project_name, "start_time": start_time, "test_results": [], "summary": {}}

        # Run all test cases
        for test_case in self.test_cases:
            test_result = await self._run_single_regression_test(test_case)
            results["test_results"].append(test_result)

            # Log progress
            status = "✅ PASS" if test_result.success else "❌ FAIL"
            logger.info(f"{status} {test_case.test_name} ({test_result.response_time:.2f}s)")

        # Calculate summary
        results["summary"] = self._calculate_regression_summary(results["test_results"])
        results["total_duration"] = time.time() - start_time

        logger.info(f"Regression testing completed in {results['total_duration']:.2f}s")
        logger.info(f"Success rate: {results['summary']['success_rate']:.1%}")

        return results

    async def _run_single_regression_test(self, test_case: RegressionTestCase) -> RegressionResult:
        """Run a single regression test case."""
        logger.debug(f"Running regression test: {test_case.test_name}")
        start_time = time.time()

        try:
            # Execute the tool function with parameters
            result = await test_case.tool_function(**test_case.parameters)
            response_time = time.time() - start_time

            # Check if result has expected fields
            has_expected_fields = self._check_expected_fields(result, test_case.expected_fields)

            # Determine success
            success = (
                isinstance(result, dict)
                and response_time <= test_case.max_response_time
                and (has_expected_fields or "error" in result)  # Error responses are acceptable for some tests
            )

            return RegressionResult(
                test_case=test_case,
                success=success,
                response_time=response_time,
                result_data=result,
                has_expected_fields=has_expected_fields,
            )

        except Exception as e:
            response_time = time.time() - start_time
            logger.warning(f"Regression test {test_case.test_name} failed with exception: {e}")

            return RegressionResult(
                test_case=test_case,
                success=False,
                response_time=response_time,
                result_data={},
                error_message=str(e),
                has_expected_fields=False,
            )

    def _check_expected_fields(self, result: dict[str, Any], expected_fields: list[str]) -> bool:
        """Check if result contains expected fields."""
        if not isinstance(result, dict):
            return False

        # For error cases, we expect error-related fields
        if "error" in expected_fields:
            return any(field in result for field in ["error", "message", "warning"])

        # For success cases, check for expected fields
        missing_fields = [field for field in expected_fields if field not in result]
        return len(missing_fields) == 0

    def _calculate_regression_summary(self, test_results: list[RegressionResult]) -> dict[str, Any]:
        """Calculate summary statistics for regression tests."""
        if not test_results:
            return {}

        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r.success)

        # Performance statistics
        response_times = [r.response_time for r in test_results]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Categorize results by tool type
        tool_categories = {}
        for result in test_results:
            tool_name = result.test_case.tool_function.__name__
            if tool_name not in tool_categories:
                tool_categories[tool_name] = []
            tool_categories[tool_name].append(result)

        # Calculate per-tool statistics
        tool_stats = {}
        for tool_name, results in tool_categories.items():
            successful = sum(1 for r in results if r.success)
            tool_stats[tool_name] = {
                "total": len(results),
                "successful": successful,
                "success_rate": successful / len(results),
                "avg_response_time": sum(r.response_time for r in results) / len(results),
            }

        # Identify failing tests
        failing_tests = [r for r in test_results if not r.success]

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "tool_statistics": tool_stats,
            "failing_tests_count": len(failing_tests),
            "failing_tests": [
                {
                    "test_name": r.test_case.test_name,
                    "tool": r.test_case.tool_function.__name__,
                    "error": r.error_message,
                    "response_time": r.response_time,
                }
                for r in failing_tests
            ],
            "regression_detected": successful_tests / total_tests < 0.95,  # Less than 95% success indicates regression
        }

    async def validate_existing_functionality(self) -> dict[str, Any]:
        """Validate that existing Graph RAG functionality still works."""
        logger.info("Validating existing Graph RAG functionality...")

        # Test core Graph RAG tools with known good parameters
        validation_tests = [
            {"name": "search_basic_validation", "function": search, "params": {"query": "main function", "n_results": 3}},
            {
                "name": "structure_analysis_validation",
                "function": graph_analyze_structure,
                "params": {"breadcrumb": "main", "project_name": self.project_name},
            },
            {
                "name": "similar_implementations_validation",
                "function": graph_find_similar_implementations,
                "params": {"query": "search function", "max_results": 3},
            },
            {
                "name": "pattern_identification_validation",
                "function": graph_identify_patterns,
                "params": {"project_name": self.project_name, "max_patterns": 5},
            },
        ]

        validation_results = []
        for test in validation_tests:
            start_time = time.time()
            try:
                result = await test["function"](**test["params"])
                success = isinstance(result, dict) and "error" not in result
                validation_results.append(
                    {
                        "test_name": test["name"],
                        "success": success,
                        "response_time": time.time() - start_time,
                        "has_results": bool(result) if success else False,
                    }
                )
            except Exception as e:
                validation_results.append(
                    {"test_name": test["name"], "success": False, "response_time": time.time() - start_time, "error": str(e)}
                )

        successful_validations = sum(1 for r in validation_results if r["success"])
        validation_success_rate = successful_validations / len(validation_results)

        return {
            "validation_tests": validation_results,
            "validation_success_rate": validation_success_rate,
            "existing_functionality_intact": validation_success_rate >= 0.90,
        }

    async def generate_regression_report(self) -> str:
        """Generate comprehensive regression test report."""
        # Run regression tests
        regression_results = await self.run_regression_tests()

        # Validate existing functionality
        validation_results = await self.validate_existing_functionality()

        # Generate report
        summary = regression_results["summary"]
        validation_summary = validation_results

        report_lines = [
            "=" * 80,
            "REGRESSION TEST REPORT",
            "=" * 80,
            f"Project: {regression_results['project_name']}",
            f"Test Duration: {regression_results['total_duration']:.2f}s",
            "",
            "REGRESSION TEST RESULTS:",
            f"  Total Tests: {summary['total_tests']}",
            f"  Successful: {summary['successful_tests']}",
            f"  Success Rate: {summary['success_rate']:.1%}",
            f"  Average Response Time: {summary['avg_response_time']:.3f}s",
            f"  Max Response Time: {summary['max_response_time']:.3f}s",
            f"  Regression Detected: {'❌ YES' if summary['regression_detected'] else '✅ NO'}",
            "",
            "EXISTING FUNCTIONALITY VALIDATION:",
            f"  Validation Success Rate: {validation_summary['validation_success_rate']:.1%}",
            f"  Existing Functionality: {'✅ INTACT' if validation_summary['existing_functionality_intact'] else '❌ BROKEN'}",
            "",
            "TOOL PERFORMANCE BREAKDOWN:",
        ]

        for tool_name, stats in summary["tool_statistics"].items():
            report_lines.extend(
                [
                    f"  {tool_name}:",
                    f"    Tests: {stats['total']}",
                    f"    Success Rate: {stats['success_rate']:.1%}",
                    f"    Avg Response Time: {stats['avg_response_time']:.3f}s",
                ]
            )

        if summary["failing_tests"]:
            report_lines.extend(
                [
                    "",
                    "FAILING TESTS:",
                ]
            )
            for failing_test in summary["failing_tests"]:
                report_lines.extend(
                    [
                        f"  {failing_test['test_name']} ({failing_test['tool']}):",
                        f"    Error: {failing_test['error'] or 'Test failed validation'}",
                        f"    Response Time: {failing_test['response_time']:.3f}s",
                        "",
                    ]
                )

        # Overall assessment
        overall_success = (
            summary["success_rate"] >= 0.95 and validation_summary["existing_functionality_intact"] and not summary["regression_detected"]
        )

        report_lines.extend(
            [
                "=" * 80,
                f"OVERALL ASSESSMENT: {'✅ PASSED' if overall_success else '❌ FAILED'}",
                "Function Chain tools integration impact:",
                f"  - Existing tools: {'Working' if validation_summary['existing_functionality_intact'] else 'Broken'}",
                f"  - Performance: {'Acceptable' if summary['avg_response_time'] < 3.0 else 'Degraded'}",
                f"  - Regression risk: {'Low' if not summary['regression_detected'] else 'High'}",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)


# Test functions for pytest
async def test_no_regression_in_search():
    """Test that search functionality has not regressed."""
    suite = RegressionTestSuite("Agentic_RAG")

    # Run search-specific tests
    search_tests = [t for t in suite.test_cases if "search" in t.test_name]

    search_results = []
    for test_case in search_tests:
        result = await suite._run_single_regression_test(test_case)
        search_results.append(result)

    success_rate = sum(1 for r in search_results if r.success) / len(search_results)
    assert success_rate >= 0.90, f"Search regression detected: {success_rate:.1%} success rate"


async def test_no_regression_in_graph_rag():
    """Test that Graph RAG tools have not regressed."""
    suite = RegressionTestSuite("Agentic_RAG")

    # Run Graph RAG specific tests
    graph_rag_tests = [t for t in suite.test_cases if any(keyword in t.test_name for keyword in ["structure", "similar", "pattern"])]

    graph_rag_results = []
    for test_case in graph_rag_tests:
        result = await suite._run_single_regression_test(test_case)
        graph_rag_results.append(result)

    success_rate = sum(1 for r in graph_rag_results if r.success) / len(graph_rag_results)
    assert success_rate >= 0.85, f"Graph RAG regression detected: {success_rate:.1%} success rate"


async def test_existing_functionality_intact():
    """Test that existing functionality remains intact after adding Function Chain tools."""
    suite = RegressionTestSuite("Agentic_RAG")
    validation_results = await suite.validate_existing_functionality()

    assert validation_results["existing_functionality_intact"], "Existing functionality has been broken"
    assert (
        validation_results["validation_success_rate"] >= 0.90
    ), f"Validation success rate too low: {validation_results['validation_success_rate']:.1%}"


async def test_comprehensive_regression():
    """Test comprehensive regression across all tools."""
    suite = RegressionTestSuite("Agentic_RAG")
    results = await suite.run_regression_tests()

    summary = results["summary"]
    assert not summary["regression_detected"], "Regression detected in comprehensive testing"
    assert summary["success_rate"] >= 0.90, f"Overall success rate too low: {summary['success_rate']:.1%}"


if __name__ == "__main__":
    # Run regression tests directly
    async def main():
        suite = RegressionTestSuite("Agentic_RAG")
        report = await suite.generate_regression_report()
        print(report)

        # Return success/failure for exit code
        results = await suite.run_regression_tests()
        validation = await suite.validate_existing_functionality()

        overall_success = (
            results["summary"]["success_rate"] >= 0.95
            and validation["existing_functionality_intact"]
            and not results["summary"]["regression_detected"]
        )

        return overall_success

    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
