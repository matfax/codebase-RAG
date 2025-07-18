"""
End-to-End Integration Tests for Function Chain MCP Tools.

This module provides comprehensive integration tests that validate the complete
workflow of all function chain tools, ensuring they work together seamlessly
and meet performance requirements.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

import pytest

from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.services.implementation_chain_service import get_implementation_chain_service
from src.tools.graph_rag.function_chain_analysis import trace_function_chain
from src.tools.graph_rag.function_path_finding import find_function_path
from src.tools.graph_rag.project_chain_analysis import analyze_project_chains
from src.utils.complexity_calculator import create_complexity_calculator, get_default_complexity_weights
from src.utils.output_formatters import OutputFormat, format_comprehensive_output

logger = logging.getLogger(__name__)


class FunctionChainIntegrationTester:
    """Integration tester for function chain tools."""

    def __init__(self, project_name: str = "test_project"):
        """Initialize the integration tester."""
        self.project_name = project_name
        self.breadcrumb_resolver = BreadcrumbResolver()
        self.implementation_chain_service = get_implementation_chain_service()
        self.complexity_calculator = create_complexity_calculator()
        self.test_results = {}

    async def run_comprehensive_tests(self) -> dict[str, Any]:
        """
        Run comprehensive end-to-end tests for all function chain tools.

        Returns:
            Dictionary containing test results and performance metrics
        """
        logger.info(f"Starting comprehensive integration tests for project: {self.project_name}")
        start_time = time.time()

        test_results = {
            "project_name": self.project_name,
            "start_time": start_time,
            "tests": {},
            "performance": {},
            "overall_status": "unknown",
        }

        try:
            # Test 1: Trace Function Chain Tool
            test_results["tests"]["trace_function_chain"] = await self._test_trace_function_chain()

            # Test 2: Find Function Path Tool
            test_results["tests"]["find_function_path"] = await self._test_find_function_path()

            # Test 3: Analyze Project Chains Tool
            test_results["tests"]["analyze_project_chains"] = await self._test_analyze_project_chains()

            # Test 4: Integration Workflow
            test_results["tests"]["integration_workflow"] = await self._test_integration_workflow()

            # Test 5: Performance Testing
            test_results["tests"]["performance"] = await self._test_performance()

            # Test 6: Output Format Consistency
            test_results["tests"]["output_formats"] = await self._test_output_formats()

            # Test 7: Error Handling
            test_results["tests"]["error_handling"] = await self._test_error_handling()

            # Calculate overall performance
            end_time = time.time()
            test_results["total_duration"] = end_time - start_time
            test_results["overall_status"] = self._calculate_overall_status(test_results["tests"])

            logger.info(f"Integration tests completed in {test_results['total_duration']:.2f}s")
            return test_results

        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            test_results["error"] = str(e)
            test_results["overall_status"] = "failed"
            return test_results

    async def _test_trace_function_chain(self) -> dict[str, Any]:
        """Test the trace_function_chain tool."""
        logger.info("Testing trace_function_chain tool...")
        start_time = time.time()

        try:
            # Test with different configurations
            test_cases = [
                {"entry_point": "main", "direction": "forward", "max_depth": 5, "output_format": "arrow"},
                {"entry_point": "process_data", "direction": "bidirectional", "max_depth": 10, "output_format": "both"},
            ]

            results = []
            for test_case in test_cases:
                result = await trace_function_chain(
                    entry_point=test_case["entry_point"],
                    project_name=self.project_name,
                    direction=test_case["direction"],
                    max_depth=test_case["max_depth"],
                    output_format=test_case["output_format"],
                )
                results.append({"test_case": test_case, "result": result, "success": "error" not in result})

            duration = time.time() - start_time
            success_rate = sum(1 for r in results if r["success"]) / len(results)

            return {
                "status": "passed" if success_rate >= 0.8 else "failed",
                "duration": duration,
                "success_rate": success_rate,
                "test_cases": len(test_cases),
                "results": results,
            }

        except Exception as e:
            return {"status": "failed", "duration": time.time() - start_time, "error": str(e)}

    async def _test_find_function_path(self) -> dict[str, Any]:
        """Test the find_function_path tool."""
        logger.info("Testing find_function_path tool...")
        start_time = time.time()

        try:
            # Test path finding between different functions
            test_cases = [
                {"start_function": "main", "end_function": "process_data", "strategy": "optimal", "max_paths": 3},
                {"start_function": "init_system", "end_function": "cleanup", "strategy": "shortest", "max_paths": 1},
            ]

            results = []
            for test_case in test_cases:
                result = await find_function_path(
                    start_function=test_case["start_function"],
                    end_function=test_case["end_function"],
                    project_name=self.project_name,
                    strategy=test_case["strategy"],
                    max_paths=test_case["max_paths"],
                )
                results.append({"test_case": test_case, "result": result, "success": "error" not in result and "paths" in result})

            duration = time.time() - start_time
            success_rate = sum(1 for r in results if r["success"]) / len(results)

            return {
                "status": "passed" if success_rate >= 0.8 else "failed",
                "duration": duration,
                "success_rate": success_rate,
                "test_cases": len(test_cases),
                "results": results,
            }

        except Exception as e:
            return {"status": "failed", "duration": time.time() - start_time, "error": str(e)}

    async def _test_analyze_project_chains(self) -> dict[str, Any]:
        """Test the analyze_project_chains tool."""
        logger.info("Testing analyze_project_chains tool...")
        start_time = time.time()

        try:
            # Test project-wide analysis
            test_cases = [
                {"analysis_scope": "full_project", "include_hotspot_analysis": True, "output_format": "comprehensive"},
                {"analysis_scope": "function_patterns", "breadcrumb_patterns": ["*main*", "*process*"], "complexity_threshold": 0.7},
            ]

            results = []
            for test_case in test_cases:
                result = await analyze_project_chains(
                    project_name=self.project_name,
                    analysis_scope=test_case["analysis_scope"],
                    **{k: v for k, v in test_case.items() if k != "analysis_scope"},
                )
                results.append({"test_case": test_case, "result": result, "success": "error" not in result and "analysis" in result})

            duration = time.time() - start_time
            success_rate = sum(1 for r in results if r["success"]) / len(results)

            return {
                "status": "passed" if success_rate >= 0.8 else "failed",
                "duration": duration,
                "success_rate": success_rate,
                "test_cases": len(test_cases),
                "results": results,
            }

        except Exception as e:
            return {"status": "failed", "duration": time.time() - start_time, "error": str(e)}

    async def _test_integration_workflow(self) -> dict[str, Any]:
        """Test complete integration workflow between tools."""
        logger.info("Testing integration workflow...")
        start_time = time.time()

        try:
            # Step 1: Analyze project to find hotspots
            project_analysis = await analyze_project_chains(
                project_name=self.project_name, analysis_scope="full_project", include_hotspot_analysis=True
            )

            # Step 2: If hotspots found, trace their chains
            traces = []
            if "hotspots" in project_analysis:
                for hotspot in project_analysis["hotspots"][:2]:  # Test first 2 hotspots
                    trace_result = await trace_function_chain(
                        entry_point=hotspot.get("breadcrumb", "main"), project_name=self.project_name, direction="forward", max_depth=10
                    )
                    traces.append(trace_result)

            # Step 3: Find paths between identified functions
            paths = []
            if len(traces) >= 2:
                path_result = await find_function_path(
                    start_function=traces[0].get("entry_point", "main"),
                    end_function=traces[1].get("entry_point", "process_data"),
                    project_name=self.project_name,
                    strategy="optimal",
                )
                paths.append(path_result)

            duration = time.time() - start_time
            workflow_success = (
                "error" not in project_analysis
                and all("error" not in trace for trace in traces)
                and all("error" not in path for path in paths)
            )

            return {
                "status": "passed" if workflow_success else "failed",
                "duration": duration,
                "steps_completed": {
                    "project_analysis": "error" not in project_analysis,
                    "function_traces": len([t for t in traces if "error" not in t]),
                    "path_finding": len([p for p in paths if "error" not in p]),
                },
                "workflow_success": workflow_success,
            }

        except Exception as e:
            return {"status": "failed", "duration": time.time() - start_time, "error": str(e)}

    async def _test_performance(self) -> dict[str, Any]:
        """Test performance requirements (<2 seconds response time)."""
        logger.info("Testing performance requirements...")

        performance_tests = []

        # Test 1: Trace function chain performance
        start_time = time.time()
        try:
            result = await trace_function_chain(entry_point="main", project_name=self.project_name, direction="forward", max_depth=10)
            duration = time.time() - start_time
            performance_tests.append(
                {
                    "test": "trace_function_chain",
                    "duration": duration,
                    "meets_requirement": duration < 2.0,
                    "success": "error" not in result,
                }
            )
        except Exception as e:
            performance_tests.append(
                {"test": "trace_function_chain", "duration": time.time() - start_time, "meets_requirement": False, "error": str(e)}
            )

        # Test 2: Find function path performance
        start_time = time.time()
        try:
            result = await find_function_path(
                start_function="main", end_function="process_data", project_name=self.project_name, strategy="optimal"
            )
            duration = time.time() - start_time
            performance_tests.append(
                {"test": "find_function_path", "duration": duration, "meets_requirement": duration < 2.0, "success": "error" not in result}
            )
        except Exception as e:
            performance_tests.append(
                {"test": "find_function_path", "duration": time.time() - start_time, "meets_requirement": False, "error": str(e)}
            )

        # Test 3: Analyze project chains performance
        start_time = time.time()
        try:
            result = await analyze_project_chains(
                project_name=self.project_name, analysis_scope="scoped_breadcrumbs", breadcrumb_patterns=["*main*"]
            )
            duration = time.time() - start_time
            performance_tests.append(
                {
                    "test": "analyze_project_chains",
                    "duration": duration,
                    "meets_requirement": duration < 2.0,
                    "success": "error" not in result,
                }
            )
        except Exception as e:
            performance_tests.append(
                {"test": "analyze_project_chains", "duration": time.time() - start_time, "meets_requirement": False, "error": str(e)}
            )

        # Calculate overall performance metrics
        total_tests = len(performance_tests)
        successful_tests = sum(1 for test in performance_tests if test.get("success", False))
        meeting_requirements = sum(1 for test in performance_tests if test.get("meets_requirement", False))
        avg_duration = sum(test["duration"] for test in performance_tests) / total_tests

        return {
            "status": "passed" if meeting_requirements == total_tests else "failed",
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "meeting_requirements": meeting_requirements,
            "performance_success_rate": meeting_requirements / total_tests,
            "average_duration": avg_duration,
            "max_duration": max(test["duration"] for test in performance_tests),
            "individual_tests": performance_tests,
        }

    async def _test_output_formats(self) -> dict[str, Any]:
        """Test output format consistency and validity."""
        logger.info("Testing output format consistency...")
        start_time = time.time()

        try:
            # Test different output formats
            format_tests = []

            # Test arrow format
            result_arrow = await trace_function_chain(entry_point="main", project_name=self.project_name, output_format="arrow")
            format_tests.append(
                {
                    "format": "arrow",
                    "success": "arrow_format" in result_arrow,
                    "has_content": bool(result_arrow.get("arrow_format", "").strip()),
                }
            )

            # Test mermaid format
            result_mermaid = await trace_function_chain(entry_point="main", project_name=self.project_name, output_format="mermaid")
            format_tests.append(
                {
                    "format": "mermaid",
                    "success": "mermaid_format" in result_mermaid,
                    "has_content": bool(result_mermaid.get("mermaid_format", "").strip()),
                }
            )

            # Test both formats
            result_both = await trace_function_chain(entry_point="main", project_name=self.project_name, output_format="both")
            format_tests.append(
                {
                    "format": "both",
                    "success": "arrow_format" in result_both and "mermaid_format" in result_both,
                    "has_content": (
                        bool(result_both.get("arrow_format", "").strip()) and bool(result_both.get("mermaid_format", "").strip())
                    ),
                }
            )

            duration = time.time() - start_time
            success_rate = sum(1 for test in format_tests if test["success"]) / len(format_tests)

            return {
                "status": "passed" if success_rate == 1.0 else "failed",
                "duration": duration,
                "success_rate": success_rate,
                "format_tests": format_tests,
            }

        except Exception as e:
            return {"status": "failed", "duration": time.time() - start_time, "error": str(e)}

    async def _test_error_handling(self) -> dict[str, Any]:
        """Test error handling and graceful degradation."""
        logger.info("Testing error handling...")
        start_time = time.time()

        error_tests = []

        # Test 1: Invalid entry point
        try:
            result = await trace_function_chain(entry_point="nonexistent_function_12345", project_name=self.project_name)
            error_tests.append(
                {
                    "test": "invalid_entry_point",
                    "graceful": "error" in result or "warning" in result,
                    "has_fallback": "suggestions" in result or "alternatives" in result,
                }
            )
        except Exception:
            error_tests.append({"test": "invalid_entry_point", "graceful": False, "exception_raised": True})

        # Test 2: Invalid project name
        try:
            result = await find_function_path(start_function="main", end_function="process_data", project_name="nonexistent_project_12345")
            error_tests.append(
                {"test": "invalid_project", "graceful": "error" in result or "warning" in result, "has_fallback": "message" in result}
            )
        except Exception:
            error_tests.append({"test": "invalid_project", "graceful": False, "exception_raised": True})

        # Test 3: Invalid parameters
        try:
            result = await analyze_project_chains(
                project_name=self.project_name,
                max_functions_per_chain=-1,  # Invalid negative value
                complexity_threshold=2.0,  # Invalid threshold > 1.0
            )
            error_tests.append(
                {
                    "test": "invalid_parameters",
                    "graceful": "error" in result or "warning" in result,
                    "parameter_validation": "validation" in result or "corrected" in result,
                }
            )
        except Exception:
            error_tests.append({"test": "invalid_parameters", "graceful": False, "exception_raised": True})

        duration = time.time() - start_time
        graceful_handling = sum(1 for test in error_tests if test.get("graceful", False))

        return {
            "status": "passed" if graceful_handling >= len(error_tests) * 0.8 else "failed",
            "duration": duration,
            "graceful_handling_rate": graceful_handling / len(error_tests),
            "error_tests": error_tests,
        }

    def _calculate_overall_status(self, test_results: dict[str, Any]) -> str:
        """Calculate overall test status."""
        passed_tests = sum(1 for test in test_results.values() if test.get("status") == "passed")
        total_tests = len(test_results)

        if passed_tests == total_tests:
            return "passed"
        elif passed_tests >= total_tests * 0.8:
            return "mostly_passed"
        else:
            return "failed"


# Pytest test functions
@pytest.mark.asyncio
async def test_function_chain_integration():
    """Test complete function chain tool integration."""
    tester = FunctionChainIntegrationTester("test_project")
    results = await tester.run_comprehensive_tests()

    assert results["overall_status"] in ["passed", "mostly_passed"], f"Integration tests failed: {results}"
    assert results["total_duration"] < 30.0, f"Tests took too long: {results['total_duration']:.2f}s"


@pytest.mark.asyncio
async def test_performance_requirements():
    """Test that all tools meet performance requirements."""
    tester = FunctionChainIntegrationTester("test_project")
    performance_results = await tester._test_performance()

    assert performance_results["status"] == "passed", f"Performance tests failed: {performance_results}"
    assert performance_results["performance_success_rate"] >= 0.8, "Too many tools failing performance requirements"


@pytest.mark.asyncio
async def test_output_format_consistency():
    """Test output format consistency across tools."""
    tester = FunctionChainIntegrationTester("test_project")
    format_results = await tester._test_output_formats()

    assert format_results["status"] == "passed", f"Output format tests failed: {format_results}"
    assert format_results["success_rate"] == 1.0, "Some output formats are inconsistent"


if __name__ == "__main__":
    # Run integration tests directly
    async def main():
        tester = FunctionChainIntegrationTester("Agentic_RAG")
        results = await tester.run_comprehensive_tests()

        print("\n" + "=" * 80)
        print("FUNCTION CHAIN INTEGRATION TEST RESULTS")
        print("=" * 80)
        print(f"Project: {results['project_name']}")
        print(f"Overall Status: {results['overall_status']}")
        print(f"Total Duration: {results['total_duration']:.2f}s")
        print("\nTest Results:")

        for test_name, test_result in results["tests"].items():
            status = test_result.get("status", "unknown")
            duration = test_result.get("duration", 0)
            print(f"  {test_name}: {status} ({duration:.2f}s)")

        return results["overall_status"] in ["passed", "mostly_passed"]

    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)
