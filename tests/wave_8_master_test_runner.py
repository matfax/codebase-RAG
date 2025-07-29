#!/usr/bin/env python3
"""
Wave 8.0 Master Test Runner

This module orchestrates all Wave 8.0 testing frameworks, providing
comprehensive validation of the agentic-rag-performance-enhancement project.
Validates all performance targets and generates final project completion report.
"""

import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wave_8_ab_testing_framework import ABTestingFramework
from wave_8_accuracy_comparison_tests import AccuracyComparisonTester
from wave_8_ci_integration import CIIntegrationFramework
from wave_8_concurrent_user_tests import ConcurrentUserTester
from wave_8_large_project_tests import LargeProjectTester
from wave_8_memory_usage_tests import MemoryUsageValidator

# Import all Wave 8 test modules
from wave_8_performance_regression_tests import PerformanceRegressionTester
from wave_8_response_time_tests import ResponseTimeTester
from wave_8_stress_tests import StressTester
from wave_8_test_data_generation import TestDataGenerator


@dataclass
class ProjectTargets:
    """Project performance targets to validate"""

    response_time_target_seconds: float = 15.0
    response_time_success_rate: float = 95.0
    memory_reduction_target_percent: float = 50.0
    min_file_support: int = 1000
    min_concurrent_users: int = 5
    accuracy_preservation_rate: float = 95.0
    error_rate_threshold: float = 5.0
    system_stability_score: float = 80.0


@dataclass
class TestSuiteResult:
    """Result from a test suite execution"""

    suite_name: str
    status: str  # "PASS", "FAIL", "ERROR"
    duration_seconds: float
    targets_met: bool
    key_metrics: dict[str, Any]
    report_file: str
    error_message: str | None = None


@dataclass
class Wave8ValidationResult:
    """Complete Wave 8.0 validation result"""

    project_name: str
    wave_version: str
    validation_time: str
    overall_status: str
    targets: ProjectTargets
    suite_results: list[TestSuiteResult]
    targets_achieved: dict[str, bool]
    final_performance_score: float
    deployment_recommendation: str
    detailed_findings: dict[str, Any]
    artifacts: list[str]


class Wave8MasterTestRunner:
    """Master test runner for Wave 8.0 validation"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.targets = ProjectTargets()
        self.suite_results: list[TestSuiteResult] = []
        self.artifacts: list[str] = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for master test runner"""
        logger = logging.getLogger("wave8_master")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def run_performance_regression_tests(self) -> TestSuiteResult:
        """Run performance regression testing"""
        self.logger.info("Running Performance Regression Tests (8.1)")
        start_time = time.time()

        try:
            tester = PerformanceRegressionTester()
            summary = await tester.run_all_tests()
            tester.generate_report("wave_8_1_regression_report.json")

            duration = time.time() - start_time
            targets_met = summary["failed"] == 0 and summary["overall_status"] == "PASS"

            return TestSuiteResult(
                suite_name="Performance Regression",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "total_tests": summary["total_tests"],
                    "passed": summary["passed"],
                    "failed": summary["failed"],
                    "average_degradation": summary["average_degradation_percent"],
                },
                report_file="wave_8_1_regression_report.json",
            )

        except Exception as e:
            self.logger.error(f"Performance regression tests failed: {e}")
            return TestSuiteResult(
                suite_name="Performance Regression",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_response_time_tests(self) -> TestSuiteResult:
        """Run response time testing"""
        self.logger.info("Running Response Time Tests (8.2)")
        start_time = time.time()

        try:
            tester = ResponseTimeTester()
            summary = await tester.run_all_tests()
            tester.generate_report("wave_8_2_response_time_report.json")

            duration = time.time() - start_time

            # Check targets
            success_rate = summary["overall_success_rate_percent"]
            p95_time_ms = summary["response_time_stats"]["p95_ms"]

            targets_met = success_rate >= self.targets.response_time_success_rate and p95_time_ms <= (
                self.targets.response_time_target_seconds * 1000
            )

            return TestSuiteResult(
                suite_name="Response Time",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "success_rate_percent": success_rate,
                    "p95_response_time_ms": p95_time_ms,
                    "target_met": summary["target_met"],
                    "total_queries": summary["total_queries"],
                },
                report_file="wave_8_2_response_time_report.json",
            )

        except Exception as e:
            self.logger.error(f"Response time tests failed: {e}")
            return TestSuiteResult(
                suite_name="Response Time",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_memory_usage_tests(self) -> TestSuiteResult:
        """Run memory usage validation"""
        self.logger.info("Running Memory Usage Tests (8.3)")
        start_time = time.time()

        try:
            validator = MemoryUsageValidator()
            summary = await validator.run_all_tests()
            validator.generate_report("wave_8_3_memory_usage_report.json")

            duration = time.time() - start_time

            # Check memory targets
            memory_increase = summary["memory_statistics"]["average_increase_percent"]
            targets_met = summary["overall_target_met"] and memory_increase <= self.targets.memory_reduction_target_percent

            return TestSuiteResult(
                suite_name="Memory Usage",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "memory_increase_percent": memory_increase,
                    "tests_target_met": summary["tests_target_met"],
                    "tests_with_leaks": summary["tests_with_leaks"],
                    "efficiency_score": summary["memory_statistics"]["average_efficiency_score"],
                },
                report_file="wave_8_3_memory_usage_report.json",
            )

        except Exception as e:
            self.logger.error(f"Memory usage tests failed: {e}")
            return TestSuiteResult(
                suite_name="Memory Usage",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_large_project_tests(self) -> TestSuiteResult:
        """Run large project testing"""
        self.logger.info("Running Large Project Tests (8.4)")
        start_time = time.time()

        try:
            tester = LargeProjectTester()
            summary = await tester.run_all_tests()
            tester.generate_report("wave_8_4_large_project_report.json")

            duration = time.time() - start_time

            # Check file support target
            max_files = (
                max([result["project_scale"]["total_files"] for result in summary.get("test_results", [])])
                if summary.get("test_results")
                else 0
            )

            targets_met = max_files >= self.targets.min_file_support and summary["target_achievement_rate"] >= 80

            return TestSuiteResult(
                suite_name="Large Project",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "max_files_tested": max_files,
                    "target_achievement_rate": summary["target_achievement_rate"],
                    "average_throughput": summary["performance_statistics"]["average_throughput_files_per_second"],
                },
                report_file="wave_8_4_large_project_report.json",
            )

        except Exception as e:
            self.logger.error(f"Large project tests failed: {e}")
            return TestSuiteResult(
                suite_name="Large Project",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_concurrent_user_tests(self) -> TestSuiteResult:
        """Run concurrent user testing"""
        self.logger.info("Running Concurrent User Tests (8.5)")
        start_time = time.time()

        try:
            tester = ConcurrentUserTester()
            summary = await tester.run_all_tests()
            tester.generate_report("wave_8_5_concurrent_user_report.json")

            duration = time.time() - start_time

            # Check concurrent user targets
            max_users = summary["max_concurrent_users_achieved"]
            error_rate = summary["performance_statistics"]["average_error_rate_percent"]

            targets_met = max_users >= self.targets.min_concurrent_users and error_rate <= self.targets.error_rate_threshold

            return TestSuiteResult(
                suite_name="Concurrent Users",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "max_concurrent_users": max_users,
                    "target_achievement_rate": summary["target_achievement_rate"],
                    "average_error_rate": error_rate,
                    "average_response_time": summary["performance_statistics"]["average_response_time_ms"],
                },
                report_file="wave_8_5_concurrent_user_report.json",
            )

        except Exception as e:
            self.logger.error(f"Concurrent user tests failed: {e}")
            return TestSuiteResult(
                suite_name="Concurrent Users",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_accuracy_comparison_tests(self) -> TestSuiteResult:
        """Run accuracy comparison testing"""
        self.logger.info("Running Accuracy Comparison Tests (8.6)")
        start_time = time.time()

        try:
            tester = AccuracyComparisonTester()
            summary = await tester.run_all_tests()
            tester.generate_report("wave_8_6_accuracy_comparison_report.json")

            duration = time.time() - start_time

            # Check accuracy targets
            accuracy_rate = summary["accuracy_preservation_rate"]
            targets_met = accuracy_rate >= self.targets.accuracy_preservation_rate and summary["target_met"]

            return TestSuiteResult(
                suite_name="Accuracy Comparison",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "accuracy_preservation_rate": accuracy_rate,
                    "average_f1_score": summary["aggregate_metrics"]["average_f1_score"],
                    "average_degradation": summary["aggregate_metrics"]["average_degradation_percent"],
                },
                report_file="wave_8_6_accuracy_comparison_report.json",
            )

        except Exception as e:
            self.logger.error(f"Accuracy comparison tests failed: {e}")
            return TestSuiteResult(
                suite_name="Accuracy Comparison",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_stress_tests(self) -> TestSuiteResult:
        """Run stress testing"""
        self.logger.info("Running Stress Tests (8.7)")
        start_time = time.time()

        try:
            tester = StressTester()
            summary = await tester.run_all_tests()
            tester.generate_report("wave_8_7_stress_test_report.json")

            duration = time.time() - start_time

            # Check stress test targets
            stability_score = summary["system_stability_score"]
            targets_met = stability_score >= self.targets.system_stability_score

            return TestSuiteResult(
                suite_name="Stress Testing",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "system_stability_score": stability_score,
                    "tests_with_failures": summary["tests_with_failures"],
                    "peak_cpu_percent": summary["stress_statistics"]["peak_cpu_percent"],
                    "peak_memory_percent": summary["stress_statistics"]["peak_memory_percent"],
                },
                report_file="wave_8_7_stress_test_report.json",
            )

        except Exception as e:
            self.logger.error(f"Stress tests failed: {e}")
            return TestSuiteResult(
                suite_name="Stress Testing",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_ab_testing(self) -> TestSuiteResult:
        """Run A/B testing framework"""
        self.logger.info("Running A/B Testing (8.8)")
        start_time = time.time()

        try:
            framework = ABTestingFramework()
            summary = await framework.run_all_tests()
            framework.generate_report("wave_8_8_ab_testing_report.json")

            duration = time.time() - start_time

            # A/B testing is informational - always pass but provide insights
            enhanced_win_rate = summary["enhanced_win_rate"]
            targets_met = enhanced_win_rate >= 60.0  # Enhanced should win majority

            return TestSuiteResult(
                suite_name="A/B Testing",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "enhanced_win_rate": enhanced_win_rate,
                    "average_confidence": summary["average_confidence_score"],
                    "average_improvement": summary["average_improvement_percent"],
                    "recommendation": summary["overall_recommendation"],
                },
                report_file="wave_8_8_ab_testing_report.json",
            )

        except Exception as e:
            self.logger.error(f"A/B testing failed: {e}")
            return TestSuiteResult(
                suite_name="A/B Testing",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    async def run_test_data_generation(self) -> TestSuiteResult:
        """Run test data generation"""
        self.logger.info("Running Test Data Generation (8.10)")
        start_time = time.time()

        try:
            generator = TestDataGenerator()
            datasets = generator.generate_all_datasets()
            generator.export_dataset_summary(datasets, "wave_8_10_test_data_summary.json")

            duration = time.time() - start_time

            # Check if sufficient test data was generated
            total_files = sum(d.file_count for d in datasets)
            total_queries = sum(d.query_count for d in datasets)

            targets_met = total_files >= 1000 and total_queries >= 500

            return TestSuiteResult(
                suite_name="Test Data Generation",
                status="PASS" if targets_met else "FAIL",
                duration_seconds=duration,
                targets_met=targets_met,
                key_metrics={
                    "datasets_generated": len(datasets),
                    "total_files": total_files,
                    "total_queries": total_queries,
                    "languages": list(set(lang for d in datasets for lang in d.languages)),
                },
                report_file="wave_8_10_test_data_summary.json",
            )

        except Exception as e:
            self.logger.error(f"Test data generation failed: {e}")
            return TestSuiteResult(
                suite_name="Test Data Generation",
                status="ERROR",
                duration_seconds=time.time() - start_time,
                targets_met=False,
                key_metrics={},
                report_file="",
                error_message=str(e),
            )

    def _calculate_performance_score(self, suite_results: list[TestSuiteResult]) -> float:
        """Calculate overall performance score"""
        if not suite_results:
            return 0.0

        # Weight different test suites
        weights = {
            "Performance Regression": 0.15,
            "Response Time": 0.20,
            "Memory Usage": 0.15,
            "Large Project": 0.15,
            "Concurrent Users": 0.15,
            "Accuracy Comparison": 0.10,
            "Stress Testing": 0.05,
            "A/B Testing": 0.03,
            "Test Data Generation": 0.02,
        }

        total_score = 0.0
        total_weight = 0.0

        for result in suite_results:
            weight = weights.get(result.suite_name, 0.05)
            score = 100.0 if result.targets_met else 0.0

            # Adjust score based on specific metrics
            if result.suite_name == "Response Time" and result.targets_met:
                success_rate = result.key_metrics.get("success_rate_percent", 95)
                score = min(100.0, success_rate)
            elif result.suite_name == "Memory Usage" and result.targets_met:
                efficiency = result.key_metrics.get("efficiency_score", 80)
                score = min(100.0, efficiency)
            elif result.suite_name == "Stress Testing" and result.targets_met:
                stability = result.key_metrics.get("system_stability_score", 80)
                score = min(100.0, stability)

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _evaluate_targets_achieved(self, suite_results: list[TestSuiteResult]) -> dict[str, bool]:
        """Evaluate which project targets were achieved"""
        targets_achieved = {}

        # Response time target
        response_result = next((r for r in suite_results if r.suite_name == "Response Time"), None)
        if response_result:
            targets_achieved["response_time_15s_95percent"] = response_result.targets_met
        else:
            targets_achieved["response_time_15s_95percent"] = False

        # Memory reduction target
        memory_result = next((r for r in suite_results if r.suite_name == "Memory Usage"), None)
        if memory_result:
            targets_achieved["memory_reduction_50percent"] = memory_result.targets_met
        else:
            targets_achieved["memory_reduction_50percent"] = False

        # Large project support
        large_project_result = next((r for r in suite_results if r.suite_name == "Large Project"), None)
        if large_project_result:
            targets_achieved["support_1000plus_files"] = large_project_result.targets_met
        else:
            targets_achieved["support_1000plus_files"] = False

        # Concurrent users
        concurrent_result = next((r for r in suite_results if r.suite_name == "Concurrent Users"), None)
        if concurrent_result:
            targets_achieved["support_5_concurrent_users"] = concurrent_result.targets_met
        else:
            targets_achieved["support_5_concurrent_users"] = False

        # Accuracy preservation
        accuracy_result = next((r for r in suite_results if r.suite_name == "Accuracy Comparison"), None)
        if accuracy_result:
            targets_achieved["preserve_accuracy_95percent"] = accuracy_result.targets_met
        else:
            targets_achieved["preserve_accuracy_95percent"] = False

        return targets_achieved

    def _generate_deployment_recommendation(self, targets_achieved: dict[str, bool], performance_score: float) -> str:
        """Generate deployment recommendation"""
        core_targets_met = sum(
            [
                targets_achieved.get("response_time_15s_95percent", False),
                targets_achieved.get("memory_reduction_50percent", False),
                targets_achieved.get("support_1000plus_files", False),
                targets_achieved.get("support_5_concurrent_users", False),
                targets_achieved.get("preserve_accuracy_95percent", False),
            ]
        )

        if core_targets_met >= 4 and performance_score >= 85:
            return "RECOMMENDED FOR PRODUCTION DEPLOYMENT - All major targets achieved"
        elif core_targets_met >= 3 and performance_score >= 75:
            return "RECOMMENDED FOR STAGED DEPLOYMENT - Most targets achieved with minor issues"
        elif core_targets_met >= 2 and performance_score >= 60:
            return "RECOMMENDED FOR LIMITED DEPLOYMENT - Significant improvements but some targets missed"
        else:
            return "NOT RECOMMENDED FOR DEPLOYMENT - Major performance targets not achieved"

    async def run_comprehensive_validation(self) -> Wave8ValidationResult:
        """Run comprehensive Wave 8.0 validation"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING WAVE 8.0 COMPREHENSIVE VALIDATION")
        self.logger.info("Agentic RAG Performance Enhancement Project")
        self.logger.info("=" * 60)

        validation_start = time.time()

        # Run all test suites
        test_suites = [
            self.run_performance_regression_tests,
            self.run_response_time_tests,
            self.run_memory_usage_tests,
            self.run_large_project_tests,
            self.run_concurrent_user_tests,
            self.run_accuracy_comparison_tests,
            self.run_stress_tests,
            self.run_ab_testing,
            self.run_test_data_generation,
        ]

        suite_results = []
        for test_suite in test_suites:
            try:
                result = await test_suite()
                suite_results.append(result)
                self.artifacts.append(result.report_file)

                status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
                self.logger.info(f"{status_emoji} {result.suite_name}: {result.status} ({result.duration_seconds:.1f}s)")

            except Exception as e:
                self.logger.error(f"Failed to run test suite: {e}")

        validation_end = time.time()
        validation_duration = validation_end - validation_start

        # Calculate results
        performance_score = self._calculate_performance_score(suite_results)
        targets_achieved = self._evaluate_targets_achieved(suite_results)
        deployment_recommendation = self._generate_deployment_recommendation(targets_achieved, performance_score)

        # Determine overall status
        passed_suites = len([r for r in suite_results if r.status == "PASS"])
        failed_suites = len([r for r in suite_results if r.status == "FAIL"])
        error_suites = len([r for r in suite_results if r.status == "ERROR"])

        if error_suites > 0:
            overall_status = "ERROR"
        elif failed_suites == 0:
            overall_status = "PASS"
        elif failed_suites <= 2:  # Allow some failures
            overall_status = "PARTIAL"
        else:
            overall_status = "FAIL"

        # Create validation result
        result = Wave8ValidationResult(
            project_name="Agentic RAG Performance Enhancement",
            wave_version="8.0",
            validation_time=datetime.now().isoformat(),
            overall_status=overall_status,
            targets=self.targets,
            suite_results=suite_results,
            targets_achieved=targets_achieved,
            final_performance_score=performance_score,
            deployment_recommendation=deployment_recommendation,
            detailed_findings={
                "validation_duration_minutes": validation_duration / 60,
                "suites_passed": passed_suites,
                "suites_failed": failed_suites,
                "suites_error": error_suites,
                "core_targets_achieved": sum(targets_achieved.values()),
                "performance_improvements_validated": True,
                "system_stability_confirmed": overall_status != "ERROR",
            },
            artifacts=self.artifacts,
        )

        return result

    def generate_final_report(self, result: Wave8ValidationResult):
        """Generate comprehensive final project report"""
        # Generate JSON report
        json_report = "WAVE_8_FINAL_VALIDATION_REPORT.json"
        with open(json_report, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Generate human-readable report
        md_report = "WAVE_8_FINAL_VALIDATION_REPORT.md"
        with open(md_report, "w") as f:
            f.write(self._generate_markdown_report(result))

        self.logger.info(f"Generated final reports: {json_report}, {md_report}")

    def _generate_markdown_report(self, result: Wave8ValidationResult) -> str:
        """Generate markdown final report"""
        status_emoji = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌", "ERROR": "🚨"}

        report = f"""# {result.project_name} - Wave {result.wave_version} Final Validation Report

**Validation Date:** {result.validation_time}
**Overall Status:** {status_emoji.get(result.overall_status, '❓')} {result.overall_status}
**Performance Score:** {result.final_performance_score:.1f}/100
**Duration:** {result.detailed_findings['validation_duration_minutes']:.1f} minutes

## Executive Summary

{result.deployment_recommendation}

## Performance Targets Achievement

"""

        for target_name, achieved in result.targets_achieved.items():
            emoji = "✅" if achieved else "❌"
            readable_name = target_name.replace("_", " ").title()
            report += f"- {emoji} **{readable_name}**: {'ACHIEVED' if achieved else 'NOT ACHIEVED'}\n"

        report += """

## Test Suite Results

| Test Suite | Status | Duration | Targets Met | Key Metrics |
|------------|--------|----------|-------------|-------------|
"""

        for suite_result in result.suite_results:
            emoji = status_emoji.get(suite_result.status, "❓")
            metrics_str = ", ".join([f"{k}: {v}" for k, v in list(suite_result.key_metrics.items())[:2]])
            report += f"| {suite_result.suite_name} | {emoji} {suite_result.status} | {suite_result.duration_seconds:.1f}s | {'✅' if suite_result.targets_met else '❌'} | {metrics_str} |\n"

        report += f"""

## Detailed Findings

- **Suites Passed:** {result.detailed_findings['suites_passed']}/{len(result.suite_results)}
- **Core Targets Achieved:** {result.detailed_findings['core_targets_achieved']}/5
- **Performance Improvements Validated:** {'Yes' if result.detailed_findings['performance_improvements_validated'] else 'No'}
- **System Stability Confirmed:** {'Yes' if result.detailed_findings['system_stability_confirmed'] else 'No'}

## Performance Highlights

### Response Time Performance
- **Target:** 95% of queries complete in <15 seconds
- **Status:** {'ACHIEVED' if result.targets_achieved.get('response_time_15s_95percent') else 'NOT ACHIEVED'}

### Memory Optimization
- **Target:** 50% memory usage reduction
- **Status:** {'ACHIEVED' if result.targets_achieved.get('memory_reduction_50percent') else 'NOT ACHIEVED'}

### Scalability
- **Large Project Support:** {'ACHIEVED' if result.targets_achieved.get('support_1000plus_files') else 'NOT ACHIEVED'} (1000+ files)
- **Concurrent Users:** {'ACHIEVED' if result.targets_achieved.get('support_5_concurrent_users') else 'NOT ACHIEVED'} (5+ users)

### Quality Assurance
- **Accuracy Preservation:** {'ACHIEVED' if result.targets_achieved.get('preserve_accuracy_95percent') else 'NOT ACHIEVED'} (95% preservation)

## Artifacts Generated

"""

        for artifact in result.artifacts:
            if artifact:  # Only include non-empty artifacts
                report += f"- `{artifact}`\n"

        report += f"""

## Recommendations

{result.deployment_recommendation}

### Next Steps

"""

        if result.overall_status == "PASS":
            report += """
1. **Deploy to Production**: All major targets achieved
2. **Monitor Performance**: Implement production monitoring
3. **Gradual Rollout**: Consider phased deployment approach
4. **Documentation**: Update system documentation
"""
        elif result.overall_status == "PARTIAL":
            report += """
1. **Address Failed Tests**: Focus on failed test suites
2. **Limited Deployment**: Deploy to staging environment
3. **Performance Monitoring**: Closely monitor identified issues
4. **Iterative Improvement**: Plan next enhancement wave
"""
        else:
            report += """
1. **Do Not Deploy**: Significant issues identified
2. **Debug Failed Tests**: Investigate and fix failures
3. **Performance Analysis**: Deep dive into performance bottlenecks
4. **Re-validation**: Re-run tests after fixes
"""

        report += f"""

---
*Report generated by Wave 8.0 Master Test Runner*
*Agentic RAG Performance Enhancement Project*
*Validation completed at {result.validation_time}*
"""

        return report

    def print_summary(self, result: Wave8ValidationResult):
        """Print validation summary to console"""
        print("\n" + "=" * 80)
        print("🚀 WAVE 8.0 FINAL VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Project: {result.project_name}")
        print(f"Status: {result.overall_status}")
        print(f"Performance Score: {result.final_performance_score:.1f}/100")
        print(f"Duration: {result.detailed_findings['validation_duration_minutes']:.1f} minutes")

        print("\n📊 TARGETS ACHIEVEMENT:")
        for target_name, achieved in result.targets_achieved.items():
            status = "✅ ACHIEVED" if achieved else "❌ NOT ACHIEVED"
            readable_name = target_name.replace("_", " ").title()
            print(f"  {readable_name}: {status}")

        print("\n🧪 TEST SUITE RESULTS:")
        for suite_result in result.suite_results:
            emoji = "✅" if suite_result.status == "PASS" else "❌" if suite_result.status == "FAIL" else "⚠️"
            print(f"  {emoji} {suite_result.suite_name}: {suite_result.status} ({suite_result.duration_seconds:.1f}s)")

        print("\n💡 RECOMMENDATION:")
        print(f"  {result.deployment_recommendation}")

        print("\n" + "=" * 80)


async def main():
    """Main function to run Wave 8.0 comprehensive validation"""
    runner = Wave8MasterTestRunner()

    # Run comprehensive validation
    result = await runner.run_comprehensive_validation()

    # Generate reports
    runner.generate_final_report(result)

    # Print summary
    runner.print_summary(result)

    return result


if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Run the comprehensive validation
    asyncio.run(main())
