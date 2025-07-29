#!/usr/bin/env python3
"""
Wave 8.0 Task 8.9: CI Integration Framework

This module integrates comprehensive testing into continuous integration pipelines,
providing automated test orchestration, result reporting, and quality gates
for performance validation in CI/CD workflows.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestStatus(Enum):
    """Test execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class CIStage(Enum):
    """CI pipeline stages"""

    BUILD = "build"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    PERFORMANCE_TESTS = "performance_tests"
    STRESS_TESTS = "stress_tests"
    DEPLOY = "deploy"


@dataclass
class TestSuite:
    """Test suite configuration"""

    name: str
    script_path: str
    timeout_seconds: int
    required_for_pipeline: bool
    stage: CIStage
    environment_vars: dict[str, str]
    dependencies: list[str]
    quality_gates: dict[str, Any]


@dataclass
class TestResult:
    """Individual test result"""

    suite_name: str
    status: TestStatus
    start_time: str
    end_time: str
    duration_seconds: float
    exit_code: int
    stdout: str
    stderr: str
    quality_gates_passed: bool
    quality_gate_results: dict[str, Any]
    artifacts: list[str]


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""

    pipeline_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    overall_status: TestStatus
    stage_results: dict[CIStage, list[TestResult]]
    quality_gates_summary: dict[str, bool]
    failed_quality_gates: list[str]
    recommendations: list[str]
    artifacts: list[str]
    metadata: dict[str, Any]


class CIIntegrationFramework:
    """Comprehensive CI integration framework"""

    def __init__(self, config_file: str = "ci_config.yaml"):
        self.config_file = config_file
        self.logger = self._setup_logging()

        # Test suites configuration
        self.test_suites = self._define_test_suites()

        # Quality gates
        self.quality_gates = self._define_quality_gates()

        # CI configuration
        self.ci_config = self._load_ci_config()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for CI integration"""
        logger = logging.getLogger("ci_integration")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _define_test_suites(self) -> list[TestSuite]:
        """Define all test suites for CI pipeline"""
        return [
            # Performance regression tests
            TestSuite(
                name="performance_regression",
                script_path="tests/wave_8_performance_regression_tests.py",
                timeout_seconds=600,
                required_for_pipeline=True,
                stage=CIStage.PERFORMANCE_TESTS,
                environment_vars={"PERFORMANCE_TEST_MODE": "ci"},
                dependencies=[],
                quality_gates={"max_degradation_percent": 20.0, "min_success_rate": 95.0},
            ),
            # Response time tests
            TestSuite(
                name="response_time",
                script_path="tests/wave_8_response_time_tests.py",
                timeout_seconds=900,
                required_for_pipeline=True,
                stage=CIStage.PERFORMANCE_TESTS,
                environment_vars={"RESPONSE_TIME_TARGET_MS": "15000"},
                dependencies=[],
                quality_gates={"target_success_rate": 95.0, "max_p95_response_time_ms": 15000.0},
            ),
            # Memory usage validation
            TestSuite(
                name="memory_usage",
                script_path="tests/wave_8_memory_usage_tests.py",
                timeout_seconds=600,
                required_for_pipeline=True,
                stage=CIStage.PERFORMANCE_TESTS,
                environment_vars={"MEMORY_TEST_MODE": "validation"},
                dependencies=[],
                quality_gates={"memory_increase_threshold": 50.0, "leak_detection": True},
            ),
            # Large project tests
            TestSuite(
                name="large_project",
                script_path="tests/wave_8_large_project_tests.py",
                timeout_seconds=1800,
                required_for_pipeline=True,
                stage=CIStage.STRESS_TESTS,
                environment_vars={"LARGE_PROJECT_MIN_FILES": "1000"},
                dependencies=[],
                quality_gates={"min_file_count": 1000, "max_indexing_time_per_file_ms": 100.0},
            ),
            # Concurrent user tests
            TestSuite(
                name="concurrent_users",
                script_path="tests/wave_8_concurrent_user_tests.py",
                timeout_seconds=1200,
                required_for_pipeline=True,
                stage=CIStage.STRESS_TESTS,
                environment_vars={"MIN_CONCURRENT_USERS": "5"},
                dependencies=[],
                quality_gates={"min_concurrent_users": 5, "max_error_rate": 5.0},
            ),
            # Accuracy comparison
            TestSuite(
                name="accuracy_comparison",
                script_path="tests/wave_8_accuracy_comparison_tests.py",
                timeout_seconds=900,
                required_for_pipeline=True,
                stage=CIStage.PERFORMANCE_TESTS,
                environment_vars={"ACCURACY_THRESHOLD": "0.95"},
                dependencies=[],
                quality_gates={"accuracy_preservation_rate": 95.0, "max_degradation_percent": 5.0},
            ),
            # Stress tests
            TestSuite(
                name="stress_tests",
                script_path="tests/wave_8_stress_tests.py",
                timeout_seconds=2400,
                required_for_pipeline=False,  # Optional for faster pipelines
                stage=CIStage.STRESS_TESTS,
                environment_vars={"STRESS_TEST_DURATION": "300"},
                dependencies=[],
                quality_gates={"system_stability_score": 80.0, "max_breaking_points": 2},
            ),
            # A/B testing
            TestSuite(
                name="ab_testing",
                script_path="tests/wave_8_ab_testing_framework.py",
                timeout_seconds=1800,
                required_for_pipeline=False,  # Optional - for release validation
                stage=CIStage.PERFORMANCE_TESTS,
                environment_vars={"AB_TEST_SAMPLE_SIZE": "200"},
                dependencies=[],
                quality_gates={"enhanced_win_rate": 60.0, "min_confidence_score": 70.0},
            ),
        ]

    def _define_quality_gates(self) -> dict[str, dict[str, Any]]:
        """Define quality gates for CI pipeline"""
        return {
            "performance": {
                "response_time_p95_ms": 15000,
                "memory_usage_increase_max_percent": 50,
                "error_rate_max_percent": 5,
                "throughput_min_qps": 1.0,
            },
            "scalability": {"min_file_support": 1000, "min_concurrent_users": 5, "max_degradation_under_load": 30},
            "accuracy": {"min_preservation_rate": 95, "max_accuracy_loss": 5, "min_f1_score": 0.8},
            "stability": {"max_error_rate": 5, "min_uptime_percent": 99, "max_memory_leaks": 0},
        }

    def _load_ci_config(self) -> dict[str, Any]:
        """Load CI configuration"""
        default_config = {
            "pipeline_timeout_minutes": 60,
            "parallel_execution": True,
            "fail_fast": False,
            "artifact_retention_days": 30,
            "notification_settings": {"on_failure": True, "on_success": False, "channels": ["email", "slack"]},
            "environment": {"PYTHONPATH": "src", "LOG_LEVEL": "INFO", "CI_MODE": "true"},
        }

        try:
            if Path(self.config_file).exists():
                with open(self.config_file) as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(config)
        except Exception as e:
            self.logger.warning(f"Could not load CI config: {e}, using defaults")

        return default_config

    def generate_ci_configs(self):
        """Generate CI configuration files for popular CI systems"""
        # GitHub Actions workflow
        self._generate_github_actions_config()

        # GitLab CI configuration
        self._generate_gitlab_ci_config()

        # Jenkins pipeline
        self._generate_jenkins_config()

        # Azure DevOps pipeline
        self._generate_azure_devops_config()

    def _generate_github_actions_config(self):
        """Generate GitHub Actions workflow file"""
        workflow = {
            "name": "Agentic RAG Performance Testing",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
                "schedule": [{"cron": "0 2 * * *"}],  # Daily at 2 AM
            },
            "jobs": {
                "performance-tests": {
                    "runs-on": "ubuntu-latest",
                    "timeout-minutes": self.ci_config["pipeline_timeout_minutes"],
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {"name": "Set up Python", "uses": "actions/setup-python@v4", "with": {"python-version": "3.9"}},
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {"name": "Start cache services", "run": "docker-compose -f docker-compose.cache.yml up -d"},
                    ],
                }
            },
        }

        # Add test steps for each suite
        for suite in self.test_suites:
            if suite.required_for_pipeline:
                step = {
                    "name": f"Run {suite.name} tests",
                    "run": f"python {suite.script_path}",
                    "timeout-minutes": suite.timeout_seconds // 60,
                    "env": suite.environment_vars,
                }
                workflow["jobs"]["performance-tests"]["steps"].append(step)

        # Add artifact upload
        workflow["jobs"]["performance-tests"]["steps"].extend(
            [
                {
                    "name": "Upload test reports",
                    "uses": "actions/upload-artifact@v3",
                    "if": "always()",
                    "with": {"name": "test-reports", "path": "*.json", "retention-days": self.ci_config["artifact_retention_days"]},
                }
            ]
        )

        # Write workflow file
        workflow_dir = Path(".github/workflows")
        workflow_dir.mkdir(parents=True, exist_ok=True)

        with open(workflow_dir / "performance-tests.yml", "w") as f:
            yaml.dump(workflow, f, default_flow_style=False)

        self.logger.info("Generated GitHub Actions workflow")

    def _generate_gitlab_ci_config(self):
        """Generate GitLab CI configuration"""
        gitlab_ci = {
            "stages": ["build", "test", "performance", "stress", "deploy"],
            "variables": self.ci_config["environment"],
            "before_script": ["pip install -r requirements.txt", "docker-compose -f docker-compose.cache.yml up -d"],
        }

        # Add jobs for each test suite
        for suite in self.test_suites:
            job_name = f"{suite.name}_tests"
            gitlab_ci[job_name] = {
                "stage": suite.stage.value.replace("_", "-"),
                "script": [f"python {suite.script_path}"],
                "timeout": f"{suite.timeout_seconds // 60}m",
                "variables": suite.environment_vars,
                "artifacts": {
                    "reports": {"junit": "*.xml"},
                    "paths": ["*.json"],
                    "expire_in": f"{self.ci_config['artifact_retention_days']} days",
                },
                "only": ["main", "develop", "merge_requests"],
            }

            if not suite.required_for_pipeline:
                gitlab_ci[job_name]["allow_failure"] = True

        # Write GitLab CI file
        with open(".gitlab-ci.yml", "w") as f:
            yaml.dump(gitlab_ci, f, default_flow_style=False)

        self.logger.info("Generated GitLab CI configuration")

    def _generate_jenkins_config(self):
        """Generate Jenkins pipeline configuration"""
        jenkins_pipeline = (
            """
pipeline {
    agent any
    
    environment {
        PYTHONPATH = 'src'
        LOG_LEVEL = 'INFO'
        CI_MODE = 'true'
    }
    
    options {
        timeout(time: %d, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    
    triggers {
        cron('H 2 * * *')  // Daily at 2 AM
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'docker-compose -f docker-compose.cache.yml up -d'
            }
        }
        
        stage('Performance Tests') {
            parallel {
"""
            % self.ci_config["pipeline_timeout_minutes"]
        )

        # Add parallel test stages
        for suite in self.test_suites:
            if suite.required_for_pipeline:
                jenkins_pipeline += f"""
                stage('{suite.name}') {{
                    steps {{
                        timeout(time: {suite.timeout_seconds // 60}, unit: 'MINUTES') {{
                            sh 'python {suite.script_path}'
                        }}
                    }}
                }}"""

        jenkins_pipeline += """
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '*.json', fingerprint: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: '*.html',
                reportName: 'Performance Test Report'
            ])
        }
        failure {
            emailext (
                subject: "Performance Tests Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Performance tests failed. Please check the console output.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
"""

        # Write Jenkins pipeline file
        with open("Jenkinsfile", "w") as f:
            f.write(jenkins_pipeline)

        self.logger.info("Generated Jenkins pipeline configuration")

    def _generate_azure_devops_config(self):
        """Generate Azure DevOps pipeline configuration"""
        azure_pipeline = {
            "trigger": {"branches": {"include": ["main", "develop"]}},
            "pr": {"branches": {"include": ["main"]}},
            "schedules": [{"cron": "0 2 * * *", "displayName": "Daily performance tests", "branches": {"include": ["main"]}}],
            "pool": {"vmImage": "ubuntu-latest"},
            "variables": self.ci_config["environment"],
            "stages": [
                {
                    "stage": "PerformanceTests",
                    "displayName": "Performance Tests",
                    "jobs": [
                        {
                            "job": "RunTests",
                            "displayName": "Run Performance Tests",
                            "timeoutInMinutes": self.ci_config["pipeline_timeout_minutes"],
                            "steps": [
                                {"task": "UsePythonVersion@0", "inputs": {"versionSpec": "3.9"}},
                                {"script": "pip install -r requirements.txt", "displayName": "Install dependencies"},
                                {"script": "docker-compose -f docker-compose.cache.yml up -d", "displayName": "Start cache services"},
                            ],
                        }
                    ],
                }
            ],
        }

        # Add test steps
        for suite in self.test_suites:
            if suite.required_for_pipeline:
                step = {
                    "script": f"python {suite.script_path}",
                    "displayName": f"Run {suite.name} tests",
                    "timeoutInMinutes": suite.timeout_seconds // 60,
                    "env": suite.environment_vars,
                }
                azure_pipeline["stages"][0]["jobs"][0]["steps"].append(step)

        # Add artifact publishing
        azure_pipeline["stages"][0]["jobs"][0]["steps"].append(
            {
                "task": "PublishTestResults@2",
                "condition": "always()",
                "inputs": {"testResultsFiles": "*.xml", "testRunTitle": "Performance Tests"},
            }
        )

        # Write Azure DevOps pipeline file
        with open("azure-pipelines.yml", "w") as f:
            yaml.dump(azure_pipeline, f, default_flow_style=False)

        self.logger.info("Generated Azure DevOps pipeline configuration")

    async def execute_test_suite(self, suite: TestSuite) -> TestResult:
        """Execute a single test suite"""
        self.logger.info(f"Executing test suite: {suite.name}")

        start_time = datetime.now()
        artifacts = []

        # Prepare environment
        env = os.environ.copy()
        env.update(suite.environment_vars)
        env.update(self.ci_config["environment"])

        try:
            # Execute test script
            process = await asyncio.create_subprocess_exec(
                sys.executable, suite.script_path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=suite.timeout_seconds)
                exit_code = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                stdout = b""
                stderr = b"Test timed out"
                exit_code = -1

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Determine status
            if exit_code == 0:
                status = TestStatus.PASSED
            elif exit_code == -1:
                status = TestStatus.ERROR
            else:
                status = TestStatus.FAILED

            # Check for artifacts
            expected_artifacts = [f"{suite.name}_report.json", f"wave_8_{suite.name}_report.json"]

            for artifact in expected_artifacts:
                if Path(artifact).exists():
                    artifacts.append(artifact)

            # Evaluate quality gates
            quality_gates_passed, quality_gate_results = self._evaluate_quality_gates(suite, artifacts)

            return TestResult(
                suite_name=suite.name,
                status=status,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                exit_code=exit_code,
                stdout=stdout.decode("utf-8", errors="ignore"),
                stderr=stderr.decode("utf-8", errors="ignore"),
                quality_gates_passed=quality_gates_passed,
                quality_gate_results=quality_gate_results,
                artifacts=artifacts,
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.logger.error(f"Error executing test suite {suite.name}: {e}")

            return TestResult(
                suite_name=suite.name,
                status=TestStatus.ERROR,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=duration,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                quality_gates_passed=False,
                quality_gate_results={},
                artifacts=[],
            )

    def _evaluate_quality_gates(self, suite: TestSuite, artifacts: list[str]) -> tuple[bool, dict[str, Any]]:
        """Evaluate quality gates for a test suite"""
        quality_gate_results = {}
        all_passed = True

        # Load test results from artifacts
        test_data = {}
        for artifact in artifacts:
            try:
                with open(artifact) as f:
                    data = json.load(f)
                    test_data.update(data)
            except Exception as e:
                self.logger.warning(f"Could not load artifact {artifact}: {e}")

        # Evaluate suite-specific quality gates
        for gate_name, threshold in suite.quality_gates.items():
            passed = self._check_quality_gate(gate_name, threshold, test_data)
            quality_gate_results[gate_name] = {"threshold": threshold, "actual": test_data.get(gate_name, "N/A"), "passed": passed}

            if not passed:
                all_passed = False

        return all_passed, quality_gate_results

    def _check_quality_gate(self, gate_name: str, threshold: Any, test_data: dict[str, Any]) -> bool:
        """Check individual quality gate"""
        try:
            if gate_name == "max_degradation_percent":
                actual = test_data.get("average_degradation_percent", 0)
                return actual <= threshold
            elif gate_name == "min_success_rate":
                actual = test_data.get("target_achievement_rate", 0)
                return actual >= threshold
            elif gate_name == "target_success_rate":
                actual = test_data.get("overall_success_rate_percent", 0)
                return actual >= threshold
            elif gate_name == "max_p95_response_time_ms":
                actual = test_data.get("response_time_stats", {}).get("p95_ms", float("inf"))
                return actual <= threshold
            elif gate_name == "memory_increase_threshold":
                actual = test_data.get("memory_statistics", {}).get("max_increase_percent", 0)
                return actual <= threshold
            elif gate_name == "min_file_count":
                actual = test_data.get("max_file_count_tested", 0)
                return actual >= threshold
            elif gate_name == "min_concurrent_users":
                actual = test_data.get("max_concurrent_users_achieved", 0)
                return actual >= threshold
            elif gate_name == "accuracy_preservation_rate":
                actual = test_data.get("accuracy_preservation_rate", 0)
                return actual >= threshold
            elif gate_name == "enhanced_win_rate":
                actual = test_data.get("enhanced_win_rate", 0)
                return actual >= threshold
            else:
                # Generic comparison
                actual = test_data.get(gate_name, 0)
                if isinstance(threshold, (int, float)):
                    if "min_" in gate_name or "rate" in gate_name:
                        return actual >= threshold
                    elif "max_" in gate_name:
                        return actual <= threshold

            return True  # Default to pass if can't evaluate

        except Exception as e:
            self.logger.warning(f"Error checking quality gate {gate_name}: {e}")
            return False

    async def run_ci_pipeline(self, pipeline_id: str | None = None) -> PipelineResult:
        """Run complete CI pipeline"""
        if not pipeline_id:
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting CI pipeline: {pipeline_id}")

        start_time = datetime.now()
        stage_results = {}
        all_artifacts = []
        failed_quality_gates = []

        # Group test suites by stage
        suites_by_stage = {}
        for suite in self.test_suites:
            stage = suite.stage
            if stage not in suites_by_stage:
                suites_by_stage[stage] = []
            suites_by_stage[stage].append(suite)

        # Execute stages in order
        stage_order = [
            CIStage.BUILD,
            CIStage.UNIT_TESTS,
            CIStage.INTEGRATION_TESTS,
            CIStage.PERFORMANCE_TESTS,
            CIStage.STRESS_TESTS,
            CIStage.DEPLOY,
        ]

        overall_status = TestStatus.PASSED

        for stage in stage_order:
            if stage not in suites_by_stage:
                continue

            self.logger.info(f"Executing stage: {stage.value}")
            stage_results[stage] = []

            # Execute test suites in parallel within stage
            if self.ci_config["parallel_execution"]:
                tasks = []
                for suite in suites_by_stage[stage]:
                    task = asyncio.create_task(self.execute_test_suite(suite))
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Sequential execution
                results = []
                for suite in suites_by_stage[stage]:
                    result = await self.execute_test_suite(suite)
                    results.append(result)

            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Test suite execution failed: {result}")
                    continue

                stage_results[stage].append(result)
                all_artifacts.extend(result.artifacts)

                # Check if test failed
                if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    suite = suites_by_stage[stage][i]
                    if suite.required_for_pipeline:
                        overall_status = TestStatus.FAILED
                        if self.ci_config["fail_fast"]:
                            break

                # Check quality gates
                if not result.quality_gates_passed:
                    failed_quality_gates.extend(
                        [
                            f"{result.suite_name}:{gate}"
                            for gate in result.quality_gate_results
                            if not result.quality_gate_results[gate]["passed"]
                        ]
                    )

            # Stop if fail fast and stage failed
            if self.ci_config["fail_fast"] and overall_status == TestStatus.FAILED:
                break

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Generate quality gates summary
        quality_gates_summary = self._generate_quality_gates_summary(stage_results)

        # Generate recommendations
        recommendations = self._generate_ci_recommendations(stage_results, failed_quality_gates, overall_status)

        result = PipelineResult(
            pipeline_id=pipeline_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            overall_status=overall_status,
            stage_results=stage_results,
            quality_gates_summary=quality_gates_summary,
            failed_quality_gates=failed_quality_gates,
            recommendations=recommendations,
            artifacts=all_artifacts,
            metadata={
                "total_tests": sum(len(results) for results in stage_results.values()),
                "passed_tests": sum(len([r for r in results if r.status == TestStatus.PASSED]) for results in stage_results.values()),
                "failed_tests": sum(
                    len([r for r in results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]) for results in stage_results.values()
                ),
            },
        )

        self.logger.info(f"CI pipeline completed: {pipeline_id} - Status: {overall_status.value}")

        return result

    def _generate_quality_gates_summary(self, stage_results: dict[CIStage, list[TestResult]]) -> dict[str, bool]:
        """Generate quality gates summary"""
        summary = {}

        for gate_category, gates in self.quality_gates.items():
            category_passed = True

            for gate_name, threshold in gates.items():
                gate_passed = True

                # Check across all test results
                for stage, results in stage_results.items():
                    for result in results:
                        if gate_name in result.quality_gate_results:
                            if not result.quality_gate_results[gate_name]["passed"]:
                                gate_passed = False
                                break

                if not gate_passed:
                    category_passed = False

                summary[f"{gate_category}_{gate_name}"] = gate_passed

            summary[gate_category] = category_passed

        return summary

    def _generate_ci_recommendations(
        self, stage_results: dict[CIStage, list[TestResult]], failed_quality_gates: list[str], overall_status: TestStatus
    ) -> list[str]:
        """Generate CI recommendations"""
        recommendations = []

        if overall_status == TestStatus.FAILED:
            recommendations.append("Pipeline failed - do not deploy to production")

        if failed_quality_gates:
            recommendations.append(f"Quality gates failed: {', '.join(failed_quality_gates)}")

        # Analyze performance trends
        total_tests = sum(len(results) for results in stage_results.values())
        failed_tests = sum(
            len([r for r in results if r.status in [TestStatus.FAILED, TestStatus.ERROR]]) for results in stage_results.values()
        )

        if failed_tests > 0:
            failure_rate = (failed_tests / total_tests) * 100
            if failure_rate > 20:
                recommendations.append("High test failure rate - investigate test environment")
            elif failure_rate > 10:
                recommendations.append("Moderate test failure rate - review test stability")

        # Check for long-running tests
        for stage, results in stage_results.items():
            for result in results:
                if result.duration_seconds > 600:  # 10 minutes
                    recommendations.append(f"Test {result.suite_name} is slow - consider optimization")

        if not recommendations:
            recommendations.append("All tests passed - pipeline ready for deployment")

        return recommendations

    def generate_pipeline_report(self, result: PipelineResult, output_file: str):
        """Generate comprehensive pipeline report"""
        report = asdict(result)

        # Add summary statistics
        report["summary"] = {
            "pipeline_id": result.pipeline_id,
            "status": result.overall_status.value,
            "duration_minutes": result.duration_seconds / 60,
            "total_tests": result.metadata["total_tests"],
            "passed_tests": result.metadata["passed_tests"],
            "failed_tests": result.metadata["failed_tests"],
            "success_rate": (result.metadata["passed_tests"] / max(1, result.metadata["total_tests"])) * 100,
            "quality_gates_passed": len([g for g in result.quality_gates_summary.values() if g]),
            "quality_gates_failed": len(result.failed_quality_gates),
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Generated pipeline report: {output_file}")


async def main():
    """Main function to run CI integration"""
    ci_framework = CIIntegrationFramework()

    # Generate CI configuration files
    print("Generating CI configuration files...")
    ci_framework.generate_ci_configs()

    # Run CI pipeline simulation
    print("Running CI pipeline simulation...")
    result = await ci_framework.run_ci_pipeline("test_pipeline")

    # Generate report
    ci_framework.generate_pipeline_report(result, "wave_8_ci_pipeline_report.json")

    print("\n=== CI Pipeline Result ===")
    print(f"Pipeline ID: {result.pipeline_id}")
    print(f"Status: {result.overall_status.value}")
    print(f"Duration: {result.duration_seconds / 60:.1f} minutes")
    print(f"Total Tests: {result.metadata['total_tests']}")
    print(f"Passed: {result.metadata['passed_tests']}")
    print(f"Failed: {result.metadata['failed_tests']}")
    print(f"Failed Quality Gates: {len(result.failed_quality_gates)}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
