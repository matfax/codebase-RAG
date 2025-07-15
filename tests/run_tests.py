#!/usr/bin/env python3
"""
Test runner for Wave 15.0 Testing Implementation
Comprehensive test execution with coverage reporting and validation.
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class TestRunner:
    """Comprehensive test runner for cache testing suite."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.coverage_threshold = 90

    def run_unit_tests(self, verbose: bool = False) -> dict[str, any]:
        """Run all unit tests (Wave 15.1)."""
        print("🧪 Running Unit Tests (Wave 15.1)")
        print("=" * 50)

        unit_test_files = [
            "test_cache_services.py",
            "test_cache_utilities.py",
            "test_encryption_security.py",
            "test_invalidation_logic.py",
            "test_configuration.py",
        ]

        results = {}

        for test_file in unit_test_files:
            print(f"\n📋 Running {test_file}...")
            result = self._run_pytest(test_file, markers=["unit"], verbose=verbose)
            results[test_file] = result

        return results

    def run_integration_tests(self, verbose: bool = False) -> dict[str, any]:
        """Run all integration tests (Wave 15.2)."""
        print("\n🔗 Running Integration Tests (Wave 15.2)")
        print("=" * 50)

        integration_test_files = [
            "test_service_integration.py",
            "test_mcp_tools_integration.py",
            "test_redis_connectivity.py",
            "test_invalidation_workflows.py",
            "test_performance_monitoring.py",
        ]

        results = {}

        for test_file in integration_test_files:
            print(f"\n📋 Running {test_file}...")
            result = self._run_pytest(test_file, markers=["integration"], verbose=verbose)
            results[test_file] = result

        return results

    def run_all_tests(self, verbose: bool = False) -> dict[str, any]:
        """Run complete test suite."""
        print("🚀 Running Complete Test Suite (Wave 15.0)")
        print("=" * 60)

        # Run unit tests first
        unit_results = self.run_unit_tests(verbose)

        # Run integration tests
        integration_results = self.run_integration_tests(verbose)

        # Combined results
        all_results = {"unit_tests": unit_results, "integration_tests": integration_results}

        return all_results

    def run_coverage_analysis(self) -> dict[str, any]:
        """Run comprehensive coverage analysis."""
        print("\n📊 Running Coverage Analysis")
        print("=" * 40)

        # Run tests with coverage
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir),
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            f"--cov-fail-under={self.coverage_threshold}",
            "--quiet",
        ]

        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

        coverage_info = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "threshold_met": result.returncode == 0,
        }

        if coverage_info["threshold_met"]:
            print(f"✅ Coverage threshold of {self.coverage_threshold}% met!")
        else:
            print(f"❌ Coverage below {self.coverage_threshold}% threshold")

        return coverage_info

    def run_performance_benchmarks(self) -> dict[str, any]:
        """Run performance benchmarks."""
        print("\n⚡ Running Performance Benchmarks")
        print("=" * 40)

        benchmark_cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "test_performance_monitoring.py"),
            "-m",
            "performance",
            "--benchmark-only",
            "--benchmark-json=benchmark_results.json",
        ]

        result = subprocess.run(benchmark_cmd, cwd=self.project_root, capture_output=True, text=True)

        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    def validate_test_environment(self) -> bool:
        """Validate test environment setup."""
        print("🔍 Validating Test Environment")
        print("=" * 35)

        checks = []

        # Check Python version
        python_version = sys.version_info
        python_ok = python_version >= (3, 10)
        checks.append(("Python >= 3.10", python_ok))

        # Check required packages
        required_packages = ["pytest", "pytest-asyncio", "pytest-cov", "redis", "cryptography", "watchdog"]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                checks.append((f"Package {package}", True))
            except ImportError:
                checks.append((f"Package {package}", False))

        # Check Redis availability (optional)
        redis_available = self._check_redis_connection()
        checks.append(("Redis connection", redis_available))

        # Print results
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            if not passed and "Redis" not in check_name:
                all_passed = False

        if not redis_available:
            print("⚠️  Redis not available - some integration tests will be skipped")

        return all_passed

    def generate_test_report(self, results: dict[str, any]) -> str:
        """Generate comprehensive test report."""
        report_lines = []
        report_lines.append("# Wave 15.0 Testing Implementation Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        if "unit_tests" in results:
            report_lines.append("## Unit Tests (Wave 15.1)")
            for test_file, result in results["unit_tests"].items():
                status = "✅ PASSED" if result["exit_code"] == 0 else "❌ FAILED"
                report_lines.append(f"- {test_file}: {status}")
                if result.get("test_count"):
                    total_tests += result["test_count"]
                    if result["exit_code"] == 0:
                        passed_tests += result["test_count"]
                    else:
                        failed_tests += result["test_count"]
            report_lines.append("")

        if "integration_tests" in results:
            report_lines.append("## Integration Tests (Wave 15.2)")
            for test_file, result in results["integration_tests"].items():
                status = "✅ PASSED" if result["exit_code"] == 0 else "❌ FAILED"
                report_lines.append(f"- {test_file}: {status}")
            report_lines.append("")

        # Coverage information
        if "coverage" in results:
            coverage = results["coverage"]
            if coverage["threshold_met"]:
                report_lines.append(f"## Coverage: ✅ {self.coverage_threshold}%+ achieved")
            else:
                report_lines.append(f"## Coverage: ❌ Below {self.coverage_threshold}% threshold")
            report_lines.append("")

        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- Total Tests: {total_tests}")
        report_lines.append(f"- Passed: {passed_tests}")
        report_lines.append(f"- Failed: {failed_tests}")
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            report_lines.append(f"- Pass Rate: {pass_rate:.1f}%")

        return "\n".join(report_lines)

    def _run_pytest(self, test_file: str, markers: list[str] | None = None, verbose: bool = False) -> dict[str, any]:
        """Run pytest on specific test file."""
        cmd = [sys.executable, "-m", "pytest", str(self.tests_dir / test_file)]

        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        # Add coverage for individual test files
        cmd.extend(["--cov=src", "--cov-report=term"])

        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

        # Parse test count from output
        test_count = self._parse_test_count(result.stdout)

        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "test_count": test_count}

    def _parse_test_count(self, pytest_output: str) -> int:
        """Parse test count from pytest output."""
        try:
            for line in pytest_output.split("\n"):
                if " passed" in line and "failed" not in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            return int(parts[i - 1])
        except:
            pass
        return 0

    def _check_redis_connection(self) -> bool:
        """Check if Redis is available for testing."""
        try:
            import redis

            client = redis.Redis(host="localhost", port=6379, db=15)
            client.ping()
            client.close()
            return True
        except:
            return False


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Wave 15.0 Testing Implementation Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests (Wave 15.1)")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests (Wave 15.2)")
    parser.add_argument("--coverage", action="store_true", help="Run coverage analysis")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--validate", action="store_true", help="Validate test environment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", type=str, help="Generate report file")

    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.parent
    runner = TestRunner(project_root)

    # Validate environment first
    if args.validate or not any([args.unit, args.integration, args.coverage, args.benchmark]):
        if not runner.validate_test_environment():
            print("\n❌ Environment validation failed")
            sys.exit(1)

    results = {}

    # Run specific test suites
    if args.unit:
        results.update(runner.run_unit_tests(args.verbose))
    elif args.integration:
        results.update(runner.run_integration_tests(args.verbose))
    elif not any([args.coverage, args.benchmark, args.validate]):
        # Run all tests if no specific suite selected
        all_results = runner.run_all_tests(args.verbose)
        results.update(all_results)

    # Run coverage analysis
    if args.coverage or not any([args.unit, args.integration, args.benchmark, args.validate]):
        coverage_results = runner.run_coverage_analysis()
        results["coverage"] = coverage_results

    # Run benchmarks
    if args.benchmark:
        benchmark_results = runner.run_performance_benchmarks()
        results["benchmarks"] = benchmark_results

    # Generate report
    if args.report or results:
        report_content = runner.generate_test_report(results)

        if args.report:
            report_file = Path(args.report)
            report_file.write_text(report_content)
            print(f"\n📄 Report saved to: {report_file}")
        else:
            print("\n" + "=" * 60)
            print(report_content)

    # Exit with appropriate code
    if results:
        # Check if any tests failed
        failed = any(
            result.get("exit_code", 0) != 0
            for test_results in results.values()
            if isinstance(test_results, dict)
            for result in (test_results.values() if isinstance(test_results, dict) else [test_results])
            if isinstance(result, dict)
        )

        if failed:
            print("\n❌ Some tests failed")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
            sys.exit(0)


if __name__ == "__main__":
    main()
