"""
Tree-sitter Parser Diagnostics Tool

This module provides comprehensive diagnostic capabilities for verifying
Tree-sitter parser health, installation status, and functionality.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import tree_sitter
    from tree_sitter import Language, Node, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    tree_sitter = None
    Language = None
    Parser = None
    Node = None

from utils.tree_sitter_manager import TreeSitterManager


@dataclass
class ParseTestResult:
    """Result of a parser test."""

    language: str
    success: bool
    error_message: str | None = None
    parse_time_ms: float = 0.0
    node_count: int = 0
    error_count: int = 0
    sample_code: str = ""
    parsed_content: str | None = None


@dataclass
class ParserHealthReport:
    """Comprehensive parser health report."""

    # System status
    tree_sitter_available: bool = False
    tree_sitter_version: str = ""

    # Installation status
    total_languages: int = 0
    installed_languages: int = 0
    failed_languages: list[str] = field(default_factory=list)

    # Functionality tests
    parsing_tests: dict[str, ParseTestResult] = field(default_factory=dict)

    # Performance metrics
    average_parse_time_ms: float = 0.0
    fastest_language: str = ""
    slowest_language: str = ""

    # Issues and recommendations
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    test_duration_seconds: float = 0.0

    def overall_health_score(self) -> float:
        """Calculate overall parser health score (0-100)."""
        if not self.tree_sitter_available:
            return 0.0

        if self.total_languages == 0:
            return 50.0  # Tree-sitter available but no languages

        # Base score from installation success rate
        install_score = (self.installed_languages / self.total_languages) * 60

        # Parsing functionality score
        if self.parsing_tests:
            successful_tests = sum(1 for test in self.parsing_tests.values() if test.success)
            parse_score = (successful_tests / len(self.parsing_tests)) * 30
        else:
            parse_score = 0.0

        # Performance score (penalize if average parse time is too high)
        performance_score = 10.0
        if self.average_parse_time_ms > 1000:  # > 1 second is concerning
            performance_score = max(0.0, 10.0 - (self.average_parse_time_ms - 1000) / 100)

        total_score = install_score + parse_score + performance_score

        # Apply penalties for critical issues
        penalty = min(len(self.critical_issues) * 15, 50)  # Max 50 point penalty

        return max(0.0, min(100.0, total_score - penalty))

    def health_status(self) -> str:
        """Get human-readable health status."""
        score = self.overall_health_score()

        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 30:
            return "Poor"
        else:
            return "Critical"


class ParserDiagnostics:
    """
    Comprehensive diagnostic tool for Tree-sitter parser health verification.
    """

    def __init__(self):
        """Initialize the diagnostics tool."""
        self.logger = logging.getLogger(__name__)
        self.tree_sitter_manager = None

        # Sample code for testing each language
        self.test_samples = {
            "python": '''
def hello_world(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

class TestClass:
    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        return self.value

# Test variable and import
import os
test_var = 42
''',
            "javascript": """
function helloWorld(name) {
    // Return a greeting message
    return `Hello, ${name}!`;
}

class TestClass {
    constructor(value) {
        this.value = value;
    }

    getValue() {
        return this.value;
    }
}

// Test variables and imports
import { someFunction } from './module';
const testVar = 42;
export { helloWorld };
""",
            "typescript": """
interface User {
    name: string;
    age: number;
}

function helloWorld(name: string): string {
    // Return a greeting message
    return `Hello, ${name}!`;
}

class TestClass<T> {
    private value: T;

    constructor(value: T) {
        this.value = value;
    }

    getValue(): T {
        return this.value;
    }
}

// Test variables and imports
import { SomeType } from './types';
const testVar: number = 42;
export { helloWorld, TestClass };
""",
            "java": """
package com.example;

import java.util.List;
import java.util.ArrayList;

public class TestClass {
    private String name;
    private int value;

    public TestClass(String name, int value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public static String helloWorld(String name) {
        // Return a greeting message
        return "Hello, " + name + "!";
    }
}
""",
            "go": """
package main

import (
    "fmt"
    "strings"
)

type TestStruct struct {
    Name  string
    Value int
}

func (t *TestStruct) GetName() string {
    return t.Name
}

func (t *TestStruct) SetValue(value int) {
    t.Value = value
}

func helloWorld(name string) string {
    // Return a greeting message
    return fmt.Sprintf("Hello, %s!", name)
}

func main() {
    test := &TestStruct{Name: "test", Value: 42}
    fmt.Println(helloWorld(test.GetName()))
}
""",
            "rust": """
use std::collections::HashMap;

struct TestStruct {
    name: String,
    value: i32,
}

impl TestStruct {
    fn new(name: String, value: i32) -> Self {
        TestStruct { name, value }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn set_value(&mut self, value: i32) {
        self.value = value;
    }
}

fn hello_world(name: &str) -> String {
    // Return a greeting message
    format!("Hello, {}!", name)
}

fn main() {
    let mut test = TestStruct::new("test".to_string(), 42);
    println!("{}", hello_world(test.get_name()));
}
""",
            "cpp": """
#include <iostream>
#include <string>
#include <vector>
#include <memory>

namespace test {
    class TestClass {
    private:
        std::string name;
        int value;

    public:
        TestClass(const std::string& name, int value)
            : name(name), value(value) {}

        ~TestClass() = default;

        const std::string& getName() const {
            return name;
        }

        void setValue(int newValue) {
            value = newValue;
        }

        template<typename T>
        T getValue() const {
            return static_cast<T>(value);
        }
    };

    std::string helloWorld(const std::string& name) {
        // Return a greeting message
        return "Hello, " + name + "!";
    }
}

int main() {
    auto test = std::make_unique<test::TestClass>("test", 42);
    std::cout << test::helloWorld(test->getName()) << std::endl;
    return 0;
}
""",
        }

    def run_comprehensive_diagnostics(self) -> ParserHealthReport:
        """
        Run comprehensive parser diagnostics.

        Returns:
            Complete health report with all diagnostic results
        """
        start_time = time.time()
        self.logger.info("Starting comprehensive Tree-sitter parser diagnostics")

        report = ParserHealthReport()

        # Check Tree-sitter availability
        self._check_tree_sitter_availability(report)

        if not report.tree_sitter_available:
            report.critical_issues.append("Tree-sitter is not available or not properly installed")
            report.test_duration_seconds = time.time() - start_time
            return report

        # Initialize Tree-sitter manager
        try:
            self.tree_sitter_manager = TreeSitterManager()
            summary = self.tree_sitter_manager.get_initialization_summary()

            report.total_languages = summary["total_languages"]
            report.installed_languages = len(summary["successful_languages"])
            report.failed_languages = summary["failed_languages"]

        except Exception as e:
            report.critical_issues.append(f"Failed to initialize TreeSitterManager: {str(e)}")
            report.test_duration_seconds = time.time() - start_time
            return report

        # Run parser functionality tests
        self._run_parsing_tests(report)

        # Calculate performance metrics
        self._calculate_performance_metrics(report)

        # Generate health assessment
        self._assess_parser_health(report)

        # Generate recommendations
        self._generate_recommendations(report)

        report.test_duration_seconds = time.time() - start_time

        self.logger.info(
            f"Diagnostics completed in {report.test_duration_seconds:.2f}s - "
            f"Health Score: {report.overall_health_score():.1f}/100 ({report.health_status()})"
        )

        return report

    def _check_tree_sitter_availability(self, report: ParserHealthReport) -> None:
        """Check if Tree-sitter is available and get version info."""
        report.tree_sitter_available = TREE_SITTER_AVAILABLE

        if TREE_SITTER_AVAILABLE:
            try:
                # Try to get Tree-sitter version
                if hasattr(tree_sitter, "__version__"):
                    report.tree_sitter_version = tree_sitter.__version__
                else:
                    # Fallback method to get version
                    import pkg_resources

                    try:
                        version = pkg_resources.get_distribution("tree-sitter").version
                        report.tree_sitter_version = version
                    except Exception:
                        report.tree_sitter_version = "unknown"
            except Exception as e:
                report.warnings.append(f"Could not determine Tree-sitter version: {str(e)}")
                report.tree_sitter_version = "unknown"
        else:
            report.critical_issues.append("Tree-sitter library is not installed")

    def _run_parsing_tests(self, report: ParserHealthReport) -> None:
        """Run parsing tests for all available languages."""
        if not self.tree_sitter_manager:
            return

        supported_languages = self.tree_sitter_manager.get_supported_languages()

        for language in supported_languages:
            self.logger.info(f"Testing parser for {language}")
            test_result = self._test_language_parser(language)
            report.parsing_tests[language] = test_result

            if not test_result.success:
                report.warnings.append(f"Parser test failed for {language}: {test_result.error_message}")

    def _test_language_parser(self, language: str) -> ParseTestResult:
        """Test parsing functionality for a specific language."""
        test_result = ParseTestResult(language=language, success=False)

        try:
            # Get sample code for the language
            sample_code = self.test_samples.get(language, "// Test code")
            test_result.sample_code = sample_code

            # Get parser for the language
            parser = self.tree_sitter_manager.get_parser(language)
            if not parser:
                test_result.error_message = "Parser not available"
                return test_result

            # Perform parsing test
            start_time = time.time()
            tree = parser.parse(bytes(sample_code, "utf8"))
            test_result.parse_time_ms = (time.time() - start_time) * 1000

            if not tree or not tree.root_node:
                test_result.error_message = "Parsing returned no tree or root node"
                return test_result

            # Count nodes and errors
            test_result.node_count = self._count_nodes(tree.root_node)
            test_result.error_count = self._count_errors(tree.root_node)

            # Extract parsed content sample
            if tree.root_node.children:
                first_child = tree.root_node.children[0]
                if first_child.text:
                    test_result.parsed_content = first_child.text.decode("utf8")[:100] + "..."

            # Test passes if we got a tree with nodes and minimal errors
            test_result.success = test_result.node_count > 0 and test_result.error_count <= test_result.node_count * 0.1  # < 10% error rate

            if not test_result.success and test_result.error_count > 0:
                test_result.error_message = f"High error rate: {test_result.error_count} errors in {test_result.node_count} nodes"

        except Exception as e:
            test_result.error_message = str(e)
            test_result.success = False

        return test_result

    def _count_nodes(self, node: Node) -> int:
        """Recursively count all nodes in the AST."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _count_errors(self, node: Node) -> int:
        """Recursively count error nodes in the AST."""
        count = 1 if node.type == "ERROR" else 0
        for child in node.children:
            count += self._count_errors(child)
        return count

    def _calculate_performance_metrics(self, report: ParserHealthReport) -> None:
        """Calculate performance metrics from test results."""
        if not report.parsing_tests:
            return

        parse_times = [test.parse_time_ms for test in report.parsing_tests.values() if test.success]

        if parse_times:
            report.average_parse_time_ms = sum(parse_times) / len(parse_times)

            # Find fastest and slowest languages
            fastest_time = min(parse_times)
            slowest_time = max(parse_times)

            for lang, test in report.parsing_tests.items():
                if test.success and test.parse_time_ms == fastest_time:
                    report.fastest_language = lang
                if test.success and test.parse_time_ms == slowest_time:
                    report.slowest_language = lang

    def _assess_parser_health(self, report: ParserHealthReport) -> None:
        """Assess overall parser health and identify issues."""
        # Check installation success rate
        if report.total_languages > 0:
            install_success_rate = (report.installed_languages / report.total_languages) * 100
            if install_success_rate < 80:
                report.warnings.append(f"Low installation success rate: {install_success_rate:.1f}%")
            elif install_success_rate < 60:
                report.critical_issues.append(f"Critical installation failure rate: {install_success_rate:.1f}%")

        # Check parsing success rate
        if report.parsing_tests:
            successful_tests = sum(1 for test in report.parsing_tests.values() if test.success)
            parse_success_rate = (successful_tests / len(report.parsing_tests)) * 100

            if parse_success_rate < 80:
                report.warnings.append(f"Low parsing success rate: {parse_success_rate:.1f}%")
            elif parse_success_rate < 50:
                report.critical_issues.append(f"Critical parsing failure rate: {parse_success_rate:.1f}%")

        # Check performance issues
        if report.average_parse_time_ms > 2000:  # > 2 seconds
            report.critical_issues.append(f"Very slow parsing performance: {report.average_parse_time_ms:.0f}ms average")
        elif report.average_parse_time_ms > 1000:  # > 1 second
            report.warnings.append(f"Slow parsing performance: {report.average_parse_time_ms:.0f}ms average")

    def _generate_recommendations(self, report: ParserHealthReport) -> None:
        """Generate actionable recommendations based on diagnostic results."""
        recommendations = []

        # Installation recommendations
        if report.failed_languages:
            recommendations.append(
                f"Reinstall failed language parsers: {', '.join(report.failed_languages)}. "
                f"Try running: poetry install or pip install tree-sitter-<language>"
            )

        # Performance recommendations
        if report.average_parse_time_ms > 1000:
            recommendations.append(
                "Consider optimizing parsing performance by reducing file sizes or " "using incremental parsing for large files"
            )

        # Functionality recommendations
        failing_tests = [lang for lang, test in report.parsing_tests.items() if not test.success]
        if failing_tests:
            recommendations.append(
                f"Investigate parsing issues for: {', '.join(failing_tests)}. " f"Check sample code compatibility and parser versions"
            )

        # General recommendations
        if report.overall_health_score() < 75:
            recommendations.append("Run 'poetry install' to ensure all Tree-sitter dependencies are properly installed")

        if not report.tree_sitter_available:
            recommendations.append("Install Tree-sitter: pip install tree-sitter or poetry add tree-sitter")

        report.recommendations = recommendations

    def run_quick_health_check(self) -> dict[str, Any]:
        """
        Run a quick health check for immediate feedback.

        Returns:
            Dictionary with basic health status
        """
        self.logger.info("Running quick parser health check")

        result = {
            "tree_sitter_available": TREE_SITTER_AVAILABLE,
            "timestamp": datetime.now().isoformat(),
        }

        if not TREE_SITTER_AVAILABLE:
            result["status"] = "critical"
            result["message"] = "Tree-sitter is not available"
            result["recommendation"] = "Install Tree-sitter: pip install tree-sitter"
            return result

        try:
            # Quick initialization test
            manager = TreeSitterManager()
            summary = manager.get_initialization_summary()

            result["total_languages"] = summary["total_languages"]
            result["successful_languages"] = len(summary["successful_languages"])
            result["failed_languages"] = summary["failed_languages"]

            success_rate = (len(summary["successful_languages"]) / summary["total_languages"]) * 100

            if success_rate >= 90:
                result["status"] = "healthy"
                result["message"] = f"All parsers working well ({success_rate:.1f}% success rate)"
            elif success_rate >= 70:
                result["status"] = "warning"
                result["message"] = f"Most parsers working ({success_rate:.1f}% success rate)"
            else:
                result["status"] = "critical"
                result["message"] = f"Many parser failures ({success_rate:.1f}% success rate)"
                result["recommendation"] = "Run comprehensive diagnostics for detailed analysis"

        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Health check failed: {str(e)}"
            result["recommendation"] = "Check Tree-sitter installation and dependencies"

        return result

    def test_specific_language(self, language: str, sample_code: str | None = None) -> ParseTestResult:
        """
        Test parsing for a specific language with custom code.

        Args:
            language: Language to test
            sample_code: Optional custom sample code (uses default if not provided)

        Returns:
            Detailed test result for the language
        """
        self.logger.info(f"Testing specific language: {language}")

        if not TREE_SITTER_AVAILABLE:
            return ParseTestResult(
                language=language,
                success=False,
                error_message="Tree-sitter not available",
            )

        try:
            if not self.tree_sitter_manager:
                self.tree_sitter_manager = TreeSitterManager()

            if sample_code:
                # Temporarily override the sample for this test
                original_sample = self.test_samples.get(language)
                self.test_samples[language] = sample_code

                result = self._test_language_parser(language)

                # Restore original sample
                if original_sample:
                    self.test_samples[language] = original_sample
                else:
                    self.test_samples.pop(language, None)

                return result
            else:
                return self._test_language_parser(language)

        except Exception as e:
            return ParseTestResult(language=language, success=False, error_message=f"Test failed: {str(e)}")

    def generate_diagnostic_report(self, report: ParserHealthReport) -> str:
        """Generate a human-readable diagnostic report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TREE-SITTER PARSER HEALTH DIAGNOSTIC REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Test Duration: {report.test_duration_seconds:.2f} seconds")
        lines.append(f"Overall Health Score: {report.overall_health_score():.1f}/100 ({report.health_status()})")
        lines.append("")

        # System Status
        lines.append("SYSTEM STATUS")
        lines.append("-" * 20)
        lines.append(f"Tree-sitter Available: {'✓' if report.tree_sitter_available else '✗'}")
        if report.tree_sitter_version:
            lines.append(f"Tree-sitter Version: {report.tree_sitter_version}")
        lines.append("")

        # Installation Status
        lines.append("INSTALLATION STATUS")
        lines.append("-" * 20)
        lines.append(f"Total Languages: {report.total_languages}")
        lines.append(f"Successfully Installed: {report.installed_languages}")
        if report.failed_languages:
            lines.append(f"Failed Languages: {', '.join(report.failed_languages)}")
        success_rate = (report.installed_languages / max(1, report.total_languages)) * 100
        lines.append(f"Installation Success Rate: {success_rate:.1f}%")
        lines.append("")

        # Parsing Tests
        if report.parsing_tests:
            lines.append("PARSING FUNCTIONALITY TESTS")
            lines.append("-" * 30)
            for language, test in report.parsing_tests.items():
                status = "✓" if test.success else "✗"
                lines.append(f"{status} {language}: {test.parse_time_ms:.1f}ms, " f"{test.node_count} nodes, {test.error_count} errors")
                if not test.success and test.error_message:
                    lines.append(f"    Error: {test.error_message}")
            lines.append("")

        # Performance Metrics
        if report.average_parse_time_ms > 0:
            lines.append("PERFORMANCE METRICS")
            lines.append("-" * 20)
            lines.append(f"Average Parse Time: {report.average_parse_time_ms:.1f}ms")
            if report.fastest_language:
                lines.append(f"Fastest Language: {report.fastest_language}")
            if report.slowest_language:
                lines.append(f"Slowest Language: {report.slowest_language}")
            lines.append("")

        # Issues
        if report.critical_issues:
            lines.append("CRITICAL ISSUES")
            lines.append("-" * 15)
            for issue in report.critical_issues:
                lines.append(f"🔴 {issue}")
            lines.append("")

        if report.warnings:
            lines.append("WARNINGS")
            lines.append("-" * 10)
            for warning in report.warnings:
                lines.append(f"🟡 {warning}")
            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 15)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


# Global instance for easy access
parser_diagnostics = ParserDiagnostics()
