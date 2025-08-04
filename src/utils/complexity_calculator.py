"""
Complexity Calculator Utility for Function Chain MCP Tools.

This module provides comprehensive complexity calculation for functions with
configurable weights for different complexity factors. It supports AST-based
analysis for accurate complexity metrics calculation.
"""

import ast
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ComplexityMetric(Enum):
    """Types of complexity metrics that can be calculated."""

    BRANCHING_FACTOR = "branching_factor"
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    CALL_DEPTH = "call_depth"
    FUNCTION_LENGTH = "function_length"


@dataclass
class ComplexityWeights:
    """Weights for complexity calculation components."""

    branching_factor: float = 0.35  # 35% - Control flow branches
    cyclomatic_complexity: float = 0.30  # 30% - Cyclomatic complexity
    call_depth: float = 0.25  # 25% - Function call depth
    function_length: float = 0.10  # 10% - Function length in lines

    def normalize(self) -> "ComplexityWeights":
        """Normalize weights to sum to 1.0."""
        total = self.branching_factor + self.cyclomatic_complexity + self.call_depth + self.function_length

        if total == 0:
            return ComplexityWeights()

        return ComplexityWeights(
            branching_factor=self.branching_factor / total,
            cyclomatic_complexity=self.cyclomatic_complexity / total,
            call_depth=self.call_depth / total,
            function_length=self.function_length / total,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "branching_factor": self.branching_factor,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "call_depth": self.call_depth,
            "function_length": self.function_length,
        }


@dataclass
class ComplexityMetrics:
    """Individual complexity metrics for a function."""

    # Raw metrics
    branching_factor: int = 0
    cyclomatic_complexity: int = 0
    call_depth: int = 0
    function_length: int = 0

    # Normalized scores (0.0-1.0)
    branching_score: float = 0.0
    cyclomatic_score: float = 0.0
    call_depth_score: float = 0.0
    function_length_score: float = 0.0

    # Weighted scores
    weighted_branching: float = 0.0
    weighted_cyclomatic: float = 0.0
    weighted_call_depth: float = 0.0
    weighted_function_length: float = 0.0

    # Overall complexity
    overall_complexity: float = 0.0
    complexity_category: str = "low"  # low, medium, high

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_metrics": {
                "branching_factor": self.branching_factor,
                "cyclomatic_complexity": self.cyclomatic_complexity,
                "call_depth": self.call_depth,
                "function_length": self.function_length,
            },
            "normalized_scores": {
                "branching_score": self.branching_score,
                "cyclomatic_score": self.cyclomatic_score,
                "call_depth_score": self.call_depth_score,
                "function_length_score": self.function_length_score,
            },
            "weighted_scores": {
                "weighted_branching": self.weighted_branching,
                "weighted_cyclomatic": self.weighted_cyclomatic,
                "weighted_call_depth": self.weighted_call_depth,
                "weighted_function_length": self.weighted_function_length,
            },
            "overall_complexity": self.overall_complexity,
            "complexity_category": self.complexity_category,
        }


class ComplexityCalculator:
    """
    Calculator for function complexity using configurable weights.

    This calculator supports AST-based analysis for accurate complexity
    calculation across multiple programming languages.
    """

    def __init__(self, weights: ComplexityWeights | None = None):
        """
        Initialize the complexity calculator.

        Args:
            weights: Custom complexity weights (uses defaults if None)
        """
        self.weights = weights or ComplexityWeights()
        self.weights = self.weights.normalize()

        # Normalization thresholds for converting raw metrics to 0-1 scores
        self.normalization_thresholds = {
            "branching_factor": {"low": 2, "medium": 5, "high": 10},
            "cyclomatic_complexity": {"low": 5, "medium": 10, "high": 20},
            "call_depth": {"low": 3, "medium": 6, "high": 10},
            "function_length": {"low": 20, "medium": 50, "high": 100},
        }

        logger.debug(f"ComplexityCalculator initialized with weights: {self.weights.to_dict()}")

    def calculate_complexity(self, function_data: dict[str, Any]) -> ComplexityMetrics:
        """
        Calculate comprehensive complexity metrics for a function.

        Args:
            function_data: Dictionary containing function information
                         (content, name, file_path, language, etc.)

        Returns:
            ComplexityMetrics with all calculated complexity information
        """
        try:
            # Extract raw metrics
            raw_metrics = self._extract_raw_metrics(function_data)

            # Normalize metrics to 0-1 scores
            normalized_scores = self._normalize_metrics(raw_metrics)

            # Apply weights
            weighted_scores = self._apply_weights(normalized_scores)

            # Calculate overall complexity
            overall_complexity = sum(weighted_scores.values())

            # Determine complexity category
            complexity_category = self._categorize_complexity(overall_complexity)

            # Create metrics object
            metrics = ComplexityMetrics(
                # Raw metrics
                branching_factor=raw_metrics["branching_factor"],
                cyclomatic_complexity=raw_metrics["cyclomatic_complexity"],
                call_depth=raw_metrics["call_depth"],
                function_length=raw_metrics["function_length"],
                # Normalized scores
                branching_score=normalized_scores["branching_factor"],
                cyclomatic_score=normalized_scores["cyclomatic_complexity"],
                call_depth_score=normalized_scores["call_depth"],
                function_length_score=normalized_scores["function_length"],
                # Weighted scores
                weighted_branching=weighted_scores["branching_factor"],
                weighted_cyclomatic=weighted_scores["cyclomatic_complexity"],
                weighted_call_depth=weighted_scores["call_depth"],
                weighted_function_length=weighted_scores["function_length"],
                # Overall complexity
                overall_complexity=overall_complexity,
                complexity_category=complexity_category,
            )

            logger.debug(f"Calculated complexity for {function_data.get('name', 'unknown')}: {overall_complexity:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating complexity for function {function_data.get('name', 'unknown')}: {e}")
            return ComplexityMetrics()  # Return default metrics on error

    def _extract_raw_metrics(self, function_data: dict[str, Any]) -> dict[str, int]:
        """Extract raw complexity metrics from function data."""
        content = function_data.get("content", "")
        language = function_data.get("language", "").lower()

        # Function length (lines of code)
        function_length = len([line for line in content.split("\n") if line.strip()])

        # Try AST-based analysis first, fallback to heuristic analysis
        if language == "python" and content:
            return self._analyze_python_ast(content, function_length)
        else:
            return self._analyze_heuristic(content, function_length, language)

    def _analyze_python_ast(self, content: str, function_length: int) -> dict[str, int]:
        """Analyze Python code using AST for accurate complexity calculation."""
        try:
            tree = ast.parse(content)

            # Initialize metrics

            # AST visitor to calculate metrics
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.max_depth = 0
                    self.current_depth = 0
                    self.branches = 0
                    self.complexity = 1

                def visit_If(self, node):
                    self.branches += 1
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_For(self, node):
                    self.branches += 1
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_While(self, node):
                    self.branches += 1
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_Try(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_ExceptHandler(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_With(self, node):
                    self.complexity += 1
                    self.generic_visit(node)

                def visit_Call(self, node):
                    self.current_depth += 1
                    self.max_depth = max(self.max_depth, self.current_depth)
                    self.generic_visit(node)
                    self.current_depth -= 1

                def visit_FunctionDef(self, node):
                    # Only analyze the first function (avoid nested functions)
                    if self.current_depth == 0:
                        self.generic_visit(node)

                def visit_AsyncFunctionDef(self, node):
                    # Only analyze the first function (avoid nested functions)
                    if self.current_depth == 0:
                        self.generic_visit(node)

            visitor = ComplexityVisitor()
            visitor.visit(tree)

            return {
                "branching_factor": visitor.branches,
                "cyclomatic_complexity": visitor.complexity,
                "call_depth": visitor.max_depth,
                "function_length": function_length,
            }

        except SyntaxError as e:
            logger.warning(f"Python syntax error in AST analysis: {e}")
            return self._analyze_heuristic(content, function_length, "python")
        except Exception as e:
            logger.warning(f"Error in Python AST analysis: {e}")
            return self._analyze_heuristic(content, function_length, "python")

    def _analyze_heuristic(self, content: str, function_length: int, language: str) -> dict[str, int]:
        """Analyze code using heuristic patterns for complexity calculation."""
        lines = content.split("\n")

        # Initialize metrics
        branching_factor = 0
        cyclomatic_complexity = 1  # Base complexity
        call_depth = 0

        # Language-specific patterns for control flow
        if language == "python":
            branch_patterns = [r"\bif\b", r"\belif\b", r"\bfor\b", r"\bwhile\b", r"\btry\b", r"\bexcept\b"]
            call_patterns = [r"\w+\s*\(", r"\.\w+\s*\("]
        elif language in ["javascript", "typescript"]:
            branch_patterns = [r"\bif\b", r"\belse\s+if\b", r"\bfor\b", r"\bwhile\b", r"\btry\b", r"\bcatch\b", r"\bswitch\b"]
            call_patterns = [r"\w+\s*\(", r"\.\w+\s*\("]
        elif language in ["java", "c", "cpp"]:
            branch_patterns = [r"\bif\b", r"\belse\s+if\b", r"\bfor\b", r"\bwhile\b", r"\btry\b", r"\bcatch\b", r"\bswitch\b"]
            call_patterns = [r"\w+\s*\(", r"\.\w+\s*\(", r"->\w+\s*\("]
        else:
            # Generic patterns
            branch_patterns = [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\btry\b", r"\bswitch\b"]
            call_patterns = [r"\w+\s*\("]

        # Count branching statements
        for line in lines:
            line = line.strip()
            for pattern in branch_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    branching_factor += 1
                    cyclomatic_complexity += 1
                    break  # Count each line only once

        # Estimate call depth by counting nested function calls
        max_calls_per_line = 0
        for line in lines:
            line = line.strip()
            call_count = 0
            for pattern in call_patterns:
                call_count += len(re.findall(pattern, line))
            max_calls_per_line = max(max_calls_per_line, call_count)

        call_depth = min(max_calls_per_line, 10)  # Cap at reasonable maximum

        return {
            "branching_factor": branching_factor,
            "cyclomatic_complexity": cyclomatic_complexity,
            "call_depth": call_depth,
            "function_length": function_length,
        }

    def _normalize_metrics(self, raw_metrics: dict[str, int]) -> dict[str, float]:
        """Normalize raw metrics to 0-1 scores."""
        normalized = {}

        for metric_name, value in raw_metrics.items():
            thresholds = self.normalization_thresholds[metric_name]

            if value <= thresholds["low"]:
                normalized[metric_name] = value / thresholds["low"] * 0.3  # 0-0.3 for low
            elif value <= thresholds["medium"]:
                ratio = (value - thresholds["low"]) / (thresholds["medium"] - thresholds["low"])
                normalized[metric_name] = 0.3 + ratio * 0.4  # 0.3-0.7 for medium
            elif value <= thresholds["high"]:
                ratio = (value - thresholds["medium"]) / (thresholds["high"] - thresholds["medium"])
                normalized[metric_name] = 0.7 + ratio * 0.3  # 0.7-1.0 for high
            else:
                normalized[metric_name] = 1.0  # Cap at 1.0 for very high values

        return normalized

    def _apply_weights(self, normalized_scores: dict[str, float]) -> dict[str, float]:
        """Apply weights to normalized scores."""
        return {
            "branching_factor": normalized_scores["branching_factor"] * self.weights.branching_factor,
            "cyclomatic_complexity": normalized_scores["cyclomatic_complexity"] * self.weights.cyclomatic_complexity,
            "call_depth": normalized_scores["call_depth"] * self.weights.call_depth,
            "function_length": normalized_scores["function_length"] * self.weights.function_length,
        }

    def _categorize_complexity(self, overall_complexity: float) -> str:
        """Categorize overall complexity into low/medium/high."""
        if overall_complexity <= 0.3:
            return "low"
        elif overall_complexity <= 0.7:
            return "medium"
        else:
            return "high"

    def calculate_batch_complexity(self, functions: list[dict[str, Any]]) -> list[ComplexityMetrics]:
        """
        Calculate complexity for a batch of functions.

        Args:
            functions: List of function data dictionaries

        Returns:
            List of ComplexityMetrics for each function
        """
        results = []

        for function_data in functions:
            metrics = self.calculate_complexity(function_data)
            results.append(metrics)

        logger.info(f"Calculated complexity for {len(functions)} functions")
        return results

    def get_complexity_statistics(self, metrics_list: list[ComplexityMetrics]) -> dict[str, Any]:
        """
        Calculate statistics across multiple complexity metrics.

        Args:
            metrics_list: List of ComplexityMetrics

        Returns:
            Dictionary with complexity statistics
        """
        if not metrics_list:
            return {}

        # Extract overall complexity scores
        complexity_scores = [m.overall_complexity for m in metrics_list]

        # Category distribution
        categories = [m.complexity_category for m in metrics_list]
        category_counts = {
            "low": categories.count("low"),
            "medium": categories.count("medium"),
            "high": categories.count("high"),
        }

        # Raw metric statistics
        raw_stats = {}
        for metric_name in ["branching_factor", "cyclomatic_complexity", "call_depth", "function_length"]:
            values = [getattr(m, metric_name) for m in metrics_list]
            raw_stats[metric_name] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2],
            }

        return {
            "total_functions": len(metrics_list),
            "complexity_distribution": {
                "min": min(complexity_scores),
                "max": max(complexity_scores),
                "avg": sum(complexity_scores) / len(complexity_scores),
                "median": sorted(complexity_scores)[len(complexity_scores) // 2],
            },
            "category_distribution": category_counts,
            "category_percentages": {
                "low": category_counts["low"] / len(metrics_list) * 100,
                "medium": category_counts["medium"] / len(metrics_list) * 100,
                "high": category_counts["high"] / len(metrics_list) * 100,
            },
            "raw_metric_statistics": raw_stats,
            "weights_used": self.weights.to_dict(),
        }

    def update_weights(self, new_weights: ComplexityWeights):
        """Update the complexity calculation weights."""
        self.weights = new_weights.normalize()
        logger.info(f"Updated complexity weights: {self.weights.to_dict()}")

    def get_complexity_breakdown(self, metrics: ComplexityMetrics) -> dict[str, Any]:
        """
        Get detailed breakdown of complexity calculation.

        Args:
            metrics: ComplexityMetrics to analyze

        Returns:
            Detailed breakdown of how complexity was calculated
        """
        total_weighted = (
            metrics.weighted_branching + metrics.weighted_cyclomatic + metrics.weighted_call_depth + metrics.weighted_function_length
        )

        return {
            "overall_complexity": metrics.overall_complexity,
            "complexity_category": metrics.complexity_category,
            "component_contributions": {
                "branching_factor": {
                    "raw_value": metrics.branching_factor,
                    "normalized_score": metrics.branching_score,
                    "weight": self.weights.branching_factor,
                    "weighted_contribution": metrics.weighted_branching,
                    "percentage_of_total": (metrics.weighted_branching / total_weighted * 100) if total_weighted > 0 else 0,
                },
                "cyclomatic_complexity": {
                    "raw_value": metrics.cyclomatic_complexity,
                    "normalized_score": metrics.cyclomatic_score,
                    "weight": self.weights.cyclomatic_complexity,
                    "weighted_contribution": metrics.weighted_cyclomatic,
                    "percentage_of_total": (metrics.weighted_cyclomatic / total_weighted * 100) if total_weighted > 0 else 0,
                },
                "call_depth": {
                    "raw_value": metrics.call_depth,
                    "normalized_score": metrics.call_depth_score,
                    "weight": self.weights.call_depth,
                    "weighted_contribution": metrics.weighted_call_depth,
                    "percentage_of_total": (metrics.weighted_call_depth / total_weighted * 100) if total_weighted > 0 else 0,
                },
                "function_length": {
                    "raw_value": metrics.function_length,
                    "normalized_score": metrics.function_length_score,
                    "weight": self.weights.function_length,
                    "weighted_contribution": metrics.weighted_function_length,
                    "percentage_of_total": (metrics.weighted_function_length / total_weighted * 100) if total_weighted > 0 else 0,
                },
            },
        }


def create_complexity_calculator(
    branching_weight: float = 0.35, cyclomatic_weight: float = 0.30, call_depth_weight: float = 0.25, function_length_weight: float = 0.10
) -> ComplexityCalculator:
    """
    Create a complexity calculator with custom weights.

    Args:
        branching_weight: Weight for branching factor (default: 35%)
        cyclomatic_weight: Weight for cyclomatic complexity (default: 30%)
        call_depth_weight: Weight for call depth (default: 25%)
        function_length_weight: Weight for function length (default: 10%)

    Returns:
        Configured ComplexityCalculator instance
    """
    weights = ComplexityWeights(
        branching_factor=branching_weight,
        cyclomatic_complexity=cyclomatic_weight,
        call_depth=call_depth_weight,
        function_length=function_length_weight,
    )

    return ComplexityCalculator(weights)


def get_default_complexity_weights() -> ComplexityWeights:
    """Get the default complexity weights as specified in the requirements."""
    return ComplexityWeights(
        branching_factor=0.35,  # 35%
        cyclomatic_complexity=0.30,  # 30%
        call_depth=0.25,  # 25%
        function_length=0.10,  # 10%
    )
