"""
Service for calculating confidence scores for function calls based on AST context and completeness.

This service implements sophisticated confidence scoring algorithms that analyze
AST node completeness, call context, and various quality indicators to assign
confidence scores to detected function calls.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from src.models.function_call import CallType, FunctionCall

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceConfiguration:
    """
    Configuration for confidence scoring algorithms.

    This class defines weights and thresholds for various confidence factors.
    """

    # Base confidence factors (0.0 to 1.0)
    base_confidence_by_call_type: dict[CallType, float] = field(
        default_factory=lambda: {
            CallType.DIRECT: 0.95,  # Direct calls are highly reliable
            CallType.METHOD: 0.90,  # Method calls are reliable
            CallType.SELF_METHOD: 0.95,  # Self method calls are very reliable
            CallType.ATTRIBUTE: 0.80,  # Attribute calls may be less certain
            CallType.ASYNC: 0.90,  # Async calls are reliable
            CallType.ASYNC_METHOD: 0.85,  # Async method calls are reliable
            CallType.ASYNCIO: 0.85,  # Asyncio calls are well-defined
            CallType.SUPER_METHOD: 0.85,  # Super calls are reliable
            CallType.CLASS_METHOD: 0.80,  # Class method calls are reliable
            CallType.DYNAMIC: 0.50,  # Dynamic calls are uncertain
            CallType.UNPACKING: 0.75,  # Unpacking calls have some uncertainty
            CallType.MODULE_FUNCTION: 0.90,  # Module function calls are reliable
            CallType.SUBSCRIPT_METHOD: 0.70,  # Subscript method calls are less certain
        }
    )

    # AST completeness factors
    complete_ast_bonus: float = 0.1  # Bonus for complete AST nodes
    partial_ast_penalty: float = 0.2  # Penalty for partial AST nodes
    missing_ast_penalty: float = 0.4  # Penalty for missing AST information

    # Context quality factors
    type_hints_bonus: float = 0.05  # Bonus for type hints
    docstring_bonus: float = 0.03  # Bonus for docstrings
    clean_syntax_bonus: float = 0.05  # Bonus for clean syntax
    argument_match_bonus: float = 0.05  # Bonus for argument count matching

    # Pattern quality factors
    pattern_match_confidence: dict[str, float] = field(
        default_factory=lambda: {
            "direct_function_call": 0.95,
            "method_call": 0.90,
            "self_method_call": 0.95,
            "chained_attribute_call": 0.75,
            "module_function_call": 0.90,
            "super_method_call": 0.85,
            "class_method_call": 0.80,
            "dynamic_attribute_call": 0.50,
            "await_function_call": 0.90,
            "await_method_call": 0.85,
            "asyncio_gather_call": 0.85,
            "asyncio_create_task_call": 0.85,
        }
    )

    # Context penalties
    conditional_uncertainty: float = 0.05  # Uncertainty for conditional calls
    nested_uncertainty: float = 0.03  # Uncertainty for nested calls
    syntax_error_penalty: float = 0.3  # Major penalty for syntax errors
    unresolved_target_penalty: float = 0.2  # Penalty for unresolved targets

    # Breadcrumb quality factors
    full_breadcrumb_bonus: float = 0.05  # Bonus for complete breadcrumbs
    partial_breadcrumb_penalty: float = 0.1  # Penalty for incomplete breadcrumbs
    missing_breadcrumb_penalty: float = 0.3  # Penalty for missing breadcrumbs

    # Call expression quality factors
    clear_expression_bonus: float = 0.03  # Bonus for clear call expressions
    complex_expression_penalty: float = 0.05  # Penalty for overly complex expressions

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate all confidence values are in valid range
        for call_type, confidence in self.base_confidence_by_call_type.items():
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Base confidence for {call_type} must be 0.0-1.0, got {confidence}")

        for pattern, confidence in self.pattern_match_confidence.items():
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Pattern confidence for {pattern} must be 0.0-1.0, got {confidence}")

    def get_base_confidence(self, call_type: CallType) -> float:
        """Get base confidence for a call type."""
        return self.base_confidence_by_call_type.get(call_type, 0.7)

    def get_pattern_confidence(self, pattern_name: str) -> float:
        """Get confidence for a specific pattern match."""
        return self.pattern_match_confidence.get(pattern_name, 0.7)


@dataclass
class ConfidenceAnalysis:
    """
    Detailed analysis of confidence factors for a function call.

    This provides transparency into how confidence scores are calculated.
    """

    # Base confidence components
    base_confidence: float
    pattern_confidence: float

    # AST quality components
    ast_completeness_score: float
    ast_quality_bonus: float

    # Context quality components
    context_quality_score: float
    breadcrumb_quality_score: float
    expression_quality_score: float

    # Applied bonuses and penalties
    bonuses_applied: dict[str, float]
    penalties_applied: dict[str, float]

    # Final calculation
    raw_confidence: float
    normalized_confidence: float
    confidence_factors: dict[str, float]

    def get_confidence_summary(self) -> dict[str, Any]:
        """Get a summary of confidence calculation."""
        return {
            "final_confidence": self.normalized_confidence,
            "confidence_grade": self._get_confidence_grade(),
            "primary_factors": self._get_primary_factors(),
            "risk_factors": list(self.penalties_applied.keys()),
            "quality_indicators": list(self.bonuses_applied.keys()),
            "ast_completeness": self.ast_completeness_score,
            "context_quality": self.context_quality_score,
        }

    def _get_confidence_grade(self) -> str:
        """Convert confidence score to letter grade."""
        if self.normalized_confidence >= 0.9:
            return "A"
        elif self.normalized_confidence >= 0.8:
            return "B"
        elif self.normalized_confidence >= 0.7:
            return "C"
        elif self.normalized_confidence >= 0.6:
            return "D"
        else:
            return "F"

    def _get_primary_factors(self) -> list[str]:
        """Get the primary factors affecting confidence."""
        factors = []

        # Add significant positive factors
        for factor, value in self.bonuses_applied.items():
            if value >= 0.03:
                factors.append(f"+{factor}")

        # Add significant negative factors
        for factor, value in self.penalties_applied.items():
            if value >= 0.05:
                factors.append(f"-{factor}")

        # Add base quality indicators
        if self.ast_completeness_score >= 0.9:
            factors.append("+complete_ast")
        elif self.ast_completeness_score <= 0.5:
            factors.append("-incomplete_ast")

        if self.context_quality_score >= 0.8:
            factors.append("+high_context_quality")
        elif self.context_quality_score <= 0.4:
            factors.append("-low_context_quality")

        return factors


class CallConfidenceScorer:
    """
    Service for calculating confidence scores for function calls.

    This service analyzes various aspects of function calls to assign confidence scores:
    - AST node completeness and quality
    - Call context and surrounding code
    - Pattern match reliability
    - Breadcrumb resolution quality
    - Expression clarity and complexity
    """

    def __init__(self, config: ConfidenceConfiguration | None = None):
        """
        Initialize the confidence scorer with configuration.

        Args:
            config: Confidence configuration. If None, uses default configuration.
        """
        self.config = config or ConfidenceConfiguration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate_confidence(self, call: FunctionCall, ast_context: dict[str, Any] | None = None) -> tuple[float, ConfidenceAnalysis]:
        """
        Calculate confidence score for a function call with detailed analysis.

        Args:
            call: The function call to score
            ast_context: Optional AST context information

        Returns:
            Tuple of (confidence_score, detailed_analysis)
        """
        # Initialize analysis tracking
        bonuses = {}
        penalties = {}

        # 1. Calculate base confidence from call type
        base_confidence = self.config.get_base_confidence(call.call_type)

        # 2. Calculate pattern match confidence
        pattern_confidence = self.config.get_pattern_confidence(call.pattern_matched)

        # 3. Analyze AST completeness
        ast_completeness, ast_bonus = self._analyze_ast_completeness(call, ast_context)
        if ast_bonus > 0:
            bonuses["ast_completeness"] = ast_bonus
        elif ast_bonus < 0:
            penalties["ast_incompleteness"] = -ast_bonus

        # 4. Analyze context quality
        context_quality = self._analyze_context_quality(call, bonuses, penalties)

        # 5. Analyze breadcrumb quality
        breadcrumb_quality = self._analyze_breadcrumb_quality(call, bonuses, penalties)

        # 6. Analyze expression quality
        expression_quality = self._analyze_expression_quality(call, bonuses, penalties)

        # 7. Apply conditional and nested call penalties
        self._apply_context_penalties(call, penalties)

        # 8. Apply syntax error penalties
        self._apply_syntax_penalties(call, penalties)

        # 9. Calculate raw confidence
        raw_confidence = base_confidence

        # Apply pattern confidence adjustment
        raw_confidence = (raw_confidence + pattern_confidence) / 2

        # Apply bonuses
        for bonus_name, bonus_value in bonuses.items():
            raw_confidence += bonus_value

        # Apply penalties
        for penalty_name, penalty_value in penalties.items():
            raw_confidence -= penalty_value

        # 10. Normalize confidence to [0.0, 1.0]
        normalized_confidence = max(0.0, min(1.0, raw_confidence))

        # Create detailed analysis
        analysis = ConfidenceAnalysis(
            base_confidence=base_confidence,
            pattern_confidence=pattern_confidence,
            ast_completeness_score=ast_completeness,
            ast_quality_bonus=ast_bonus,
            context_quality_score=context_quality,
            breadcrumb_quality_score=breadcrumb_quality,
            expression_quality_score=expression_quality,
            bonuses_applied=bonuses,
            penalties_applied=penalties,
            raw_confidence=raw_confidence,
            normalized_confidence=normalized_confidence,
            confidence_factors={
                "base_confidence": base_confidence,
                "pattern_confidence": pattern_confidence,
                "ast_quality": ast_completeness,
                "context_quality": context_quality,
                "breadcrumb_quality": breadcrumb_quality,
                "expression_quality": expression_quality,
            },
        )

        self.logger.debug(
            f"Confidence for '{call.call_expression}': {normalized_confidence:.3f} "
            f"(base={base_confidence:.3f}, pattern={pattern_confidence:.3f}, "
            f"bonuses={len(bonuses)}, penalties={len(penalties)})"
        )

        return normalized_confidence, analysis

    def calculate_confidence_for_calls(
        self, calls: list[FunctionCall], ast_contexts: dict[str, Any] | None = None
    ) -> list[tuple[FunctionCall, ConfidenceAnalysis]]:
        """
        Calculate confidence scores for a list of function calls.

        Args:
            calls: List of function calls to score
            ast_contexts: Optional dictionary of AST contexts keyed by call identifier

        Returns:
            List of (updated_call, confidence_analysis) tuples
        """
        results = []

        for call in calls:
            # Get AST context for this call if available
            call_id = f"{call.file_path}:{call.line_number}"
            ast_context = ast_contexts.get(call_id) if ast_contexts else None

            # Calculate confidence
            confidence, analysis = self.calculate_confidence(call, ast_context)

            # Update call with confidence score
            call.confidence = confidence

            results.append((call, analysis))

        self.logger.info(
            f"Calculated confidence for {len(calls)} calls. "
            f"Average confidence: {sum(call.confidence for call, _ in results) / len(results):.3f}"
        )

        return results

    def _analyze_ast_completeness(self, call: FunctionCall, ast_context: dict[str, Any] | None) -> tuple[float, float]:
        """
        Analyze AST node completeness and quality.

        Args:
            call: Function call to analyze
            ast_context: AST context information

        Returns:
            Tuple of (completeness_score, bonus_or_penalty)
        """
        completeness_score = 0.5  # Default score
        bonus_penalty = 0.0

        # Check if we have AST context
        if ast_context is None:
            # No AST context available - but be more lenient for same-project calls
            if self._is_likely_same_project_call(call):
                completeness_score = 0.6  # More lenient for same-project
                bonus_penalty = -self.config.missing_ast_penalty * 0.5  # Reduced penalty
            else:
                completeness_score = 0.3
                bonus_penalty = -self.config.missing_ast_penalty
            return completeness_score, bonus_penalty

        # Analyze AST node quality indicators
        quality_indicators = 0
        total_indicators = 0

        # Check for complete node information
        if ast_context.get("node_type"):
            quality_indicators += 1
        total_indicators += 1

        if ast_context.get("node_text"):
            quality_indicators += 1
        total_indicators += 1

        if ast_context.get("start_point") and ast_context.get("end_point"):
            quality_indicators += 1
        total_indicators += 1

        if ast_context.get("parent_node"):
            quality_indicators += 1
        total_indicators += 1

        if ast_context.get("children_nodes"):
            quality_indicators += 1
        total_indicators += 1

        # Additional quality indicators from enhanced context
        if ast_context.get("pattern_name"):
            quality_indicators += 1
        total_indicators += 1

        if ast_context.get("chunk_context"):
            quality_indicators += 1
        total_indicators += 1

        # Calculate completeness score
        if total_indicators > 0:
            completeness_score = quality_indicators / total_indicators

        # Apply bonuses/penalties based on completeness
        if completeness_score >= 0.9:
            bonus_penalty = self.config.complete_ast_bonus
        elif completeness_score >= 0.7:
            bonus_penalty = self.config.complete_ast_bonus * 0.5  # Partial bonus
        elif completeness_score <= 0.4:
            # Reduce penalty for same-project calls
            penalty_factor = 0.5 if self._is_likely_same_project_call(call) else 1.0
            bonus_penalty = -self.config.partial_ast_penalty * penalty_factor

        return completeness_score, bonus_penalty

    def _analyze_context_quality(self, call: FunctionCall, bonuses: dict[str, float], penalties: dict[str, float]) -> float:
        """
        Analyze the quality of call context information.

        Args:
            call: Function call to analyze
            bonuses: Dictionary to add bonuses to
            penalties: Dictionary to add penalties to

        Returns:
            Context quality score (0.0 to 1.0)
        """
        quality_score = 0.0
        quality_count = 0

        # Check for type hints
        if call.has_type_hints:
            quality_score += 1.0
            bonuses["type_hints"] = self.config.type_hints_bonus
        quality_count += 1

        # Check for docstrings
        if call.has_docstring:
            quality_score += 1.0
            bonuses["docstring"] = self.config.docstring_bonus
        quality_count += 1

        # Check for clean syntax
        if not call.has_syntax_errors:
            quality_score += 1.0
            bonuses["clean_syntax"] = self.config.clean_syntax_bonus
        quality_count += 1

        # Check argument count reasonableness (0-10 is reasonable)
        if 0 <= call.arguments_count <= 10:
            quality_score += 1.0
            if call.arguments_count > 0:
                bonuses["argument_match"] = self.config.argument_match_bonus
        quality_count += 1

        # Calculate normalized quality score
        return quality_score / quality_count if quality_count > 0 else 0.0

    def _analyze_breadcrumb_quality(self, call: FunctionCall, bonuses: dict[str, float], penalties: dict[str, float]) -> float:
        """
        Analyze the quality of breadcrumb information.

        Args:
            call: Function call to analyze
            bonuses: Dictionary to add bonuses to
            penalties: Dictionary to add penalties to

        Returns:
            Breadcrumb quality score (0.0 to 1.0)
        """
        quality_score = 0.0
        quality_count = 0
        is_same_project = self._is_likely_same_project_call(call)

        # Analyze source breadcrumb
        if call.source_breadcrumb:
            if len(call.source_breadcrumb.split(".")) >= 2:
                quality_score += 1.0
                bonuses["full_source_breadcrumb"] = self.config.full_breadcrumb_bonus / 2
            else:
                quality_score += 0.5
                # Reduce penalty for same-project calls
                penalty_factor = 0.5 if is_same_project else 1.0
                penalties["partial_source_breadcrumb"] = self.config.partial_breadcrumb_penalty * penalty_factor / 2
        else:
            # Reduce penalty for same-project calls
            penalty_factor = 0.5 if is_same_project else 1.0
            penalties["missing_source_breadcrumb"] = self.config.missing_breadcrumb_penalty * penalty_factor / 2
        quality_count += 1

        # Analyze target breadcrumb with improved same-project detection
        if call.target_breadcrumb:
            target_parts = call.target_breadcrumb.split(".")

            # Better scoring for same-project breadcrumbs
            if is_same_project:
                if len(target_parts) >= 2:
                    quality_score += 1.0
                    bonuses["full_target_breadcrumb"] = self.config.full_breadcrumb_bonus / 2
                    # Additional bonus for likely same-project calls
                    bonuses["same_project_call"] = 0.05
                else:
                    quality_score += 0.7  # More generous for same-project
                    # Reduced penalty
                    penalties["partial_target_breadcrumb"] = self.config.partial_breadcrumb_penalty * 0.3 / 2
            else:
                if len(target_parts) >= 2:
                    quality_score += 1.0
                    bonuses["full_target_breadcrumb"] = self.config.full_breadcrumb_bonus / 2
                else:
                    quality_score += 0.5
                    penalties["partial_target_breadcrumb"] = self.config.partial_breadcrumb_penalty / 2
        else:
            # Reduce penalty for same-project calls
            penalty_factor = 0.3 if is_same_project else 1.0
            penalties["missing_target_breadcrumb"] = self.config.missing_breadcrumb_penalty * penalty_factor / 2
        quality_count += 1

        return quality_score / quality_count if quality_count > 0 else 0.0

    def _is_likely_same_project_call(self, call: FunctionCall) -> bool:
        """
        Determine if a function call is likely within the same project.

        Args:
            call: Function call to analyze

        Returns:
            True if likely same-project call
        """
        if not call.target_breadcrumb:
            return True  # Assume same-project if no specific target

        target = call.target_breadcrumb.lower()

        # Patterns that indicate same-project calls
        same_project_patterns = [
            "self.",
            "current_class.",
            "current_module.",
            "src.",
            ".service",
            ".model",
            ".util",
            ".tool",
            ".cache",
            "_service",
            "_resolver",
            "_extractor",
            "_analyzer",
            "_builder",
        ]

        # Patterns that indicate external calls
        external_patterns = [
            "time.",
            "os.",
            "sys.",
            "json.",
            "logging.",
            "asyncio.",
            "pathlib.",
            "datetime.",
            "collections.",
            "functools.",
            "itertools.",
            "typing.",
            "dataclasses.",
            "abc.",
            "copy.",
            "re.",
            "math.",
            "random.",
            "unknown_module.",
            "unknown.",
        ]

        # Check for external patterns first
        for pattern in external_patterns:
            if pattern in target:
                return False

        # Check for same-project patterns
        for pattern in same_project_patterns:
            if pattern in target:
                return True

        # If target breadcrumb doesn't have dots, likely same-project
        if "." not in target:
            return True

        # Default to same-project for ambiguous cases
        return True

    def _analyze_expression_quality(self, call: FunctionCall, bonuses: dict[str, float], penalties: dict[str, float]) -> float:
        """
        Analyze the quality of the call expression.

        Args:
            call: Function call to analyze
            bonuses: Dictionary to add bonuses to
            penalties: Dictionary to add penalties to

        Returns:
            Expression quality score (0.0 to 1.0)
        """
        if not call.call_expression:
            penalties["missing_expression"] = 0.1
            return 0.0

        expression = call.call_expression.strip()
        quality_score = 0.5  # Base score

        # Check for clear, readable expression
        if len(expression) > 0:
            # Bonus for reasonable length (not too short or too long)
            if 5 <= len(expression) <= 100:
                quality_score += 0.3
                bonuses["clear_expression"] = self.config.clear_expression_bonus
            elif len(expression) > 200:
                quality_score -= 0.2
                penalties["complex_expression"] = self.config.complex_expression_penalty

        # Check for balanced parentheses
        if expression.count("(") == expression.count(")"):
            quality_score += 0.2
        else:
            quality_score -= 0.3
            penalties["unbalanced_parentheses"] = 0.05

        # Check for suspicious patterns that might indicate parsing errors
        suspicious_patterns = ["..", "((", "))", ".,", ",.", ";;"]
        for pattern in suspicious_patterns:
            if pattern in expression:
                quality_score -= 0.1
                penalties["suspicious_expression_pattern"] = 0.03
                break

        return max(0.0, min(1.0, quality_score))

    def _apply_context_penalties(self, call: FunctionCall, penalties: dict[str, float]) -> None:
        """
        Apply penalties based on call context.

        Args:
            call: Function call to analyze
            penalties: Dictionary to add penalties to
        """
        # Conditional call penalty
        if call.is_conditional:
            penalties["conditional_call"] = self.config.conditional_uncertainty

        # Nested call penalty
        if call.is_nested:
            penalties["nested_call"] = self.config.nested_uncertainty

    def _apply_syntax_penalties(self, call: FunctionCall, penalties: dict[str, float]) -> None:
        """
        Apply penalties based on syntax errors.

        Args:
            call: Function call to analyze
            penalties: Dictionary to add penalties to
        """
        if call.has_syntax_errors:
            penalties["syntax_errors"] = self.config.syntax_error_penalty

            # Additional penalty based on error details
            if call.error_details:
                error_lower = call.error_details.lower()
                if any(severe in error_lower for severe in ["unexpected", "invalid", "missing"]):
                    penalties["severe_syntax_errors"] = 0.1

    def get_confidence_statistics(self, calls: list[FunctionCall]) -> dict[str, Any]:
        """
        Generate confidence-related statistics for a list of calls.

        Args:
            calls: List of function calls to analyze

        Returns:
            Dictionary containing confidence statistics
        """
        if not calls:
            return {"total_calls": 0}

        confidences = [call.confidence for call in calls]

        # Confidence distribution
        high_confidence = len([c for c in calls if c.confidence >= 0.8])
        medium_confidence = len([c for c in calls if 0.6 <= c.confidence < 0.8])
        low_confidence = len([c for c in calls if c.confidence < 0.6])

        # Confidence by call type
        confidence_by_type = {}
        for call in calls:
            call_type = call.call_type.value
            if call_type not in confidence_by_type:
                confidence_by_type[call_type] = []
            confidence_by_type[call_type].append(call.confidence)

        avg_confidence_by_type = {call_type: sum(confidences) / len(confidences) for call_type, confidences in confidence_by_type.items()}

        return {
            "total_calls": len(calls),
            "confidence_stats": {
                "min": min(confidences),
                "max": max(confidences),
                "avg": sum(confidences) / len(confidences),
                "median": sorted(confidences)[len(confidences) // 2],
            },
            "confidence_distribution": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence,
                "high_confidence_pct": high_confidence / len(calls) * 100,
                "medium_confidence_pct": medium_confidence / len(calls) * 100,
                "low_confidence_pct": low_confidence / len(calls) * 100,
            },
            "confidence_by_call_type": avg_confidence_by_type,
        }

    def update_configuration(self, new_config: ConfidenceConfiguration) -> None:
        """
        Update the confidence scoring configuration.

        Args:
            new_config: New configuration to use
        """
        self.config = new_config
        self.logger.info("Updated confidence scoring configuration")

    def get_configuration(self) -> ConfidenceConfiguration:
        """
        Get the current confidence scoring configuration.

        Returns:
            Current configuration
        """
        return self.config


# Utility functions for confidence scoring


def create_default_confidence_config() -> ConfidenceConfiguration:
    """Create a default confidence configuration."""
    return ConfidenceConfiguration()


def create_strict_confidence_config() -> ConfidenceConfiguration:
    """Create a strict confidence configuration with higher standards."""
    config = ConfidenceConfiguration()

    # Increase penalties
    config.syntax_error_penalty = 0.5
    config.missing_ast_penalty = 0.6
    config.partial_ast_penalty = 0.3
    config.unresolved_target_penalty = 0.3

    # Reduce bonuses
    config.type_hints_bonus = 0.03
    config.docstring_bonus = 0.02
    config.clean_syntax_bonus = 0.03

    return config


def create_lenient_confidence_config() -> ConfidenceConfiguration:
    """Create a lenient confidence configuration with lower standards."""
    config = ConfidenceConfiguration()

    # Reduce penalties
    config.syntax_error_penalty = 0.2
    config.missing_ast_penalty = 0.2
    config.partial_ast_penalty = 0.1
    config.unresolved_target_penalty = 0.1

    # Increase bonuses
    config.type_hints_bonus = 0.08
    config.docstring_bonus = 0.05
    config.clean_syntax_bonus = 0.08

    return config


def analyze_confidence_trends(calls_with_analysis: list[tuple[FunctionCall, ConfidenceAnalysis]]) -> dict[str, Any]:
    """
    Analyze confidence trends across a set of calls with detailed analysis.

    Args:
        calls_with_analysis: List of (call, confidence_analysis) tuples

    Returns:
        Dictionary containing trend analysis
    """
    if not calls_with_analysis:
        return {}

    # Extract analysis data
    analyses = [analysis for _, analysis in calls_with_analysis]

    # Common bonuses and penalties
    all_bonuses = {}
    all_penalties = {}

    for analysis in analyses:
        for bonus, value in analysis.bonuses_applied.items():
            all_bonuses[bonus] = all_bonuses.get(bonus, 0) + 1

        for penalty, value in analysis.penalties_applied.items():
            all_penalties[penalty] = all_penalties.get(penalty, 0) + 1

    # Calculate averages
    avg_ast_completeness = sum(a.ast_completeness_score for a in analyses) / len(analyses)
    avg_context_quality = sum(a.context_quality_score for a in analyses) / len(analyses)
    avg_breadcrumb_quality = sum(a.breadcrumb_quality_score for a in analyses) / len(analyses)
    avg_expression_quality = sum(a.expression_quality_score for a in analyses) / len(analyses)

    # Confidence grade distribution
    grades = [analysis.get_confidence_summary()["confidence_grade"] for analysis in analyses]
    grade_distribution = {grade: grades.count(grade) for grade in set(grades)}

    return {
        "total_analyzed": len(analyses),
        "average_scores": {
            "ast_completeness": avg_ast_completeness,
            "context_quality": avg_context_quality,
            "breadcrumb_quality": avg_breadcrumb_quality,
            "expression_quality": avg_expression_quality,
        },
        "common_bonuses": dict(sorted(all_bonuses.items(), key=lambda x: x[1], reverse=True)[:5]),
        "common_penalties": dict(sorted(all_penalties.items(), key=lambda x: x[1], reverse=True)[:5]),
        "confidence_grades": grade_distribution,
        "quality_recommendations": _generate_quality_recommendations(
            avg_ast_completeness, avg_context_quality, avg_breadcrumb_quality, avg_expression_quality
        ),
    }


def _generate_quality_recommendations(
    ast_score: float, context_score: float, breadcrumb_score: float, expression_score: float
) -> list[str]:
    """Generate recommendations based on quality scores."""
    recommendations = []

    if ast_score < 0.6:
        recommendations.append("Improve AST parsing completeness and error handling")

    if context_score < 0.7:
        recommendations.append("Add more type hints and documentation to improve context quality")

    if breadcrumb_score < 0.6:
        recommendations.append("Enhance breadcrumb resolution for better call target identification")

    if expression_score < 0.7:
        recommendations.append("Improve call expression extraction and parsing")

    if not recommendations:
        recommendations.append("Overall confidence quality is good - maintain current standards")

    return recommendations
