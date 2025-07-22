"""
Service for filtering function calls based on configurable weight and confidence thresholds.

This service provides comprehensive filtering capabilities with configurable thresholds,
quality filters, and advanced filtering strategies for function call analysis.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from src.models.function_call import CallType, FunctionCall

logger = logging.getLogger(__name__)


class FilterMode(Enum):
    """Enumeration of filtering modes."""

    STRICT = "strict"  # High standards, fewer results
    STANDARD = "standard"  # Balanced filtering
    LENIENT = "lenient"  # Lower standards, more results
    CUSTOM = "custom"  # User-defined thresholds


class FilterStrategy(Enum):
    """Enumeration of filtering strategies."""

    AND = "and"  # All conditions must be met
    OR = "or"  # Any condition can be met
    WEIGHTED = "weighted"  # Weighted combination of conditions
    THRESHOLD = "threshold"  # Simple threshold-based filtering


@dataclass
class FilterConfiguration:
    """
    Configuration for function call filtering with configurable thresholds.

    This class defines all filtering parameters and thresholds for different
    quality metrics and call characteristics.
    """

    # Core thresholds
    min_confidence: float = 0.6  # Minimum confidence score (0.0-1.0)
    min_weight: float = 0.3  # Minimum weight score
    min_effective_weight: float = 0.4  # Minimum effective weight (weight * frequency_factor)

    # Frequency-based filtering
    min_frequency_in_file: int = 1  # Minimum call frequency per file
    max_frequency_in_file: int = 100  # Maximum call frequency per file (noise reduction)
    frequency_factor_threshold: float = 0.5  # Minimum frequency factor

    # Call type filtering
    allowed_call_types: set[CallType] = field(default_factory=lambda: set(CallType))
    excluded_call_types: set[CallType] = field(default_factory=set)

    # Quality-based filtering
    require_type_hints: bool = False  # Whether to require type hints
    require_docstring: bool = False  # Whether to require docstrings
    allow_syntax_errors: bool = True  # Whether to allow calls with syntax errors
    max_syntax_error_severity: str = "error"  # Maximum allowed syntax error severity

    # Context-based filtering
    allow_conditional_calls: bool = True  # Whether to allow conditional calls
    allow_nested_calls: bool = True  # Whether to allow nested calls
    allow_recursive_calls: bool = True  # Whether to allow recursive calls
    allow_cross_module_calls: bool = True  # Whether to allow cross-module calls

    # Breadcrumb quality filtering
    require_source_breadcrumb: bool = True  # Whether to require source breadcrumb
    require_target_breadcrumb: bool = True  # Whether to require target breadcrumb
    min_breadcrumb_depth: int = 1  # Minimum breadcrumb depth
    max_breadcrumb_depth: int = 10  # Maximum breadcrumb depth

    # Expression quality filtering
    require_call_expression: bool = True  # Whether to require call expression
    min_expression_length: int = 3  # Minimum call expression length
    max_expression_length: int = 500  # Maximum call expression length

    # Advanced filtering options
    filter_mode: FilterMode = FilterMode.STANDARD  # Overall filtering mode
    filter_strategy: FilterStrategy = FilterStrategy.AND  # How to combine filters
    custom_filters: list[Callable[[FunctionCall], bool]] = field(default_factory=list)

    # Statistical filtering
    use_statistical_outlier_removal: bool = False  # Whether to remove statistical outliers
    outlier_z_score_threshold: float = 3.0  # Z-score threshold for outliers

    def __post_init__(self):
        """Initialize and validate configuration."""
        # Set default allowed call types if none specified
        if not self.allowed_call_types:
            self.allowed_call_types = set(CallType)

        # Remove excluded types from allowed types
        self.allowed_call_types -= self.excluded_call_types

        # Validate thresholds
        self._validate_thresholds()

    def _validate_thresholds(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be 0.0-1.0, got {self.min_confidence}")

        if self.min_weight < 0.0:
            raise ValueError(f"min_weight must be non-negative, got {self.min_weight}")

        if self.min_effective_weight < 0.0:
            raise ValueError(f"min_effective_weight must be non-negative, got {self.min_effective_weight}")

        if self.min_frequency_in_file < 0:
            raise ValueError(f"min_frequency_in_file must be non-negative, got {self.min_frequency_in_file}")

        if self.max_frequency_in_file < self.min_frequency_in_file:
            raise ValueError("max_frequency_in_file must be >= min_frequency_in_file")

        if self.min_breadcrumb_depth < 0:
            raise ValueError(f"min_breadcrumb_depth must be non-negative, got {self.min_breadcrumb_depth}")

        if self.max_breadcrumb_depth < self.min_breadcrumb_depth:
            raise ValueError("max_breadcrumb_depth must be >= min_breadcrumb_depth")

    def create_strict_config(self) -> "FilterConfiguration":
        """Create a strict filtering configuration."""
        config = FilterConfiguration()
        config.min_confidence = 0.8
        config.min_weight = 0.5
        config.min_effective_weight = 0.6
        config.require_type_hints = True
        config.require_docstring = True
        config.allow_syntax_errors = False
        config.allow_conditional_calls = False
        config.allow_nested_calls = False
        config.min_breadcrumb_depth = 2
        config.filter_mode = FilterMode.STRICT
        return config

    def create_lenient_config(self) -> "FilterConfiguration":
        """Create a lenient filtering configuration."""
        config = FilterConfiguration()
        config.min_confidence = 0.3
        config.min_weight = 0.1
        config.min_effective_weight = 0.2
        config.require_type_hints = False
        config.require_docstring = False
        config.allow_syntax_errors = True
        config.allow_conditional_calls = True
        config.allow_nested_calls = True
        config.allow_recursive_calls = True
        config.min_breadcrumb_depth = 0
        config.filter_mode = FilterMode.LENIENT
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "min_confidence": self.min_confidence,
            "min_weight": self.min_weight,
            "min_effective_weight": self.min_effective_weight,
            "min_frequency_in_file": self.min_frequency_in_file,
            "max_frequency_in_file": self.max_frequency_in_file,
            "frequency_factor_threshold": self.frequency_factor_threshold,
            "allowed_call_types": [ct.value for ct in self.allowed_call_types],
            "excluded_call_types": [ct.value for ct in self.excluded_call_types],
            "require_type_hints": self.require_type_hints,
            "require_docstring": self.require_docstring,
            "allow_syntax_errors": self.allow_syntax_errors,
            "max_syntax_error_severity": self.max_syntax_error_severity,
            "allow_conditional_calls": self.allow_conditional_calls,
            "allow_nested_calls": self.allow_nested_calls,
            "allow_recursive_calls": self.allow_recursive_calls,
            "allow_cross_module_calls": self.allow_cross_module_calls,
            "require_source_breadcrumb": self.require_source_breadcrumb,
            "require_target_breadcrumb": self.require_target_breadcrumb,
            "min_breadcrumb_depth": self.min_breadcrumb_depth,
            "max_breadcrumb_depth": self.max_breadcrumb_depth,
            "require_call_expression": self.require_call_expression,
            "min_expression_length": self.min_expression_length,
            "max_expression_length": self.max_expression_length,
            "filter_mode": self.filter_mode.value,
            "filter_strategy": self.filter_strategy.value,
            "use_statistical_outlier_removal": self.use_statistical_outlier_removal,
            "outlier_z_score_threshold": self.outlier_z_score_threshold,
        }


@dataclass
class FilterResult:
    """
    Result of filtering operation with detailed statistics.

    This provides transparency into what was filtered and why.
    """

    # Filtered results
    filtered_calls: list[FunctionCall]  # Calls that passed all filters
    rejected_calls: list[FunctionCall]  # Calls that were filtered out

    # Filter statistics
    total_input_calls: int  # Total calls before filtering
    total_output_calls: int  # Total calls after filtering
    filter_efficiency: float  # Percentage of calls that passed

    # Rejection reasons
    rejection_reasons: dict[str, int]  # Reason -> count of rejections
    rejection_by_call_type: dict[str, int]  # Call type -> rejection count

    # Quality metrics
    avg_confidence_before: float  # Average confidence before filtering
    avg_confidence_after: float  # Average confidence after filtering
    avg_weight_before: float  # Average weight before filtering
    avg_weight_after: float  # Average weight after filtering

    # Performance metrics
    filtering_time_ms: float  # Time taken for filtering

    def get_filter_summary(self) -> dict[str, Any]:
        """Get a summary of filtering results."""
        return {
            "input_calls": self.total_input_calls,
            "output_calls": self.total_output_calls,
            "filter_efficiency": f"{self.filter_efficiency:.1f}%",
            "quality_improvement": {
                "confidence_improvement": f"{self.avg_confidence_after - self.avg_confidence_before:.3f}",
                "weight_improvement": f"{self.avg_weight_after - self.avg_weight_before:.3f}",
            },
            "top_rejection_reasons": dict(sorted(self.rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5]),
            "filtering_time_ms": self.filtering_time_ms,
        }


class CallFilterService:
    """
    Service for filtering function calls based on configurable criteria.

    This service provides comprehensive filtering capabilities including:
    - Confidence and weight thresholds
    - Call type filtering
    - Quality-based filtering
    - Context-based filtering
    - Statistical outlier removal
    - Custom filter functions
    """

    def __init__(self, config: FilterConfiguration | None = None):
        """
        Initialize the filter service with configuration.

        Args:
            config: Filter configuration. If None, uses default configuration.
        """
        self.config = config or FilterConfiguration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def filter_calls(self, calls: list[FunctionCall]) -> FilterResult:
        """
        Filter function calls based on configured criteria.

        Args:
            calls: List of function calls to filter

        Returns:
            FilterResult with filtered calls and statistics
        """
        import time

        start_time = time.time()

        if not calls:
            return FilterResult(
                filtered_calls=[],
                rejected_calls=[],
                total_input_calls=0,
                total_output_calls=0,
                filter_efficiency=0.0,
                rejection_reasons={},
                rejection_by_call_type={},
                avg_confidence_before=0.0,
                avg_confidence_after=0.0,
                avg_weight_before=0.0,
                avg_weight_after=0.0,
                filtering_time_ms=0.0,
            )

        # Calculate initial metrics
        initial_avg_confidence = sum(call.confidence for call in calls) / len(calls)
        initial_avg_weight = sum(call.weight for call in calls) / len(calls)

        # Track rejection reasons
        rejection_reasons = {}
        rejection_by_call_type = {}

        # Apply filters based on strategy
        if self.config.filter_strategy == FilterStrategy.AND:
            filtered_calls, rejected_calls = self._filter_with_and_strategy(calls, rejection_reasons, rejection_by_call_type)
        elif self.config.filter_strategy == FilterStrategy.OR:
            filtered_calls, rejected_calls = self._filter_with_or_strategy(calls, rejection_reasons, rejection_by_call_type)
        elif self.config.filter_strategy == FilterStrategy.WEIGHTED:
            filtered_calls, rejected_calls = self._filter_with_weighted_strategy(calls, rejection_reasons, rejection_by_call_type)
        else:  # THRESHOLD
            filtered_calls, rejected_calls = self._filter_with_threshold_strategy(calls, rejection_reasons, rejection_by_call_type)

        # Apply statistical outlier removal if configured
        if self.config.use_statistical_outlier_removal:
            filtered_calls, outlier_rejected = self._remove_statistical_outliers(filtered_calls)
            rejected_calls.extend(outlier_rejected)
            for call in outlier_rejected:
                rejection_reasons["statistical_outlier"] = rejection_reasons.get("statistical_outlier", 0) + 1

        # Apply custom filters
        if self.config.custom_filters:
            for custom_filter in self.config.custom_filters:
                custom_filtered = []
                custom_rejected = []
                for call in filtered_calls:
                    try:
                        if custom_filter(call):
                            custom_filtered.append(call)
                        else:
                            custom_rejected.append(call)
                            rejection_reasons["custom_filter"] = rejection_reasons.get("custom_filter", 0) + 1
                    except Exception as e:
                        self.logger.warning(f"Custom filter error for call {call.call_expression}: {e}")
                        custom_rejected.append(call)
                        rejection_reasons["custom_filter_error"] = rejection_reasons.get("custom_filter_error", 0) + 1

                filtered_calls = custom_filtered
                rejected_calls.extend(custom_rejected)

        # Calculate final metrics
        final_avg_confidence = sum(call.confidence for call in filtered_calls) / len(filtered_calls) if filtered_calls else 0.0
        final_avg_weight = sum(call.weight for call in filtered_calls) / len(filtered_calls) if filtered_calls else 0.0

        filter_efficiency = (len(filtered_calls) / len(calls)) * 100
        filtering_time = (time.time() - start_time) * 1000

        result = FilterResult(
            filtered_calls=filtered_calls,
            rejected_calls=rejected_calls,
            total_input_calls=len(calls),
            total_output_calls=len(filtered_calls),
            filter_efficiency=filter_efficiency,
            rejection_reasons=rejection_reasons,
            rejection_by_call_type=rejection_by_call_type,
            avg_confidence_before=initial_avg_confidence,
            avg_confidence_after=final_avg_confidence,
            avg_weight_before=initial_avg_weight,
            avg_weight_after=final_avg_weight,
            filtering_time_ms=filtering_time,
        )

        self.logger.info(
            f"Filtered {len(calls)} calls -> {len(filtered_calls)} calls "
            f"({filter_efficiency:.1f}% efficiency) in {filtering_time:.1f}ms"
        )

        return result

    def _filter_with_and_strategy(
        self, calls: list[FunctionCall], rejection_reasons: dict[str, int], rejection_by_call_type: dict[str, int]
    ) -> tuple[list[FunctionCall], list[FunctionCall]]:
        """Filter using AND strategy - all conditions must be met."""
        filtered_calls = []
        rejected_calls = []

        for call in calls:
            passed_all_filters = True
            rejection_reason = None

            # Apply each filter
            if not self._check_confidence_threshold(call):
                passed_all_filters = False
                rejection_reason = "low_confidence"
            elif not self._check_weight_threshold(call):
                passed_all_filters = False
                rejection_reason = "low_weight"
            elif not self._check_call_type_filter(call):
                passed_all_filters = False
                rejection_reason = "excluded_call_type"
            elif not self._check_quality_filters(call):
                passed_all_filters = False
                rejection_reason = "quality_requirements"
            elif not self._check_context_filters(call):
                passed_all_filters = False
                rejection_reason = "context_restrictions"
            elif not self._check_breadcrumb_filters(call):
                passed_all_filters = False
                rejection_reason = "breadcrumb_requirements"
            elif not self._check_expression_filters(call):
                passed_all_filters = False
                rejection_reason = "expression_requirements"
            elif not self._check_frequency_filters(call):
                passed_all_filters = False
                rejection_reason = "frequency_threshold"

            if passed_all_filters:
                filtered_calls.append(call)
            else:
                rejected_calls.append(call)
                rejection_reasons[rejection_reason] = rejection_reasons.get(rejection_reason, 0) + 1
                call_type = call.call_type.value
                rejection_by_call_type[call_type] = rejection_by_call_type.get(call_type, 0) + 1

        return filtered_calls, rejected_calls

    def _filter_with_or_strategy(
        self, calls: list[FunctionCall], rejection_reasons: dict[str, int], rejection_by_call_type: dict[str, int]
    ) -> tuple[list[FunctionCall], list[FunctionCall]]:
        """Filter using OR strategy - any condition can be met."""
        filtered_calls = []
        rejected_calls = []

        for call in calls:
            # Check if call passes any of the filters
            passed_any_filter = (
                self._check_confidence_threshold(call)
                or self._check_weight_threshold(call)
                or (self._check_call_type_filter(call) and self._check_quality_filters(call))
            )

            if passed_any_filter:
                filtered_calls.append(call)
            else:
                rejected_calls.append(call)
                rejection_reasons["failed_all_conditions"] = rejection_reasons.get("failed_all_conditions", 0) + 1
                call_type = call.call_type.value
                rejection_by_call_type[call_type] = rejection_by_call_type.get(call_type, 0) + 1

        return filtered_calls, rejected_calls

    def _filter_with_weighted_strategy(
        self, calls: list[FunctionCall], rejection_reasons: dict[str, int], rejection_by_call_type: dict[str, int]
    ) -> tuple[list[FunctionCall], list[FunctionCall]]:
        """Filter using weighted strategy - combine multiple factors."""
        filtered_calls = []
        rejected_calls = []

        for call in calls:
            # Calculate weighted score
            score = 0.0
            max_score = 0.0

            # Confidence factor (weight: 0.4)
            if call.confidence >= self.config.min_confidence:
                score += 0.4 * call.confidence
            max_score += 0.4

            # Weight factor (weight: 0.3)
            if call.effective_weight >= self.config.min_effective_weight:
                score += 0.3 * min(1.0, call.effective_weight)
            max_score += 0.3

            # Quality factor (weight: 0.2)
            quality_score = 0.0
            if not call.has_syntax_errors:
                quality_score += 0.5
            if call.has_type_hints:
                quality_score += 0.3
            if call.has_docstring:
                quality_score += 0.2
            score += 0.2 * quality_score
            max_score += 0.2

            # Context factor (weight: 0.1)
            context_score = 1.0
            if call.is_conditional:
                context_score -= 0.2
            if call.is_nested:
                context_score -= 0.1
            if call.has_syntax_errors:
                context_score -= 0.5
            score += 0.1 * max(0.0, context_score)
            max_score += 0.1

            # Normalize score
            normalized_score = score / max_score if max_score > 0 else 0.0

            # Apply threshold (default: 0.6 for weighted strategy)
            threshold = 0.6
            if normalized_score >= threshold:
                filtered_calls.append(call)
            else:
                rejected_calls.append(call)
                rejection_reasons["low_weighted_score"] = rejection_reasons.get("low_weighted_score", 0) + 1
                call_type = call.call_type.value
                rejection_by_call_type[call_type] = rejection_by_call_type.get(call_type, 0) + 1

        return filtered_calls, rejected_calls

    def _filter_with_threshold_strategy(
        self, calls: list[FunctionCall], rejection_reasons: dict[str, int], rejection_by_call_type: dict[str, int]
    ) -> tuple[list[FunctionCall], list[FunctionCall]]:
        """Filter using simple threshold strategy."""
        filtered_calls = []
        rejected_calls = []

        for call in calls:
            # Simple threshold checks
            if (
                call.confidence >= self.config.min_confidence
                and call.effective_weight >= self.config.min_effective_weight
                and call.call_type in self.config.allowed_call_types
            ):
                filtered_calls.append(call)
            else:
                rejected_calls.append(call)
                if call.confidence < self.config.min_confidence:
                    rejection_reasons["confidence_threshold"] = rejection_reasons.get("confidence_threshold", 0) + 1
                elif call.effective_weight < self.config.min_effective_weight:
                    rejection_reasons["weight_threshold"] = rejection_reasons.get("weight_threshold", 0) + 1
                else:
                    rejection_reasons["call_type_filter"] = rejection_reasons.get("call_type_filter", 0) + 1

                call_type = call.call_type.value
                rejection_by_call_type[call_type] = rejection_by_call_type.get(call_type, 0) + 1

        return filtered_calls, rejected_calls

    def _check_confidence_threshold(self, call: FunctionCall) -> bool:
        """Check if call meets confidence threshold."""
        return call.confidence >= self.config.min_confidence

    def _check_weight_threshold(self, call: FunctionCall) -> bool:
        """Check if call meets weight thresholds."""
        return call.weight >= self.config.min_weight and call.effective_weight >= self.config.min_effective_weight

    def _check_call_type_filter(self, call: FunctionCall) -> bool:
        """Check if call type is allowed."""
        return call.call_type in self.config.allowed_call_types

    def _check_quality_filters(self, call: FunctionCall) -> bool:
        """Check if call meets quality requirements."""
        if self.config.require_type_hints and not call.has_type_hints:
            return False

        if self.config.require_docstring and not call.has_docstring:
            return False

        if not self.config.allow_syntax_errors and call.has_syntax_errors:
            return False

        return True

    def _check_context_filters(self, call: FunctionCall) -> bool:
        """Check if call meets context requirements."""
        if not self.config.allow_conditional_calls and call.is_conditional:
            return False

        if not self.config.allow_nested_calls and call.is_nested:
            return False

        if not self.config.allow_recursive_calls and call.is_recursive_call():
            return False

        if not self.config.allow_cross_module_calls and call.is_cross_module_call():
            return False

        return True

    def _check_breadcrumb_filters(self, call: FunctionCall) -> bool:
        """Check if call meets breadcrumb requirements."""
        if self.config.require_source_breadcrumb and not call.source_breadcrumb:
            return False

        if self.config.require_target_breadcrumb and not call.target_breadcrumb:
            return False

        # Check breadcrumb depth
        source_depth = call.get_call_depth()
        target_depth = call.get_target_depth()

        if source_depth < self.config.min_breadcrumb_depth or source_depth > self.config.max_breadcrumb_depth:
            return False

        if target_depth < self.config.min_breadcrumb_depth or target_depth > self.config.max_breadcrumb_depth:
            return False

        return True

    def _check_expression_filters(self, call: FunctionCall) -> bool:
        """Check if call meets expression requirements."""
        if self.config.require_call_expression and not call.call_expression:
            return False

        if call.call_expression:
            expr_len = len(call.call_expression)
            if expr_len < self.config.min_expression_length or expr_len > self.config.max_expression_length:
                return False

        return True

    def _check_frequency_filters(self, call: FunctionCall) -> bool:
        """Check if call meets frequency requirements."""
        freq = call.frequency_in_file
        if freq < self.config.min_frequency_in_file or freq > self.config.max_frequency_in_file:
            return False

        if call.frequency_factor < self.config.frequency_factor_threshold:
            return False

        return True

    def _remove_statistical_outliers(self, calls: list[FunctionCall]) -> tuple[list[FunctionCall], list[FunctionCall]]:
        """Remove statistical outliers based on Z-score."""
        if len(calls) < 3:  # Need at least 3 points for meaningful statistics
            return calls, []

        # Calculate statistics for confidence and weight
        confidences = [call.confidence for call in calls]
        weights = [call.effective_weight for call in calls]

        # Calculate means and standard deviations
        conf_mean = sum(confidences) / len(confidences)
        weight_mean = sum(weights) / len(weights)

        conf_std = (sum((c - conf_mean) ** 2 for c in confidences) / len(confidences)) ** 0.5
        weight_std = (sum((w - weight_mean) ** 2 for w in weights) / len(weights)) ** 0.5

        if conf_std == 0 and weight_std == 0:
            return calls, []  # No variance, no outliers

        # Identify outliers
        filtered_calls = []
        outlier_calls = []

        for call in calls:
            is_outlier = False

            # Check confidence outlier
            if conf_std > 0:
                conf_z_score = abs(call.confidence - conf_mean) / conf_std
                if conf_z_score > self.config.outlier_z_score_threshold:
                    is_outlier = True

            # Check weight outlier
            if weight_std > 0:
                weight_z_score = abs(call.effective_weight - weight_mean) / weight_std
                if weight_z_score > self.config.outlier_z_score_threshold:
                    is_outlier = True

            if is_outlier:
                outlier_calls.append(call)
            else:
                filtered_calls.append(call)

        return filtered_calls, outlier_calls

    def update_configuration(self, new_config: FilterConfiguration) -> None:
        """Update the filter configuration."""
        self.config = new_config
        self.logger.info("Updated filter configuration")

    def get_configuration(self) -> FilterConfiguration:
        """Get the current filter configuration."""
        return self.config

    def add_custom_filter(self, filter_func: Callable[[FunctionCall], bool]) -> None:
        """Add a custom filter function."""
        self.config.custom_filters.append(filter_func)
        self.logger.info("Added custom filter function")

    def remove_custom_filters(self) -> None:
        """Remove all custom filter functions."""
        self.config.custom_filters.clear()
        self.logger.info("Removed all custom filter functions")


# Predefined filter configurations


def create_high_quality_filter_config() -> FilterConfiguration:
    """Create a filter configuration optimized for high quality calls."""
    config = FilterConfiguration()
    config.min_confidence = 0.8
    config.min_weight = 0.6
    config.min_effective_weight = 0.7
    config.require_type_hints = True
    config.require_docstring = True
    config.allow_syntax_errors = False
    config.allow_conditional_calls = False
    config.min_breadcrumb_depth = 2
    config.filter_mode = FilterMode.STRICT
    config.filter_strategy = FilterStrategy.AND
    return config


def create_production_filter_config() -> FilterConfiguration:
    """Create a filter configuration suitable for production use."""
    config = FilterConfiguration()
    config.min_confidence = 0.7
    config.min_weight = 0.4
    config.min_effective_weight = 0.5
    config.allow_syntax_errors = False
    config.allow_recursive_calls = False
    config.min_breadcrumb_depth = 1
    config.filter_mode = FilterMode.STANDARD
    config.filter_strategy = FilterStrategy.WEIGHTED
    config.use_statistical_outlier_removal = True
    return config


def create_exploratory_filter_config() -> FilterConfiguration:
    """Create a filter configuration for exploratory analysis."""
    config = FilterConfiguration()
    config.min_confidence = 0.4
    config.min_weight = 0.2
    config.min_effective_weight = 0.3
    config.allow_syntax_errors = True
    config.allow_conditional_calls = True
    config.allow_nested_calls = True
    config.allow_recursive_calls = True
    config.min_breadcrumb_depth = 0
    config.filter_mode = FilterMode.LENIENT
    config.filter_strategy = FilterStrategy.OR
    return config


# Utility functions for filtering


def create_call_type_filter(allowed_types: list[CallType]) -> Callable[[FunctionCall], bool]:
    """Create a custom filter for specific call types."""
    allowed_set = set(allowed_types)
    return lambda call: call.call_type in allowed_set


def create_pattern_filter(required_patterns: list[str]) -> Callable[[FunctionCall], bool]:
    """Create a custom filter for specific patterns."""
    pattern_set = set(required_patterns)
    return lambda call: call.pattern_matched in pattern_set


def create_file_filter(allowed_files: list[str]) -> Callable[[FunctionCall], bool]:
    """Create a custom filter for specific files."""
    allowed_set = set(allowed_files)
    return lambda call: call.file_path in allowed_set


def create_breadcrumb_filter(required_substrings: list[str]) -> Callable[[FunctionCall], bool]:
    """Create a custom filter for breadcrumbs containing specific substrings."""

    def filter_func(call: FunctionCall) -> bool:
        return any(substring in call.target_breadcrumb for substring in required_substrings)

    return filter_func
