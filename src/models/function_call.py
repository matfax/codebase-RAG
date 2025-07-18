"""
Data models for function call relationships and detection metadata.

This module contains data models that represent function call relationships
detected through AST analysis, including weight calculations and confidence scoring.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class CallType(Enum):
    """Enumeration of different function call types for weight calculation."""

    # Direct function calls: function_name()
    DIRECT = "direct"

    # Method calls: object.method()
    METHOD = "method"

    # Attribute chain calls: obj.attr.method()
    ATTRIBUTE = "attribute"

    # Self method calls: self.method()
    SELF_METHOD = "self_method"

    # Async function calls: await function()
    ASYNC = "async"

    # Async method calls: await obj.method()
    ASYNC_METHOD = "async_method"

    # Asyncio library calls: asyncio.gather(), asyncio.create_task()
    ASYNCIO = "asyncio"

    # Super method calls: super().method()
    SUPER_METHOD = "super_method"

    # Class method calls: cls.method() or Class.method()
    CLASS_METHOD = "class_method"

    # Dynamic calls: getattr(obj, 'method')()
    DYNAMIC = "dynamic"

    # Calls with unpacking: func(*args, **kwargs)
    UNPACKING = "unpacking"

    # Module function calls: module.function()
    MODULE_FUNCTION = "module_function"

    # Subscript method calls: obj[key].method()
    SUBSCRIPT_METHOD = "subscript_method"


@dataclass
class FunctionCall:
    """
    Represents a function call relationship detected through AST analysis.

    This model captures the source and target of function calls with metadata
    for weight calculation, confidence scoring, and relationship filtering.
    """

    # Core relationship identification
    source_breadcrumb: str  # Full breadcrumb path of calling function (e.g., "module.class.method")
    target_breadcrumb: str  # Full breadcrumb path of called function (e.g., "module.utils.helper")
    call_type: CallType  # Type of function call (direct, method, attribute, etc.)

    # Location information
    line_number: int  # Line number where the call occurs
    file_path: str  # Source file containing the call

    # Weight and confidence scoring
    confidence: float  # Confidence score (0.0-1.0) based on AST completeness and context
    weight: float  # Calculated weight based on call type and frequency

    # Call context and analysis metadata
    call_expression: str  # Raw call expression text (e.g., "self.progress_tracker.set_total_items(100)")
    arguments_count: int = 0  # Number of arguments in the call
    is_conditional: bool = False  # Whether call is inside conditional block (if/try/etc.)
    is_nested: bool = False  # Whether call is nested within another call

    # Frequency and pattern analysis
    frequency_in_file: int = 1  # How many times this specific call appears in the file
    frequency_factor: float = 1.0  # Frequency-based weight multiplier

    # AST node information for debugging and validation
    ast_node_type: str = ""  # Tree-sitter node type that matched the pattern
    pattern_matched: str = ""  # Name of the Tree-sitter pattern that detected this call

    # Processing metadata
    detected_at: datetime = None  # When this call was detected
    content_hash: str = ""  # Hash of the surrounding code context for change detection

    # Additional context for confidence calculation
    has_type_hints: bool = False  # Whether the call target has type annotations
    has_docstring: bool = False  # Whether the call target has documentation
    has_syntax_errors: bool = False  # Whether there were syntax errors around the call
    error_details: str = ""  # Details about any syntax errors

    def __post_init__(self):
        """Initialize default values and validate constraints."""
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()

        # Validate confidence score range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Validate weight is positive
        if self.weight < 0.0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")

        # Validate frequency factor
        if self.frequency_factor < 0.0:
            raise ValueError(f"Frequency factor must be non-negative, got {self.frequency_factor}")

    @property
    def effective_weight(self) -> float:
        """Calculate the effective weight including frequency factor."""
        return self.weight * self.frequency_factor

    @property
    def weighted_confidence(self) -> float:
        """Calculate confidence weighted by effective weight."""
        return self.confidence * self.effective_weight

    @property
    def is_high_confidence(self) -> bool:
        """Check if this call has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    @property
    def is_low_confidence(self) -> bool:
        """Check if this call has low confidence (<= 0.3)."""
        return self.confidence <= 0.3

    @property
    def call_context_score(self) -> float:
        """
        Calculate a context quality score based on available metadata.

        Returns:
            Score from 0.0 to 1.0 based on context richness
        """
        score = 0.0

        # Base score for having breadcrumbs
        if self.source_breadcrumb and self.target_breadcrumb:
            score += 0.3

        # Score for type hints and documentation
        if self.has_type_hints:
            score += 0.2
        if self.has_docstring:
            score += 0.2

        # Score for clean syntax
        if not self.has_syntax_errors:
            score += 0.2

        # Score for call expression quality
        if self.call_expression and len(self.call_expression.strip()) > 0:
            score += 0.1

        return min(score, 1.0)

    def get_call_depth(self) -> int:
        """
        Calculate the depth of the call based on source breadcrumb.

        Returns:
            Number of levels in the source breadcrumb hierarchy
        """
        if not self.source_breadcrumb:
            return 0

        # Count separators to determine depth
        dot_count = self.source_breadcrumb.count(".")
        double_colon_count = self.source_breadcrumb.count("::")
        return max(dot_count, double_colon_count) + 1

    def get_target_depth(self) -> int:
        """
        Calculate the depth of the target based on target breadcrumb.

        Returns:
            Number of levels in the target breadcrumb hierarchy
        """
        if not self.target_breadcrumb:
            return 0

        # Count separators to determine depth
        dot_count = self.target_breadcrumb.count(".")
        double_colon_count = self.target_breadcrumb.count("::")
        return max(dot_count, double_colon_count) + 1

    def is_cross_module_call(self) -> bool:
        """
        Check if this is a cross-module function call.

        Returns:
            True if source and target are in different modules
        """
        if not self.source_breadcrumb or not self.target_breadcrumb:
            return False

        source_components = self.source_breadcrumb.split(".")
        target_components = self.target_breadcrumb.split(".")

        # Compare first component (module level)
        if len(source_components) > 0 and len(target_components) > 0:
            return source_components[0] != target_components[0]

        return False

    def is_recursive_call(self) -> bool:
        """
        Check if this appears to be a recursive function call.

        Returns:
            True if source and target breadcrumbs are the same
        """
        return self.source_breadcrumb == self.target_breadcrumb

    def get_call_relationship_type(self) -> str:
        """
        Determine the type of relationship this call represents.

        Returns:
            String describing the relationship type
        """
        if self.is_recursive_call():
            return "recursive"
        elif self.is_cross_module_call():
            return "cross_module"
        elif self.call_type in [CallType.SELF_METHOD, CallType.SUPER_METHOD]:
            return "inheritance"
        elif self.call_type in [CallType.ASYNC, CallType.ASYNC_METHOD, CallType.ASYNCIO]:
            return "async"
        elif self.call_type == CallType.DYNAMIC:
            return "dynamic"
        else:
            return "standard"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the FunctionCall to a dictionary for serialization.

        This is used for storing in databases and API responses.
        """
        return {
            "source_breadcrumb": self.source_breadcrumb,
            "target_breadcrumb": self.target_breadcrumb,
            "call_type": self.call_type.value,
            "line_number": self.line_number,
            "file_path": self.file_path,
            "confidence": self.confidence,
            "weight": self.weight,
            "call_expression": self.call_expression,
            "arguments_count": self.arguments_count,
            "is_conditional": self.is_conditional,
            "is_nested": self.is_nested,
            "frequency_in_file": self.frequency_in_file,
            "frequency_factor": self.frequency_factor,
            "ast_node_type": self.ast_node_type,
            "pattern_matched": self.pattern_matched,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "content_hash": self.content_hash,
            "has_type_hints": self.has_type_hints,
            "has_docstring": self.has_docstring,
            "has_syntax_errors": self.has_syntax_errors,
            "error_details": self.error_details,
            # Computed properties
            "effective_weight": self.effective_weight,
            "weighted_confidence": self.weighted_confidence,
            "is_high_confidence": self.is_high_confidence,
            "is_low_confidence": self.is_low_confidence,
            "call_context_score": self.call_context_score,
            "call_depth": self.get_call_depth(),
            "target_depth": self.get_target_depth(),
            "is_cross_module_call": self.is_cross_module_call(),
            "is_recursive_call": self.is_recursive_call(),
            "call_relationship_type": self.get_call_relationship_type(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionCall":
        """
        Create a FunctionCall instance from a dictionary.

        This is used for deserializing from databases and API requests.
        """
        # Convert string call_type back to enum
        call_type = CallType(data["call_type"])

        # Convert ISO datetime string back to datetime object
        detected_at = None
        if data.get("detected_at"):
            detected_at = datetime.fromisoformat(data["detected_at"])

        return cls(
            source_breadcrumb=data["source_breadcrumb"],
            target_breadcrumb=data["target_breadcrumb"],
            call_type=call_type,
            line_number=data["line_number"],
            file_path=data["file_path"],
            confidence=data["confidence"],
            weight=data["weight"],
            call_expression=data.get("call_expression", ""),
            arguments_count=data.get("arguments_count", 0),
            is_conditional=data.get("is_conditional", False),
            is_nested=data.get("is_nested", False),
            frequency_in_file=data.get("frequency_in_file", 1),
            frequency_factor=data.get("frequency_factor", 1.0),
            ast_node_type=data.get("ast_node_type", ""),
            pattern_matched=data.get("pattern_matched", ""),
            detected_at=detected_at,
            content_hash=data.get("content_hash", ""),
            has_type_hints=data.get("has_type_hints", False),
            has_docstring=data.get("has_docstring", False),
            has_syntax_errors=data.get("has_syntax_errors", False),
            error_details=data.get("error_details", ""),
        )


@dataclass
class CallDetectionResult:
    """
    Result of function call detection for a specific file or code chunk.

    This contains all detected function calls and related metadata.
    """

    calls: list[FunctionCall]  # All detected function calls
    file_path: str  # Source file path
    detection_success: bool  # Whether detection completed successfully
    processing_time_ms: float = 0.0  # Time taken for call detection
    pattern_matches: dict[str, int] = None  # Count of matches per pattern type
    error_count: int = 0  # Number of errors encountered
    total_call_expressions: int = 0  # Total call expressions found (before filtering)

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.pattern_matches is None:
            self.pattern_matches = {}

    @property
    def call_count(self) -> int:
        """Get the total number of detected calls."""
        return len(self.calls)

    @property
    def high_confidence_calls(self) -> list[FunctionCall]:
        """Get all high-confidence function calls."""
        return [call for call in self.calls if call.is_high_confidence]

    @property
    def low_confidence_calls(self) -> list[FunctionCall]:
        """Get all low-confidence function calls."""
        return [call for call in self.calls if call.is_low_confidence]

    @property
    def call_types_distribution(self) -> dict[str, int]:
        """Get distribution of call types."""
        distribution = {}
        for call in self.calls:
            call_type = call.call_type.value
            distribution[call_type] = distribution.get(call_type, 0) + 1
        return distribution

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all calls."""
        if not self.calls:
            return 0.0
        return sum(call.confidence for call in self.calls) / len(self.calls)

    @property
    def average_weight(self) -> float:
        """Calculate average weight across all calls."""
        if not self.calls:
            return 0.0
        return sum(call.weight for call in self.calls) / len(self.calls)

    def get_calls_by_type(self, call_type: CallType) -> list[FunctionCall]:
        """Get all calls of a specific type."""
        return [call for call in self.calls if call.call_type == call_type]

    def get_calls_by_confidence_range(self, min_confidence: float, max_confidence: float) -> list[FunctionCall]:
        """Get calls within a specific confidence range."""
        return [call for call in self.calls if min_confidence <= call.confidence <= max_confidence]

    def filter_calls_by_weight(self, min_weight: float) -> list[FunctionCall]:
        """Filter calls by minimum effective weight."""
        return [call for call in self.calls if call.effective_weight >= min_weight]

    def to_dict(self) -> dict[str, Any]:
        """Convert CallDetectionResult to dictionary for logging and debugging."""
        return {
            "file_path": self.file_path,
            "detection_success": self.detection_success,
            "call_count": self.call_count,
            "processing_time_ms": self.processing_time_ms,
            "pattern_matches": self.pattern_matches,
            "error_count": self.error_count,
            "total_call_expressions": self.total_call_expressions,
            "call_types_distribution": self.call_types_distribution,
            "average_confidence": self.average_confidence,
            "average_weight": self.average_weight,
            "high_confidence_count": len(self.high_confidence_calls),
            "low_confidence_count": len(self.low_confidence_calls),
            "calls": [call.to_dict() for call in self.calls],
        }


# Utility functions for working with function calls


def group_calls_by_target(calls: list[FunctionCall]) -> dict[str, list[FunctionCall]]:
    """
    Group function calls by their target breadcrumb.

    Args:
        calls: List of function calls to group

    Returns:
        Dictionary mapping target breadcrumbs to lists of calls
    """
    grouped = {}
    for call in calls:
        target = call.target_breadcrumb
        if target not in grouped:
            grouped[target] = []
        grouped[target].append(call)
    return grouped


def group_calls_by_source(calls: list[FunctionCall]) -> dict[str, list[FunctionCall]]:
    """
    Group function calls by their source breadcrumb.

    Args:
        calls: List of function calls to group

    Returns:
        Dictionary mapping source breadcrumbs to lists of calls
    """
    grouped = {}
    for call in calls:
        source = call.source_breadcrumb
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(call)
    return grouped


def calculate_call_frequency_map(calls: list[FunctionCall]) -> dict[str, int]:
    """
    Calculate frequency map for target functions.

    Args:
        calls: List of function calls

    Returns:
        Dictionary mapping target breadcrumbs to call frequencies
    """
    frequency_map = {}
    for call in calls:
        target = call.target_breadcrumb
        frequency_map[target] = frequency_map.get(target, 0) + 1
    return frequency_map


def filter_calls_by_confidence_threshold(calls: list[FunctionCall], threshold: float) -> list[FunctionCall]:
    """
    Filter function calls by minimum confidence threshold.

    Args:
        calls: List of function calls to filter
        threshold: Minimum confidence threshold (0.0-1.0)

    Returns:
        Filtered list of function calls
    """
    return [call for call in calls if call.confidence >= threshold]


def get_call_statistics(calls: list[FunctionCall]) -> dict[str, Any]:
    """
    Generate comprehensive statistics for a list of function calls.

    Args:
        calls: List of function calls to analyze

    Returns:
        Dictionary containing various statistics
    """
    if not calls:
        return {"total_calls": 0}

    return {
        "total_calls": len(calls),
        "unique_targets": len({call.target_breadcrumb for call in calls}),
        "unique_sources": len({call.source_breadcrumb for call in calls}),
        "call_types": {call_type.value: len([c for c in calls if c.call_type == call_type]) for call_type in CallType},
        "confidence_stats": {
            "min": min(call.confidence for call in calls),
            "max": max(call.confidence for call in calls),
            "avg": sum(call.confidence for call in calls) / len(calls),
        },
        "weight_stats": {
            "min": min(call.weight for call in calls),
            "max": max(call.weight for call in calls),
            "avg": sum(call.weight for call in calls) / len(calls),
        },
        "high_confidence_count": len([c for c in calls if c.is_high_confidence]),
        "low_confidence_count": len([c for c in calls if c.is_low_confidence]),
        "cross_module_calls": len([c for c in calls if c.is_cross_module_call()]),
        "recursive_calls": len([c for c in calls if c.is_recursive_call()]),
    }
