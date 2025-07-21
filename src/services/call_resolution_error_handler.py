"""
Call Resolution Error Handler Service.

This service provides comprehensive error handling for function call resolution
failures, implementing confidence degradation strategies and fallback mechanisms
to ensure graceful handling of unresolvable calls.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.models.function_call import FunctionCall
from src.services.breadcrumb_resolver_service import BreadcrumbCandidate, BreadcrumbResolutionResult


class ResolutionErrorType(Enum):
    """Types of resolution errors that can occur."""

    INVALID_SYNTAX = "invalid_syntax"
    MISSING_IMPORTS = "missing_imports"
    UNKNOWN_MODULE = "unknown_module"
    UNKNOWN_FUNCTION = "unknown_function"
    AMBIGUOUS_RESOLUTION = "ambiguous_resolution"
    CROSS_FILE_FAILURE = "cross_file_failure"
    ATTRIBUTE_CHAIN_FAILURE = "attribute_chain_failure"
    SEARCH_TIMEOUT = "search_timeout"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CONFIDENCE_TOO_LOW = "confidence_too_low"


@dataclass
class ResolutionError:
    """Information about a resolution error."""

    error_type: ResolutionErrorType
    error_message: str
    original_confidence: float
    degraded_confidence: float
    suggested_fallback: str | None
    recovery_strategy: str
    context_info: dict[str, Any]


@dataclass
class FallbackResolution:
    """A fallback resolution for unresolvable calls."""

    breadcrumb: str
    confidence: float
    reasoning: str
    fallback_type: str
    error_details: str


class CallResolutionErrorHandler:
    """
    Service for handling function call resolution errors with confidence degradation.

    This service implements various strategies for dealing with unresolvable calls,
    including confidence degradation, fallback resolution, and error recovery.
    """

    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)

        # Configuration for confidence degradation
        self.degradation_factors = {
            ResolutionErrorType.INVALID_SYNTAX: 0.1,
            ResolutionErrorType.MISSING_IMPORTS: 0.3,
            ResolutionErrorType.UNKNOWN_MODULE: 0.4,
            ResolutionErrorType.UNKNOWN_FUNCTION: 0.5,
            ResolutionErrorType.AMBIGUOUS_RESOLUTION: 0.6,
            ResolutionErrorType.CROSS_FILE_FAILURE: 0.4,
            ResolutionErrorType.ATTRIBUTE_CHAIN_FAILURE: 0.3,
            ResolutionErrorType.SEARCH_TIMEOUT: 0.7,
            ResolutionErrorType.INSUFFICIENT_CONTEXT: 0.5,
            ResolutionErrorType.CONFIDENCE_TOO_LOW: 0.2,
        }

        # Minimum confidence thresholds
        self.min_fallback_confidence = 0.1
        self.enable_aggressive_fallbacks = True
        self.enable_error_recovery = True

        # Error statistics
        self.error_stats = {error_type: 0 for error_type in ResolutionErrorType}

        self.logger.info("CallResolutionErrorHandler initialized")

    async def handle_resolution_failure(
        self,
        function_call: FunctionCall,
        error_type: ResolutionErrorType,
        error_message: str,
        original_result: BreadcrumbResolutionResult | None = None,
        context: dict[str, Any] = None,
    ) -> BreadcrumbResolutionResult:
        """
        Handle a function call resolution failure with appropriate error handling.

        Args:
            function_call: FunctionCall that failed to resolve
            error_type: Type of resolution error
            error_message: Detailed error message
            original_result: Original resolution result (if any)
            context: Additional context information

        Returns:
            BreadcrumbResolutionResult with error handling applied
        """
        start_time = time.time()

        try:
            # Update error statistics
            self.error_stats[error_type] += 1

            # Create resolution error info
            resolution_error = self._create_resolution_error(function_call, error_type, error_message, context or {})

            # Attempt error recovery if enabled
            if self.enable_error_recovery:
                recovery_result = await self._attempt_error_recovery(function_call, resolution_error, original_result)
                if recovery_result and recovery_result.success:
                    return recovery_result

            # Generate fallback resolution
            fallback_resolution = self._generate_fallback_resolution(function_call, resolution_error)

            # Create the final result with degraded confidence
            if fallback_resolution:
                candidate = BreadcrumbCandidate(
                    breadcrumb=fallback_resolution.breadcrumb,
                    confidence_score=fallback_resolution.confidence,
                    source_chunk=None,
                    reasoning=fallback_resolution.reasoning,
                    match_type="fallback",
                    file_path=function_call.file_path,
                    line_start=function_call.line_number,
                    line_end=function_call.line_number,
                    chunk_type="unknown",
                    language="python",
                )

                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=True,
                    primary_candidate=candidate,
                    error_message=f"Fallback resolution: {fallback_resolution.error_details}",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )
            else:
                # Complete failure - return error result
                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=False,
                    error_message=f"Resolution failed: {error_message}",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )

        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression,
                success=False,
                error_message=f"Error handling failed: {str(e)}",
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

    def _create_resolution_error(
        self, function_call: FunctionCall, error_type: ResolutionErrorType, error_message: str, context: dict[str, Any]
    ) -> ResolutionError:
        """
        Create a ResolutionError object with degraded confidence.

        Args:
            function_call: Original function call
            error_type: Type of error
            error_message: Error message
            context: Additional context

        Returns:
            ResolutionError object
        """
        original_confidence = function_call.confidence
        degradation_factor = self.degradation_factors.get(error_type, 0.5)
        degraded_confidence = max(original_confidence * degradation_factor, self.min_fallback_confidence)

        # Generate suggested fallback
        suggested_fallback = self._suggest_fallback_breadcrumb(function_call, error_type)

        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error_type)

        return ResolutionError(
            error_type=error_type,
            error_message=error_message,
            original_confidence=original_confidence,
            degraded_confidence=degraded_confidence,
            suggested_fallback=suggested_fallback,
            recovery_strategy=recovery_strategy,
            context_info=context,
        )

    def _suggest_fallback_breadcrumb(self, function_call: FunctionCall, error_type: ResolutionErrorType) -> str | None:
        """
        Suggest a fallback breadcrumb based on the error type and call pattern.

        Args:
            function_call: Function call that failed
            error_type: Type of resolution error

        Returns:
            Suggested fallback breadcrumb or None
        """
        call_expr = function_call.call_expression

        # Extract function name for fallback
        if "(" in call_expr:
            func_part = call_expr.split("(")[0].strip()
        else:
            func_part = call_expr.strip()

        if error_type == ResolutionErrorType.UNKNOWN_MODULE:
            # For unknown modules, create a generic module.function breadcrumb
            if "." in func_part:
                return func_part  # Use as-is
            else:
                return f"unknown_module.{func_part}"

        elif error_type == ResolutionErrorType.UNKNOWN_FUNCTION:
            # For unknown functions, use the function name with unknown context
            if "." in func_part:
                function_name = func_part.split(".")[-1]
                return f"unknown_class.{function_name}"
            else:
                return f"unknown_module.{func_part}"

        elif error_type == ResolutionErrorType.ATTRIBUTE_CHAIN_FAILURE:
            # For attribute chain failures, use partial resolution
            if "." in func_part:
                parts = func_part.split(".")
                if len(parts) >= 2:
                    return f"{parts[0]}.unknown_attribute.{parts[-1]}"
            return f"unknown_object.{func_part}"

        elif error_type == ResolutionErrorType.CROSS_FILE_FAILURE:
            # For cross-file failures, assume it's in a related module
            if "." in func_part:
                return func_part
            else:
                return f"external_module.{func_part}"

        elif error_type == ResolutionErrorType.MISSING_IMPORTS:
            # For missing imports, suggest standard library or common modules
            if "." in func_part:
                module_part = func_part.split(".")[0]
                function_part = func_part.split(".")[-1]

                # Check for common module patterns
                if module_part in ["os", "sys", "json", "re", "time", "math"]:
                    return func_part  # Standard library
                else:
                    return f"imported_module.{function_part}"
            else:
                return f"imported_module.{func_part}"

        # Default fallback
        if "." in func_part:
            return func_part
        else:
            return f"unknown.{func_part}"

    def _determine_recovery_strategy(self, error_type: ResolutionErrorType) -> str:
        """
        Determine the appropriate recovery strategy for an error type.

        Args:
            error_type: Type of resolution error

        Returns:
            Recovery strategy description
        """
        strategies = {
            ResolutionErrorType.INVALID_SYNTAX: "syntax_correction",
            ResolutionErrorType.MISSING_IMPORTS: "import_analysis",
            ResolutionErrorType.UNKNOWN_MODULE: "module_search",
            ResolutionErrorType.UNKNOWN_FUNCTION: "function_search",
            ResolutionErrorType.AMBIGUOUS_RESOLUTION: "disambiguation",
            ResolutionErrorType.CROSS_FILE_FAILURE: "cross_file_retry",
            ResolutionErrorType.ATTRIBUTE_CHAIN_FAILURE: "partial_chain_resolution",
            ResolutionErrorType.SEARCH_TIMEOUT: "retry_with_timeout",
            ResolutionErrorType.INSUFFICIENT_CONTEXT: "context_expansion",
            ResolutionErrorType.CONFIDENCE_TOO_LOW: "threshold_adjustment",
        }

        return strategies.get(error_type, "generic_fallback")

    async def _attempt_error_recovery(
        self, function_call: FunctionCall, resolution_error: ResolutionError, original_result: BreadcrumbResolutionResult | None
    ) -> BreadcrumbResolutionResult | None:
        """
        Attempt to recover from a resolution error using specific strategies.

        Args:
            function_call: Original function call
            resolution_error: Error information
            original_result: Original resolution result

        Returns:
            Recovered result or None if recovery fails
        """
        error_type = resolution_error.error_type

        try:
            if error_type == ResolutionErrorType.AMBIGUOUS_RESOLUTION:
                return await self._handle_ambiguous_resolution(function_call, original_result)

            elif error_type == ResolutionErrorType.CONFIDENCE_TOO_LOW:
                return await self._handle_low_confidence(function_call, original_result)

            elif error_type == ResolutionErrorType.SEARCH_TIMEOUT:
                return await self._handle_search_timeout(function_call)

            elif error_type == ResolutionErrorType.INSUFFICIENT_CONTEXT:
                return await self._handle_insufficient_context(function_call)

            # Add more recovery strategies as needed

        except Exception as e:
            self.logger.debug(f"Error recovery failed: {e}")

        return None

    async def _handle_ambiguous_resolution(
        self, function_call: FunctionCall, original_result: BreadcrumbResolutionResult | None
    ) -> BreadcrumbResolutionResult | None:
        """
        Handle ambiguous resolution by selecting the best candidate.

        Args:
            function_call: Function call
            original_result: Original result with multiple candidates

        Returns:
            Result with best candidate or None
        """
        if not original_result or not original_result.alternative_candidates:
            return None

        # Select the candidate with highest confidence
        all_candidates = [original_result.primary_candidate] if original_result.primary_candidate else []
        all_candidates.extend(original_result.alternative_candidates)

        best_candidate = max(all_candidates, key=lambda c: c.confidence_score)

        # Apply slight confidence degradation for ambiguity
        best_candidate.confidence_score *= 0.9
        best_candidate.reasoning += " (disambiguation applied)"

        return BreadcrumbResolutionResult(
            query=function_call.call_expression,
            success=True,
            primary_candidate=best_candidate,
            error_message="Resolved ambiguity by selecting highest confidence candidate",
        )

    async def _handle_low_confidence(
        self, function_call: FunctionCall, original_result: BreadcrumbResolutionResult | None
    ) -> BreadcrumbResolutionResult | None:
        """
        Handle low confidence by applying threshold adjustment.

        Args:
            function_call: Function call
            original_result: Original low-confidence result

        Returns:
            Result with adjusted threshold or None
        """
        if not original_result or not original_result.primary_candidate:
            return None

        candidate = original_result.primary_candidate

        # If confidence is close to threshold, allow it with degradation
        if candidate.confidence_score >= self.min_fallback_confidence:
            candidate.confidence_score *= 0.8  # Apply degradation
            candidate.reasoning += " (low confidence accepted with degradation)"
            candidate.match_type = "low_confidence"

            return BreadcrumbResolutionResult(
                query=function_call.call_expression,
                success=True,
                primary_candidate=candidate,
                error_message="Low confidence result accepted with degradation",
            )

        return None

    async def _handle_search_timeout(self, function_call: FunctionCall) -> BreadcrumbResolutionResult | None:
        """
        Handle search timeout by creating a timeout fallback.

        Args:
            function_call: Function call

        Returns:
            Timeout fallback result
        """
        # Create a simple fallback based on the call expression
        call_expr = function_call.call_expression
        if "(" in call_expr:
            func_part = call_expr.split("(")[0].strip()
        else:
            func_part = call_expr.strip()

        fallback_breadcrumb = func_part if "." in func_part else f"timeout.{func_part}"

        candidate = BreadcrumbCandidate(
            breadcrumb=fallback_breadcrumb,
            confidence_score=0.4,  # Moderate confidence for timeout fallback
            source_chunk=None,
            reasoning="Search timeout - using expression-based fallback",
            match_type="timeout_fallback",
            file_path=function_call.file_path,
            line_start=function_call.line_number,
            line_end=function_call.line_number,
            chunk_type="unknown",
            language="python",
        )

        return BreadcrumbResolutionResult(
            query=function_call.call_expression,
            success=True,
            primary_candidate=candidate,
            error_message="Search timeout - fallback resolution applied",
        )

    async def _handle_insufficient_context(self, function_call: FunctionCall) -> BreadcrumbResolutionResult | None:
        """
        Handle insufficient context by creating a context-based fallback.

        Args:
            function_call: Function call

        Returns:
            Context-based fallback result
        """
        # Use the source breadcrumb context to create a reasonable fallback
        source_breadcrumb = function_call.source_breadcrumb
        call_expr = function_call.call_expression

        if "(" in call_expr:
            func_part = call_expr.split("(")[0].strip()
        else:
            func_part = call_expr.strip()

        # Extract function name
        if "." in func_part:
            function_name = func_part.split(".")[-1]
        else:
            function_name = func_part

        # Create fallback based on source context
        if source_breadcrumb:
            source_parts = source_breadcrumb.split(".")
            if len(source_parts) > 1:
                fallback_breadcrumb = f"{'.'.join(source_parts[:-1])}.{function_name}"
            else:
                fallback_breadcrumb = f"{source_breadcrumb}.{function_name}"
        else:
            fallback_breadcrumb = f"context_unknown.{function_name}"

        candidate = BreadcrumbCandidate(
            breadcrumb=fallback_breadcrumb,
            confidence_score=0.3,
            source_chunk=None,
            reasoning="Insufficient context - using source-based fallback",
            match_type="context_fallback",
            file_path=function_call.file_path,
            line_start=function_call.line_number,
            line_end=function_call.line_number,
            chunk_type="unknown",
            language="python",
        )

        return BreadcrumbResolutionResult(
            query=function_call.call_expression,
            success=True,
            primary_candidate=candidate,
            error_message="Insufficient context - source-based fallback applied",
        )

    def _generate_fallback_resolution(self, function_call: FunctionCall, resolution_error: ResolutionError) -> FallbackResolution | None:
        """
        Generate a fallback resolution for an unresolvable call.

        Args:
            function_call: Function call that failed
            resolution_error: Error information

        Returns:
            FallbackResolution or None if no fallback possible
        """
        if not self.enable_aggressive_fallbacks:
            return None

        suggested_fallback = resolution_error.suggested_fallback
        if not suggested_fallback:
            return None

        # Generate reasoning for the fallback
        reasoning_parts = [
            f"Resolution failed: {resolution_error.error_message}",
            f"Error type: {resolution_error.error_type.value}",
            f"Applied confidence degradation: {resolution_error.original_confidence:.2f} -> {resolution_error.degraded_confidence:.2f}",
            f"Fallback strategy: {resolution_error.recovery_strategy}",
        ]

        return FallbackResolution(
            breadcrumb=suggested_fallback,
            confidence=resolution_error.degraded_confidence,
            reasoning=" | ".join(reasoning_parts),
            fallback_type=resolution_error.error_type.value,
            error_details=resolution_error.error_message,
        )

    def get_error_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive error statistics.

        Returns:
            Dictionary with error statistics and configuration
        """
        total_errors = sum(self.error_stats.values())

        return {
            "total_errors": total_errors,
            "error_breakdown": dict(self.error_stats),
            "error_rates": {
                error_type.value: (count / total_errors if total_errors > 0 else 0.0) for error_type, count in self.error_stats.items()
            },
            "configuration": {
                "min_fallback_confidence": self.min_fallback_confidence,
                "enable_aggressive_fallbacks": self.enable_aggressive_fallbacks,
                "enable_error_recovery": self.enable_error_recovery,
                "degradation_factors": {error_type.value: factor for error_type, factor in self.degradation_factors.items()},
            },
        }

    def reset_error_statistics(self):
        """Reset error statistics."""
        self.error_stats = {error_type: 0 for error_type in ResolutionErrorType}
        self.logger.info("Error statistics reset")

    def configure_degradation_factor(self, error_type: ResolutionErrorType, factor: float):
        """
        Configure the confidence degradation factor for a specific error type.

        Args:
            error_type: Type of error
            factor: Degradation factor (0.0 to 1.0)
        """
        if 0.0 <= factor <= 1.0:
            self.degradation_factors[error_type] = factor
            self.logger.info(f"Updated degradation factor for {error_type.value}: {factor}")
        else:
            raise ValueError("Degradation factor must be between 0.0 and 1.0")

    def set_fallback_configuration(
        self, min_fallback_confidence: float = None, enable_aggressive_fallbacks: bool = None, enable_error_recovery: bool = None
    ):
        """
        Configure fallback behavior.

        Args:
            min_fallback_confidence: Minimum confidence for fallback results
            enable_aggressive_fallbacks: Whether to enable aggressive fallback strategies
            enable_error_recovery: Whether to enable error recovery attempts
        """
        if min_fallback_confidence is not None:
            self.min_fallback_confidence = max(0.0, min(1.0, min_fallback_confidence))

        if enable_aggressive_fallbacks is not None:
            self.enable_aggressive_fallbacks = enable_aggressive_fallbacks

        if enable_error_recovery is not None:
            self.enable_error_recovery = enable_error_recovery

        self.logger.info("Fallback configuration updated")
