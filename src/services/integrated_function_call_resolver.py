"""
Integrated Function Call Resolver Service.

This service integrates all Wave 3.0 components to provide a comprehensive
function call resolution system that combines extraction, breadcrumb resolution,
cross-file resolution, attribute chain resolution, and error handling.
"""

import logging
import time
from typing import Any, Optional, Union

from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, FunctionCall
from src.services.attribute_chain_resolver_service import AttributeChainResolver
from src.services.breadcrumb_resolver_service import BreadcrumbResolutionResult, BreadcrumbResolver
from src.services.call_resolution_error_handler import (
    CallResolutionErrorHandler,
    ResolutionErrorType,
)
from src.services.cross_file_resolver_service import (
    CrossFileResolutionContext,
    CrossFileResolver,
)
from src.services.function_call_extractor_service import FunctionCallExtractor


class IntegratedFunctionCallResolver:
    """
    Comprehensive function call resolution service integrating all Wave 3.0 components.

    This service orchestrates the entire function call detection and resolution pipeline,
    from AST extraction through breadcrumb resolution with comprehensive error handling.
    """

    def __init__(self):
        """Initialize the integrated resolver with all component services."""
        self.logger = logging.getLogger(__name__)

        # Initialize all component services
        self.function_call_extractor = FunctionCallExtractor()
        self.breadcrumb_resolver = BreadcrumbResolver()
        self.cross_file_resolver = CrossFileResolver()
        self.attribute_chain_resolver = AttributeChainResolver()
        self.error_handler = CallResolutionErrorHandler()

        # Configuration
        self.enable_cross_file_resolution = True
        self.enable_attribute_chain_resolution = True
        self.enable_error_recovery = True
        self.max_resolution_attempts = 3

        # Resolution statistics
        self.resolution_stats = {
            "total_calls_processed": 0,
            "successful_resolutions": 0,
            "cross_file_resolutions": 0,
            "attribute_chain_resolutions": 0,
            "error_recoveries": 0,
            "fallback_resolutions": 0,
            "failed_resolutions": 0,
        }

        self.logger.info("IntegratedFunctionCallResolver initialized")

    async def extract_and_resolve_calls(
        self,
        chunk: CodeChunk,
        source_breadcrumb: str,
        content_lines: list[str],
        target_projects: list[str] | None = None,
        resolution_context: CrossFileResolutionContext | None = None,
    ) -> CallDetectionResult:
        """
        Extract function calls from a chunk and resolve all target breadcrumbs.

        Args:
            chunk: Code chunk to analyze
            source_breadcrumb: Breadcrumb of the source function
            content_lines: Lines of the source file
            target_projects: Projects to search in for resolution
            resolution_context: Context for cross-file resolution

        Returns:
            CallDetectionResult with extracted calls and resolved targets
        """
        start_time = time.time()

        try:
            # Step 1: Extract function calls using FunctionCallExtractor
            extraction_result = await self.function_call_extractor.extract_calls_from_chunk(chunk, source_breadcrumb, content_lines)

            if not extraction_result.detection_success:
                return extraction_result

            # Step 2: Resolve each function call target
            resolved_calls = []
            for function_call in extraction_result.calls:
                resolved_call = await self._resolve_function_call_target(function_call, resolution_context, target_projects)
                resolved_calls.append(resolved_call)
                self.resolution_stats["total_calls_processed"] += 1

            # Step 3: Update the result with resolved calls
            extraction_result.calls = resolved_calls
            extraction_result.processing_time_ms += (time.time() - start_time) * 1000

            return extraction_result

        except Exception as e:
            self.logger.error(f"Error in integrated call extraction and resolution: {e}")
            return CallDetectionResult(
                calls=[],
                file_path=chunk.file_path,
                detection_success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
                error_count=1,
            )

    async def _resolve_function_call_target(
        self, function_call: FunctionCall, resolution_context: CrossFileResolutionContext | None, target_projects: list[str] | None
    ) -> FunctionCall:
        """
        Resolve the target breadcrumb for a single function call using all available strategies.

        Args:
            function_call: Function call to resolve
            resolution_context: Context for resolution
            target_projects: Projects to search in

        Returns:
            FunctionCall with resolved target breadcrumb
        """
        original_target = function_call.target_breadcrumb

        for attempt in range(self.max_resolution_attempts):
            try:
                # Strategy 1: Try basic breadcrumb resolution first
                basic_result = await self._attempt_basic_resolution(function_call, target_projects)
                if basic_result.success and basic_result.primary_candidate:
                    function_call.target_breadcrumb = basic_result.primary_candidate.breadcrumb
                    function_call.confidence = min(function_call.confidence, basic_result.primary_candidate.confidence_score)
                    self.resolution_stats["successful_resolutions"] += 1
                    return function_call

                # Strategy 2: Try attribute chain resolution if applicable
                if self.enable_attribute_chain_resolution and self.attribute_chain_resolver.is_attribute_chain_call(
                    function_call.call_expression
                ):
                    chain_result = await self._attempt_attribute_chain_resolution(function_call, target_projects)
                    if chain_result.success and chain_result.primary_candidate:
                        function_call.target_breadcrumb = chain_result.primary_candidate.breadcrumb
                        function_call.confidence = min(function_call.confidence, chain_result.primary_candidate.confidence_score)
                        self.resolution_stats["attribute_chain_resolutions"] += 1
                        return function_call

                # Strategy 3: Try cross-file resolution if applicable and context available
                if (
                    self.enable_cross_file_resolution
                    and resolution_context
                    and self.cross_file_resolver._is_likely_cross_file_call(function_call)
                ):
                    cross_file_result = await self._attempt_cross_file_resolution(function_call, resolution_context, target_projects)
                    if cross_file_result.success and cross_file_result.primary_candidate:
                        function_call.target_breadcrumb = cross_file_result.primary_candidate.breadcrumb
                        function_call.confidence = min(function_call.confidence, cross_file_result.primary_candidate.confidence_score)
                        self.resolution_stats["cross_file_resolutions"] += 1
                        return function_call

                # Strategy 4: If all strategies fail, apply error handling
                if attempt == self.max_resolution_attempts - 1:
                    error_result = await self._apply_error_handling(function_call, target_projects)
                    if error_result.success and error_result.primary_candidate:
                        function_call.target_breadcrumb = error_result.primary_candidate.breadcrumb
                        function_call.confidence = error_result.primary_candidate.confidence_score
                        if error_result.primary_candidate.match_type == "fallback":
                            self.resolution_stats["fallback_resolutions"] += 1
                        else:
                            self.resolution_stats["error_recoveries"] += 1
                        return function_call

            except Exception as e:
                self.logger.debug(f"Resolution attempt {attempt + 1} failed: {e}")
                continue

        # Complete failure - mark as failed and keep original target
        self.resolution_stats["failed_resolutions"] += 1
        function_call.confidence *= 0.1  # Severely degrade confidence
        return function_call

    async def _attempt_basic_resolution(self, function_call: FunctionCall, target_projects: list[str] | None) -> BreadcrumbResolutionResult:
        """
        Attempt basic breadcrumb resolution using the BreadcrumbResolver.

        Args:
            function_call: Function call to resolve
            target_projects: Projects to search in

        Returns:
            BreadcrumbResolutionResult
        """
        try:
            return await self.breadcrumb_resolver.resolve_function_call_target(function_call, target_projects)
        except Exception as e:
            self.logger.debug(f"Basic resolution failed: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression, success=False, error_message=f"Basic resolution error: {str(e)}"
            )

    async def _attempt_attribute_chain_resolution(
        self, function_call: FunctionCall, target_projects: list[str] | None
    ) -> BreadcrumbResolutionResult:
        """
        Attempt attribute chain resolution using the AttributeChainResolver.

        Args:
            function_call: Function call to resolve
            target_projects: Projects to search in

        Returns:
            BreadcrumbResolutionResult
        """
        try:
            # Create source context from function call
            source_context = {
                "breadcrumb": function_call.source_breadcrumb,
                "file_path": function_call.file_path,
                "class_name": self._extract_class_name_from_breadcrumb(function_call.source_breadcrumb),
            }

            return await self.attribute_chain_resolver.resolve_attribute_chain_call(function_call, source_context, target_projects)
        except Exception as e:
            self.logger.debug(f"Attribute chain resolution failed: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression, success=False, error_message=f"Attribute chain resolution error: {str(e)}"
            )

    async def _attempt_cross_file_resolution(
        self, function_call: FunctionCall, resolution_context: CrossFileResolutionContext, target_projects: list[str] | None
    ) -> BreadcrumbResolutionResult:
        """
        Attempt cross-file resolution using the CrossFileResolver.

        Args:
            function_call: Function call to resolve
            resolution_context: Cross-file resolution context
            target_projects: Projects to search in

        Returns:
            BreadcrumbResolutionResult
        """
        try:
            return await self.cross_file_resolver.resolve_cross_file_call(function_call, resolution_context, target_projects)
        except Exception as e:
            self.logger.debug(f"Cross-file resolution failed: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression, success=False, error_message=f"Cross-file resolution error: {str(e)}"
            )

    async def _apply_error_handling(self, function_call: FunctionCall, target_projects: list[str] | None) -> BreadcrumbResolutionResult:
        """
        Apply comprehensive error handling with fallback resolution.

        Args:
            function_call: Function call that failed to resolve
            target_projects: Projects to search in

        Returns:
            BreadcrumbResolutionResult with error handling applied
        """
        try:
            # Determine the most appropriate error type
            error_type = self._classify_resolution_failure(function_call)

            # Apply error handling
            return await self.error_handler.handle_resolution_failure(
                function_call, error_type, "All resolution strategies failed", context={"target_projects": target_projects}
            )
        except Exception as e:
            self.logger.debug(f"Error handling failed: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression, success=False, error_message=f"Error handling failed: {str(e)}"
            )

    def _extract_class_name_from_breadcrumb(self, breadcrumb: str) -> str | None:
        """
        Extract class name from a breadcrumb path.

        Args:
            breadcrumb: Breadcrumb path

        Returns:
            Class name or None
        """
        if not breadcrumb:
            return None

        parts = breadcrumb.split(".")
        # Assume class names are capitalized and method names are not
        for part in parts:
            if part and part[0].isupper():
                return part

        return None

    def _classify_resolution_failure(self, function_call: FunctionCall) -> ResolutionErrorType:
        """
        Classify the type of resolution failure based on the function call.

        Args:
            function_call: Failed function call

        Returns:
            Appropriate ResolutionErrorType
        """
        call_expr = function_call.call_expression

        # Check for syntax issues
        if not call_expr or "(" not in call_expr:
            return ResolutionErrorType.INVALID_SYNTAX

        # Check for cross-file patterns
        if self.cross_file_resolver._is_likely_cross_file_call(function_call):
            return ResolutionErrorType.CROSS_FILE_FAILURE

        # Check for attribute chains
        if self.attribute_chain_resolver.is_attribute_chain_call(call_expr):
            return ResolutionErrorType.ATTRIBUTE_CHAIN_FAILURE

        # Check for module patterns
        if "." in call_expr and not call_expr.startswith("self."):
            return ResolutionErrorType.UNKNOWN_MODULE

        # Default to unknown function
        return ResolutionErrorType.UNKNOWN_FUNCTION

    async def batch_extract_and_resolve(
        self,
        chunks: list[CodeChunk],
        file_content: str,
        target_projects: list[str] | None = None,
        resolution_context: CrossFileResolutionContext | None = None,
    ) -> list[CallDetectionResult]:
        """
        Process multiple chunks and resolve all function calls efficiently.

        Args:
            chunks: List of code chunks to process
            file_content: Complete file content
            target_projects: Projects to search in
            resolution_context: Context for cross-file resolution

        Returns:
            List of CallDetectionResult objects
        """
        content_lines = file_content.split("\n")
        results = []

        for chunk in chunks:
            if chunk.chunk_type.value in ["function", "method", "class"]:
                source_breadcrumb = self._build_source_breadcrumb(chunk)
                result = await self.extract_and_resolve_calls(chunk, source_breadcrumb, content_lines, target_projects, resolution_context)
                results.append(result)

        return results

    def _build_source_breadcrumb(self, chunk: CodeChunk) -> str:
        """Build a source breadcrumb from chunk metadata."""
        if chunk.parent_class and chunk.name:
            return f"{chunk.parent_class}.{chunk.name}"
        elif chunk.name:
            return chunk.name
        else:
            return f"unknown_{chunk.chunk_type.value}"

    def get_resolution_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive resolution statistics.

        Returns:
            Dictionary with detailed statistics
        """
        total_processed = self.resolution_stats["total_calls_processed"]

        return {
            "resolution_stats": dict(self.resolution_stats),
            "success_rate": (self.resolution_stats["successful_resolutions"] / total_processed if total_processed > 0 else 0.0),
            "error_recovery_rate": (
                (self.resolution_stats["error_recoveries"] + self.resolution_stats["fallback_resolutions"]) / total_processed
                if total_processed > 0
                else 0.0
            ),
            "component_stats": {
                "extractor": self.function_call_extractor.get_statistics(),
                "breadcrumb_resolver": self.breadcrumb_resolver.get_cache_stats(),
                "cross_file_resolver": self.cross_file_resolver.get_statistics(),
                "attribute_chain_resolver": self.attribute_chain_resolver.get_statistics(),
                "error_handler": self.error_handler.get_error_statistics(),
            },
            "configuration": {
                "enable_cross_file_resolution": self.enable_cross_file_resolution,
                "enable_attribute_chain_resolution": self.enable_attribute_chain_resolution,
                "enable_error_recovery": self.enable_error_recovery,
                "max_resolution_attempts": self.max_resolution_attempts,
            },
        }

    def reset_statistics(self):
        """Reset all resolution statistics."""
        self.resolution_stats = dict.fromkeys(self.resolution_stats, 0)
        self.error_handler.reset_error_statistics()
        self.logger.info("Resolution statistics reset")

    def configure(
        self,
        enable_cross_file_resolution: bool = None,
        enable_attribute_chain_resolution: bool = None,
        enable_error_recovery: bool = None,
        max_resolution_attempts: int = None,
    ):
        """
        Configure the integrated resolver behavior.

        Args:
            enable_cross_file_resolution: Enable cross-file resolution
            enable_attribute_chain_resolution: Enable attribute chain resolution
            enable_error_recovery: Enable error recovery
            max_resolution_attempts: Maximum resolution attempts per call
        """
        if enable_cross_file_resolution is not None:
            self.enable_cross_file_resolution = enable_cross_file_resolution

        if enable_attribute_chain_resolution is not None:
            self.enable_attribute_chain_resolution = enable_attribute_chain_resolution

        if enable_error_recovery is not None:
            self.enable_error_recovery = enable_error_recovery

        if max_resolution_attempts is not None:
            self.max_resolution_attempts = max(1, max_resolution_attempts)

        self.logger.info("IntegratedFunctionCallResolver configuration updated")

    def clear_all_caches(self):
        """Clear all caches across all component services."""
        self.function_call_extractor.clear_query_cache()
        self.breadcrumb_resolver.clear_cache()
        self.cross_file_resolver.clear_caches()
        self.attribute_chain_resolver.clear_caches()
        self.logger.info("All caches cleared")
