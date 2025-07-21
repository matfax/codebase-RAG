"""
Function Call Extractor Service for AST-based call detection.

This service integrates Wave 1.0's Tree-sitter patterns with Wave 2.0's data models
to extract function call relationships from code ASTs. It provides comprehensive
function call detection with confidence scoring and breadcrumb integration.
"""

import hashlib
import logging
import time
from typing import Optional

import tree_sitter
from tree_sitter import Node

from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, CallType, FunctionCall
from src.services.ast_extraction_service import AstExtractionService
from src.services.call_confidence_scorer import CallConfidenceScorer
from src.services.call_frequency_analyzer import CallFrequencyAnalyzer
from src.services.call_weight_calculator_service import CallWeightCalculator
from src.utils.python_call_patterns import PythonCallPatterns
from src.utils.tree_sitter_manager import TreeSitterManager


class FunctionCallExtractor:
    """
    Service for extracting function call relationships from AST nodes.

    This service uses Tree-sitter query patterns from Wave 1.0 to detect function calls
    in code and creates FunctionCall data models from Wave 2.0 with comprehensive metadata.
    """

    def __init__(self):
        """Initialize the function call extractor."""
        self.logger = logging.getLogger(__name__)

        # Initialize dependencies
        self.tree_sitter_manager = TreeSitterManager()
        self.ast_extraction_service = AstExtractionService()
        self.weight_calculator = CallWeightCalculator()
        self.confidence_scorer = CallConfidenceScorer()
        self.frequency_analyzer = CallFrequencyAnalyzer()

        # Cache for compiled queries
        self._compiled_queries: dict[str, tree_sitter.Query] = {}

        # Configuration
        self.enable_async_patterns = True
        self.enable_advanced_patterns = True
        self.min_confidence_threshold = 0.1  # Very low threshold, filtering done elsewhere

        self.logger.info("FunctionCallExtractor initialized")

    async def extract_calls_from_chunk(self, chunk: CodeChunk, source_breadcrumb: str, content_lines: list[str]) -> CallDetectionResult:
        """
        Extract function calls from a code chunk.

        Args:
            chunk: Code chunk to analyze
            source_breadcrumb: Breadcrumb path of the source function/class
            content_lines: Lines of the source file for context

        Returns:
            CallDetectionResult containing all detected function calls
        """
        start_time = time.time()

        try:
            # Parse the chunk content to get AST
            parser = self.tree_sitter_manager.get_parser(chunk.language)
            if not parser:
                return self._create_error_result(chunk.file_path, f"No parser available for language: {chunk.language}", start_time)

            # Parse chunk content
            tree = parser.parse(chunk.content.encode("utf-8"))
            if not tree.root_node:
                return self._create_error_result(chunk.file_path, "Failed to parse chunk content", start_time)

            # Extract calls based on language
            if chunk.language == "python":
                calls = await self._extract_python_calls(tree.root_node, chunk, source_breadcrumb, content_lines)
            else:
                # Future: Add support for other languages
                calls = []

            # Calculate frequency factors
            calls_with_frequency = self.frequency_analyzer.calculate_frequency_factors(calls)

            # Create result
            processing_time = (time.time() - start_time) * 1000
            pattern_matches = self._count_pattern_matches(calls_with_frequency)

            return CallDetectionResult(
                calls=calls_with_frequency,
                file_path=chunk.file_path,
                detection_success=True,
                processing_time_ms=processing_time,
                pattern_matches=pattern_matches,
                total_call_expressions=len(calls_with_frequency),
            )

        except Exception as e:
            self.logger.error(f"Error extracting calls from chunk {chunk.chunk_id}: {e}")
            return self._create_error_result(chunk.file_path, f"Exception during extraction: {str(e)}", start_time)

    async def extract_calls_from_file(self, file_path: str, content: str, language: str, chunks: list[CodeChunk]) -> CallDetectionResult:
        """
        Extract function calls from an entire file.

        Args:
            file_path: Path to the source file
            content: File content
            language: Programming language
            chunks: Pre-extracted code chunks from the file

        Returns:
            CallDetectionResult containing all detected function calls
        """
        start_time = time.time()
        content_lines = content.split("\n")
        all_calls = []

        try:
            # Extract calls from each chunk
            for chunk in chunks:
                if chunk.chunk_type.value in ["function", "method", "class"]:
                    # Build source breadcrumb from chunk metadata
                    source_breadcrumb = self._build_source_breadcrumb(chunk)

                    # Extract calls from this chunk
                    chunk_result = await self.extract_calls_from_chunk(chunk, source_breadcrumb, content_lines)

                    if chunk_result.detection_success:
                        all_calls.extend(chunk_result.calls)

            # Calculate global frequency factors
            all_calls_with_frequency = self.frequency_analyzer.calculate_frequency_factors(all_calls)

            # Create final result
            processing_time = (time.time() - start_time) * 1000
            pattern_matches = self._count_pattern_matches(all_calls_with_frequency)

            return CallDetectionResult(
                calls=all_calls_with_frequency,
                file_path=file_path,
                detection_success=True,
                processing_time_ms=processing_time,
                pattern_matches=pattern_matches,
                total_call_expressions=len(all_calls_with_frequency),
            )

        except Exception as e:
            self.logger.error(f"Error extracting calls from file {file_path}: {e}")
            return self._create_error_result(file_path, f"Exception during file extraction: {str(e)}", start_time)

    async def _extract_python_calls(
        self, root_node: Node, chunk: CodeChunk, source_breadcrumb: str, content_lines: list[str]
    ) -> list[FunctionCall]:
        """
        Extract Python function calls using Tree-sitter patterns from Wave 1.0.

        Args:
            root_node: Root AST node of the chunk
            chunk: Source code chunk
            source_breadcrumb: Breadcrumb of the calling function
            content_lines: Lines of the source file

        Returns:
            List of detected function calls
        """
        calls = []

        # Get appropriate patterns based on configuration
        if self.enable_advanced_patterns:
            patterns = PythonCallPatterns.get_all_patterns()
        else:
            patterns = PythonCallPatterns.get_basic_patterns()

        # Process each pattern type
        for pattern_name, pattern_query in patterns.items():
            try:
                # Get or compile the query
                query = self._get_compiled_query(pattern_query, "python")
                if not query:
                    continue

                # Execute query on the AST using matches (not captures)
                matches = query.matches(root_node)

                # Process each match
                for match_id, captures_dict in matches:
                    # Process each capture in this match
                    for capture_name, nodes in captures_dict.items():
                        # Each capture can have multiple nodes
                        for node in nodes:
                            call = await self._create_function_call_from_capture(
                                node, capture_name, pattern_name, chunk, source_breadcrumb, content_lines
                            )

                            if call and call.confidence >= self.min_confidence_threshold:
                                calls.append(call)

            except Exception as e:
                self.logger.debug(f"Error processing pattern {pattern_name}: {e}")
                continue

        return calls

    async def _create_function_call_from_capture(
        self, node: Node, capture_name: str, pattern_name: str, chunk: CodeChunk, source_breadcrumb: str, content_lines: list[str]
    ) -> FunctionCall | None:
        """
        Create a FunctionCall object from a Tree-sitter capture.

        Args:
            node: Captured AST node
            capture_name: Name of the capture from the query
            pattern_name: Name of the pattern that matched
            chunk: Source code chunk
            source_breadcrumb: Breadcrumb of the calling function
            content_lines: Lines of the source file

        Returns:
            FunctionCall object or None if creation failed
        """
        try:
            # Determine call type from pattern name
            call_type = self._map_pattern_to_call_type(pattern_name)

            # Extract call expression
            call_expression = node.text.decode("utf-8") if node.text else ""

            # Extract target breadcrumb (placeholder - will be resolved in Task 3.2)
            target_breadcrumb = self._extract_target_breadcrumb_placeholder(node, call_expression, pattern_name)

            # Calculate line number (Tree-sitter is 0-indexed)
            line_number = node.start_point[0] + 1 + chunk.start_line - 1

            # Create a temporary FunctionCall object for confidence calculation
            temp_call = FunctionCall(
                source_breadcrumb=source_breadcrumb,
                target_breadcrumb=target_breadcrumb,
                call_type=call_type,
                line_number=line_number,
                file_path=chunk.file_path,
                confidence=0.5,  # Temporary value
                weight=1.0,  # Temporary value
                call_expression=call_expression,
                arguments_count=self._count_arguments(node),
            )

            # Calculate confidence score with AST context
            ast_context = {"node": node, "content_lines": content_lines, "line_number": line_number}
            confidence_score, confidence_analysis = self.confidence_scorer.calculate_confidence(temp_call, ast_context)

            # Extract additional metadata
            argument_count = self._count_arguments(node)
            is_conditional = self._is_in_conditional_block(node)
            is_nested = self._is_nested_call(node)

            # Create content hash
            content_hash = hashlib.md5(call_expression.encode("utf-8")).hexdigest()[:8]

            # Determine type hints and docstring presence
            has_type_hints = self._has_type_hints(node, content_lines)
            has_docstring = self._has_docstring(node, content_lines)
            has_syntax_errors = self.ast_extraction_service.count_errors(node) > 0

            # Create FunctionCall object with temporary weight
            function_call = FunctionCall(
                source_breadcrumb=source_breadcrumb,
                target_breadcrumb=target_breadcrumb,
                call_type=call_type,
                line_number=line_number,
                file_path=chunk.file_path,
                confidence=confidence_score,
                weight=1.0,  # Temporary weight
                call_expression=call_expression,
                arguments_count=argument_count,
                is_conditional=is_conditional,
                is_nested=is_nested,
                ast_node_type=node.type,
                pattern_matched=pattern_name,
                content_hash=content_hash,
                has_type_hints=has_type_hints,
                has_docstring=has_docstring,
                has_syntax_errors=has_syntax_errors,
            )

            # Calculate final weight using the FunctionCall object
            final_weight = self.weight_calculator.calculate_weight(function_call)

            # Update the function call with final weight
            function_call.weight = final_weight

            return function_call

        except Exception as e:
            self.logger.debug(f"Error creating function call from capture: {e}")
            return None

    def _get_compiled_query(self, pattern_query: str, language: str) -> tree_sitter.Query | None:
        """Get or compile a Tree-sitter query."""
        cache_key = f"{language}:{hashlib.md5(pattern_query.encode()).hexdigest()[:8]}"

        if cache_key in self._compiled_queries:
            return self._compiled_queries[cache_key]

        try:
            parser = self.tree_sitter_manager.get_parser(language)
            if not parser:
                return None

            query = parser.language.query(pattern_query)
            self._compiled_queries[cache_key] = query
            return query

        except Exception as e:
            self.logger.debug(f"Error compiling query: {e}")
            return None

    def _map_pattern_to_call_type(self, pattern_name: str) -> CallType:
        """Map Tree-sitter pattern names to CallType enum values."""
        mapping = {
            "direct_function_call": CallType.DIRECT,
            "method_call": CallType.METHOD,
            "self_method_call": CallType.SELF_METHOD,
            "chained_attribute_call": CallType.ATTRIBUTE,
            "module_function_call": CallType.MODULE_FUNCTION,
            "subscript_method_call": CallType.SUBSCRIPT_METHOD,
            "super_method_call": CallType.SUPER_METHOD,
            "class_method_call": CallType.CLASS_METHOD,
            "dynamic_attribute_call": CallType.DYNAMIC,
            "unpacking_call": CallType.UNPACKING,
            "await_function_call": CallType.ASYNC,
            "await_method_call": CallType.ASYNC_METHOD,
            "await_self_method_call": CallType.ASYNC_METHOD,
            "await_chained_call": CallType.ASYNC_METHOD,
            "asyncio_gather_call": CallType.ASYNCIO,
            "asyncio_create_task_call": CallType.ASYNCIO,
            "asyncio_run_call": CallType.ASYNCIO,
            "asyncio_wait_call": CallType.ASYNCIO,
            "asyncio_wait_for_call": CallType.ASYNCIO,
            "asyncio_generic_call": CallType.ASYNCIO,
            "await_asyncio_call": CallType.ASYNCIO,
        }

        return mapping.get(pattern_name, CallType.DIRECT)

    def _extract_target_breadcrumb_placeholder(self, node: Node, call_expression: str, pattern_name: str) -> str:
        """
        Extract a placeholder target breadcrumb from the call expression.

        This will be replaced with proper breadcrumb resolution in Task 3.2.
        """
        # Simple extraction for now - will be enhanced in Task 3.2
        if "." in call_expression:
            # Method or attribute call
            parts = call_expression.split("(")[0].split(".")
            return ".".join(parts)
        else:
            # Direct function call
            func_name = call_expression.split("(")[0].strip()
            return func_name

    def _count_arguments(self, node: Node) -> int:
        """Count the number of arguments in a function call."""
        # Look for argument_list child
        for child in node.children:
            if child.type == "argument_list":
                # Count comma-separated arguments (simplified)
                if child.text:
                    arg_text = child.text.decode("utf-8")
                    # Simple heuristic: count commas + 1, but handle empty args
                    if arg_text.strip() in ["()", "(,)"]:
                        return 0
                    return arg_text.count(",") + 1 if arg_text.strip() != "()" else 0
        return 0

    def _is_in_conditional_block(self, node: Node) -> bool:
        """Check if the call is inside a conditional block."""
        # Traverse up the AST to find conditional parents
        current = node.parent
        while current:
            if current.type in [
                "if_statement",
                "elif_clause",
                "else_clause",
                "try_statement",
                "except_clause",
                "for_statement",
                "while_statement",
                "with_statement",
            ]:
                return True
            current = current.parent
        return False

    def _is_nested_call(self, node: Node) -> bool:
        """Check if this call is nested within another call."""
        current = node.parent
        while current:
            if current.type == "call":
                return True
            current = current.parent
        return False

    def _has_type_hints(self, node: Node, content_lines: list[str]) -> bool:
        """Check if the surrounding code has type hints."""
        # Simple heuristic: look for type annotations in nearby lines
        line_start = max(0, node.start_point[0] - 2)
        line_end = min(len(content_lines), node.end_point[0] + 3)

        for i in range(line_start, line_end):
            if i < len(content_lines):
                line = content_lines[i]
                if ":" in line and ("->" in line or "List[" in line or "Dict[" in line):
                    return True
        return False

    def _has_docstring(self, node: Node, content_lines: list[str]) -> bool:
        """Check if the surrounding code has docstrings."""
        # Look for docstring patterns in nearby lines
        line_start = max(0, node.start_point[0] - 5)
        line_end = min(len(content_lines), node.end_point[0] + 2)

        for i in range(line_start, line_end):
            if i < len(content_lines):
                line = content_lines[i].strip()
                if line.startswith('"""') or line.startswith("'''"):
                    return True
        return False

    def _build_source_breadcrumb(self, chunk: CodeChunk) -> str:
        """Build a breadcrumb path from chunk metadata."""
        if chunk.parent_name and chunk.name:
            return f"{chunk.parent_name}.{chunk.name}"
        elif chunk.name:
            return chunk.name
        else:
            return f"unknown_{chunk.chunk_type.value}"

    def _count_pattern_matches(self, calls: list[FunctionCall]) -> dict[str, int]:
        """Count matches per pattern type."""
        pattern_counts = {}
        for call in calls:
            pattern = call.pattern_matched
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        return pattern_counts

    def _create_error_result(self, file_path: str, error_message: str, start_time: float) -> CallDetectionResult:
        """Create an error result for failed extraction."""
        processing_time = (time.time() - start_time) * 1000

        return CallDetectionResult(
            calls=[], file_path=file_path, detection_success=False, processing_time_ms=processing_time, error_count=1
        )

    def configure(self, enable_async_patterns: bool = True, enable_advanced_patterns: bool = True, min_confidence_threshold: float = 0.1):
        """Configure the extractor behavior."""
        self.enable_async_patterns = enable_async_patterns
        self.enable_advanced_patterns = enable_advanced_patterns
        self.min_confidence_threshold = min_confidence_threshold

        self.logger.info(
            f"FunctionCallExtractor configured: async={enable_async_patterns}, "
            f"advanced={enable_advanced_patterns}, min_confidence={min_confidence_threshold}"
        )

    def clear_query_cache(self):
        """Clear the compiled query cache."""
        self._compiled_queries.clear()
        self.logger.info("Function call extractor query cache cleared")

    def get_statistics(self) -> dict:
        """Get extractor statistics."""
        return {
            "compiled_queries_count": len(self._compiled_queries),
            "configuration": {
                "enable_async_patterns": self.enable_async_patterns,
                "enable_advanced_patterns": self.enable_advanced_patterns,
                "min_confidence_threshold": self.min_confidence_threshold,
            },
        }
