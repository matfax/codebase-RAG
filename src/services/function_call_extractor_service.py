"""
Function Call Extractor Service for AST-based call detection.

This service integrates Wave 1.0's Tree-sitter patterns with Wave 2.0's data models
to extract function call relationships from code ASTs. It provides comprehensive
function call detection with confidence scoring and breadcrumb integration.
"""

import hashlib
import logging
import time
from typing import Optional, Union

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

        # MCP Performance Mode - for faster processing
        self.mcp_performance_mode = True  # Enable aggressive performance optimizations for MCP tools

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

        Enhanced with deduplication logic to prevent Tree-sitter over-matching.

        Args:
            root_node: Root AST node of the chunk
            chunk: Source code chunk
            source_breadcrumb: Breadcrumb of the calling function
            content_lines: Lines of the source file

        Returns:
            List of detected function calls (deduplicated)
        """
        # Deduplication tracking: (start_line, end_line, call_expression) -> FunctionCall
        seen_calls: dict[tuple[int, int, str], FunctionCall] = {}
        total_calls_processed = 0

        # Get appropriate patterns based on configuration
        if self.mcp_performance_mode:
            # Use only the most essential patterns for MCP tools - prioritize speed
            patterns = {
                "direct_function_call": PythonCallPatterns.DIRECT_FUNCTION_CALL,
                "method_call": PythonCallPatterns.METHOD_CALL,
            }
            self.logger.debug("MCP performance mode: using minimal patterns for speed")
        elif self.enable_advanced_patterns:
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
                    # Look for the primary capture that represents the complete call
                    primary_captures = self._find_primary_captures(captures_dict, pattern_name)

                    # Process only primary captures to avoid partial matches
                    for capture_name, nodes in primary_captures.items():
                        for node in nodes:
                            # Skip if this looks like a partial capture (too short or missing parentheses)
                            if not self._is_valid_call_node(node, pattern_name):
                                continue

                            call = await self._create_function_call_from_capture(
                                node, capture_name, pattern_name, chunk, source_breadcrumb, content_lines
                            )

                            if call and call.confidence >= self.min_confidence_threshold:
                                total_calls_processed += 1

                                # Create deduplication key using more context
                                dedup_key = (call.line_number, call.call_expression.strip(), call.target_breadcrumb)

                                # Check if we've seen this exact call before
                                if dedup_key in seen_calls:
                                    existing_call = seen_calls[dedup_key]
                                    # Keep the call with higher confidence or more specific pattern
                                    if call.confidence > existing_call.confidence or self._is_more_specific_pattern(
                                        pattern_name, existing_call.pattern_matched
                                    ):
                                        seen_calls[dedup_key] = call
                                        self.logger.debug(
                                            f"Replaced duplicate call at line {call.line_number}: {existing_call.pattern_matched} -> {pattern_name}"
                                        )
                                    else:
                                        self.logger.debug(
                                            f"Skipped duplicate call at line {call.line_number}: {pattern_name} (keeping {existing_call.pattern_matched})"
                                        )
                                else:
                                    seen_calls[dedup_key] = call

            except Exception as e:
                self.logger.debug(f"Error processing pattern {pattern_name}: {e}")
                continue

        # Convert deduplicated calls to list
        deduplicated_calls = list(seen_calls.values())

        # Log deduplication results
        if total_calls_processed > len(deduplicated_calls):
            duplicates_removed = total_calls_processed - len(deduplicated_calls)
            self.logger.info(
                f"Function call deduplication: {total_calls_processed} -> {len(deduplicated_calls)} (removed {duplicates_removed} duplicates)"
            )
        elif len(deduplicated_calls) > 0:
            self.logger.debug(f"Function call extraction completed: {len(deduplicated_calls)} unique calls found")

        return deduplicated_calls

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

            # Calculate confidence score with enhanced AST context
            ast_context = {
                "node": node,
                "node_type": node.type,
                "node_text": node.text.decode("utf-8") if node.text else "",
                "start_point": node.start_point,
                "end_point": node.end_point,
                "parent_node": node.parent,
                "children_nodes": list(node.children),
                "content_lines": content_lines,
                "line_number": line_number,
                "pattern_name": pattern_name,
                "chunk_context": {
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_name": chunk.name,
                    "parent_name": chunk.parent_name,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                },
            }
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
        Extract an intelligent target breadcrumb from the call expression.

        Enhanced breadcrumb resolution for same-project function calls.
        """
        # Extract the function/method name from call expression
        if "(" in call_expression:
            func_part = call_expression.split("(")[0].strip()
        else:
            func_part = call_expression.strip()

        # Handle different call patterns
        if "." in func_part:
            # Method or attribute call - analyze the pattern
            parts = func_part.split(".")

            # Self method calls (self.method_name)
            if parts[0] == "self":
                if len(parts) >= 2:
                    # Try to infer class context from node's parent
                    class_context = self._infer_class_context_from_node(node)
                    if class_context:
                        return f"{class_context}.{parts[1]}"
                    else:
                        # Use source breadcrumb context if available
                        return f"current_class.{parts[1]}"

            # Module function calls (time.time, os.path.join)
            elif len(parts) == 2:
                module_name, func_name = parts
                # Check if it's a likely standard library call
                if self._is_likely_standard_library(module_name):
                    return func_part  # Keep as-is for stdlib calls
                else:
                    # Assume it's a same-project module
                    return f"src.{module_name}.{func_name}"

            # Chained attribute calls (obj.method.call)
            elif len(parts) > 2:
                # For complex chains, keep the last two parts
                return ".".join(parts[-2:])

        else:
            # Direct function call - assume it's in current module/project
            func_name = func_part

            # Check if it's a likely builtin function
            if self._is_likely_builtin_function(func_name):
                return func_name  # Keep as-is for builtins
            else:
                # Assume it's a same-project function
                return f"current_module.{func_name}"

        # Fallback
        return func_part

    def _infer_class_context_from_node(self, node: Node) -> str | None:
        """Infer class context by traversing up the AST."""
        current = node.parent
        max_depth = 20  # Safety limit to prevent infinite loops
        depth = 0

        while current and depth < max_depth:
            if current.type == "class_definition":
                # Look for class name in the node's children
                for child in current.children:
                    if child.type == "identifier":
                        return child.text.decode("utf-8")
            current = current.parent
            depth += 1

        return None

    def _is_likely_standard_library(self, module_name: str) -> bool:
        """Check if a module name is likely from Python standard library."""
        stdlib_modules = {
            "os",
            "sys",
            "time",
            "datetime",
            "json",
            "re",
            "math",
            "random",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "urllib",
            "http",
            "logging",
            "threading",
            "asyncio",
            "typing",
            "dataclasses",
            "abc",
            "copy",
            "pickle",
            "csv",
            "sqlite3",
            "hashlib",
            "uuid",
            "tempfile",
            "shutil",
            "glob",
            "subprocess",
            "socket",
            "ssl",
        }
        return module_name.lower() in stdlib_modules

    def _is_likely_builtin_function(self, func_name: str) -> bool:
        """Check if a function name is likely a Python builtin."""
        builtin_functions = {
            "print",
            "len",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "list",
            "dict",
            "set",
            "tuple",
            "str",
            "int",
            "float",
            "bool",
            "min",
            "max",
            "sum",
            "any",
            "all",
            "sorted",
            "reversed",
            "open",
            "input",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "dir",
            "vars",
            "globals",
            "locals",
            "eval",
            "exec",
            "compile",
            "repr",
            "format",
            "abs",
            "round",
        }
        return func_name.lower() in builtin_functions

    def _is_more_specific_pattern(self, new_pattern: str, existing_pattern: str) -> bool:
        """
        Determine if a new pattern is more specific than an existing pattern.

        More specific patterns should take precedence during deduplication.

        Args:
            new_pattern: Name of the new pattern being evaluated
            existing_pattern: Name of the existing pattern to compare against

        Returns:
            True if new_pattern is more specific than existing_pattern
        """
        # Define pattern specificity hierarchy (higher values = more specific)
        pattern_specificity = {
            # Async patterns are more specific than sync versions
            "await_self_method_call": 10,
            "await_chained_call": 9,
            "await_method_call": 8,
            "await_function_call": 7,
            "await_asyncio_call": 9,
            # Specific method types are more specific than generic ones
            "self_method_call": 8,
            "super_method_call": 8,
            "chained_attribute_call": 7,
            "subscript_method_call": 7,
            "class_method_call": 6,
            "dynamic_attribute_call": 6,
            # Asyncio-specific patterns
            "asyncio_gather_call": 9,
            "asyncio_create_task_call": 9,
            "asyncio_run_call": 9,
            "asyncio_wait_call": 9,
            "asyncio_wait_for_call": 9,
            "asyncio_generic_call": 5,
            # General patterns (less specific)
            "method_call": 5,
            "module_function_call": 4,
            "unpacking_call": 4,
            "direct_function_call": 3,
        }

        new_specificity = pattern_specificity.get(new_pattern, 1)
        existing_specificity = pattern_specificity.get(existing_pattern, 1)

        return new_specificity > existing_specificity

    def _find_primary_captures(self, captures_dict: dict, pattern_name: str) -> dict:
        """
        Find primary captures that represent complete function calls.

        Filter out partial captures like individual identifiers or argument lists.

        Args:
            captures_dict: Dictionary of capture names to node lists
            pattern_name: Name of the pattern being processed

        Returns:
            Dictionary containing only primary captures
        """
        primary_captures = {}

        # Define primary capture patterns for different query types
        primary_suffixes = [
            "call.direct",
            "call.method",
            "call.self_method",
            "call.chained",
            "call.module_function",
            "call.subscript_method",
            "call.super_method",
            "call.class_method",
            "call.dynamic",
            "call.unpacking",
            "async_call.await_function",
            "async_call.await_method",
            "async_call.await_self_method",
            "async_call.await_chained",
            "asyncio_call.gather",
            "asyncio_call.create_task",
            "asyncio_call.run",
            "asyncio_call.wait",
            "asyncio_call.wait_for",
            "asyncio_call.generic",
            "async_call.await_asyncio",
        ]

        # Look for primary capture names (these represent complete calls)
        for capture_name, nodes in captures_dict.items():
            # Include captures that end with primary suffixes
            is_primary = any(capture_name.endswith(suffix) for suffix in primary_suffixes)

            # Also include captures that represent the whole call node
            if not is_primary and pattern_name in capture_name:
                is_primary = True

            if is_primary:
                primary_captures[capture_name] = nodes

        # If no primary captures found, fall back to all captures (backwards compatibility)
        if not primary_captures:
            return captures_dict

        return primary_captures

    def _is_valid_call_node(self, node: Node, pattern_name: str) -> bool:
        """
        Check if a node represents a valid function call.

        Filter out partial matches like bare identifiers or argument lists.

        Args:
            node: Tree-sitter node to validate
            pattern_name: Pattern name for context

        Returns:
            True if node represents a valid call
        """
        if not node or not node.text:
            return False

        node_text = node.text.decode("utf-8").strip()

        # Skip empty or very short captures
        if len(node_text) < 2:
            return False

        # Skip if it's just parentheses or argument list
        if node_text.startswith("(") and node_text.endswith(")") and "(" not in node_text[1:-1]:
            return False

        # For function calls, expect either:
        # 1. Complete call with parentheses: func() or obj.method()
        # 2. Await expression: await func()
        # 3. Node type should be 'call' for most patterns
        if node.type == "call":
            return True
        elif node.type == "await" and "await" in pattern_name:
            return True
        elif "(" in node_text and ")" in node_text:
            return True
        elif node.type == "identifier" and len(node_text) <= 3:
            # Skip short identifiers that are likely partial captures
            return False

        return True

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
