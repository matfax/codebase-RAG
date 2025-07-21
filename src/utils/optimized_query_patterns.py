"""
Optimized Tree-sitter Query Patterns for Performance on Large Codebases.

This module provides performance-optimized Tree-sitter query patterns with
intelligent caching, batch processing, and adaptive optimization strategies
designed specifically for handling large codebases efficiently.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import tree_sitter
from tree_sitter import Node


class OptimizationLevel(Enum):
    """Query optimization levels for different performance requirements."""

    MINIMAL = "minimal"  # Basic patterns only
    BALANCED = "balanced"  # Standard patterns with some optimizations
    AGGRESSIVE = "aggressive"  # All patterns with full optimizations
    CUSTOM = "custom"  # User-defined optimization strategy


@dataclass
class QueryPerformanceStats:
    """Performance statistics for query execution."""

    pattern_name: str
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    matches_found: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def record_execution(self, execution_time_ms: float, matches_count: int, was_cached: bool):
        """Record a query execution."""
        self.execution_count += 1
        self.total_execution_time_ms += execution_time_ms
        self.average_execution_time_ms = self.total_execution_time_ms / self.execution_count
        self.matches_found += matches_count

        if was_cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


@dataclass
class OptimizedQueryConfig:
    """Configuration for optimized query execution."""

    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    enable_query_caching: bool = True
    enable_result_caching: bool = True
    cache_size_limit: int = 1000
    enable_performance_monitoring: bool = True
    enable_adaptive_optimization: bool = True
    batch_size: int = 100
    timeout_ms: float = 5000.0  # 5 second timeout per query

    # Pattern-specific optimizations
    use_focused_patterns: bool = True  # Use specific patterns instead of broad ones
    enable_early_termination: bool = True  # Stop when sufficient matches found
    max_matches_per_query: int = 1000  # Limit matches to prevent performance issues
    enable_depth_limiting: bool = True  # Limit AST traversal depth
    max_traversal_depth: int = 50  # Maximum AST depth to traverse

    @classmethod
    def for_large_codebase(cls) -> "OptimizedQueryConfig":
        """Create configuration optimized for large codebases."""
        return cls(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_query_caching=True,
            enable_result_caching=True,
            cache_size_limit=2000,
            batch_size=200,
            timeout_ms=10000.0,
            use_focused_patterns=True,
            enable_early_termination=True,
            max_matches_per_query=500,
            enable_depth_limiting=True,
            max_traversal_depth=30,
        )

    @classmethod
    def for_small_codebase(cls) -> "OptimizedQueryConfig":
        """Create configuration optimized for small codebases."""
        return cls(
            optimization_level=OptimizationLevel.MINIMAL,
            enable_query_caching=False,
            enable_result_caching=False,
            batch_size=50,
            timeout_ms=2000.0,
            use_focused_patterns=False,
            enable_early_termination=False,
            max_matches_per_query=100,
            enable_depth_limiting=False,
        )


class OptimizedPythonCallPatterns:
    """
    Performance-optimized Tree-sitter query patterns for Python function calls.

    This class provides optimized query patterns with intelligent caching,
    batch processing, and adaptive optimization for large codebase performance.
    """

    # High-performance focused patterns (most common cases first)
    FOCUSED_FUNCTION_CALL = """
    ; Optimized direct function calls (most common pattern)
    (call
        function: (identifier) @function_name
        arguments: (argument_list)) @call_node
    """

    FOCUSED_METHOD_CALL = """
    ; Optimized method calls (second most common)
    (call
        function: (attribute
            object: (identifier) @object_name
            attribute: (identifier) @method_name)
        arguments: (argument_list)) @call_node
    """

    FOCUSED_SELF_METHOD_CALL = """
    ; Optimized self method calls (common in classes)
    (call
        function: (attribute
            object: (identifier) @self_ref
            attribute: (identifier) @method_name)
        arguments: (argument_list)) @call_node
    (#eq? @self_ref "self")
    """

    # Composite pattern for maximum efficiency
    OPTIMIZED_COMPOSITE_CALL_PATTERN = """
    ; Composite pattern covering most call types efficiently
    [
        ; Direct function calls
        (call
            function: (identifier) @call.function.name
            arguments: (argument_list) @call.arguments) @call.direct

        ; Method calls (obj.method)
        (call
            function: (attribute
                object: (identifier) @call.object.name
                attribute: (identifier) @call.method.name)
            arguments: (argument_list) @call.arguments) @call.method

        ; Chained calls (obj.attr.method) - limited depth for performance
        (call
            function: (attribute
                object: (attribute
                    object: (identifier) @call.base_object
                    attribute: (identifier) @call.attr)
                attribute: (identifier) @call.method.name)
            arguments: (argument_list) @call.arguments) @call.chained
    ] @any_call
    """

    # Async patterns optimized for performance
    OPTIMIZED_ASYNC_PATTERNS = """
    ; Optimized async call patterns
    [
        ; Await function calls
        (await
            (call
                function: (identifier) @async.function.name
                arguments: (argument_list) @async.arguments)) @async.await_function

        ; Await method calls
        (await
            (call
                function: (attribute
                    object: (identifier) @async.object
                    attribute: (identifier) @async.method.name)
                arguments: (argument_list) @async.arguments)) @async.await_method
    ] @any_async_call
    """

    # Special patterns for specific frameworks (cached separately)
    FRAMEWORK_SPECIFIC_PATTERNS = {
        "asyncio": """
        ; Asyncio-specific patterns
        (call
            function: (attribute
                object: (identifier) @asyncio.module
                attribute: (identifier) @asyncio.function)
            arguments: (argument_list)) @asyncio.call
        (#eq? @asyncio.module "asyncio")
        """,
        "django": """
        ; Django ORM patterns
        (call
            function: (attribute
                object: (attribute
                    object: (identifier) @django.model
                    attribute: (identifier) @django.objects)
                attribute: (identifier) @django.method)
            arguments: (argument_list)) @django.orm_call
        (#eq? @django.objects "objects")
        """,
        "flask": """
        ; Flask route and app patterns
        (call
            function: (attribute
                object: (identifier) @flask.app
                attribute: (identifier) @flask.method)
            arguments: (argument_list)) @flask.call
        """,
    }

    # Batch patterns for processing multiple nodes efficiently
    BATCH_PROCESSING_PATTERNS = """
    ; Batch pattern for processing multiple call types at once
    (call
        function: [
            (identifier) @batch.function.name
            (attribute
                object: (_) @batch.object
                attribute: (identifier) @batch.method.name)
        ]
        arguments: (argument_list) @batch.arguments) @batch.call
    """

    @classmethod
    def get_patterns_for_optimization_level(cls, level: OptimizationLevel) -> dict[str, str]:
        """Get query patterns based on optimization level."""
        if level == OptimizationLevel.MINIMAL:
            return {"function_calls": cls.FOCUSED_FUNCTION_CALL, "method_calls": cls.FOCUSED_METHOD_CALL}
        elif level == OptimizationLevel.BALANCED:
            return {"composite_calls": cls.OPTIMIZED_COMPOSITE_CALL_PATTERN, "async_calls": cls.OPTIMIZED_ASYNC_PATTERNS}
        elif level == OptimizationLevel.AGGRESSIVE:
            patterns = {
                "composite_calls": cls.OPTIMIZED_COMPOSITE_CALL_PATTERN,
                "async_calls": cls.OPTIMIZED_ASYNC_PATTERNS,
                "batch_calls": cls.BATCH_PROCESSING_PATTERNS,
            }
            patterns.update(cls.FRAMEWORK_SPECIFIC_PATTERNS)
            return patterns
        else:  # CUSTOM
            return {"all_patterns": cls.OPTIMIZED_COMPOSITE_CALL_PATTERN}


class OptimizedQueryExecutor:
    """
    High-performance Tree-sitter query executor with caching and optimization.

    This executor provides intelligent caching, batch processing, and adaptive
    optimization strategies for processing large codebases efficiently.
    """

    def __init__(self, config: OptimizedQueryConfig | None = None):
        """
        Initialize the optimized query executor.

        Args:
            config: Query execution configuration
        """
        self.config = config or OptimizedQueryConfig()
        self.logger = logging.getLogger(__name__)

        # Query and result caches
        self._compiled_query_cache: dict[str, tree_sitter.Query] = {}
        self._result_cache: dict[str, tuple[list[Any], float]] = {}  # (results, timestamp)

        # Performance monitoring
        self._performance_stats: dict[str, QueryPerformanceStats] = {}
        self._total_queries_executed = 0
        self._total_cache_hits = 0

        # Adaptive optimization
        self._pattern_performance_history: dict[str, list[float]] = {}
        self._optimization_adjustments = 0

        self.logger.info(f"OptimizedQueryExecutor initialized with level: {self.config.optimization_level}")

    async def execute_optimized_query(
        self,
        parser: tree_sitter.Parser,
        source_code: str,
        pattern_name: str,
        node: Node | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[tuple[Node, dict[str, Any]]]:
        """
        Execute an optimized query with caching and performance monitoring.

        Args:
            parser: Tree-sitter parser
            source_code: Source code to parse
            pattern_name: Name of the pattern to execute
            node: Optional specific node to query (for focused queries)
            context: Optional context for adaptive optimization

        Returns:
            List of (node, capture_dict) tuples
        """
        start_time = time.time()

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(source_code, pattern_name, node)

            # Check result cache
            if self.config.enable_result_caching and cache_key in self._result_cache:
                cached_results, timestamp = self._result_cache[cache_key]

                # Check if cache is still valid (5 minute TTL)
                if time.time() - timestamp < 300:
                    self._record_performance(pattern_name, 0.0, len(cached_results), True)
                    return cached_results
                else:
                    # Remove expired cache entry
                    del self._result_cache[cache_key]

            # Get optimized patterns
            patterns = OptimizedPythonCallPatterns.get_patterns_for_optimization_level(self.config.optimization_level)

            if pattern_name not in patterns:
                self.logger.warning(f"Pattern {pattern_name} not found for optimization level {self.config.optimization_level}")
                return []

            # Compile query with caching
            query = await self._get_compiled_query(parser.language, patterns[pattern_name], pattern_name)

            if not query:
                return []

            # Parse source code if needed
            if node is None:
                tree = parser.parse(source_code.encode("utf-8"))
                if not tree.root_node:
                    return []
                query_node = tree.root_node
            else:
                query_node = node

            # Execute query with optimizations
            results = await self._execute_query_with_optimizations(query, query_node, source_code, pattern_name, context)

            # Cache results if enabled
            if self.config.enable_result_caching and len(self._result_cache) < self.config.cache_size_limit:
                self._result_cache[cache_key] = (results, time.time())

            execution_time = (time.time() - start_time) * 1000
            self._record_performance(pattern_name, execution_time, len(results), False)

            return results

        except Exception as e:
            self.logger.error(f"Error executing optimized query {pattern_name}: {e}")
            return []

    async def _get_compiled_query(self, language: tree_sitter.Language, pattern: str, pattern_name: str) -> tree_sitter.Query | None:
        """Get a compiled query with caching."""
        cache_key = f"{language}:{hashlib.md5(pattern.encode()).hexdigest()}"

        if self.config.enable_query_caching and cache_key in self._compiled_query_cache:
            return self._compiled_query_cache[cache_key]

        try:
            query = language.query(pattern)

            if self.config.enable_query_caching and len(self._compiled_query_cache) < self.config.cache_size_limit:
                self._compiled_query_cache[cache_key] = query

            return query

        except Exception as e:
            self.logger.error(f"Error compiling query {pattern_name}: {e}")
            return None

    async def _execute_query_with_optimizations(
        self, query: tree_sitter.Query, node: Node, source_code: str, pattern_name: str, context: dict[str, Any] | None
    ) -> list[tuple[Node, dict[str, Any]]]:
        """Execute query with performance optimizations."""
        results = []
        matches_processed = 0

        try:
            # Set up timeout if configured
            start_time = time.time()
            timeout_s = self.config.timeout_ms / 1000.0

            # Execute query with captures
            captures = query.captures(node)

            for capture_node, capture_name in captures:
                # Check timeout
                if time.time() - start_time > timeout_s:
                    self.logger.warning(f"Query {pattern_name} timed out after {timeout_s}s")
                    break

                # Check depth limiting
                if self.config.enable_depth_limiting:
                    depth = self._calculate_node_depth(capture_node)
                    if depth > self.config.max_traversal_depth:
                        continue

                # Create capture dictionary
                capture_dict = {capture_name: capture_node}

                # Add additional context if available
                if context:
                    capture_dict.update(context)

                results.append((capture_node, capture_dict))
                matches_processed += 1

                # Check early termination
                if self.config.enable_early_termination and matches_processed >= self.config.max_matches_per_query:
                    break

            return results

        except Exception as e:
            self.logger.error(f"Error in optimized query execution for {pattern_name}: {e}")
            return results

    def _generate_cache_key(self, source_code: str, pattern_name: str, node: Node | None) -> str:
        """Generate a cache key for the query."""
        # Use hash of source code + pattern + node info for cache key
        content_hash = hashlib.md5(source_code.encode()).hexdigest()[:16]
        node_info = f"{node.start_point}-{node.end_point}" if node else "root"
        return f"{pattern_name}:{content_hash}:{node_info}"

    def _calculate_node_depth(self, node: Node) -> int:
        """Calculate the depth of a node in the AST."""
        depth = 0
        current = node.parent
        while current:
            depth += 1
            current = current.parent
        return depth

    def _record_performance(self, pattern_name: str, execution_time_ms: float, matches_count: int, was_cached: bool):
        """Record performance statistics for a query execution."""
        if not self.config.enable_performance_monitoring:
            return

        if pattern_name not in self._performance_stats:
            self._performance_stats[pattern_name] = QueryPerformanceStats(pattern_name)

        self._performance_stats[pattern_name].record_execution(execution_time_ms, matches_count, was_cached)
        self._total_queries_executed += 1

        if was_cached:
            self._total_cache_hits += 1

        # Adaptive optimization
        if self.config.enable_adaptive_optimization:
            self._update_adaptive_optimization(pattern_name, execution_time_ms)

    def _update_adaptive_optimization(self, pattern_name: str, execution_time_ms: float):
        """Update adaptive optimization based on performance."""
        if pattern_name not in self._pattern_performance_history:
            self._pattern_performance_history[pattern_name] = []

        history = self._pattern_performance_history[pattern_name]
        history.append(execution_time_ms)

        # Keep only recent history
        if len(history) > 100:
            history.pop(0)

        # Check if pattern is consistently slow
        if len(history) >= 10:
            avg_time = sum(history[-10:]) / 10
            if avg_time > 100.0:  # More than 100ms average
                # Suggest optimization adjustments
                self.logger.info(f"Pattern {pattern_name} is slow (avg: {avg_time:.2f}ms), consider optimization")
                self._optimization_adjustments += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "total_queries_executed": self._total_queries_executed,
            "total_cache_hits": self._total_cache_hits,
            "cache_hit_rate": (self._total_cache_hits / self._total_queries_executed * 100) if self._total_queries_executed > 0 else 0.0,
            "optimization_adjustments": self._optimization_adjustments,
            "pattern_stats": {},
        }

        for pattern_name, pattern_stats in self._performance_stats.items():
            stats["pattern_stats"][pattern_name] = {
                "execution_count": pattern_stats.execution_count,
                "average_execution_time_ms": pattern_stats.average_execution_time_ms,
                "total_matches_found": pattern_stats.matches_found,
                "cache_hit_rate": pattern_stats.cache_hit_rate,
            }

        return stats

    def optimize_for_codebase_size(self, estimated_file_count: int, estimated_total_lines: int):
        """Automatically optimize configuration based on codebase size."""
        if estimated_file_count > 1000 or estimated_total_lines > 100000:
            # Large codebase optimization
            self.config = OptimizedQueryConfig.for_large_codebase()
            self.logger.info("Optimized configuration for large codebase")
        elif estimated_file_count < 50 and estimated_total_lines < 5000:
            # Small codebase optimization
            self.config = OptimizedQueryConfig.for_small_codebase()
            self.logger.info("Optimized configuration for small codebase")
        else:
            # Keep balanced configuration
            self.logger.info("Using balanced configuration for medium codebase")

    def clear_caches(self):
        """Clear all caches to free memory."""
        self._compiled_query_cache.clear()
        self._result_cache.clear()
        self.logger.info("Query and result caches cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache usage information."""
        return {
            "compiled_query_cache_size": len(self._compiled_query_cache),
            "result_cache_size": len(self._result_cache),
            "cache_size_limit": self.config.cache_size_limit,
            "cache_utilization": {
                "query_cache": len(self._compiled_query_cache) / self.config.cache_size_limit * 100,
                "result_cache": len(self._result_cache) / self.config.cache_size_limit * 100,
            },
        }
