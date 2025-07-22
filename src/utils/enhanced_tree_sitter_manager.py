"""
Enhanced Tree-sitter Manager with Performance Optimization.

This module extends the Tree-sitter manager with performance optimization capabilities,
including optimized query execution, caching, and adaptive performance tuning for
large codebase processing.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from tree_sitter import Language, Node, Parser
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")

from src.utils.optimized_query_patterns import OptimizationLevel, OptimizedPythonCallPatterns, OptimizedQueryConfig, OptimizedQueryExecutor
from src.utils.tree_sitter_manager import TreeSitterManager


class EnhancedTreeSitterManager(TreeSitterManager):
    """
    Enhanced Tree-sitter manager with performance optimization capabilities.

    This manager extends the base TreeSitterManager with:
    - Optimized query execution with caching
    - Adaptive performance tuning
    - Batch processing capabilities
    - Language-specific optimization strategies
    - Performance monitoring and analytics
    """

    def __init__(self, optimization_config: OptimizedQueryConfig | None = None):
        """
        Initialize the enhanced Tree-sitter manager.

        Args:
            optimization_config: Configuration for query optimization
        """
        super().__init__()

        # Optimization components
        self.optimization_config = optimization_config or OptimizedQueryConfig()
        self.query_executor = OptimizedQueryExecutor(self.optimization_config)

        # Performance tracking
        self._parsing_stats: dict[str, dict[str, Any]] = {}
        self._optimization_stats = {"total_optimizations_applied": 0, "cache_hits": 0, "cache_misses": 0, "performance_improvements": {}}

        # Language-specific optimizations
        self._language_optimizations: dict[str, dict[str, Any]] = {
            "python": {
                "preferred_patterns": ["composite_calls", "async_calls"],
                "optimization_level": OptimizationLevel.BALANCED,
                "batch_size": 100,
            },
            "javascript": {"preferred_patterns": ["composite_calls"], "optimization_level": OptimizationLevel.BALANCED, "batch_size": 80},
            "typescript": {
                "preferred_patterns": ["composite_calls"],
                "optimization_level": OptimizationLevel.AGGRESSIVE,
                "batch_size": 120,
            },
        }

        self.logger.info(f"EnhancedTreeSitterManager initialized with optimization level: {self.optimization_config.optimization_level}")

    async def parse_with_optimization(
        self, source_code: str, language: str, enable_caching: bool = True, context: dict[str, Any] | None = None
    ) -> Node | None:
        """
        Parse source code with optimization and caching.

        Args:
            source_code: Source code to parse
            language: Programming language
            enable_caching: Whether to use caching
            context: Optional context for optimization decisions

        Returns:
            Root node of the parsed AST, or None if parsing failed
        """
        start_time = time.time()

        try:
            # Get optimized parser
            parser = await self._get_optimized_parser(language, context)
            if not parser:
                return None

            # Parse with timeout protection
            try:
                tree = await asyncio.wait_for(self._async_parse(parser, source_code), timeout=self.optimization_config.timeout_ms / 1000.0)

                if tree and tree.root_node:
                    parsing_time = (time.time() - start_time) * 1000
                    self._record_parsing_stats(language, parsing_time, len(source_code), True)
                    return tree.root_node
                else:
                    self._record_parsing_stats(language, 0, len(source_code), False)
                    return None

            except asyncio.TimeoutError:
                self.logger.warning(f"Parsing timeout for {language} after {self.optimization_config.timeout_ms}ms")
                self._record_parsing_stats(language, self.optimization_config.timeout_ms, len(source_code), False)
                return None

        except Exception as e:
            self.logger.error(f"Error in optimized parsing for {language}: {e}")
            return None

    async def extract_calls_optimized(
        self,
        source_code: str,
        language: str,
        pattern_name: str | None = None,
        node: Node | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[tuple[Node, dict[str, Any]]]:
        """
        Extract function calls using optimized query patterns.

        Args:
            source_code: Source code to analyze
            language: Programming language
            pattern_name: Specific pattern to use (optional)
            node: Specific node to query (optional)
            context: Optional context for optimization

        Returns:
            List of (node, capture_dict) tuples for detected calls
        """
        try:
            # Get parser
            parser = await self._get_optimized_parser(language, context)
            if not parser:
                return []

            # Determine optimal pattern
            if not pattern_name:
                pattern_name = self._get_optimal_pattern_for_language(language, context)

            # Execute optimized query
            results = await self.query_executor.execute_optimized_query(
                parser=parser, source_code=source_code, pattern_name=pattern_name, node=node, context=context
            )

            self._optimization_stats["total_optimizations_applied"] += 1

            return results

        except Exception as e:
            self.logger.error(f"Error in optimized call extraction for {language}: {e}")
            return []

    async def batch_extract_calls(
        self,
        sources: list[tuple[str, str]],  # (source_code, language) pairs
        pattern_name: str | None = None,
        progress_callback: callable | None = None,
    ) -> dict[int, list[tuple[Node, dict[str, Any]]]]:
        """
        Extract calls from multiple sources using batch processing.

        Args:
            sources: List of (source_code, language) tuples
            pattern_name: Pattern to use for all sources
            progress_callback: Optional progress callback

        Returns:
            Dictionary mapping source index to extraction results
        """
        results = {}
        batch_size = self.optimization_config.batch_size

        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]

            # Process batch concurrently
            batch_tasks = []
            for j, (source_code, language) in enumerate(batch):
                task = self.extract_calls_optimized(
                    source_code=source_code, language=language, pattern_name=pattern_name, context={"batch_index": i + j}
                )
                batch_tasks.append(task)

            # Execute batch
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"Batch extraction failed for source {i + j}: {result}")
                        results[i + j] = []
                    else:
                        results[i + j] = result

                # Progress callback
                if progress_callback:
                    progress = min(1.0, (i + len(batch)) / len(sources))
                    progress_callback(f"Processed {i + len(batch)}/{len(sources)} sources", progress)

            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
                # Fill failed batch with empty results
                for j in range(len(batch)):
                    results[i + j] = []

        return results

    async def _get_optimized_parser(self, language: str, context: dict[str, Any] | None = None) -> Parser | None:
        """Get an optimized parser for the language."""
        # Use base class method but with optimization context
        parser = self.get_parser(language)

        if parser and context:
            # Apply language-specific optimizations
            lang_opts = self._language_optimizations.get(language, {})

            # Adjust parser settings based on context
            if "file_size" in context:
                file_size = context["file_size"]
                if file_size > 100000:  # Large file
                    # Could implement parser-specific optimizations here
                    pass

        return parser

    def _get_optimal_pattern_for_language(self, language: str, context: dict[str, Any] | None = None) -> str:
        """Determine the optimal pattern for a language and context."""
        lang_opts = self._language_optimizations.get(language, {})
        preferred_patterns = lang_opts.get("preferred_patterns", ["composite_calls"])

        # Context-based pattern selection
        if context:
            if context.get("file_size", 0) > 50000:  # Large file
                return "composite_calls"  # More comprehensive
            elif context.get("async_heavy", False):
                return "async_calls"
            elif context.get("simple_calls_only", False):
                return "function_calls"

        # Default to first preferred pattern
        return preferred_patterns[0] if preferred_patterns else "composite_calls"

    async def _async_parse(self, parser: Parser, source_code: str):
        """Asynchronous wrapper for parser.parse()."""
        # Tree-sitter parsing is CPU-bound, run in executor for true async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, parser.parse, source_code.encode("utf-8"))

    def _record_parsing_stats(self, language: str, parsing_time_ms: float, code_length: int, success: bool):
        """Record parsing performance statistics."""
        if language not in self._parsing_stats:
            self._parsing_stats[language] = {
                "total_parses": 0,
                "successful_parses": 0,
                "total_time_ms": 0.0,
                "total_code_length": 0,
                "average_time_ms": 0.0,
                "average_chars_per_ms": 0.0,
            }

        stats = self._parsing_stats[language]
        stats["total_parses"] += 1
        stats["total_code_length"] += code_length

        if success:
            stats["successful_parses"] += 1
            stats["total_time_ms"] += parsing_time_ms
            stats["average_time_ms"] = stats["total_time_ms"] / stats["successful_parses"]
            stats["average_chars_per_ms"] = stats["total_code_length"] / stats["total_time_ms"] if stats["total_time_ms"] > 0 else 0

    def optimize_for_codebase(self, codebase_info: dict[str, Any]):
        """
        Optimize the manager based on codebase characteristics.

        Args:
            codebase_info: Information about the codebase (file counts, sizes, languages)
        """
        total_files = codebase_info.get("total_files", 0)
        total_lines = codebase_info.get("total_lines", 0)
        languages = codebase_info.get("languages", {})

        # Optimize query executor
        self.query_executor.optimize_for_codebase_size(total_files, total_lines)

        # Language-specific optimizations
        for language, file_count in languages.items():
            if language in self._language_optimizations:
                lang_opts = self._language_optimizations[language]

                if file_count > 100:  # Many files in this language
                    lang_opts["optimization_level"] = OptimizationLevel.AGGRESSIVE
                    lang_opts["batch_size"] = min(200, lang_opts["batch_size"] * 2)
                elif file_count < 10:  # Few files
                    lang_opts["optimization_level"] = OptimizationLevel.MINIMAL
                    lang_opts["batch_size"] = max(20, lang_opts["batch_size"] // 2)

        self.logger.info(f"Optimized manager for codebase: {total_files} files, {total_lines} lines")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "parsing_stats": self._parsing_stats,
            "optimization_stats": self._optimization_stats,
            "query_executor_stats": self.query_executor.get_performance_stats(),
            "cache_info": self.query_executor.get_cache_info(),
            "language_optimizations": self._language_optimizations,
            "configuration": {
                "optimization_level": self.optimization_config.optimization_level.value,
                "cache_enabled": self.optimization_config.enable_query_caching,
                "batch_size": self.optimization_config.batch_size,
                "timeout_ms": self.optimization_config.timeout_ms,
            },
        }

    def apply_performance_tuning(self, performance_data: dict[str, Any]):
        """
        Apply performance tuning based on observed performance data.

        Args:
            performance_data: Performance data from previous operations
        """
        # Analyze performance patterns
        for language, stats in performance_data.get("parsing_stats", {}).items():
            avg_time = stats.get("average_time_ms", 0)

            if language in self._language_optimizations:
                lang_opts = self._language_optimizations[language]

                if avg_time > 200:  # Slow parsing
                    # Reduce batch size, increase timeout
                    lang_opts["batch_size"] = max(20, lang_opts["batch_size"] // 2)
                    self.optimization_config.timeout_ms *= 1.5
                    self.logger.info(f"Applied performance tuning for slow {language} parsing")
                elif avg_time < 50:  # Fast parsing
                    # Increase batch size
                    lang_opts["batch_size"] = min(300, lang_opts["batch_size"] * 1.5)
                    self.logger.info(f"Applied performance tuning for fast {language} parsing")

        self._optimization_stats["total_optimizations_applied"] += 1

    def clear_caches(self):
        """Clear all caches to free memory."""
        self.query_executor.clear_caches()
        self.logger.info("Enhanced Tree-sitter manager caches cleared")

    async def benchmark_patterns(self, language: str, sample_code: str) -> dict[str, float]:
        """
        Benchmark different patterns for a language with sample code.

        Args:
            language: Programming language to benchmark
            sample_code: Sample code for benchmarking

        Returns:
            Dictionary mapping pattern names to execution times
        """
        patterns = OptimizedPythonCallPatterns.get_patterns_for_optimization_level(OptimizationLevel.AGGRESSIVE)

        benchmark_results = {}

        for pattern_name in patterns.keys():
            start_time = time.time()

            try:
                results = await self.extract_calls_optimized(source_code=sample_code, language=language, pattern_name=pattern_name)

                execution_time = (time.time() - start_time) * 1000
                benchmark_results[pattern_name] = execution_time

                self.logger.debug(f"Pattern {pattern_name}: {execution_time:.2f}ms, {len(results)} matches")

            except Exception as e:
                self.logger.warning(f"Benchmark failed for pattern {pattern_name}: {e}")
                benchmark_results[pattern_name] = float("inf")

        return benchmark_results
