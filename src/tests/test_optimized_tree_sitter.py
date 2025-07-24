"""
Tests for optimized Tree-sitter query patterns and performance enhancements.

This module tests the performance optimization features for Tree-sitter query
execution, including caching, adaptive optimization, and batch processing.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.utils.enhanced_tree_sitter_manager import EnhancedTreeSitterManager
from src.utils.optimized_query_patterns import (
    OptimizationLevel,
    OptimizedPythonCallPatterns,
    OptimizedQueryConfig,
    OptimizedQueryExecutor,
    QueryPerformanceStats,
)


class TestOptimizationLevel:
    """Test optimization level enumeration."""

    def test_optimization_levels(self):
        """Test all optimization levels are defined."""
        assert OptimizationLevel.MINIMAL == "minimal"
        assert OptimizationLevel.BALANCED == "balanced"
        assert OptimizationLevel.AGGRESSIVE == "aggressive"
        assert OptimizationLevel.CUSTOM == "custom"


class TestQueryPerformanceStats:
    """Test query performance statistics tracking."""

    def test_stats_initialization(self):
        """Test performance stats initialization."""
        stats = QueryPerformanceStats("test_pattern")

        assert stats.pattern_name == "test_pattern"
        assert stats.execution_count == 0
        assert stats.total_execution_time_ms == 0.0
        assert stats.average_execution_time_ms == 0.0
        assert stats.matches_found == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0

    def test_record_execution(self):
        """Test recording execution statistics."""
        stats = QueryPerformanceStats("test_pattern")

        # Record first execution (cache miss)
        stats.record_execution(100.0, 5, False)

        assert stats.execution_count == 1
        assert stats.total_execution_time_ms == 100.0
        assert stats.average_execution_time_ms == 100.0
        assert stats.matches_found == 5
        assert stats.cache_hits == 0
        assert stats.cache_misses == 1

        # Record second execution (cache hit)
        stats.record_execution(50.0, 3, True)

        assert stats.execution_count == 2
        assert stats.total_execution_time_ms == 150.0
        assert stats.average_execution_time_ms == 75.0
        assert stats.matches_found == 8
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = QueryPerformanceStats("test_pattern")

        # No executions yet
        assert stats.cache_hit_rate == 0.0

        # Add some executions
        stats.record_execution(100.0, 5, False)  # miss
        stats.record_execution(50.0, 3, True)  # hit
        stats.record_execution(25.0, 2, True)  # hit

        # 2 hits out of 3 total = 66.67%
        assert stats.cache_hit_rate == pytest.approx(66.67, rel=1e-2)


class TestOptimizedQueryConfig:
    """Test optimized query configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizedQueryConfig()

        assert config.optimization_level == OptimizationLevel.BALANCED
        assert config.enable_query_caching is True
        assert config.enable_result_caching is True
        assert config.cache_size_limit == 1000
        assert config.batch_size == 100
        assert config.timeout_ms == 5000.0
        assert config.use_focused_patterns is True
        assert config.max_matches_per_query == 1000

    def test_large_codebase_config(self):
        """Test configuration optimized for large codebases."""
        config = OptimizedQueryConfig.for_large_codebase()

        assert config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert config.cache_size_limit == 2000
        assert config.batch_size == 200
        assert config.timeout_ms == 10000.0
        assert config.max_matches_per_query == 500
        assert config.max_traversal_depth == 30

    def test_small_codebase_config(self):
        """Test configuration optimized for small codebases."""
        config = OptimizedQueryConfig.for_small_codebase()

        assert config.optimization_level == OptimizationLevel.MINIMAL
        assert config.enable_query_caching is False
        assert config.enable_result_caching is False
        assert config.batch_size == 50
        assert config.timeout_ms == 2000.0
        assert config.max_matches_per_query == 100
        assert config.enable_depth_limiting is False


class TestOptimizedPythonCallPatterns:
    """Test optimized Python call patterns."""

    def test_focused_patterns(self):
        """Test focused pattern definitions."""
        assert "function: (identifier)" in OptimizedPythonCallPatterns.FOCUSED_FUNCTION_CALL
        assert "object: (identifier)" in OptimizedPythonCallPatterns.FOCUSED_METHOD_CALL
        assert '#eq? @self_ref "self"' in OptimizedPythonCallPatterns.FOCUSED_SELF_METHOD_CALL

    def test_composite_pattern(self):
        """Test composite pattern includes multiple call types."""
        pattern = OptimizedPythonCallPatterns.OPTIMIZED_COMPOSITE_CALL_PATTERN

        assert "Direct function calls" in pattern
        assert "Method calls" in pattern
        assert "Chained calls" in pattern
        assert "@call.direct" in pattern
        assert "@call.method" in pattern
        assert "@call.chained" in pattern

    def test_async_patterns(self):
        """Test async pattern definitions."""
        pattern = OptimizedPythonCallPatterns.OPTIMIZED_ASYNC_PATTERNS

        assert "(await" in pattern
        assert "@async.await_function" in pattern
        assert "@async.await_method" in pattern

    def test_framework_patterns(self):
        """Test framework-specific patterns."""
        patterns = OptimizedPythonCallPatterns.FRAMEWORK_SPECIFIC_PATTERNS

        assert "asyncio" in patterns
        assert "django" in patterns
        assert "flask" in patterns

        # Check asyncio pattern
        asyncio_pattern = patterns["asyncio"]
        assert "asyncio.module" in asyncio_pattern
        assert '#eq? @asyncio.module "asyncio"' in asyncio_pattern

    def test_patterns_for_optimization_level(self):
        """Test pattern selection by optimization level."""
        # Minimal patterns
        minimal_patterns = OptimizedPythonCallPatterns.get_patterns_for_optimization_level(OptimizationLevel.MINIMAL)
        assert "function_calls" in minimal_patterns
        assert "method_calls" in minimal_patterns
        assert len(minimal_patterns) == 2

        # Balanced patterns
        balanced_patterns = OptimizedPythonCallPatterns.get_patterns_for_optimization_level(OptimizationLevel.BALANCED)
        assert "composite_calls" in balanced_patterns
        assert "async_calls" in balanced_patterns
        assert len(balanced_patterns) == 2

        # Aggressive patterns
        aggressive_patterns = OptimizedPythonCallPatterns.get_patterns_for_optimization_level(OptimizationLevel.AGGRESSIVE)
        assert "composite_calls" in aggressive_patterns
        assert "batch_calls" in aggressive_patterns
        assert "asyncio" in aggressive_patterns
        assert len(aggressive_patterns) >= 5


@pytest.mark.asyncio
class TestOptimizedQueryExecutor:
    """Test optimized query executor."""

    async def test_executor_initialization(self):
        """Test executor initialization."""
        config = OptimizedQueryConfig()
        executor = OptimizedQueryExecutor(config)

        assert executor.config == config
        assert len(executor._compiled_query_cache) == 0
        assert len(executor._result_cache) == 0
        assert len(executor._performance_stats) == 0

    @patch("src.utils.optimized_query_patterns.tree_sitter")
    async def test_query_caching(self, mock_tree_sitter):
        """Test query compilation caching."""
        # Mock language and query
        mock_language = Mock()
        mock_query = Mock()
        mock_language.query.return_value = mock_query

        config = OptimizedQueryConfig(enable_query_caching=True)
        executor = OptimizedQueryExecutor(config)

        pattern = "test pattern"

        # First call should compile and cache
        result1 = await executor._get_compiled_query(mock_language, pattern, "test_pattern")
        assert result1 == mock_query
        assert mock_language.query.call_count == 1

        # Second call should use cache
        result2 = await executor._get_compiled_query(mock_language, pattern, "test_pattern")
        assert result2 == mock_query
        assert mock_language.query.call_count == 1  # Should not increase

    async def test_cache_key_generation(self):
        """Test cache key generation."""
        executor = OptimizedQueryExecutor()

        source_code = "def test(): pass"
        pattern_name = "test_pattern"

        # Test without node
        key1 = executor._generate_cache_key(source_code, pattern_name, None)
        assert pattern_name in key1
        assert "root" in key1

        # Test with mock node
        mock_node = Mock()
        mock_node.start_point = (1, 0)
        mock_node.end_point = (2, 10)

        key2 = executor._generate_cache_key(source_code, pattern_name, mock_node)
        assert pattern_name in key2
        assert "(1, 0)-(2, 10)" in key2

        # Keys should be different
        assert key1 != key2

    async def test_node_depth_calculation(self):
        """Test AST node depth calculation."""
        executor = OptimizedQueryExecutor()

        # Mock node hierarchy
        root_node = Mock()
        root_node.parent = None

        level1_node = Mock()
        level1_node.parent = root_node

        level2_node = Mock()
        level2_node.parent = level1_node

        # Test depth calculations
        assert executor._calculate_node_depth(root_node) == 0
        assert executor._calculate_node_depth(level1_node) == 1
        assert executor._calculate_node_depth(level2_node) == 2

    async def test_performance_recording(self):
        """Test performance statistics recording."""
        config = OptimizedQueryConfig(enable_performance_monitoring=True)
        executor = OptimizedQueryExecutor(config)

        pattern_name = "test_pattern"

        # Record some executions
        executor._record_performance(pattern_name, 100.0, 5, False)  # miss
        executor._record_performance(pattern_name, 50.0, 3, True)  # hit

        # Check statistics
        assert pattern_name in executor._performance_stats
        stats = executor._performance_stats[pattern_name]

        assert stats.execution_count == 2
        assert stats.average_execution_time_ms == 75.0
        assert stats.matches_found == 8
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1

    async def test_adaptive_optimization(self):
        """Test adaptive optimization updates."""
        config = OptimizedQueryConfig(enable_adaptive_optimization=True)
        executor = OptimizedQueryExecutor(config)

        pattern_name = "slow_pattern"

        # Record multiple slow executions
        for _ in range(15):
            executor._update_adaptive_optimization(pattern_name, 150.0)  # Slow execution

        assert pattern_name in executor._pattern_performance_history
        history = executor._pattern_performance_history[pattern_name]
        assert len(history) == 15
        assert executor._optimization_adjustments > 0

    def test_cache_management(self):
        """Test cache clearing and info."""
        executor = OptimizedQueryExecutor()

        # Add some mock cache entries
        executor._compiled_query_cache["test1"] = Mock()
        executor._result_cache["test2"] = ([], time.time())

        # Test cache info
        cache_info = executor.get_cache_info()
        assert cache_info["compiled_query_cache_size"] == 1
        assert cache_info["result_cache_size"] == 1

        # Test cache clearing
        executor.clear_caches()
        assert len(executor._compiled_query_cache) == 0
        assert len(executor._result_cache) == 0

    def test_codebase_size_optimization(self):
        """Test automatic optimization based on codebase size."""
        executor = OptimizedQueryExecutor()

        # Test large codebase optimization
        original_config = executor.config
        executor.optimize_for_codebase_size(2000, 150000)  # Large codebase

        # Should have switched to large codebase config
        assert executor.config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert executor.config.cache_size_limit >= original_config.cache_size_limit

        # Test small codebase optimization
        executor.optimize_for_codebase_size(30, 3000)  # Small codebase

        # Should have switched to small codebase config
        assert executor.config.optimization_level == OptimizationLevel.MINIMAL


@pytest.mark.asyncio
class TestEnhancedTreeSitterManager:
    """Test enhanced Tree-sitter manager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        config = OptimizedQueryConfig()
        manager = EnhancedTreeSitterManager(config)

        assert manager.optimization_config == config
        assert manager.query_executor is not None
        assert len(manager._parsing_stats) == 0
        assert len(manager._language_optimizations) > 0

    def test_language_optimizations(self):
        """Test language-specific optimizations."""
        manager = EnhancedTreeSitterManager()

        # Check default language optimizations
        assert "python" in manager._language_optimizations
        assert "javascript" in manager._language_optimizations
        assert "typescript" in manager._language_optimizations

        python_opts = manager._language_optimizations["python"]
        assert "preferred_patterns" in python_opts
        assert "optimization_level" in python_opts
        assert "batch_size" in python_opts

    def test_optimal_pattern_selection(self):
        """Test optimal pattern selection for languages."""
        manager = EnhancedTreeSitterManager()

        # Test without context
        pattern = manager._get_optimal_pattern_for_language("python")
        assert pattern in ["composite_calls", "async_calls"]

        # Test with large file context
        pattern = manager._get_optimal_pattern_for_language("python", {"file_size": 100000})
        assert pattern == "composite_calls"

        # Test with async context
        pattern = manager._get_optimal_pattern_for_language("python", {"async_heavy": True})
        assert pattern == "async_calls"

        # Test with simple calls context
        pattern = manager._get_optimal_pattern_for_language("python", {"simple_calls_only": True})
        assert pattern == "function_calls"

    def test_parsing_stats_recording(self):
        """Test parsing statistics recording."""
        manager = EnhancedTreeSitterManager()

        language = "python"

        # Record successful parse
        manager._record_parsing_stats(language, 100.0, 1000, True)

        assert language in manager._parsing_stats
        stats = manager._parsing_stats[language]

        assert stats["total_parses"] == 1
        assert stats["successful_parses"] == 1
        assert stats["total_time_ms"] == 100.0
        assert stats["total_code_length"] == 1000
        assert stats["average_time_ms"] == 100.0

        # Record failed parse
        manager._record_parsing_stats(language, 0.0, 500, False)

        stats = manager._parsing_stats[language]
        assert stats["total_parses"] == 2
        assert stats["successful_parses"] == 1  # Still 1 successful
        assert stats["total_code_length"] == 1500

    def test_codebase_optimization(self):
        """Test codebase-based optimization."""
        manager = EnhancedTreeSitterManager()

        codebase_info = {"total_files": 1500, "total_lines": 200000, "languages": {"python": 800, "javascript": 500, "typescript": 200}}

        # Apply optimization
        manager.optimize_for_codebase(codebase_info)

        # Check that language optimizations were updated
        python_opts = manager._language_optimizations["python"]
        assert python_opts["optimization_level"] == OptimizationLevel.AGGRESSIVE

        javascript_opts = manager._language_optimizations["javascript"]
        assert javascript_opts["optimization_level"] == OptimizationLevel.AGGRESSIVE

        typescript_opts = manager._language_optimizations["typescript"]
        assert typescript_opts["optimization_level"] == OptimizationLevel.AGGRESSIVE

    def test_performance_report(self):
        """Test performance report generation."""
        manager = EnhancedTreeSitterManager()

        # Add some stats
        manager._record_parsing_stats("python", 100.0, 1000, True)
        manager._optimization_stats["total_optimizations_applied"] = 5

        report = manager.get_performance_report()

        assert "parsing_stats" in report
        assert "optimization_stats" in report
        assert "query_executor_stats" in report
        assert "cache_info" in report
        assert "language_optimizations" in report
        assert "configuration" in report

        # Check specific values
        assert report["parsing_stats"]["python"]["total_parses"] == 1
        assert report["optimization_stats"]["total_optimizations_applied"] == 5

    def test_performance_tuning(self):
        """Test automatic performance tuning."""
        manager = EnhancedTreeSitterManager()

        performance_data = {"parsing_stats": {"python": {"average_time_ms": 300}, "javascript": {"average_time_ms": 30}}}  # Slow  # Fast

        original_python_batch = manager._language_optimizations["python"]["batch_size"]
        original_js_batch = manager._language_optimizations["javascript"]["batch_size"]

        # Apply tuning
        manager.apply_performance_tuning(performance_data)

        # Check adjustments
        python_batch = manager._language_optimizations["python"]["batch_size"]
        js_batch = manager._language_optimizations["javascript"]["batch_size"]

        assert python_batch < original_python_batch  # Should be reduced for slow parsing
        assert js_batch > original_js_batch  # Should be increased for fast parsing

    @patch("src.utils.enhanced_tree_sitter_manager.OptimizedPythonCallPatterns")
    async def test_pattern_benchmarking(self, mock_patterns):
        """Test pattern benchmarking functionality."""
        manager = EnhancedTreeSitterManager()

        # Mock patterns
        mock_patterns.get_patterns_for_optimization_level.return_value = {"pattern1": "query1", "pattern2": "query2"}

        # Mock the extract_calls_optimized method
        async def mock_extract(source_code, language, pattern_name):
            if pattern_name == "pattern1":
                await asyncio.sleep(0.01)  # Simulate 10ms
                return [Mock(), Mock()]  # 2 results
            else:
                await asyncio.sleep(0.005)  # Simulate 5ms
                return [Mock()]  # 1 result

        manager.extract_calls_optimized = mock_extract

        # Run benchmark
        results = await manager.benchmark_patterns("python", "def test(): pass")

        assert "pattern1" in results
        assert "pattern2" in results
        assert results["pattern1"] >= 10  # Should be at least 10ms
        assert results["pattern2"] >= 5  # Should be at least 5ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
