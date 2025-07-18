"""
Comprehensive Unit Tests for Graph Performance Optimizer

This module provides thorough testing for performance optimization functionality,
covering batch processing, progress tracking, and memory management.
"""

import asyncio
import time
from typing import Any, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.code_chunk import ChunkType
from src.services.graph_performance_optimizer import (
    BatchProcessingConfig,
    GraphPerformanceOptimizer,
    MemoryMonitor,
    OptimizationStrategy,
    PerformanceMetrics,
    ProcessingPhase,
    ProgressUpdate,
    get_graph_performance_optimizer,
)
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


class TestGraphPerformanceOptimizer:
    """Test suite for Graph Performance Optimizer."""

    @pytest.fixture
    def optimizer(self):
        """GraphPerformanceOptimizer instance for testing."""
        return GraphPerformanceOptimizer()

    @pytest.fixture
    def sample_graph_small(self):
        """Small sample graph for testing."""
        nodes = {
            f"node_{i}": GraphNode(
                breadcrumb=f"node_{i}",
                name=f"node_{i}",
                chunk_type=ChunkType.FUNCTION,
                file_path=f"/test/file_{i}.py",
                depth=1,
                parent_breadcrumb=None,
                children_breadcrumbs=[],
            )
            for i in range(10)
        }

        edges = [GraphEdge(f"node_{i}", f"node_{i+1}", "calls", 1.0) for i in range(9)]

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["node_0"],
            project_name="small_test_project",
        )

    @pytest.fixture
    def sample_graph_large(self):
        """Large sample graph for testing."""
        nodes = {
            f"node_{i}": GraphNode(
                breadcrumb=f"node_{i}",
                name=f"node_{i}",
                chunk_type=ChunkType.FUNCTION,
                file_path=f"/test/file_{i}.py",
                depth=i % 10,
                parent_breadcrumb=f"node_{i-1}" if i > 0 else None,
                children_breadcrumbs=[f"node_{i+1}"] if i < 999 else [],
            )
            for i in range(1000)
        }

        edges = [GraphEdge(f"node_{i}", f"node_{i+1}", "calls", 1.0) for i in range(999)]

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["node_0"],
            project_name="large_test_project",
        )

    def test_determine_optimization_strategy_small_project(self, optimizer):
        """Test optimization strategy determination for small projects."""
        strategy = optimizer.determine_optimization_strategy(
            project_name="small_project",
            estimated_nodes=500,
            estimated_edges=300,
            available_memory_mb=4096,
        )

        assert strategy == OptimizationStrategy.SMALL_PROJECT

    def test_determine_optimization_strategy_medium_project(self, optimizer):
        """Test optimization strategy determination for medium projects."""
        strategy = optimizer.determine_optimization_strategy(
            project_name="medium_project",
            estimated_nodes=5000,
            estimated_edges=3000,
            available_memory_mb=2048,
        )

        assert strategy == OptimizationStrategy.MEDIUM_PROJECT

    def test_determine_optimization_strategy_large_project(self, optimizer):
        """Test optimization strategy determination for large projects."""
        strategy = optimizer.determine_optimization_strategy(
            project_name="large_project",
            estimated_nodes=50000,
            estimated_edges=30000,
            available_memory_mb=1024,
        )

        assert strategy == OptimizationStrategy.LARGE_PROJECT

    def test_determine_optimization_strategy_enterprise_project(self, optimizer):
        """Test optimization strategy determination for enterprise projects."""
        strategy = optimizer.determine_optimization_strategy(
            project_name="enterprise_project",
            estimated_nodes=500000,
            estimated_edges=300000,
            available_memory_mb=512,
        )

        assert strategy == OptimizationStrategy.ENTERPRISE_PROJECT

    def test_strategy_caching(self, optimizer):
        """Test that optimization strategy is cached."""
        # First call
        strategy1 = optimizer.determine_optimization_strategy(
            project_name="test_project",
            estimated_nodes=1000,
            estimated_edges=500,
            available_memory_mb=2048,
        )

        # Second call with same parameters should use cache
        strategy2 = optimizer.determine_optimization_strategy(
            project_name="test_project",
            estimated_nodes=1000,
            estimated_edges=500,
            available_memory_mb=2048,
        )

        assert strategy1 == strategy2
        assert len(optimizer._strategy_cache) == 1

    def test_get_batch_config_small_project(self, optimizer):
        """Test batch configuration for small projects."""
        config = optimizer.get_batch_config(
            strategy=OptimizationStrategy.SMALL_PROJECT,
            total_items=1000,
            available_memory_mb=4096,
        )

        assert isinstance(config, BatchProcessingConfig)
        assert config.batch_size <= 500
        assert config.max_concurrent_batches == 8
        assert config.enable_gc_between_batches is False

    def test_get_batch_config_large_project(self, optimizer):
        """Test batch configuration for large projects."""
        config = optimizer.get_batch_config(
            strategy=OptimizationStrategy.LARGE_PROJECT,
            total_items=100000,
            available_memory_mb=1024,
        )

        assert isinstance(config, BatchProcessingConfig)
        assert config.batch_size <= 100
        assert config.max_concurrent_batches == 4
        assert config.enable_gc_between_batches is True

    def test_get_batch_config_minimum_batch_size(self, optimizer):
        """Test that minimum batch size is enforced."""
        config = optimizer.get_batch_config(
            strategy=OptimizationStrategy.ENTERPRISE_PROJECT,
            total_items=1,  # Very small
            available_memory_mb=512,
        )

        assert config.batch_size >= 1

    @pytest.mark.asyncio
    async def test_process_in_batches_basic(self, optimizer):
        """Test basic batch processing functionality."""
        items = list(range(100))

        def processor(batch: list[int]) -> list[int]:
            return [x * 2 for x in batch]

        config = BatchProcessingConfig(
            batch_size=10,
            max_concurrent_batches=2,
            enable_progress_tracking=False,
        )

        results = await optimizer.process_in_batches(
            items=items,
            processor=processor,
            config=config,
        )

        # Should have doubled all values
        expected = [x * 2 for x in items]
        assert results == expected

    @pytest.mark.asyncio
    async def test_process_in_batches_async_processor(self, optimizer):
        """Test batch processing with async processor."""
        items = list(range(50))

        async def async_processor(batch: list[int]) -> list[int]:
            await asyncio.sleep(0.01)  # Simulate async work
            return [x + 1 for x in batch]

        config = BatchProcessingConfig(
            batch_size=10,
            max_concurrent_batches=3,
            enable_progress_tracking=False,
        )

        results = await optimizer.process_in_batches(
            items=items,
            processor=async_processor,
            config=config,
        )

        expected = [x + 1 for x in items]
        assert results == expected

    @pytest.mark.asyncio
    async def test_process_in_batches_with_progress_tracking(self, optimizer):
        """Test batch processing with progress tracking."""
        items = list(range(30))
        progress_updates = []

        def progress_callback(update: ProgressUpdate):
            progress_updates.append(update)

        def processor(batch: list[int]) -> list[int]:
            return batch

        config = BatchProcessingConfig(
            batch_size=10,
            max_concurrent_batches=2,
            enable_progress_tracking=True,
            progress_callback=progress_callback,
        )

        optimizer.add_progress_callback(progress_callback)

        results = await optimizer.process_in_batches(
            items=items,
            processor=processor,
            config=config,
        )

        assert len(results) == 30
        assert len(progress_updates) > 0

        # Check that progress updates are valid
        for update in progress_updates:
            assert isinstance(update, ProgressUpdate)
            assert 0 <= update.percentage <= 100
            assert update.current_step <= update.total_steps

    @pytest.mark.asyncio
    async def test_process_in_batches_error_handling(self, optimizer):
        """Test batch processing error handling."""
        items = list(range(20))

        def failing_processor(batch: list[int]) -> list[int]:
            if 10 in batch:  # Fail on batch containing 10
                raise ValueError("Test error")
            return batch

        config = BatchProcessingConfig(
            batch_size=5,
            max_concurrent_batches=2,
            enable_progress_tracking=False,
        )

        # Should continue processing other batches despite failure
        results = await optimizer.process_in_batches(
            items=items,
            processor=failing_processor,
            config=config,
        )

        # Should have some results (failed batch excluded)
        assert len(results) < len(items)

    @pytest.mark.asyncio
    async def test_optimize_graph_structure_small_graph(self, optimizer, sample_graph_small):
        """Test graph structure optimization for small graphs."""
        optimized_graph = await optimizer._optimize_graph_structure(sample_graph_small, OptimizationStrategy.SMALL_PROJECT)

        assert isinstance(optimized_graph, StructureGraph)
        assert len(optimized_graph.nodes) <= len(sample_graph_small.nodes)
        assert len(optimized_graph.edges) <= len(sample_graph_small.edges)

    @pytest.mark.asyncio
    async def test_optimize_graph_structure_large_graph(self, optimizer, sample_graph_large):
        """Test graph structure optimization for large graphs."""
        optimized_graph = await optimizer._optimize_graph_structure(sample_graph_large, OptimizationStrategy.LARGE_PROJECT)

        assert isinstance(optimized_graph, StructureGraph)
        # For large graphs, memory layout optimization should be applied
        assert len(optimized_graph.nodes) <= len(sample_graph_large.nodes)

    @pytest.mark.asyncio
    async def test_optimize_graph_structure_invalid_nodes(self, optimizer):
        """Test graph optimization removes invalid nodes."""
        # Create graph with invalid nodes
        nodes = {
            "valid_node": GraphNode(
                breadcrumb="valid_node",
                name="valid",
                chunk_type=ChunkType.FUNCTION,
                file_path="/test.py",
                depth=1,
                parent_breadcrumb=None,
                children_breadcrumbs=[],
            ),
            "": GraphNode(  # Invalid empty breadcrumb
                breadcrumb="",
                name="invalid",
                chunk_type=ChunkType.FUNCTION,
                file_path="/test.py",
                depth=1,
                parent_breadcrumb=None,
                children_breadcrumbs=[],
            ),
            "   ": GraphNode(  # Invalid whitespace breadcrumb
                breadcrumb="   ",
                name="invalid2",
                chunk_type=ChunkType.FUNCTION,
                file_path="/test.py",
                depth=1,
                parent_breadcrumb=None,
                children_breadcrumbs=[],
            ),
        }

        graph = StructureGraph(
            nodes=nodes,
            edges=[],
            root_nodes=["valid_node"],
            project_name="test_project",
        )

        optimized_graph = await optimizer._optimize_graph_structure(graph, OptimizationStrategy.MEDIUM_PROJECT)

        # Should only have the valid node
        assert len(optimized_graph.nodes) == 1
        assert "valid_node" in optimized_graph.nodes

    @pytest.mark.asyncio
    async def test_optimize_graph_analysis(self, optimizer, sample_graph_small):
        """Test optimized graph analysis."""
        analysis_functions = [
            lambda graph: {"nodes": len(graph.nodes)},
            lambda graph: {"edges": len(graph.edges)},
        ]

        results = await optimizer.optimize_graph_analysis(
            graph=sample_graph_small,
            analysis_functions=analysis_functions,
            strategy=OptimizationStrategy.SMALL_PROJECT,
        )

        assert "analysis_results" in results
        assert "performance_metrics" in results
        assert "optimization_applied" in results
        assert results["optimization_applied"] is True

        analysis_results = results["analysis_results"]
        assert len(analysis_results) == 2

    @pytest.mark.asyncio
    async def test_optimize_graph_analysis_with_async_functions(self, optimizer, sample_graph_small):
        """Test optimized graph analysis with async functions."""

        async def async_analysis(graph):
            await asyncio.sleep(0.01)
            return {"async_nodes": len(graph.nodes)}

        analysis_functions = [async_analysis]

        results = await optimizer.optimize_graph_analysis(
            graph=sample_graph_small,
            analysis_functions=analysis_functions,
            strategy=OptimizationStrategy.SMALL_PROJECT,
        )

        assert "analysis_results" in results
        assert "async_analysis" in results["analysis_results"]

    def test_progress_callback_management(self, optimizer):
        """Test progress callback management."""

        def callback1(update):
            pass

        def callback2(update):
            pass

        # Add callbacks
        optimizer.add_progress_callback(callback1)
        optimizer.add_progress_callback(callback2)

        assert len(optimizer._progress_callbacks) == 2

        # Add duplicate (should not add)
        optimizer.add_progress_callback(callback1)
        assert len(optimizer._progress_callbacks) == 2

        # Remove callback
        optimizer.remove_progress_callback(callback1)
        assert len(optimizer._progress_callbacks) == 1
        assert callback2 in optimizer._progress_callbacks

    def test_get_optimization_recommendations(self, optimizer):
        """Test optimization recommendations generation."""
        # Create metrics with high memory usage
        high_memory_metrics = PerformanceMetrics(
            total_execution_time_ms=1000,
            peak_memory_usage_mb=3000,  # High memory
            average_memory_usage_mb=1500,
            total_items_processed=1000,
            average_processing_rate=50,  # Low rate
            cache_hit_rate=0.3,  # Low cache hit rate
            batch_count=10,
            optimization_strategy=OptimizationStrategy.SMALL_PROJECT,
        )

        recommendations = optimizer.get_optimization_recommendations("test_project", high_memory_metrics)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should recommend memory optimizations
        memory_recommendations = [r for r in recommendations if "memory" in r.lower()]
        assert len(memory_recommendations) > 0

    def test_metrics_tracking(self, optimizer):
        """Test performance metrics tracking."""
        initial_metrics = optimizer.get_performance_metrics()
        assert len(initial_metrics) == 0

        # Simulate adding metrics
        test_metrics = PerformanceMetrics(
            total_execution_time_ms=500,
            peak_memory_usage_mb=256,
            average_memory_usage_mb=128,
            total_items_processed=100,
            average_processing_rate=200,
            cache_hit_rate=0.8,
            batch_count=5,
            optimization_strategy=OptimizationStrategy.MEDIUM_PROJECT,
        )

        optimizer._metrics_history.append(test_metrics)

        metrics = optimizer.get_performance_metrics()
        assert len(metrics) == 1
        assert metrics[0] == test_metrics

    def test_reset_metrics(self, optimizer):
        """Test metrics reset functionality."""
        # Add some metrics
        test_metrics = PerformanceMetrics(
            total_execution_time_ms=500,
            peak_memory_usage_mb=256,
            average_memory_usage_mb=128,
            total_items_processed=100,
            average_processing_rate=200,
            cache_hit_rate=0.8,
            batch_count=5,
            optimization_strategy=OptimizationStrategy.MEDIUM_PROJECT,
        )

        optimizer._metrics_history.append(test_metrics)
        optimizer._current_metrics = test_metrics

        # Reset
        optimizer.reset_metrics()

        assert len(optimizer._metrics_history) == 0
        assert optimizer._current_metrics is None


class TestMemoryMonitor:
    """Test suite for Memory Monitor."""

    @pytest.fixture
    def memory_monitor(self):
        """MemoryMonitor instance for testing."""
        return MemoryMonitor()

    def test_get_current_usage(self, memory_monitor):
        """Test current memory usage retrieval."""
        usage = memory_monitor.get_current_usage_mb()
        assert isinstance(usage, float)
        assert usage > 0  # Should always have some memory usage

    def test_get_memory_delta(self, memory_monitor):
        """Test memory delta calculation."""
        delta = memory_monitor.get_memory_delta_mb()
        assert isinstance(delta, float)
        # Delta can be positive or negative

    def test_get_system_memory_info(self, memory_monitor):
        """Test system memory information retrieval."""
        info = memory_monitor.get_system_memory_info()

        assert isinstance(info, dict)
        assert "total_mb" in info
        assert "available_mb" in info
        assert "used_mb" in info
        assert "percent_used" in info

        # Validate values
        assert info["total_mb"] > 0
        assert info["available_mb"] >= 0
        assert info["used_mb"] >= 0
        assert 0 <= info["percent_used"] <= 100


class TestSingletonAccess:
    """Test suite for singleton access pattern."""

    def test_singleton_access(self):
        """Test singleton access pattern."""
        optimizer1 = get_graph_performance_optimizer()
        optimizer2 = get_graph_performance_optimizer()

        assert optimizer1 is optimizer2
        assert isinstance(optimizer1, GraphPerformanceOptimizer)

    def test_singleton_reset(self):
        """Test singleton reset for clean testing."""
        # Reset singleton for clean test
        import src.services.graph_performance_optimizer

        src.services.graph_performance_optimizer._performance_optimizer_instance = None

        # Should create new instance
        optimizer = get_graph_performance_optimizer()
        assert isinstance(optimizer, GraphPerformanceOptimizer)


class TestPerformanceDataClasses:
    """Test suite for performance data classes."""

    def test_progress_update_creation(self):
        """Test ProgressUpdate creation and properties."""
        update = ProgressUpdate(
            phase=ProcessingPhase.GRAPH_BUILDING,
            current_step=5,
            total_steps=10,
            percentage=50.0,
            message="Building graph",
            elapsed_time_ms=1000.0,
            estimated_remaining_ms=1000.0,
            memory_usage_mb=256.0,
            items_processed=50,
            items_per_second=50.0,
        )

        assert update.phase == ProcessingPhase.GRAPH_BUILDING
        assert update.current_step == 5
        assert update.total_steps == 10
        assert update.percentage == 50.0
        assert update.message == "Building graph"

    def test_batch_processing_config_defaults(self):
        """Test BatchProcessingConfig default values."""
        config = BatchProcessingConfig()

        assert config.batch_size == 100
        assert config.max_concurrent_batches == 4
        assert config.memory_threshold_mb == 1024
        assert config.enable_gc_between_batches is True
        assert config.enable_progress_tracking is True

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation and properties."""
        metrics = PerformanceMetrics(
            total_execution_time_ms=1000.0,
            peak_memory_usage_mb=512.0,
            average_memory_usage_mb=256.0,
            total_items_processed=1000,
            average_processing_rate=1000.0,
            cache_hit_rate=0.8,
            batch_count=10,
            optimization_strategy=OptimizationStrategy.LARGE_PROJECT,
        )

        assert metrics.total_execution_time_ms == 1000.0
        assert metrics.peak_memory_usage_mb == 512.0
        assert metrics.optimization_strategy == OptimizationStrategy.LARGE_PROJECT
        assert isinstance(metrics.memory_optimizations_applied, list)
        assert isinstance(metrics.time_optimizations_applied, list)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=src.services.graph_performance_optimizer",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
