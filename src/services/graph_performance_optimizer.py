"""
Graph Performance Optimizer Service

This service provides performance optimizations for graph analysis operations
on large projects, including batch processing, progress tracking, and memory management.
"""

import asyncio
import gc
import logging
import sys
import time
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil

from src.models.code_chunk import ChunkType, CodeChunk
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Optimization strategies for different project sizes."""

    SMALL_PROJECT = "small_project"
    MEDIUM_PROJECT = "medium_project"
    LARGE_PROJECT = "large_project"
    ENTERPRISE_PROJECT = "enterprise_project"


class ProcessingPhase(Enum):
    """Phases of graph processing for progress tracking."""

    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    GRAPH_BUILDING = "graph_building"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    PATTERN_DETECTION = "pattern_detection"
    REPORT_GENERATION = "report_generation"
    CLEANUP = "cleanup"


@dataclass
class ProgressUpdate:
    """Progress update information."""

    phase: ProcessingPhase
    current_step: int
    total_steps: int
    percentage: float
    message: str
    elapsed_time_ms: float
    estimated_remaining_ms: float
    memory_usage_mb: float
    items_processed: int = 0
    items_per_second: float = 0.0


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations."""

    batch_size: int = 100
    max_concurrent_batches: int = 4
    memory_threshold_mb: int = 1024  # 1GB
    enable_gc_between_batches: bool = True
    progress_callback: Callable[[ProgressUpdate], None] | None = None
    enable_progress_tracking: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking."""

    total_execution_time_ms: float
    peak_memory_usage_mb: float
    average_memory_usage_mb: float
    total_items_processed: int
    average_processing_rate: float
    cache_hit_rate: float
    batch_count: int
    optimization_strategy: OptimizationStrategy
    memory_optimizations_applied: list[str] = field(default_factory=list)
    time_optimizations_applied: list[str] = field(default_factory=list)


class GraphPerformanceOptimizer:
    """
    Service for optimizing graph analysis performance on large projects.

    Provides batch processing, memory management, progress tracking,
    and adaptive optimization strategies.
    """

    def __init__(self):
        """Initialize the performance optimizer."""
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self._metrics_history: list[PerformanceMetrics] = []
        self._current_metrics: PerformanceMetrics | None = None

        # Memory management
        self._memory_monitor = MemoryMonitor()
        self._gc_threshold_mb = 512  # Trigger GC at 512MB

        # Progress tracking
        self._progress_callbacks: list[Callable[[ProgressUpdate], None]] = []
        self._current_progress: ProgressUpdate | None = None

        # Optimization caches
        self._strategy_cache: dict[str, OptimizationStrategy] = {}
        self._batch_size_cache: dict[str, int] = {}

        self.logger.info("GraphPerformanceOptimizer initialized")

    def determine_optimization_strategy(
        self, project_name: str, estimated_nodes: int, estimated_edges: int, available_memory_mb: int | None = None
    ) -> OptimizationStrategy:
        """
        Determine the optimal strategy based on project characteristics.

        Args:
            project_name: Name of the project
            estimated_nodes: Estimated number of nodes in the graph
            estimated_edges: Estimated number of edges in the graph
            available_memory_mb: Available memory in MB

        Returns:
            Recommended optimization strategy
        """
        # Check cache first
        cache_key = f"{project_name}:{estimated_nodes}:{estimated_edges}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]

        # Get available memory if not provided
        if available_memory_mb is None:
            available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)

        # Determine strategy based on size and memory
        total_components = estimated_nodes + estimated_edges

        if total_components < 1000 and available_memory_mb > 2048:
            strategy = OptimizationStrategy.SMALL_PROJECT
        elif total_components < 10000 and available_memory_mb > 1024:
            strategy = OptimizationStrategy.MEDIUM_PROJECT
        elif total_components < 100000 and available_memory_mb > 512:
            strategy = OptimizationStrategy.LARGE_PROJECT
        else:
            strategy = OptimizationStrategy.ENTERPRISE_PROJECT

        # Cache the strategy
        self._strategy_cache[cache_key] = strategy

        self.logger.info(
            f"Determined optimization strategy for {project_name}: {strategy.value} "
            f"({total_components} components, {available_memory_mb}MB memory)"
        )

        return strategy

    def get_batch_config(
        self, strategy: OptimizationStrategy, total_items: int, available_memory_mb: int | None = None
    ) -> BatchProcessingConfig:
        """
        Get optimized batch processing configuration.

        Args:
            strategy: Optimization strategy to use
            total_items: Total number of items to process
            available_memory_mb: Available memory in MB

        Returns:
            Optimized batch processing configuration
        """
        if available_memory_mb is None:
            available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)

        # Strategy-based configurations
        configs = {
            OptimizationStrategy.SMALL_PROJECT: BatchProcessingConfig(
                batch_size=min(total_items, 500),
                max_concurrent_batches=8,
                memory_threshold_mb=available_memory_mb // 2,
                enable_gc_between_batches=False,
            ),
            OptimizationStrategy.MEDIUM_PROJECT: BatchProcessingConfig(
                batch_size=min(total_items // 10, 200),
                max_concurrent_batches=6,
                memory_threshold_mb=available_memory_mb // 3,
                enable_gc_between_batches=True,
            ),
            OptimizationStrategy.LARGE_PROJECT: BatchProcessingConfig(
                batch_size=min(total_items // 20, 100),
                max_concurrent_batches=4,
                memory_threshold_mb=available_memory_mb // 4,
                enable_gc_between_batches=True,
            ),
            OptimizationStrategy.ENTERPRISE_PROJECT: BatchProcessingConfig(
                batch_size=min(total_items // 50, 50),
                max_concurrent_batches=2,
                memory_threshold_mb=available_memory_mb // 6,
                enable_gc_between_batches=True,
            ),
        }

        config = configs.get(strategy, configs[OptimizationStrategy.LARGE_PROJECT])

        # Ensure minimum batch size
        config.batch_size = max(config.batch_size, 1)

        self.logger.debug(
            f"Batch config for {strategy.value}: "
            f"batch_size={config.batch_size}, "
            f"max_concurrent={config.max_concurrent_batches}, "
            f"memory_threshold={config.memory_threshold_mb}MB"
        )

        return config

    async def process_in_batches(
        self,
        items: list[Any],
        processor: Callable[[list[Any]], Any],
        config: BatchProcessingConfig,
        phase: ProcessingPhase = ProcessingPhase.DATA_LOADING,
        description: str = "Processing items",
    ) -> list[Any]:
        """
        Process items in batches with optimization and progress tracking.

        Args:
            items: Items to process
            processor: Function to process each batch
            config: Batch processing configuration
            phase: Current processing phase
            description: Description for progress updates

        Returns:
            List of processed results
        """
        start_time = time.time()
        results = []
        total_batches = (len(items) + config.batch_size - 1) // config.batch_size

        self.logger.info(f"Starting batch processing: {len(items)} items in {total_batches} batches " f"(batch_size={config.batch_size})")

        # Initialize progress tracking
        if config.enable_progress_tracking:
            await self._update_progress(
                phase=phase,
                current_step=0,
                total_steps=total_batches,
                message=f"Starting {description}",
                elapsed_time_ms=0,
                items_processed=0,
            )

        # Process batches
        semaphore = asyncio.Semaphore(config.max_concurrent_batches)
        batch_tasks = []

        for batch_idx in range(total_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(items))
            batch = items[start_idx:end_idx]

            # Create batch processing task
            task = asyncio.create_task(
                self._process_batch_with_optimization(
                    batch, processor, semaphore, config, batch_idx, total_batches, phase, description, start_time
                )
            )
            batch_tasks.append(task)

        # Execute batches and collect results
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Process results and handle exceptions
        for batch_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch {batch_idx} failed: {result}")
                # Continue processing other batches
                continue

            if result is not None:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

        # Final progress update
        total_time = (time.time() - start_time) * 1000
        if config.enable_progress_tracking:
            await self._update_progress(
                phase=phase,
                current_step=total_batches,
                total_steps=total_batches,
                message=f"Completed {description}",
                elapsed_time_ms=total_time,
                items_processed=len(items),
            )

        self.logger.info(
            f"Batch processing completed: {len(results)} results in {total_time:.2f}ms " f"({len(items)/total_time*1000:.2f} items/sec)"
        )

        return results

    async def _process_batch_with_optimization(
        self,
        batch: list[Any],
        processor: Callable[[list[Any]], Any],
        semaphore: asyncio.Semaphore,
        config: BatchProcessingConfig,
        batch_idx: int,
        total_batches: int,
        phase: ProcessingPhase,
        description: str,
        start_time: float,
    ) -> Any:
        """Process a single batch with memory and performance optimization."""
        async with semaphore:
            batch_start_time = time.time()

            try:
                # Check memory usage before processing
                memory_usage = self._memory_monitor.get_current_usage_mb()

                if memory_usage > config.memory_threshold_mb:
                    self.logger.warning(
                        f"Memory usage ({memory_usage}MB) exceeds threshold " f"({config.memory_threshold_mb}MB), triggering GC"
                    )
                    gc.collect()
                    await asyncio.sleep(0.1)  # Allow GC to complete

                # Process the batch
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(batch)
                else:
                    # Run CPU-intensive processor in thread pool
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(executor, processor, batch)

                # Memory cleanup between batches if enabled
                if config.enable_gc_between_batches:
                    gc.collect()

                # Update progress
                if config.enable_progress_tracking:
                    elapsed_time = (time.time() - start_time) * 1000
                    items_processed = (batch_idx + 1) * len(batch)

                    await self._update_progress(
                        phase=phase,
                        current_step=batch_idx + 1,
                        total_steps=total_batches,
                        message=f"{description} - batch {batch_idx + 1}/{total_batches}",
                        elapsed_time_ms=elapsed_time,
                        items_processed=items_processed,
                    )

                batch_time = (time.time() - batch_start_time) * 1000
                self.logger.debug(f"Batch {batch_idx + 1}/{total_batches} completed in {batch_time:.2f}ms " f"({len(batch)} items)")

                return result

            except Exception as e:
                self.logger.error(f"Error processing batch {batch_idx}: {e}")
                raise

    async def optimize_graph_analysis(
        self,
        graph: StructureGraph,
        analysis_functions: list[Callable],
        strategy: OptimizationStrategy,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> dict[str, Any]:
        """
        Optimize graph analysis operations for large projects.

        Args:
            graph: Structure graph to analyze
            analysis_functions: List of analysis functions to run
            strategy: Optimization strategy to use
            progress_callback: Optional progress callback

        Returns:
            Dictionary with analysis results and performance metrics
        """
        start_time = time.time()

        if progress_callback:
            self.add_progress_callback(progress_callback)

        try:
            # Initialize metrics tracking
            self._current_metrics = PerformanceMetrics(
                total_execution_time_ms=0,
                peak_memory_usage_mb=0,
                average_memory_usage_mb=0,
                total_items_processed=0,
                average_processing_rate=0,
                cache_hit_rate=0,
                batch_count=0,
                optimization_strategy=strategy,
            )

            # Optimize graph structure for analysis
            optimized_graph = await self._optimize_graph_structure(graph, strategy)

            # Run analysis functions with optimization
            results = {}
            for i, analysis_func in enumerate(analysis_functions):
                await self._update_progress(
                    phase=ProcessingPhase.RELATIONSHIP_ANALYSIS,
                    current_step=i,
                    total_steps=len(analysis_functions),
                    message=f"Running analysis function {i + 1}/{len(analysis_functions)}",
                    elapsed_time_ms=(time.time() - start_time) * 1000,
                    items_processed=i,
                )

                function_name = getattr(analysis_func, "__name__", f"analysis_{i}")

                try:
                    if asyncio.iscoroutinefunction(analysis_func):
                        result = await analysis_func(optimized_graph)
                    else:
                        result = analysis_func(optimized_graph)

                    results[function_name] = result

                except Exception as e:
                    self.logger.error(f"Analysis function {function_name} failed: {e}")
                    results[function_name] = {"error": str(e)}

            # Finalize metrics
            total_time = (time.time() - start_time) * 1000
            self._current_metrics.total_execution_time_ms = total_time
            self._current_metrics.average_processing_rate = (
                self._current_metrics.total_items_processed / total_time * 1000 if total_time > 0 else 0
            )

            # Add metrics to history
            self._metrics_history.append(self._current_metrics)

            # Final progress update
            await self._update_progress(
                phase=ProcessingPhase.CLEANUP,
                current_step=len(analysis_functions),
                total_steps=len(analysis_functions),
                message="Analysis completed",
                elapsed_time_ms=total_time,
                items_processed=len(analysis_functions),
            )

            return {
                "analysis_results": results,
                "performance_metrics": self._current_metrics,
                "optimization_applied": True,
                "strategy_used": strategy.value,
            }

        except Exception as e:
            self.logger.error(f"Error in optimized graph analysis: {e}")
            raise

        finally:
            if progress_callback:
                self.remove_progress_callback(progress_callback)

    async def _optimize_graph_structure(self, graph: StructureGraph, strategy: OptimizationStrategy) -> StructureGraph:
        """
        Optimize graph structure for analysis performance.

        Args:
            graph: Original structure graph
            strategy: Optimization strategy

        Returns:
            Optimized structure graph
        """
        await self._update_progress(
            phase=ProcessingPhase.GRAPH_BUILDING,
            current_step=0,
            total_steps=3,
            message="Optimizing graph structure",
            elapsed_time_ms=0,
            items_processed=0,
        )

        optimizations_applied = []

        # Optimization 1: Node deduplication and cleanup
        await self._update_progress(
            phase=ProcessingPhase.GRAPH_BUILDING,
            current_step=1,
            total_steps=3,
            message="Deduplicating nodes",
            elapsed_time_ms=0,
            items_processed=0,
        )

        optimized_nodes = {}
        for breadcrumb, node in graph.nodes.items():
            # Remove nodes with empty breadcrumbs or invalid data
            if breadcrumb and breadcrumb.strip():
                optimized_nodes[breadcrumb] = node

        if len(optimized_nodes) < len(graph.nodes):
            removed_count = len(graph.nodes) - len(optimized_nodes)
            optimizations_applied.append(f"removed_{removed_count}_invalid_nodes")
            self.logger.info(f"Removed {removed_count} invalid nodes during optimization")

        # Optimization 2: Edge optimization
        await self._update_progress(
            phase=ProcessingPhase.GRAPH_BUILDING,
            current_step=2,
            total_steps=3,
            message="Optimizing edges",
            elapsed_time_ms=0,
            items_processed=len(optimized_nodes),
        )

        optimized_edges = []
        for edge in graph.edges:
            # Only keep edges where both nodes exist
            if edge.source_breadcrumb in optimized_nodes and edge.target_breadcrumb in optimized_nodes:
                optimized_edges.append(edge)

        if len(optimized_edges) < len(graph.edges):
            removed_count = len(graph.edges) - len(optimized_edges)
            optimizations_applied.append(f"removed_{removed_count}_orphaned_edges")
            self.logger.info(f"Removed {removed_count} orphaned edges during optimization")

        # Optimization 3: Memory layout optimization for large graphs
        if strategy in [OptimizationStrategy.LARGE_PROJECT, OptimizationStrategy.ENTERPRISE_PROJECT]:
            await self._update_progress(
                phase=ProcessingPhase.GRAPH_BUILDING,
                current_step=3,
                total_steps=3,
                message="Applying memory layout optimizations",
                elapsed_time_ms=0,
                items_processed=len(optimized_nodes),
            )

            # Sort nodes by breadcrumb for better memory locality
            sorted_nodes = dict(sorted(optimized_nodes.items()))
            optimized_nodes = sorted_nodes
            optimizations_applied.append("memory_layout_optimization")

        # Update metrics
        if self._current_metrics:
            self._current_metrics.memory_optimizations_applied.extend(optimizations_applied)
            self._current_metrics.total_items_processed += len(optimized_nodes)

        # Create optimized graph
        optimized_graph = StructureGraph(
            nodes=optimized_nodes,
            edges=optimized_edges,
            root_nodes=[node for node in graph.root_nodes if node in optimized_nodes],
            project_name=graph.project_name,
        )

        self.logger.info(
            f"Graph optimization completed: {len(optimized_nodes)} nodes, "
            f"{len(optimized_edges)} edges (applied: {', '.join(optimizations_applied)})"
        )

        return optimized_graph

    async def _update_progress(
        self,
        phase: ProcessingPhase,
        current_step: int,
        total_steps: int,
        message: str,
        elapsed_time_ms: float,
        items_processed: int,
    ) -> None:
        """Update progress tracking."""
        if not self._progress_callbacks:
            return

        percentage = (current_step / total_steps * 100) if total_steps > 0 else 0

        # Estimate remaining time
        if current_step > 0 and elapsed_time_ms > 0:
            time_per_step = elapsed_time_ms / current_step
            remaining_steps = total_steps - current_step
            estimated_remaining_ms = time_per_step * remaining_steps
        else:
            estimated_remaining_ms = 0

        # Calculate processing rate
        items_per_second = (items_processed / elapsed_time_ms * 1000) if elapsed_time_ms > 0 else 0

        # Get current memory usage
        memory_usage_mb = self._memory_monitor.get_current_usage_mb()

        # Update peak memory usage
        if self._current_metrics:
            self._current_metrics.peak_memory_usage_mb = max(self._current_metrics.peak_memory_usage_mb, memory_usage_mb)

        progress = ProgressUpdate(
            phase=phase,
            current_step=current_step,
            total_steps=total_steps,
            percentage=percentage,
            message=message,
            elapsed_time_ms=elapsed_time_ms,
            estimated_remaining_ms=estimated_remaining_ms,
            memory_usage_mb=memory_usage_mb,
            items_processed=items_processed,
            items_per_second=items_per_second,
        )

        self._current_progress = progress

        # Notify all callbacks
        for callback in self._progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")

    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Add a progress callback."""
        if callback not in self._progress_callbacks:
            self._progress_callbacks.append(callback)

    def remove_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    def get_current_progress(self) -> ProgressUpdate | None:
        """Get the current progress update."""
        return self._current_progress

    def get_performance_metrics(self) -> list[PerformanceMetrics]:
        """Get performance metrics history."""
        return self._metrics_history.copy()

    def get_optimization_recommendations(self, project_name: str, current_performance: PerformanceMetrics) -> list[str]:
        """
        Get optimization recommendations based on performance metrics.

        Args:
            project_name: Name of the project
            current_performance: Current performance metrics

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Memory-based recommendations
        if current_performance.peak_memory_usage_mb > 2048:
            recommendations.append("Consider reducing batch size or enabling more aggressive garbage collection")

        if current_performance.average_memory_usage_mb > 1024:
            recommendations.append("Enable memory layout optimizations for large projects")

        # Performance-based recommendations
        if current_performance.average_processing_rate < 100:
            recommendations.append("Consider increasing batch size or concurrent processing limits")

        if current_performance.cache_hit_rate < 0.5:
            recommendations.append("Implement caching strategies to improve performance")

        # Strategy-based recommendations
        if (
            current_performance.optimization_strategy == OptimizationStrategy.SMALL_PROJECT
            and current_performance.total_items_processed > 10000
        ):
            recommendations.append("Project size suggests upgrading to MEDIUM_PROJECT optimization strategy")

        if (
            current_performance.optimization_strategy == OptimizationStrategy.MEDIUM_PROJECT
            and current_performance.total_items_processed > 100000
        ):
            recommendations.append("Project size suggests upgrading to LARGE_PROJECT optimization strategy")

        return recommendations

    def reset_metrics(self) -> None:
        """Reset performance metrics history."""
        self._metrics_history.clear()
        self._current_metrics = None
        self.logger.info("Performance metrics reset")


class MemoryMonitor:
    """Monitor memory usage during graph operations."""

    def __init__(self):
        """Initialize memory monitor."""
        self.process = psutil.Process()
        self.baseline_memory_mb = self.get_current_usage_mb()

    def get_current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def get_memory_delta_mb(self) -> float:
        """Get memory usage delta from baseline."""
        return self.get_current_usage_mb() - self.baseline_memory_mb

    def get_system_memory_info(self) -> dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_mb": memory.used / (1024 * 1024),
            "percent_used": memory.percent,
        }


# Singleton instance for global access
_performance_optimizer_instance: GraphPerformanceOptimizer | None = None


def get_graph_performance_optimizer() -> GraphPerformanceOptimizer:
    """
    Get the global Graph Performance Optimizer instance.

    Returns:
        GraphPerformanceOptimizer singleton instance
    """
    global _performance_optimizer_instance

    if _performance_optimizer_instance is None:
        _performance_optimizer_instance = GraphPerformanceOptimizer()

    return _performance_optimizer_instance
