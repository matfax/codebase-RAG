"""
Incremental Call Detection Service for Modified Files Only.

This service provides intelligent incremental processing for function call detection,
only processing files that have been modified since the last analysis. It integrates
with the caching, concurrent processing, and change detection infrastructure.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models.breadcrumb_cache_models import FileModificationTracker
from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, FunctionCall
from src.services.breadcrumb_cache_service import BreadcrumbCacheService
from src.services.change_detector_service import ChangeDetectionResult, ChangeDetectorService, ChangeType
from src.services.concurrent_call_extractor_service import ConcurrentCallExtractor
from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class IncrementalProcessingConfig:
    """Configuration for incremental call detection processing."""

    enable_incremental_processing: bool = True
    change_detection_interval_seconds: float = 300.0  # 5 minutes
    enable_dependency_tracking: bool = True
    enable_cascade_reprocessing: bool = True  # Reprocess dependent files
    max_cascade_depth: int = 3
    force_reprocess_after_hours: float = 24.0  # Force reprocess after 24 hours
    batch_size_incremental: int = 50
    concurrent_incremental_files: int = 5
    enable_smart_invalidation: bool = True
    enable_performance_optimization: bool = True

    @classmethod
    def from_env(cls) -> "IncrementalProcessingConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            enable_incremental_processing=os.getenv("INCREMENTAL_CALL_DETECT_ENABLED", "true").lower() == "true",
            change_detection_interval_seconds=float(os.getenv("INCREMENTAL_CALL_DETECT_INTERVAL", "300")),
            enable_dependency_tracking=os.getenv("INCREMENTAL_CALL_DETECT_DEPENDENCIES", "true").lower() == "true",
            enable_cascade_reprocessing=os.getenv("INCREMENTAL_CALL_DETECT_CASCADE", "true").lower() == "true",
            max_cascade_depth=int(os.getenv("INCREMENTAL_CALL_DETECT_MAX_DEPTH", "3")),
            force_reprocess_after_hours=float(os.getenv("INCREMENTAL_CALL_DETECT_FORCE_HOURS", "24")),
            batch_size_incremental=int(os.getenv("INCREMENTAL_CALL_DETECT_BATCH_SIZE", "50")),
            concurrent_incremental_files=int(os.getenv("INCREMENTAL_CALL_DETECT_CONCURRENCY", "5")),
            enable_smart_invalidation=os.getenv("INCREMENTAL_CALL_DETECT_SMART_INVALIDATION", "true").lower() == "true",
            enable_performance_optimization=os.getenv("INCREMENTAL_CALL_DETECT_PERFORMANCE_OPT", "true").lower() == "true",
        )


@dataclass
class IncrementalProcessingResult:
    """Result of incremental call detection processing."""

    total_files_analyzed: int
    modified_files_processed: int
    unchanged_files_skipped: int
    cascade_files_processed: int
    total_calls_detected: int
    processing_time_ms: float
    cache_hits: int
    cache_misses: int
    performance_improvement_percent: float
    change_detection_result: ChangeDetectionResult | None = None
    error_files: list[str] = field(default_factory=list)

    @property
    def efficiency_ratio(self) -> float:
        """Calculate processing efficiency (files processed vs total files)."""
        return (self.modified_files_processed / self.total_files_analyzed * 100) if self.total_files_analyzed > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


@dataclass
class DependencyTracker:
    """Tracks file dependencies for cascade reprocessing."""

    file_path: str
    dependencies: set[str] = field(default_factory=set)  # Files this file depends on
    dependents: set[str] = field(default_factory=set)  # Files that depend on this file
    last_updated: float = field(default_factory=time.time)

    def add_dependency(self, dependency_path: str):
        """Add a dependency for this file."""
        self.dependencies.add(dependency_path)
        self.last_updated = time.time()

    def add_dependent(self, dependent_path: str):
        """Add a file that depends on this file."""
        self.dependents.add(dependent_path)
        self.last_updated = time.time()

    def get_cascade_files(self, max_depth: int) -> set[str]:
        """Get all files that should be reprocessed due to dependencies."""
        cascade_files = set()
        visited = set()

        def traverse_dependents(file_path: str, current_depth: int):
            if current_depth >= max_depth or file_path in visited:
                return

            visited.add(file_path)
            cascade_files.add(file_path)

            # Add files that depend on this file
            for dependent in self.dependents:
                traverse_dependents(dependent, current_depth + 1)

        traverse_dependents(self.file_path, 0)
        cascade_files.discard(self.file_path)  # Don't include self

        return cascade_files


class IncrementalCallDetectionService:
    """
    Service for incremental function call detection on modified files only.

    This service provides intelligent incremental processing that:
    - Detects file changes using existing change detection infrastructure
    - Only processes modified files to save computation time
    - Tracks dependencies and cascades reprocessing when needed
    - Integrates with caching and concurrent processing systems
    - Provides comprehensive performance monitoring and optimization
    """

    def __init__(
        self,
        config: IncrementalProcessingConfig | None = None,
        change_detector: ChangeDetectorService | None = None,
        concurrent_extractor: ConcurrentCallExtractor | None = None,
        cache_service: BreadcrumbCacheService | None = None,
    ):
        """
        Initialize the incremental call detection service.

        Args:
            config: Configuration for incremental processing
            change_detector: Change detection service
            concurrent_extractor: Concurrent call extraction service
            cache_service: Breadcrumb cache service
        """
        self.config = config or IncrementalProcessingConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Core services
        self.change_detector = change_detector
        self.concurrent_extractor = concurrent_extractor
        self.cache_service = cache_service

        # Dependency tracking
        self._dependency_graph: dict[str, DependencyTracker] = {}
        self._last_full_processing: dict[str, float] = {}  # project -> timestamp

        # Performance monitoring
        if self.config.enable_performance_optimization:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        # Processing statistics
        self._stats = {
            "total_incremental_operations": 0,
            "total_files_skipped": 0,
            "total_processing_time_saved_ms": 0.0,
            "average_efficiency_ratio": 0.0,
            "cascade_operations": 0,
            "dependency_updates": 0,
        }

        self.logger.info(f"IncrementalCallDetectionService initialized with config: {self.config}")

    async def process_project_incrementally(
        self,
        project_name: str,
        project_directory: str,
        all_project_chunks: dict[str, list[CodeChunk]],
        breadcrumb_mapping: dict[str, str],
        progress_callback: callable | None = None,
    ) -> IncrementalProcessingResult:
        """
        Process a project using incremental call detection.

        Args:
            project_name: Name of the project
            project_directory: Project root directory
            all_project_chunks: All project chunks for reference
            breadcrumb_mapping: Mapping of chunks to breadcrumbs
            progress_callback: Optional progress callback

        Returns:
            IncrementalProcessingResult with processing details
        """
        start_time = time.time()

        if not self.config.enable_incremental_processing:
            # Fall back to full processing
            return await self._process_full_project(project_name, all_project_chunks, breadcrumb_mapping, progress_callback)

        try:
            # Detect changes
            if progress_callback:
                progress_callback("Detecting file changes...", 0.1)

            change_result = await self._detect_project_changes(project_name, project_directory, all_project_chunks)

            if not change_result or not change_result.has_changes:
                # No changes detected
                self.logger.info(f"No changes detected in project {project_name}")
                return IncrementalProcessingResult(
                    total_files_analyzed=len(all_project_chunks),
                    modified_files_processed=0,
                    unchanged_files_skipped=len(all_project_chunks),
                    cascade_files_processed=0,
                    total_calls_detected=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    cache_hits=0,
                    cache_misses=0,
                    performance_improvement_percent=100.0,  # No processing needed
                    change_detection_result=change_result,
                )

            # Determine files to process
            if progress_callback:
                progress_callback("Analyzing dependencies...", 0.2)

            files_to_process = await self._determine_files_to_process(change_result, all_project_chunks, project_name)

            if not files_to_process:
                return IncrementalProcessingResult(
                    total_files_analyzed=len(all_project_chunks),
                    modified_files_processed=0,
                    unchanged_files_skipped=len(all_project_chunks),
                    cascade_files_processed=0,
                    total_calls_detected=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    cache_hits=0,
                    cache_misses=0,
                    performance_improvement_percent=100.0,
                    change_detection_result=change_result,
                )

            # Process modified files
            if progress_callback:
                progress_callback(f"Processing {len(files_to_process)} modified files...", 0.3)

            processing_result = await self._process_modified_files(
                files_to_process, all_project_chunks, breadcrumb_mapping, progress_callback
            )

            # Update dependencies
            if self.config.enable_dependency_tracking:
                await self._update_dependency_graph(files_to_process, all_project_chunks)

            # Invalidate caches for modified files
            if self.config.enable_smart_invalidation and self.cache_service:
                await self._invalidate_caches_for_modified_files(files_to_process)

            # Calculate performance metrics
            processing_time = (time.time() - start_time) * 1000
            total_files = len(all_project_chunks)
            processed_files = len(files_to_process)

            # Estimate time saved
            estimated_full_time = processing_time * (total_files / processed_files) if processed_files > 0 else processing_time
            time_saved = max(0, estimated_full_time - processing_time)
            performance_improvement = (time_saved / estimated_full_time * 100) if estimated_full_time > 0 else 0

            # Update statistics
            self._stats["total_incremental_operations"] += 1
            self._stats["total_files_skipped"] += total_files - processed_files
            self._stats["total_processing_time_saved_ms"] += time_saved

            # Calculate cascade files
            cascade_count = 0
            for file_path in files_to_process:
                if file_path in self._dependency_graph:
                    cascade_files = self._dependency_graph[file_path].get_cascade_files(self.config.max_cascade_depth)
                    cascade_count += len(cascade_files)

            result = IncrementalProcessingResult(
                total_files_analyzed=total_files,
                modified_files_processed=processed_files,
                unchanged_files_skipped=total_files - processed_files,
                cascade_files_processed=cascade_count,
                total_calls_detected=processing_result.get("total_calls", 0),
                processing_time_ms=processing_time,
                cache_hits=processing_result.get("cache_hits", 0),
                cache_misses=processing_result.get("cache_misses", 0),
                performance_improvement_percent=performance_improvement,
                change_detection_result=change_result,
                error_files=processing_result.get("error_files", []),
            )

            if progress_callback:
                progress_callback("Incremental processing complete", 1.0)

            self.logger.info(
                f"Incremental processing completed for {project_name}: "
                f"{processed_files}/{total_files} files processed, "
                f"{performance_improvement:.1f}% time saved"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in incremental processing for {project_name}: {e}")

            # Fall back to full processing
            return await self._process_full_project(project_name, all_project_chunks, breadcrumb_mapping, progress_callback)

    async def _detect_project_changes(
        self, project_name: str, project_directory: str, all_project_chunks: dict[str, list[CodeChunk]]
    ) -> ChangeDetectionResult | None:
        """Detect changes in the project files."""
        if not self.change_detector:
            # No change detector available, assume all files changed
            return None

        try:
            # Get list of current files
            current_files = list(all_project_chunks.keys())

            # Detect changes
            changes = self.change_detector.detect_changes(
                project_name=project_name, current_files=current_files, project_root=project_directory
            )

            return changes

        except Exception as e:
            self.logger.error(f"Error detecting changes for project {project_name}: {e}")
            return None

    async def _determine_files_to_process(
        self, change_result: ChangeDetectionResult, all_project_chunks: dict[str, list[CodeChunk]], project_name: str
    ) -> list[str]:
        """Determine which files need to be processed based on changes and dependencies."""
        files_to_process = set()

        # Add modified and new files
        for change in change_result.modified_files + change_result.added_files:
            if change.file_path in all_project_chunks:
                files_to_process.add(change.file_path)

        # Handle cascade reprocessing for dependencies
        if self.config.enable_cascade_reprocessing:
            cascade_files = set()

            for file_path in files_to_process.copy():
                if file_path in self._dependency_graph:
                    cascade = self._dependency_graph[file_path].get_cascade_files(self.config.max_cascade_depth)
                    cascade_files.update(cascade)

            # Add cascade files that exist in the project
            for cascade_file in cascade_files:
                if cascade_file in all_project_chunks:
                    files_to_process.add(cascade_file)

            if cascade_files:
                self._stats["cascade_operations"] += 1
                self.logger.info(f"Added {len(cascade_files)} cascade files for reprocessing")

        # Check for forced reprocessing (files not processed in too long)
        force_reprocess_threshold = time.time() - (self.config.force_reprocess_after_hours * 3600)
        last_processing = self._last_full_processing.get(project_name, 0)

        if last_processing < force_reprocess_threshold:
            # Force reprocess all files if it's been too long
            files_to_process.update(all_project_chunks.keys())
            self._last_full_processing[project_name] = time.time()
            self.logger.info(f"Forcing full reprocessing for {project_name} (last processed: {datetime.fromtimestamp(last_processing)})")

        return list(files_to_process)

    async def _process_modified_files(
        self,
        files_to_process: list[str],
        all_project_chunks: dict[str, list[CodeChunk]],
        breadcrumb_mapping: dict[str, str],
        progress_callback: callable | None = None,
    ) -> dict[str, Any]:
        """Process the modified files using concurrent extraction."""
        if not self.concurrent_extractor:
            # No concurrent extractor available
            return {"total_calls": 0, "cache_hits": 0, "cache_misses": 0, "error_files": []}

        # Prepare chunks for processing
        modified_chunks = {}
        for file_path in files_to_process:
            if file_path in all_project_chunks:
                modified_chunks[file_path] = all_project_chunks[file_path]

        if not modified_chunks:
            return {"total_calls": 0, "cache_hits": 0, "cache_misses": 0, "error_files": []}

        try:
            # Use concurrent extractor with incremental configuration
            batch_result = await self.concurrent_extractor.extract_calls_from_files(
                file_chunks=modified_chunks, breadcrumb_mapping=breadcrumb_mapping
            )

            return {
                "total_calls": batch_result.total_calls_detected,
                "cache_hits": batch_result.performance_metrics.get("cache_hits", 0),
                "cache_misses": batch_result.performance_metrics.get("cache_misses", 0),
                "error_files": [result.file_path for result in batch_result.file_results if not result.success],
            }

        except Exception as e:
            self.logger.error(f"Error processing modified files: {e}")
            return {"total_calls": 0, "cache_hits": 0, "cache_misses": 0, "error_files": files_to_process}

    async def _update_dependency_graph(self, processed_files: list[str], all_project_chunks: dict[str, list[CodeChunk]]):
        """Update the dependency graph based on processed files."""
        for file_path in processed_files:
            if file_path not in self._dependency_graph:
                self._dependency_graph[file_path] = DependencyTracker(file_path)

            # Analyze chunks for dependencies
            chunks = all_project_chunks.get(file_path, [])
            tracker = self._dependency_graph[file_path]

            for chunk in chunks:
                # Look for import statements or function calls that indicate dependencies
                if hasattr(chunk, "imports_used") and chunk.imports_used:
                    for import_path in chunk.imports_used:
                        # Find corresponding file path
                        for potential_file in all_project_chunks.keys():
                            if import_path in potential_file or potential_file.endswith(f"{import_path}.py"):
                                tracker.add_dependency(potential_file)

                                # Update reverse dependency
                                if potential_file not in self._dependency_graph:
                                    self._dependency_graph[potential_file] = DependencyTracker(potential_file)
                                self._dependency_graph[potential_file].add_dependent(file_path)

        self._stats["dependency_updates"] += len(processed_files)

    async def _invalidate_caches_for_modified_files(self, modified_files: list[str]):
        """Invalidate caches for modified files."""
        if not self.cache_service:
            return

        invalidated_count = 0
        for file_path in modified_files:
            count = await self.cache_service.invalidate_by_file(file_path)
            invalidated_count += count

        if invalidated_count > 0:
            self.logger.info(f"Invalidated {invalidated_count} cache entries for modified files")

    async def _process_full_project(
        self,
        project_name: str,
        all_project_chunks: dict[str, list[CodeChunk]],
        breadcrumb_mapping: dict[str, str],
        progress_callback: callable | None = None,
    ) -> IncrementalProcessingResult:
        """Fall back to full project processing."""
        start_time = time.time()

        if progress_callback:
            progress_callback("Processing all files (full processing)...", 0.5)

        try:
            if self.concurrent_extractor:
                batch_result = await self.concurrent_extractor.extract_calls_from_files(
                    file_chunks=all_project_chunks, breadcrumb_mapping=breadcrumb_mapping
                )

                total_calls = batch_result.total_calls_detected
                cache_hits = batch_result.performance_metrics.get("cache_hits", 0)
                cache_misses = batch_result.performance_metrics.get("cache_misses", 0)
                error_files = [result.file_path for result in batch_result.file_results if not result.success]
            else:
                total_calls = 0
                cache_hits = 0
                cache_misses = 0
                error_files = []

            processing_time = (time.time() - start_time) * 1000
            total_files = len(all_project_chunks)

            # Update last full processing time
            self._last_full_processing[project_name] = time.time()

            return IncrementalProcessingResult(
                total_files_analyzed=total_files,
                modified_files_processed=total_files,
                unchanged_files_skipped=0,
                cascade_files_processed=0,
                total_calls_detected=total_calls,
                processing_time_ms=processing_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                performance_improvement_percent=0.0,  # No improvement for full processing
                error_files=error_files,
            )

        except Exception as e:
            self.logger.error(f"Error in full project processing for {project_name}: {e}")

            return IncrementalProcessingResult(
                total_files_analyzed=len(all_project_chunks),
                modified_files_processed=0,
                unchanged_files_skipped=0,
                cascade_files_processed=0,
                total_calls_detected=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                cache_hits=0,
                cache_misses=0,
                performance_improvement_percent=0.0,
                error_files=list(all_project_chunks.keys()),
            )

    def get_dependency_graph_info(self) -> dict[str, Any]:
        """Get information about the dependency graph."""
        total_dependencies = sum(len(tracker.dependencies) for tracker in self._dependency_graph.values())
        total_dependents = sum(len(tracker.dependents) for tracker in self._dependency_graph.values())

        return {
            "total_tracked_files": len(self._dependency_graph),
            "total_dependencies": total_dependencies,
            "total_dependents": total_dependents,
            "average_dependencies_per_file": total_dependencies / len(self._dependency_graph) if self._dependency_graph else 0,
            "files_with_most_dependencies": sorted(
                [(path, len(tracker.dependencies)) for path, tracker in self._dependency_graph.items()], key=lambda x: x[1], reverse=True
            )[:10],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive service statistics."""
        return {
            "incremental_stats": self._stats.copy(),
            "dependency_graph_info": self.get_dependency_graph_info(),
            "configuration": self.config.__dict__,
            "last_full_processing": {
                project: datetime.fromtimestamp(timestamp).isoformat() for project, timestamp in self._last_full_processing.items()
            },
        }

    async def force_full_reprocessing(self, project_name: str):
        """Force full reprocessing for a project on next run."""
        self._last_full_processing[project_name] = 0
        self.logger.info(f"Marked project {project_name} for forced full reprocessing")

    def clear_dependency_graph(self):
        """Clear the dependency graph to free memory."""
        self._dependency_graph.clear()
        self.logger.info("Dependency graph cleared")

    async def shutdown(self):
        """Shutdown the incremental service."""
        self.logger.info("Shutting down IncrementalCallDetectionService")
        self.clear_dependency_graph()
        self.logger.info("IncrementalCallDetectionService shutdown complete")
