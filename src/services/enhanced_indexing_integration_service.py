"""
Enhanced Indexing Integration Service for Concurrent Call Extraction.

This service integrates the concurrent function call extraction capabilities
with the existing indexing infrastructure, providing seamless performance
optimization for large codebases.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, FunctionCall
from src.services.batch_call_processing_service import BatchCallProcessingService, BatchProcessingSummary, BatchSchedulingConfig
from src.services.concurrent_call_extractor_service import ConcurrentProcessingConfig
from src.services.indexing_service import IndexingService
from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class IntegrationConfig:
    """Configuration for indexing integration."""

    enable_concurrent_processing: bool = True
    concurrent_threshold_files: int = 5  # Use concurrent processing for 5+ files
    concurrent_threshold_chunks: int = 50  # Or 50+ chunks
    enable_progress_callbacks: bool = True
    enable_performance_monitoring: bool = True
    save_intermediate_results: bool = True
    integration_timeout_seconds: float = 1800.0  # 30 minutes

    @classmethod
    def from_env(cls) -> "IntegrationConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            enable_concurrent_processing=os.getenv("INDEXING_CONCURRENT_ENABLED", "true").lower() == "true",
            concurrent_threshold_files=int(os.getenv("INDEXING_CONCURRENT_THRESHOLD_FILES", "5")),
            concurrent_threshold_chunks=int(os.getenv("INDEXING_CONCURRENT_THRESHOLD_CHUNKS", "50")),
            enable_progress_callbacks=os.getenv("INDEXING_PROGRESS_CALLBACKS", "true").lower() == "true",
            enable_performance_monitoring=os.getenv("INDEXING_PERFORMANCE_MONITOR", "true").lower() == "true",
            save_intermediate_results=os.getenv("INDEXING_SAVE_INTERMEDIATE", "true").lower() == "true",
            integration_timeout_seconds=float(os.getenv("INDEXING_INTEGRATION_TIMEOUT", "1800")),
        )


@dataclass
class IndexingResult:
    """Result of enhanced indexing with concurrent call extraction."""

    project_name: str
    total_files_processed: int
    total_chunks_processed: int
    total_calls_detected: int
    processing_time_ms: float
    used_concurrent_processing: bool
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    batch_summary: BatchProcessingSummary | None = None
    error_message: str | None = None
    success: bool = True

    @property
    def calls_per_file(self) -> float:
        """Calculate average calls per file."""
        return self.total_calls_detected / self.total_files_processed if self.total_files_processed > 0 else 0.0

    @property
    def processing_rate_chunks_per_second(self) -> float:
        """Calculate processing rate in chunks per second."""
        seconds = self.processing_time_ms / 1000.0
        return self.total_chunks_processed / seconds if seconds > 0 else 0.0


class EnhancedIndexingIntegrationService:
    """
    Service that integrates concurrent function call extraction with existing indexing.

    This service provides:
    - Intelligent decision making for when to use concurrent processing
    - Integration with existing IndexingService workflows
    - Performance monitoring and optimization
    - Progress tracking and reporting
    - Seamless fallback to sequential processing when appropriate
    """

    def __init__(
        self,
        integration_config: IntegrationConfig | None = None,
        scheduling_config: BatchSchedulingConfig | None = None,
        processing_config: ConcurrentProcessingConfig | None = None,
    ):
        """
        Initialize the enhanced indexing integration service.

        Args:
            integration_config: Integration configuration
            scheduling_config: Batch scheduling configuration
            processing_config: Concurrent processing configuration
        """
        self.integration_config = integration_config or IntegrationConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Initialize batch processing service
        self.batch_processor = BatchCallProcessingService(scheduling_config=scheduling_config, processing_config=processing_config)

        # Performance monitoring
        if self.integration_config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        # Integration statistics
        self._stats = {
            "total_indexing_operations": 0,
            "concurrent_operations": 0,
            "sequential_operations": 0,
            "total_files_processed": 0,
            "total_calls_detected": 0,
            "total_processing_time_ms": 0.0,
            "average_concurrent_speedup": 0.0,
        }

        self.logger.info(f"EnhancedIndexingIntegrationService initialized with config: {self.integration_config}")

    async def process_project_with_call_extraction(
        self, project_name: str, project_chunks: dict[str, list[CodeChunk]], progress_callback: callable | None = None
    ) -> IndexingResult:
        """
        Process a project with enhanced function call extraction.

        Args:
            project_name: Name of the project being processed
            project_chunks: Dictionary mapping file paths to code chunks
            progress_callback: Optional callback for progress updates

        Returns:
            IndexingResult with detailed processing information
        """
        start_time = time.time()

        self.logger.info(f"Starting enhanced indexing for project: {project_name}")

        try:
            # Update statistics
            self._stats["total_indexing_operations"] += 1

            # Determine processing strategy
            use_concurrent = await self._should_use_concurrent_processing(project_chunks)

            if use_concurrent:
                result = await self._process_with_concurrent_extraction(
                    project_name=project_name, project_chunks=project_chunks, progress_callback=progress_callback
                )
                self._stats["concurrent_operations"] += 1
            else:
                result = await self._process_with_sequential_extraction(
                    project_name=project_name, project_chunks=project_chunks, progress_callback=progress_callback
                )
                self._stats["sequential_operations"] += 1

            # Update global statistics
            self._stats["total_files_processed"] += result.total_files_processed
            self._stats["total_calls_detected"] += result.total_calls_detected
            self._stats["total_processing_time_ms"] += result.processing_time_ms

            # Calculate performance metrics
            result.performance_metrics.update(await self._collect_performance_metrics())

            self.logger.info(
                f"Enhanced indexing completed for {project_name}: "
                f"{result.total_files_processed} files, {result.total_calls_detected} calls, "
                f"{'concurrent' if use_concurrent else 'sequential'} processing"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in enhanced indexing for {project_name}: {e}")

            return IndexingResult(
                project_name=project_name,
                total_files_processed=len(project_chunks),
                total_chunks_processed=sum(len(chunks) for chunks in project_chunks.values()),
                total_calls_detected=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                used_concurrent_processing=False,
                error_message=str(e),
                success=False,
            )

    async def _should_use_concurrent_processing(self, project_chunks: dict[str, list[CodeChunk]]) -> bool:
        """
        Determine whether to use concurrent processing based on project characteristics.

        Args:
            project_chunks: Project chunks to analyze

        Returns:
            True if concurrent processing should be used
        """
        if not self.integration_config.enable_concurrent_processing:
            return False

        total_files = len(project_chunks)
        total_chunks = sum(len(chunks) for chunks in project_chunks.values())

        # Check thresholds
        meets_file_threshold = total_files >= self.integration_config.concurrent_threshold_files
        meets_chunk_threshold = total_chunks >= self.integration_config.concurrent_threshold_chunks

        # Additional heuristics
        avg_chunks_per_file = total_chunks / total_files if total_files > 0 else 0

        # Language analysis
        languages = set()
        for chunks in project_chunks.values():
            for chunk in chunks:
                if chunk.language:
                    languages.add(chunk.language)

        has_supported_languages = any(lang in ["python", "javascript", "typescript", "java"] for lang in languages)

        decision = (meets_file_threshold or meets_chunk_threshold) and has_supported_languages

        self.logger.info(
            f"Concurrent processing decision: {decision} "
            f"(files: {total_files}, chunks: {total_chunks}, "
            f"avg_chunks_per_file: {avg_chunks_per_file:.1f}, "
            f"languages: {languages})"
        )

        return decision

    async def _process_with_concurrent_extraction(
        self, project_name: str, project_chunks: dict[str, list[CodeChunk]], progress_callback: callable | None = None
    ) -> IndexingResult:
        """
        Process project using concurrent extraction.

        Args:
            project_name: Project name
            project_chunks: Project chunks
            progress_callback: Progress callback

        Returns:
            IndexingResult with concurrent processing results
        """
        start_time = time.time()

        try:
            # Create progress wrapper if needed
            wrapped_callback = None
            if progress_callback and self.integration_config.enable_progress_callbacks:

                def wrapped_callback(message: str, progress: float):
                    progress_callback(f"[Concurrent] {message}", progress)

            # Process with batch processor
            batch_summary = await asyncio.wait_for(
                self.batch_processor.process_codebase_calls(project_chunks=project_chunks, progress_callback=wrapped_callback),
                timeout=self.integration_config.integration_timeout_seconds,
            )

            # Create result
            result = IndexingResult(
                project_name=project_name,
                total_files_processed=batch_summary.total_files,
                total_chunks_processed=sum(
                    sum(len(file_result.call_detection_results) for file_result in batch_result.file_results)
                    for batch_result in batch_summary.batch_results
                ),
                total_calls_detected=batch_summary.total_calls_detected,
                processing_time_ms=(time.time() - start_time) * 1000,
                used_concurrent_processing=True,
                batch_summary=batch_summary,
                performance_metrics=batch_summary.performance_metrics,
            )

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"Concurrent processing timed out for {project_name}")

            return IndexingResult(
                project_name=project_name,
                total_files_processed=len(project_chunks),
                total_chunks_processed=sum(len(chunks) for chunks in project_chunks.values()),
                total_calls_detected=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                used_concurrent_processing=True,
                error_message="Processing timeout",
                success=False,
            )

        except Exception as e:
            self.logger.error(f"Error in concurrent processing for {project_name}: {e}")

            # Fallback to sequential processing
            self.logger.info(f"Falling back to sequential processing for {project_name}")
            return await self._process_with_sequential_extraction(
                project_name=project_name, project_chunks=project_chunks, progress_callback=progress_callback
            )

    async def _process_with_sequential_extraction(
        self, project_name: str, project_chunks: dict[str, list[CodeChunk]], progress_callback: callable | None = None
    ) -> IndexingResult:
        """
        Process project using sequential extraction (fallback).

        Args:
            project_name: Project name
            project_chunks: Project chunks
            progress_callback: Progress callback

        Returns:
            IndexingResult with sequential processing results
        """
        start_time = time.time()

        try:
            from src.services.function_call_extractor_service import FunctionCallExtractor

            extractor = FunctionCallExtractor()
            total_calls_detected = 0
            total_files = len(project_chunks)
            processed_files = 0

            for file_path, chunks in project_chunks.items():
                if progress_callback and self.integration_config.enable_progress_callbacks:
                    progress = processed_files / total_files
                    progress_callback(f"[Sequential] Processing {Path(file_path).name}", progress)

                for chunk in chunks:
                    try:
                        # Read file content
                        content_lines = []
                        try:
                            path = Path(file_path)
                            if path.exists():
                                content_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        except Exception:
                            pass

                        # Extract calls
                        breadcrumb = chunk.breadcrumb or f"{Path(file_path).stem}.{chunk.name}"
                        result = await extractor.extract_calls_from_chunk(
                            chunk=chunk, source_breadcrumb=breadcrumb, content_lines=content_lines
                        )

                        if result.success:
                            total_calls_detected += len(result.function_calls)

                    except Exception as e:
                        self.logger.warning(f"Error processing chunk {chunk.name} in {file_path}: {e}")
                        continue

                processed_files += 1

            if progress_callback:
                progress_callback("[Sequential] Processing complete", 1.0)

            return IndexingResult(
                project_name=project_name,
                total_files_processed=total_files,
                total_chunks_processed=sum(len(chunks) for chunks in project_chunks.values()),
                total_calls_detected=total_calls_detected,
                processing_time_ms=(time.time() - start_time) * 1000,
                used_concurrent_processing=False,
            )

        except Exception as e:
            self.logger.error(f"Error in sequential processing for {project_name}: {e}")

            return IndexingResult(
                project_name=project_name,
                total_files_processed=len(project_chunks),
                total_chunks_processed=sum(len(chunks) for chunks in project_chunks.values()),
                total_calls_detected=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                used_concurrent_processing=False,
                error_message=str(e),
                success=False,
            )

    async def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect comprehensive performance metrics."""
        metrics = {
            "integration_stats": self._stats.copy(),
            "batch_processor_stats": self.batch_processor.get_statistics(),
            "configuration": {
                "integration": self.integration_config.__dict__,
                "concurrent_threshold_files": self.integration_config.concurrent_threshold_files,
                "concurrent_threshold_chunks": self.integration_config.concurrent_threshold_chunks,
            },
        }

        if self.performance_monitor:
            metrics["performance_monitor"] = self.performance_monitor.get_metrics()

        return metrics

    async def optimize_for_codebase(self, project_chunks: dict[str, list[CodeChunk]]) -> dict[str, Any]:
        """
        Analyze codebase and provide optimization recommendations.

        Args:
            project_chunks: Project chunks to analyze

        Returns:
            Dictionary with optimization recommendations
        """
        total_files = len(project_chunks)
        total_chunks = sum(len(chunks) for chunks in project_chunks.values())

        # Language distribution
        language_counts = {}
        chunk_type_counts = {}
        file_size_distribution = []

        for file_path, chunks in project_chunks.items():
            file_size_distribution.append(len(chunks))

            for chunk in chunks:
                if chunk.language:
                    language_counts[chunk.language] = language_counts.get(chunk.language, 0) + 1
                if chunk.chunk_type:
                    chunk_type_counts[chunk.chunk_type] = chunk_type_counts.get(chunk.chunk_type, 0) + 1

        # Calculate statistics
        avg_chunks_per_file = total_chunks / total_files if total_files > 0 else 0
        max_chunks_per_file = max(file_size_distribution) if file_size_distribution else 0
        min_chunks_per_file = min(file_size_distribution) if file_size_distribution else 0

        # Generate recommendations
        recommendations = []

        if total_files >= self.integration_config.concurrent_threshold_files:
            recommendations.append("Use concurrent processing for improved performance")
        else:
            recommendations.append("Sequential processing recommended for small codebase")

        if avg_chunks_per_file > 20:
            recommendations.append("Consider increasing batch size for large files")

        if max_chunks_per_file > 100:
            recommendations.append("Very large files detected - consider file-level parallelization")

        if "python" in language_counts and language_counts["python"] > total_chunks * 0.8:
            recommendations.append("Python-dominant codebase - optimize for Python call patterns")

        return {
            "analysis": {
                "total_files": total_files,
                "total_chunks": total_chunks,
                "avg_chunks_per_file": avg_chunks_per_file,
                "max_chunks_per_file": max_chunks_per_file,
                "min_chunks_per_file": min_chunks_per_file,
                "language_distribution": language_counts,
                "chunk_type_distribution": chunk_type_counts,
            },
            "recommendations": recommendations,
            "optimal_config": {
                "use_concurrent": total_files >= self.integration_config.concurrent_threshold_files,
                "recommended_batch_size": min(100, max(10, total_files // 10)),
                "recommended_concurrency": min(20, max(2, total_files // 50)),
            },
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get integration service statistics."""
        return {
            "integration_stats": self._stats.copy(),
            "batch_processor_stats": self.batch_processor.get_statistics(),
            "configuration": self.integration_config.__dict__,
        }

    async def shutdown(self):
        """Shutdown the integration service."""
        self.logger.info("Shutting down EnhancedIndexingIntegrationService")
        await self.batch_processor.shutdown()
        self.logger.info("EnhancedIndexingIntegrationService shutdown complete")
