"""
Batch Call Processing Service for Large Codebase Support.

This service coordinates concurrent function call extraction across large codebases,
integrating with the existing indexing infrastructure and providing intelligent
batch scheduling and resource management.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, FunctionCall
from src.services.concurrent_call_extractor_service import BatchProcessingResult, ConcurrentCallExtractor, ConcurrentProcessingConfig
from src.services.indexing_service import IndexingService
from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class BatchSchedulingConfig:
    """Configuration for batch scheduling and processing."""

    max_files_per_batch: int = 100
    max_chunks_per_batch: int = 1000
    batch_overlap_chunks: int = 10
    adaptive_batch_sizing: bool = True
    prioritize_by_file_size: bool = True
    prioritize_by_language: bool = True
    language_priorities: dict[str, int] = field(
        default_factory=lambda: {"python": 1, "javascript": 2, "typescript": 2, "java": 3, "cpp": 4}
    )
    min_batch_size: int = 10
    max_batch_size: int = 500
    target_processing_time_ms: float = 30000.0  # 30 seconds per batch
    enable_smart_grouping: bool = True

    @classmethod
    def from_env(cls) -> "BatchSchedulingConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            max_files_per_batch=int(os.getenv("BATCH_CALL_PROCESS_MAX_FILES", "100")),
            max_chunks_per_batch=int(os.getenv("BATCH_CALL_PROCESS_MAX_CHUNKS", "1000")),
            batch_overlap_chunks=int(os.getenv("BATCH_CALL_PROCESS_OVERLAP", "10")),
            adaptive_batch_sizing=os.getenv("BATCH_CALL_PROCESS_ADAPTIVE", "true").lower() == "true",
            prioritize_by_file_size=os.getenv("BATCH_CALL_PROCESS_PRIORITIZE_SIZE", "true").lower() == "true",
            prioritize_by_language=os.getenv("BATCH_CALL_PROCESS_PRIORITIZE_LANG", "true").lower() == "true",
            min_batch_size=int(os.getenv("BATCH_CALL_PROCESS_MIN_BATCH", "10")),
            max_batch_size=int(os.getenv("BATCH_CALL_PROCESS_MAX_BATCH", "500")),
            target_processing_time_ms=float(os.getenv("BATCH_CALL_PROCESS_TARGET_TIME", "30000")),
            enable_smart_grouping=os.getenv("BATCH_CALL_PROCESS_SMART_GROUP", "true").lower() == "true",
        )


@dataclass
class FileBatchInfo:
    """Information about a file for batch processing."""

    file_path: str
    chunks: list[CodeChunk]
    total_chunks: int
    language: str
    estimated_size: int
    priority_score: float
    estimated_processing_time_ms: float = 0.0

    @property
    def size_category(self) -> str:
        """Categorize file size for batch grouping."""
        if self.total_chunks < 10:
            return "small"
        elif self.total_chunks < 50:
            return "medium"
        elif self.total_chunks < 200:
            return "large"
        else:
            return "xlarge"


@dataclass
class ProcessingBatch:
    """A batch of files for concurrent processing."""

    batch_id: str
    files: list[FileBatchInfo]
    total_files: int
    total_chunks: int
    estimated_processing_time_ms: float
    language_distribution: dict[str, int] = field(default_factory=dict)
    size_distribution: dict[str, int] = field(default_factory=dict)

    def to_file_chunks_dict(self) -> dict[str, list[CodeChunk]]:
        """Convert batch to file_chunks dictionary for processing."""
        return {file_info.file_path: file_info.chunks for file_info in self.files}

    def create_breadcrumb_mapping(self) -> dict[str, str]:
        """Create breadcrumb mapping for all chunks in the batch."""
        mapping = {}
        for file_info in self.files:
            for chunk in file_info.chunks:
                if chunk.breadcrumb:
                    mapping[chunk.name] = chunk.breadcrumb
                else:
                    # Generate fallback breadcrumb
                    file_stem = Path(file_info.file_path).stem
                    mapping[chunk.name] = f"{file_stem}.{chunk.name}"
        return mapping


@dataclass
class BatchProcessingSummary:
    """Summary of complete batch processing operation."""

    total_files: int
    total_batches: int
    successful_batches: int
    failed_batches: int
    total_calls_detected: int
    total_processing_time_ms: float
    batch_results: list[BatchProcessingResult] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        return (self.successful_batches / self.total_batches * 100) if self.total_batches > 0 else 0.0

    @property
    def calls_per_minute(self) -> float:
        """Calculate calls detected per minute."""
        minutes = self.total_processing_time_ms / (60 * 1000)
        return self.total_calls_detected / minutes if minutes > 0 else 0.0


class BatchCallProcessingService:
    """
    Service for batch processing function call extraction across large codebases.

    This service coordinates with the existing indexing infrastructure to provide:
    - Intelligent batch scheduling based on file characteristics
    - Adaptive resource management
    - Progress tracking and monitoring
    - Integration with concurrent extraction
    - Performance optimization for large codebases
    """

    def __init__(
        self,
        scheduling_config: BatchSchedulingConfig | None = None,
        processing_config: ConcurrentProcessingConfig | None = None,
    ):
        """
        Initialize the batch call processing service.

        Args:
            scheduling_config: Batch scheduling configuration
            processing_config: Concurrent processing configuration
        """
        self.scheduling_config = scheduling_config or BatchSchedulingConfig.from_env()
        self.processing_config = processing_config or ConcurrentProcessingConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Initialize concurrent extractor
        self.concurrent_extractor = ConcurrentCallExtractor(self.processing_config)

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()

        # Batch statistics
        self._batch_stats = {
            "total_batches_processed": 0,
            "total_files_processed": 0,
            "total_calls_detected": 0,
            "adaptive_adjustments": 0,
            "optimal_batch_size": self.scheduling_config.max_files_per_batch,
        }

        self.logger.info(f"BatchCallProcessingService initialized with scheduling_config: {self.scheduling_config}")

    async def process_codebase_calls(
        self, project_chunks: dict[str, list[CodeChunk]], progress_callback: callable | None = None
    ) -> BatchProcessingSummary:
        """
        Process function call extraction for an entire codebase.

        Args:
            project_chunks: Dictionary mapping file paths to their code chunks
            progress_callback: Optional callback for progress updates

        Returns:
            BatchProcessingSummary with complete processing results
        """
        start_time = time.time()

        self.logger.info(f"Starting batch call processing for {len(project_chunks)} files")

        try:
            # Prepare file information for batch scheduling
            file_infos = await self._prepare_file_infos(project_chunks)

            # Create processing batches
            batches = await self._create_processing_batches(file_infos)

            self.logger.info(f"Created {len(batches)} processing batches")

            # Process batches
            batch_results = []
            successful_batches = 0
            failed_batches = 0
            total_calls = 0

            for i, batch in enumerate(batches):
                if progress_callback:
                    progress_callback(f"Processing batch {i+1}/{len(batches)}", i / len(batches))

                try:
                    # Process batch
                    batch_result = await self._process_batch(batch)
                    batch_results.append(batch_result)

                    if batch_result.successful_files > 0:
                        successful_batches += 1
                        total_calls += batch_result.total_calls_detected
                    else:
                        failed_batches += 1

                    # Update statistics
                    self._batch_stats["total_batches_processed"] += 1
                    self._batch_stats["total_files_processed"] += batch_result.total_files
                    self._batch_stats["total_calls_detected"] += batch_result.total_calls_detected

                    # Adaptive batch size adjustment
                    if self.scheduling_config.adaptive_batch_sizing:
                        await self._adjust_batch_size(batch, batch_result)

                except Exception as e:
                    self.logger.error(f"Failed to process batch {i+1}: {e}")
                    failed_batches += 1

                    # Create error result
                    error_result = BatchProcessingResult(
                        total_files=batch.total_files,
                        successful_files=0,
                        failed_files=batch.total_files,
                        total_calls_detected=0,
                        total_processing_time_ms=0.0,
                        performance_metrics={"error": str(e)},
                    )
                    batch_results.append(error_result)

            # Create summary
            total_processing_time = (time.time() - start_time) * 1000

            summary = BatchProcessingSummary(
                total_files=len(project_chunks),
                total_batches=len(batches),
                successful_batches=successful_batches,
                failed_batches=failed_batches,
                total_calls_detected=total_calls,
                total_processing_time_ms=total_processing_time,
                batch_results=batch_results,
                performance_metrics=await self._collect_performance_metrics(),
            )

            self.logger.info(
                f"Batch processing completed: {summary.successful_batches}/{len(batches)} batches, "
                f"{total_calls} calls detected in {total_processing_time:.2f}ms"
            )

            if progress_callback:
                progress_callback("Processing complete", 1.0)

            return summary

        except Exception as e:
            self.logger.error(f"Fatal error in batch processing: {e}")
            raise

    async def _prepare_file_infos(self, project_chunks: dict[str, list[CodeChunk]]) -> list[FileBatchInfo]:
        """Prepare file information for batch scheduling."""
        file_infos = []

        for file_path, chunks in project_chunks.items():
            if not chunks:
                continue

            # Determine primary language
            languages = [chunk.language for chunk in chunks if chunk.language]
            primary_language = max(set(languages), key=languages.count) if languages else "unknown"

            # Calculate priority score
            priority_score = await self._calculate_priority_score(file_path, chunks, primary_language)

            # Estimate processing time
            estimated_time = await self._estimate_processing_time(chunks, primary_language)

            file_info = FileBatchInfo(
                file_path=file_path,
                chunks=chunks,
                total_chunks=len(chunks),
                language=primary_language,
                estimated_size=sum(len(chunk.content) for chunk in chunks),
                priority_score=priority_score,
                estimated_processing_time_ms=estimated_time,
            )

            file_infos.append(file_info)

        # Sort by priority score (descending)
        file_infos.sort(key=lambda x: x.priority_score, reverse=True)

        return file_infos

    async def _calculate_priority_score(self, file_path: str, chunks: list[CodeChunk], language: str) -> float:
        """Calculate priority score for file processing order."""
        score = 0.0

        # Language priority
        if self.scheduling_config.prioritize_by_language:
            lang_priority = self.scheduling_config.language_priorities.get(language, 10)
            score += (10 - lang_priority) * 10  # Higher priority = lower number

        # Size priority (smaller files first for faster initial results)
        if self.scheduling_config.prioritize_by_file_size:
            chunk_count = len(chunks)
            if chunk_count <= 10:
                score += 50  # Small files get high priority
            elif chunk_count <= 50:
                score += 30  # Medium files get medium priority
            else:
                score += 10  # Large files get lower priority

        # Function/class density (more likely to have calls)
        function_chunks = sum(1 for chunk in chunks if chunk.chunk_type in ["function", "method", "class"])
        density = function_chunks / len(chunks) if chunks else 0
        score += density * 20

        # File type bonus
        path = Path(file_path)
        if path.suffix in [".py", ".js", ".ts", ".java"]:
            score += 20

        return score

    async def _estimate_processing_time(self, chunks: list[CodeChunk], language: str) -> float:
        """Estimate processing time for chunks based on historical data."""
        # Base time per chunk (in ms)
        base_time_per_chunk = 50.0

        # Language complexity multiplier
        language_multipliers = {"python": 1.0, "javascript": 1.2, "typescript": 1.3, "java": 1.4, "cpp": 1.6, "unknown": 2.0}

        multiplier = language_multipliers.get(language, 1.5)

        # Content complexity factor
        total_content_length = sum(len(chunk.content) for chunk in chunks)
        complexity_factor = min(2.0, 1.0 + (total_content_length / 10000))

        return len(chunks) * base_time_per_chunk * multiplier * complexity_factor

    async def _create_processing_batches(self, file_infos: list[FileBatchInfo]) -> list[ProcessingBatch]:
        """Create optimized processing batches from file information."""
        batches = []
        current_batch_files = []
        current_batch_chunks = 0
        current_batch_time = 0.0
        batch_id = 0

        for file_info in file_infos:
            # Check if adding this file would exceed batch limits
            would_exceed_files = len(current_batch_files) >= self.scheduling_config.max_files_per_batch
            would_exceed_chunks = (current_batch_chunks + file_info.total_chunks) > self.scheduling_config.max_chunks_per_batch
            would_exceed_time = (
                current_batch_time + file_info.estimated_processing_time_ms
            ) > self.scheduling_config.target_processing_time_ms

            # Start new batch if limits exceeded or smart grouping suggests it
            if current_batch_files and (
                would_exceed_files
                or would_exceed_chunks
                or (self.scheduling_config.adaptive_batch_sizing and would_exceed_time)
                or (self.scheduling_config.enable_smart_grouping and not self._should_group_together(current_batch_files[-1], file_info))
            ):
                # Create batch from current files
                batch = await self._create_batch(batch_id, current_batch_files)
                batches.append(batch)

                # Reset for next batch
                batch_id += 1
                current_batch_files = []
                current_batch_chunks = 0
                current_batch_time = 0.0

            # Add file to current batch
            current_batch_files.append(file_info)
            current_batch_chunks += file_info.total_chunks
            current_batch_time += file_info.estimated_processing_time_ms

        # Create final batch if there are remaining files
        if current_batch_files:
            batch = await self._create_batch(batch_id, current_batch_files)
            batches.append(batch)

        return batches

    def _should_group_together(self, file1: FileBatchInfo, file2: FileBatchInfo) -> bool:
        """Determine if two files should be grouped in the same batch."""
        if not self.scheduling_config.enable_smart_grouping:
            return True

        # Group by language
        if file1.language != file2.language:
            return False

        # Group by similar size categories
        if file1.size_category != file2.size_category:
            # Allow grouping adjacent size categories
            size_order = ["small", "medium", "large", "xlarge"]
            try:
                idx1 = size_order.index(file1.size_category)
                idx2 = size_order.index(file2.size_category)
                return abs(idx1 - idx2) <= 1
            except ValueError:
                return True

        return True

    async def _create_batch(self, batch_id: int, file_infos: list[FileBatchInfo]) -> ProcessingBatch:
        """Create a processing batch from file information."""
        total_chunks = sum(file_info.total_chunks for file_info in file_infos)
        estimated_time = sum(file_info.estimated_processing_time_ms for file_info in file_infos)

        # Calculate distributions
        language_dist = defaultdict(int)
        size_dist = defaultdict(int)

        for file_info in file_infos:
            language_dist[file_info.language] += 1
            size_dist[file_info.size_category] += 1

        return ProcessingBatch(
            batch_id=f"batch_{batch_id}",
            files=file_infos,
            total_files=len(file_infos),
            total_chunks=total_chunks,
            estimated_processing_time_ms=estimated_time,
            language_distribution=dict(language_dist),
            size_distribution=dict(size_dist),
        )

    async def _process_batch(self, batch: ProcessingBatch) -> BatchProcessingResult:
        """Process a single batch using concurrent extraction."""
        self.logger.info(f"Processing {batch.batch_id} with {batch.total_files} files, {batch.total_chunks} chunks")

        # Convert batch to format expected by concurrent extractor
        file_chunks = batch.to_file_chunks_dict()
        breadcrumb_mapping = batch.create_breadcrumb_mapping()

        # Process with concurrent extractor
        result = await self.concurrent_extractor.extract_calls_from_files(file_chunks=file_chunks, breadcrumb_mapping=breadcrumb_mapping)

        # Add batch information to result
        result.performance_metrics.update(
            {
                "batch_id": batch.batch_id,
                "estimated_time_ms": batch.estimated_processing_time_ms,
                "language_distribution": batch.language_distribution,
                "size_distribution": batch.size_distribution,
            }
        )

        return result

    async def _adjust_batch_size(self, batch: ProcessingBatch, result: BatchProcessingResult):
        """Adjust batch sizing based on processing results."""
        # Calculate efficiency metrics
        actual_time = result.total_processing_time_ms
        estimated_time = batch.estimated_processing_time_ms
        time_accuracy = min(actual_time, estimated_time) / max(actual_time, estimated_time) if max(actual_time, estimated_time) > 0 else 0

        # Adjust optimal batch size
        if actual_time < self.scheduling_config.target_processing_time_ms * 0.5:
            # Batch processed too quickly, can increase size
            self._batch_stats["optimal_batch_size"] = min(
                self.scheduling_config.max_batch_size, int(self._batch_stats["optimal_batch_size"] * 1.2)
            )
        elif actual_time > self.scheduling_config.target_processing_time_ms * 1.5:
            # Batch took too long, decrease size
            self._batch_stats["optimal_batch_size"] = max(
                self.scheduling_config.min_batch_size, int(self._batch_stats["optimal_batch_size"] * 0.8)
            )

        self._batch_stats["adaptive_adjustments"] += 1

        self.logger.debug(
            f"Batch size adjustment: optimal_size={self._batch_stats['optimal_batch_size']}, " f"time_accuracy={time_accuracy:.2f}"
        )

    async def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect comprehensive performance metrics."""
        extractor_stats = self.concurrent_extractor.get_statistics()

        return {
            "batch_stats": self._batch_stats.copy(),
            "extractor_stats": extractor_stats,
            "performance_monitor": self.performance_monitor.get_metrics(),
            "config": {
                "scheduling": {
                    "max_files_per_batch": self.scheduling_config.max_files_per_batch,
                    "adaptive_batch_sizing": self.scheduling_config.adaptive_batch_sizing,
                    "enable_smart_grouping": self.scheduling_config.enable_smart_grouping,
                },
                "processing": {
                    "max_concurrent_files": self.processing_config.max_concurrent_files,
                    "enable_adaptive_concurrency": self.processing_config.enable_adaptive_concurrency,
                },
            },
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get current processing statistics."""
        return {
            "batch_stats": self._batch_stats.copy(),
            "concurrent_extractor_stats": self.concurrent_extractor.get_statistics(),
            "configuration": {"scheduling": self.scheduling_config.__dict__, "processing": self.processing_config.__dict__},
        }

    async def shutdown(self):
        """Shutdown the batch processing service."""
        self.logger.info("Shutting down BatchCallProcessingService")
        await self.concurrent_extractor.shutdown()
        self.logger.info("BatchCallProcessingService shutdown complete")
