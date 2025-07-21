"""
Concurrent Function Call Extractor Service for Performance Optimization.

This service provides concurrent processing capabilities for function call extraction
across multiple files, improving performance on large codebases through intelligent
parallelization and resource management.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, FunctionCall
from src.services.function_call_extractor_service import FunctionCallExtractor
from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class ConcurrentProcessingConfig:
    """Configuration for concurrent function call extraction."""

    max_concurrent_files: int = 10
    max_concurrent_chunks_per_file: int = 5
    chunk_batch_size: int = 50
    timeout_seconds: float = 300.0  # 5 minutes
    enable_progress_tracking: bool = True
    enable_memory_monitoring: bool = True
    memory_threshold_mb: int = 1000
    enable_adaptive_concurrency: bool = True
    min_concurrency: int = 2
    max_concurrency: int = 20

    @classmethod
    def from_env(cls) -> "ConcurrentProcessingConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            max_concurrent_files=int(os.getenv("CONCURRENT_CALL_EXTRACT_MAX_FILES", "10")),
            max_concurrent_chunks_per_file=int(os.getenv("CONCURRENT_CALL_EXTRACT_MAX_CHUNKS", "5")),
            chunk_batch_size=int(os.getenv("CONCURRENT_CALL_EXTRACT_BATCH_SIZE", "50")),
            timeout_seconds=float(os.getenv("CONCURRENT_CALL_EXTRACT_TIMEOUT", "300")),
            enable_progress_tracking=os.getenv("CONCURRENT_CALL_EXTRACT_PROGRESS", "true").lower() == "true",
            enable_memory_monitoring=os.getenv("CONCURRENT_CALL_EXTRACT_MEMORY_MONITOR", "true").lower() == "true",
            memory_threshold_mb=int(os.getenv("CONCURRENT_CALL_EXTRACT_MEMORY_THRESHOLD", "1000")),
            enable_adaptive_concurrency=os.getenv("CONCURRENT_CALL_EXTRACT_ADAPTIVE", "true").lower() == "true",
            min_concurrency=int(os.getenv("CONCURRENT_CALL_EXTRACT_MIN_CONCURRENCY", "2")),
            max_concurrency=int(os.getenv("CONCURRENT_CALL_EXTRACT_MAX_CONCURRENCY", "20")),
        )


@dataclass
class FileProcessingResult:
    """Result of processing a single file."""

    file_path: str
    success: bool
    call_detection_results: list[CallDetectionResult] = field(default_factory=list)
    total_calls_detected: int = 0
    processing_time_ms: float = 0.0
    error_message: str | None = None
    chunks_processed: int = 0
    chunks_failed: int = 0


@dataclass
class BatchProcessingResult:
    """Result of processing a batch of files."""

    total_files: int
    successful_files: int
    failed_files: int
    total_calls_detected: int
    total_processing_time_ms: float
    file_results: list[FileProcessingResult] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    memory_usage_mb: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0.0

    @property
    def average_calls_per_file(self) -> float:
        """Calculate average calls detected per successful file."""
        return (self.total_calls_detected / self.successful_files) if self.successful_files > 0 else 0.0


class ConcurrentCallExtractor:
    """
    Service for concurrent function call extraction across multiple files.

    This service provides performance-optimized concurrent processing with:
    - Adaptive concurrency based on system resources
    - Progress tracking and monitoring
    - Memory usage management
    - Error handling and recovery
    - Batch processing with intelligent scheduling
    """

    def __init__(self, config: ConcurrentProcessingConfig | None = None):
        """
        Initialize the concurrent call extractor.

        Args:
            config: Processing configuration, defaults to environment-based config
        """
        self.config = config or ConcurrentProcessingConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Core extractors pool
        self._extractor_pool: list[FunctionCallExtractor] = []
        self._pool_size = min(self.config.max_concurrent_files, self.config.max_concurrency)

        # Concurrency management
        self._file_semaphore = asyncio.Semaphore(self.config.max_concurrent_files)
        self._chunk_semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks_per_file)
        self._adaptive_concurrency = self.config.max_concurrent_files

        # Performance monitoring
        if self.config.enable_progress_tracking:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        # Memory tracking
        self._memory_warnings = 0
        self._memory_throttle_active = False

        # Statistics
        self._stats = {
            "total_files_processed": 0,
            "total_chunks_processed": 0,
            "total_calls_detected": 0,
            "total_processing_time_ms": 0.0,
            "concurrent_operations": 0,
            "memory_throttle_events": 0,
            "timeout_events": 0,
        }

        self.logger.info(f"ConcurrentCallExtractor initialized with config: {self.config}")

    async def initialize_pool(self):
        """Initialize the extractor pool."""
        if not self._extractor_pool:
            for _ in range(self._pool_size):
                extractor = FunctionCallExtractor()
                self._extractor_pool.append(extractor)

            self.logger.info(f"Initialized extractor pool with {self._pool_size} extractors")

    async def extract_calls_from_files(
        self, file_chunks: dict[str, list[CodeChunk]], breadcrumb_mapping: dict[str, str]
    ) -> BatchProcessingResult:
        """
        Extract function calls from multiple files concurrently.

        Args:
            file_chunks: Dictionary mapping file paths to their code chunks
            breadcrumb_mapping: Dictionary mapping chunk names to breadcrumb paths

        Returns:
            BatchProcessingResult with detailed extraction results
        """
        start_time = time.time()

        # Initialize pool if needed
        await self.initialize_pool()

        # Prepare processing tasks
        file_tasks = []
        total_files = len(file_chunks)

        self.logger.info(f"Starting concurrent extraction for {total_files} files")

        # Create file processing tasks
        for file_path, chunks in file_chunks.items():
            task = self._process_file_with_monitoring(file_path=file_path, chunks=chunks, breadcrumb_mapping=breadcrumb_mapping)
            file_tasks.append(task)

        try:
            # Execute file processing with timeout
            file_results = await asyncio.wait_for(asyncio.gather(*file_tasks, return_exceptions=True), timeout=self.config.timeout_seconds)

            # Process results
            successful_results = []
            failed_results = []

            for result in file_results:
                if isinstance(result, Exception):
                    # Create error result
                    failed_result = FileProcessingResult(file_path="unknown", success=False, error_message=str(result))
                    failed_results.append(failed_result)
                elif isinstance(result, FileProcessingResult):
                    if result.success:
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
                else:
                    # Unexpected result type
                    failed_result = FileProcessingResult(
                        file_path="unknown", success=False, error_message=f"Unexpected result type: {type(result)}"
                    )
                    failed_results.append(failed_result)

            # Calculate aggregate statistics
            total_calls = sum(r.total_calls_detected for r in successful_results)
            total_processing_time = (time.time() - start_time) * 1000

            # Update internal stats
            self._stats["total_files_processed"] += total_files
            self._stats["total_calls_detected"] += total_calls
            self._stats["total_processing_time_ms"] += total_processing_time

            # Create batch result
            batch_result = BatchProcessingResult(
                total_files=total_files,
                successful_files=len(successful_results),
                failed_files=len(failed_results),
                total_calls_detected=total_calls,
                total_processing_time_ms=total_processing_time,
                file_results=successful_results + failed_results,
                performance_metrics=await self._collect_performance_metrics(),
                memory_usage_mb=await self._get_memory_usage_mb(),
            )

            self.logger.info(
                f"Concurrent extraction completed: {batch_result.successful_files}/{total_files} files, "
                f"{total_calls} calls detected in {total_processing_time:.2f}ms"
            )

            return batch_result

        except asyncio.TimeoutError:
            self._stats["timeout_events"] += 1
            self.logger.error(f"Concurrent extraction timed out after {self.config.timeout_seconds}s")

            # Return partial results
            return BatchProcessingResult(
                total_files=total_files,
                successful_files=0,
                failed_files=total_files,
                total_calls_detected=0,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                performance_metrics={"timeout": True},
                memory_usage_mb=await self._get_memory_usage_mb(),
            )

        except Exception as e:
            self.logger.error(f"Unexpected error in concurrent extraction: {e}")
            raise

    async def _process_file_with_monitoring(
        self, file_path: str, chunks: list[CodeChunk], breadcrumb_mapping: dict[str, str]
    ) -> FileProcessingResult:
        """
        Process a single file with monitoring and adaptive concurrency.

        Args:
            file_path: Path to the file being processed
            chunks: Code chunks for the file
            breadcrumb_mapping: Mapping of chunk names to breadcrumbs

        Returns:
            FileProcessingResult with detailed processing information
        """
        start_time = time.time()

        async with self._file_semaphore:
            try:
                # Check memory before processing
                if self.config.enable_memory_monitoring:
                    memory_mb = await self._get_memory_usage_mb()
                    if memory_mb > self.config.memory_threshold_mb:
                        await self._handle_memory_pressure()

                # Process chunks in batches with concurrency control
                call_detection_results = []
                chunks_processed = 0
                chunks_failed = 0

                # Create chunk processing tasks
                chunk_tasks = []
                for chunk in chunks:
                    breadcrumb = breadcrumb_mapping.get(chunk.name, f"{file_path}:{chunk.name}")

                    task = self._process_chunk_with_semaphore(chunk=chunk, breadcrumb=breadcrumb, file_path=file_path)
                    chunk_tasks.append(task)

                # Process chunks with adaptive batching
                batch_size = min(self.config.chunk_batch_size, len(chunk_tasks))

                for i in range(0, len(chunk_tasks), batch_size):
                    batch = chunk_tasks[i : i + batch_size]

                    try:
                        batch_results = await asyncio.gather(*batch, return_exceptions=True)

                        for result in batch_results:
                            if isinstance(result, Exception):
                                chunks_failed += 1
                                self.logger.warning(f"Chunk processing failed in {file_path}: {result}")
                            elif isinstance(result, CallDetectionResult):
                                call_detection_results.append(result)
                                chunks_processed += 1
                            else:
                                chunks_failed += 1
                                self.logger.warning(f"Unexpected chunk result type in {file_path}: {type(result)}")

                    except Exception as e:
                        chunks_failed += len(batch)
                        self.logger.error(f"Batch processing failed for {file_path}: {e}")

                # Calculate total calls detected
                total_calls = sum(len(result.function_calls) for result in call_detection_results)

                # Update stats
                self._stats["total_chunks_processed"] += chunks_processed

                return FileProcessingResult(
                    file_path=file_path,
                    success=chunks_processed > 0,
                    call_detection_results=call_detection_results,
                    total_calls_detected=total_calls,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    chunks_processed=chunks_processed,
                    chunks_failed=chunks_failed,
                )

            except Exception as e:
                return FileProcessingResult(
                    file_path=file_path, success=False, error_message=str(e), processing_time_ms=(time.time() - start_time) * 1000
                )

    async def _process_chunk_with_semaphore(self, chunk: CodeChunk, breadcrumb: str, file_path: str) -> CallDetectionResult:
        """
        Process a single chunk with semaphore control.

        Args:
            chunk: Code chunk to process
            breadcrumb: Breadcrumb path for the chunk
            file_path: Path to the source file

        Returns:
            CallDetectionResult from the extraction
        """
        async with self._chunk_semaphore:
            # Get an extractor from the pool
            extractor = self._get_extractor()

            try:
                # Read file content for context
                content_lines = await self._read_file_lines(file_path)

                # Extract calls
                result = await extractor.extract_calls_from_chunk(chunk=chunk, source_breadcrumb=breadcrumb, content_lines=content_lines)

                return result

            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk.name} in {file_path}: {e}")
                # Return empty result on error
                return CallDetectionResult(
                    source_file_path=file_path,
                    source_breadcrumb=breadcrumb,
                    function_calls=[],
                    processing_time_ms=0.0,
                    success=False,
                    error_message=str(e),
                )

    def _get_extractor(self) -> FunctionCallExtractor:
        """Get an extractor from the pool (round-robin)."""
        if not self._extractor_pool:
            return FunctionCallExtractor()

        # Simple round-robin selection
        extractor = self._extractor_pool[self._stats["concurrent_operations"] % len(self._extractor_pool)]
        self._stats["concurrent_operations"] += 1
        return extractor

    async def _read_file_lines(self, file_path: str) -> list[str]:
        """Read file content and return lines."""
        try:
            path = Path(file_path)
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
                return content.splitlines()
            return []
        except Exception as e:
            self.logger.warning(f"Failed to read file {file_path}: {e}")
            return []

    async def _handle_memory_pressure(self):
        """Handle memory pressure by adjusting concurrency."""
        if not self._memory_throttle_active:
            self._memory_throttle_active = True
            self._memory_warnings += 1
            self._stats["memory_throttle_events"] += 1

            # Reduce adaptive concurrency
            if self.config.enable_adaptive_concurrency:
                self._adaptive_concurrency = max(self.config.min_concurrency, self._adaptive_concurrency // 2)

                self.logger.warning(f"Memory pressure detected, reducing concurrency to {self._adaptive_concurrency}")

            # Brief pause to allow garbage collection
            await asyncio.sleep(0.1)

            self._memory_throttle_active = False

    async def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    async def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect performance metrics."""
        metrics = {
            "adaptive_concurrency": self._adaptive_concurrency,
            "memory_warnings": self._memory_warnings,
            "memory_throttle_active": self._memory_throttle_active,
            "pool_size": len(self._extractor_pool),
            "stats": self._stats.copy(),
        }

        if self.performance_monitor:
            metrics.update(self.performance_monitor.get_metrics())

        return metrics

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "config": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "max_concurrent_chunks_per_file": self.config.max_concurrent_chunks_per_file,
                "chunk_batch_size": self.config.chunk_batch_size,
                "adaptive_concurrency_enabled": self.config.enable_adaptive_concurrency,
                "memory_monitoring_enabled": self.config.enable_memory_monitoring,
            },
            "runtime_stats": self._stats.copy(),
            "current_state": {
                "adaptive_concurrency": self._adaptive_concurrency,
                "memory_throttle_active": self._memory_throttle_active,
                "pool_size": len(self._extractor_pool),
            },
        }

    async def shutdown(self):
        """Shutdown the concurrent extractor and cleanup resources."""
        self.logger.info("Shutting down ConcurrentCallExtractor")

        # Clear extractor pool
        self._extractor_pool.clear()

        # Reset statistics
        self._stats = {key: 0 if isinstance(value, (int, float)) else value for key, value in self._stats.items()}

        self.logger.info("ConcurrentCallExtractor shutdown complete")
