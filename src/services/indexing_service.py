import os
import tempfile
import shutil
import multiprocessing
import gc
import psutil
import logging
import threading
import time
from git import Repo, GitCommandError
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from services.project_analysis_service import ProjectAnalysisService
from utils.performance_monitor import MemoryMonitor, ProgressTracker
from utils.stage_logger import get_file_discovery_logger, get_file_reading_logger, log_timing, log_batch_summary

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]

class IndexingService:
    def __init__(self):
        self.project_analysis_service = ProjectAnalysisService()
        self._lock = Lock()  # For thread-safe operations
        self._error_files = []  # Track files that failed processing
        self._processed_count = 0  # Atomic counter for processed files
        self._reset_counters()  # Reset processing counters
        self._setup_thread_safe_logging()
        
        # Initialize memory monitoring with configurable threshold
        memory_threshold = float(os.getenv('MEMORY_WARNING_THRESHOLD_MB', '1000'))
        self.memory_monitor = MemoryMonitor(warning_threshold_mb=memory_threshold)
        
        # Initialize stage-specific loggers
        self.file_discovery_logger = get_file_discovery_logger()
        self.file_reading_logger = get_file_reading_logger()
        
        # Progress tracker for external monitoring
        self.progress_tracker: Optional[ProgressTracker] = None

    def process_codebase_for_indexing(self, source_path: str) -> List[Chunk]:
        self.logger.info(f"Processing codebase from: {source_path}")

        is_git_url = source_path.startswith(('http://', 'https://', 'git@'))

        if is_git_url:
            temp_dir = tempfile.mkdtemp()
            try:
                self.logger.info(f"Cloning {source_path} into {temp_dir}")
                Repo.clone_from(source_path, temp_dir)
                directory_to_index = temp_dir
            except GitCommandError as e:
                self.logger.error(f"Error cloning repository: {e}")
                shutil.rmtree(temp_dir)
                return []
        else:
            directory_to_index = source_path

        # Stage 1: File Discovery with detailed logging
        with self.file_discovery_logger.stage("file_discovery", directory=directory_to_index) as stage:
            discovery_start = time.time()
            relevant_files = self.project_analysis_service.get_relevant_files(directory_to_index)
            discovery_duration = time.time() - discovery_start
            
            stage.item_count = len(relevant_files)
            stage.processed_count = len(relevant_files)
            
            log_timing(self.file_discovery_logger, "file_discovery", discovery_duration, 
                      files_found=len(relevant_files), directory=directory_to_index)

        if not relevant_files:
            self.logger.warning("No relevant files found to process.")
            if is_git_url: shutil.rmtree(temp_dir)
            return []

        chunks = []
        self._error_files = []  # Reset error tracking
        
        # Get and validate concurrency settings
        max_workers = self._get_optimal_worker_count()
        batch_size = int(os.getenv('INDEXING_BATCH_SIZE', '20'))
        
        self.logger.info(f"Processing {len(relevant_files)} files with {max_workers} workers (batch size: {batch_size})...")
        
        # Initialize progress tracking
        self.progress_tracker = ProgressTracker(len(relevant_files), "Indexing codebase files")
        
        # Monitor initial memory usage
        initial_memory = self.memory_monitor.check_memory_usage(self.logger)
        self.logger.info(f"Initial memory usage: {initial_memory['memory_mb']} MB")
        
        # Stage 2: File Reading and Processing with detailed logging
        with self.file_reading_logger.stage("file_processing", item_count=len(relevant_files)) as processing_stage:
            # Process files in batches to manage memory
            for batch_start in range(0, len(relevant_files), batch_size):
                batch_end = min(batch_start + batch_size, len(relevant_files))
                batch_files = relevant_files[batch_start:batch_end]
                batch_num = batch_start // batch_size + 1
                
                self.file_reading_logger.info(f"Processing batch {batch_num}: files {batch_start+1}-{batch_end}")
                batch_start_time = time.time()
                
                # Use ThreadPoolExecutor with proper resource management
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    try:
                        # Submit batch processing tasks
                        future_to_file = {executor.submit(self._process_single_file, file_path): file_path 
                                         for file_path in batch_files}
                        
                        batch_processed = 0
                        batch_failed = 0
                        
                        # Collect results as they complete
                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            thread_id = threading.get_ident()
                            try:
                                chunk = future.result()
                                if chunk:  # Only add non-None chunks
                                    chunks.append(chunk)
                                    batch_processed += 1
                                    self.file_reading_logger.log_item_processed("file_processing", file_path=file_path)
                                    with self._lock:
                                        self._processed_count += 1
                                        processing_stage.processed_count = self._processed_count
                                        self.progress_tracker.increment_processed()
                                        self.logger.debug(f"[Thread-{thread_id}] Successfully processed file {file_path} ({self._processed_count}/{len(relevant_files)})")
                            except Exception as e:
                                batch_failed += 1
                                self.file_reading_logger.log_item_failed("file_processing", error=str(e), file_path=file_path)
                                with self._lock:
                                    self._error_files.append((file_path, str(e)))
                                    self.progress_tracker.increment_failed()
                                self.logger.error(f"[Thread-{thread_id}] Error processing file {file_path}: {e}")
                            finally:
                                # Clean up future reference
                                del future_to_file[future]
                    
                    except Exception as e:
                        self.logger.error(f"Error in batch processing: {e}")
                    
                    finally:
                        # Ensure ThreadPoolExecutor cleanup
                        executor.shutdown(wait=True)
                
                # Log batch completion
                batch_duration = time.time() - batch_start_time
                log_batch_summary(self.file_reading_logger, batch_num, len(batch_files), 
                                batch_processed, batch_failed, batch_duration)
                
                # Memory cleanup between batches
                self._cleanup_memory()
                
                # Monitor memory usage with automatic warnings
                memory_stats = self.memory_monitor.check_memory_usage(self.logger)
                self.logger.info(f"Batch completed. Memory usage: {memory_stats['memory_mb']} MB ({memory_stats['memory_percent']}%)")
        
        # Report any errors
        if self._error_files:
            self.logger.error(f"Failed to process {len(self._error_files)} files:")
            for file_path, error in self._error_files:
                self.logger.error(f"  - {file_path}: {error}")
        
        # Final processing summary
        self.logger.info(f"Processing completed: {len(chunks)} chunks created, {len(self._error_files)} errors")
        
        if is_git_url:
            self.logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        return chunks
    
    def get_progress_summary(self) -> Optional[Dict[str, Any]]:
        """Get current progress summary for external monitoring."""
        if self.progress_tracker is None:
            return None
        
        return self.progress_tracker.get_progress_summary()

    def _process_single_file(self, file_path: str) -> Chunk:
        """Process a single file and return a Chunk. Thread-safe worker function."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking: treat entire file as one chunk for now
            # More sophisticated chunking can be added here (e.g., based on AST, lines, etc.)
            return Chunk(
                content=content,
                metadata={
                    "file_path": file_path,
                    "chunk_index": 0, # Assuming single chunk for now
                    "line_start": 1,
                    "line_end": len(content.splitlines()),
                    "language": self._detect_language(file_path) # Add language detection
                }
            )
        except Exception as e:
            # Let the calling code handle the exception
            raise e
    
    def _get_optimal_worker_count(self) -> int:
        """Calculate optimal worker count based on CPU cores and configuration."""
        # Get configured concurrency or use default
        configured_workers = int(os.getenv('INDEXING_CONCURRENCY', '4'))
        
        # Get CPU count for optimization
        cpu_count = multiprocessing.cpu_count()
        
        # For I/O-bound operations like file reading, we can use more threads than CPU cores
        # But cap it at 2x CPU count to avoid too much context switching
        max_recommended = min(cpu_count * 2, 8)  # Cap at 8 to be conservative
        
        # Use the smaller of configured or recommended
        optimal_workers = min(configured_workers, max_recommended)
        
        # Ensure at least 1 worker
        optimal_workers = max(1, optimal_workers)
        
        if optimal_workers != configured_workers:
            self.logger.info(f"Adjusted worker count from {configured_workers} to {optimal_workers} based on CPU cores ({cpu_count})")
        
        return optimal_workers
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate and return configuration settings with safe defaults."""
        config = {}
        
        # Validate concurrency settings
        try:
            config['concurrency'] = max(1, int(os.getenv('INDEXING_CONCURRENCY', '4')))
        except ValueError:
            self.logger.warning("Invalid INDEXING_CONCURRENCY value, using default: 4")
            config['concurrency'] = 4
        
        try:
            config['batch_size'] = max(1, int(os.getenv('INDEXING_BATCH_SIZE', '20')))
        except ValueError:
            self.logger.warning("Invalid INDEXING_BATCH_SIZE value, using default: 20")
            config['batch_size'] = 20
        
        return config
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _cleanup_memory(self) -> None:
        """Force garbage collection to free memory."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Additional cleanup for large objects
            if hasattr(gc, 'collect'):
                # Run multiple collection cycles for thorough cleanup
                for _ in range(3):
                    collected = gc.collect()
                    if collected == 0:
                        break
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def _reset_counters(self) -> None:
        """Reset processing counters for new indexing operation."""
        with self._lock:
            self._processed_count = 0
            self._error_files = []
    
    def _setup_thread_safe_logging(self) -> None:
        """Setup thread-safe logging configuration."""
        # Create logger for this service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Only configure if not already configured
        if not self.logger.handlers:
            # Set level from environment or default to INFO
            log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Create thread-safe formatter with thread ID
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s'
            )
            
            # Create console handler (thread-safe by default)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Prevent duplicate logging
            self.logger.propagate = False
    
    def _detect_language(self, file_path: str) -> str:
        # Basic language detection based on file extension
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".py":
            return "python"
        elif extension == ".js":
            return "javascript"
        elif extension == ".ts":
            return "typescript"
        elif extension == ".java":
            return "java"
        elif extension == ".go":
            return "go"
        elif extension == ".rs":
            return "rust"
        elif extension == ".md":
            return "markdown"
        elif extension == ".json":
            return "json"
        elif extension == ".yaml" or extension == ".yml":
            return "yaml"
        else:
            return "unknown"