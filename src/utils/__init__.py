"""
Utility modules for the Agentic RAG MCP server.
"""

from .performance_monitor import (
    ProgressTracker,
    MemoryMonitor,
    ProcessingStats,
    format_duration,
    format_memory_size
)

from .stage_logger import (
    StageLogger,
    StageMetrics,
    create_stage_logger,
    get_file_discovery_logger,
    get_file_reading_logger,
    get_embedding_logger,
    get_database_logger,
    log_timing,
    log_batch_summary
)

from .file_system_utils import (
    get_file_mtime,
    get_file_size,
    file_exists,
    is_file_readable,
    get_file_stats,
    batch_get_file_stats,
    compare_file_times,
    find_files_newer_than,
    create_directory_if_not_exists,
    is_file_binary,
    get_relative_path,
    calculate_directory_size,
    format_file_size,
    format_timestamp,
    FileSystemWatcher
)

from .file_hash_utils import (
    calculate_file_hash,
    batch_calculate_hashes,
    verify_file_hash,
    compare_file_hashes,
    find_duplicate_files,
    get_file_hash_info,
    HashVerifier
)

__all__ = [
    "ProgressTracker",
    "MemoryMonitor", 
    "ProcessingStats",
    "format_duration",
    "format_memory_size",
    "StageLogger",
    "StageMetrics",
    "create_stage_logger",
    "get_file_discovery_logger",
    "get_file_reading_logger", 
    "get_embedding_logger",
    "get_database_logger",
    "log_timing",
    "log_batch_summary",
    "get_file_mtime",
    "get_file_size",
    "file_exists",
    "is_file_readable",
    "get_file_stats",
    "batch_get_file_stats",
    "compare_file_times",
    "find_files_newer_than",
    "create_directory_if_not_exists",
    "is_file_binary",
    "get_relative_path",
    "calculate_directory_size",
    "format_file_size",
    "format_timestamp",
    "FileSystemWatcher",
    "calculate_file_hash",
    "batch_calculate_hashes",
    "verify_file_hash",
    "compare_file_hashes",
    "find_duplicate_files",
    "get_file_hash_info",
    "HashVerifier"
]