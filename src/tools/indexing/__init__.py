"""Indexing tools for codebase processing and search.

This module contains tools for indexing, searching, and analyzing codebases.
"""

from tools.indexing.chunking_tools import (
    get_chunking_metrics,
    register_chunking_tools,
    reset_chunking_metrics,
)
from tools.indexing.index_tools import index_directory, index_directory_sync
from tools.indexing.parser_tools import diagnose_parser_health, register_parser_tools
from tools.indexing.progress_tools import (
    check_index_status,
    get_indexing_progress,
    register_progress_tools,
)
from tools.indexing.search_tools import (
    analyze_repository_tool,
    check_index_status_tool,
    get_file_filtering_stats_tool,
    search,
    search_sync,
)

__all__ = [
    # Core indexing and search
    "index_directory",
    "index_directory_sync",
    "search",
    "search_sync",
    # Analysis tools
    "analyze_repository_tool",
    "get_file_filtering_stats_tool",
    "check_index_status_tool",
    # Chunking tools
    "get_chunking_metrics",
    "reset_chunking_metrics",
    "register_chunking_tools",
    # Parser tools
    "diagnose_parser_health",
    "register_parser_tools",
    # Progress tools
    "get_indexing_progress",
    "check_index_status",
    "register_progress_tools",
]
