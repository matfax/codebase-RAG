"""
Backward compatibility layer for mcp_tools imports.

This module provides backward compatibility for code that still imports from mcp_tools.
All functions are now imported from the new modular structure.
"""

import warnings

# Import functions from the new modular structure
from tools.core.health import health_check
from tools.indexing.index_tools import index_directory
from tools.indexing.search_tools import search
from tools.indexing.chunking_tools import get_chunking_metrics, reset_chunking_metrics
from tools.indexing.parser_tools import diagnose_parser_health
from tools.indexing.progress_tools import get_indexing_progress, check_index_status
from tools.project.project_tools import get_project_info, list_indexed_projects, clear_project_data
from tools.project.file_tools import get_file_metadata, clear_file_metadata, reindex_file
from tools.project.project_utils import get_current_project, get_collection_name
from tools.database.qdrant_utils import (
    get_qdrant_client, ensure_collection, check_qdrant_health, 
    retry_qdrant_operation, log_database_metrics
)
from tools.core.memory_utils import (
    get_memory_usage_mb, log_memory_usage, force_memory_cleanup,
    should_cleanup_memory, get_adaptive_batch_size, clear_processing_variables
)
from tools.core.retry_utils import retry_operation
from tools.core.errors import *
from tools import register_tools

# Compatibility aliases for renamed functions
analyze_repository_tool = None  # Will be imported from search_tools
get_file_filtering_stats_tool = None
clear_project_collections = clear_project_data

def deprecation_warning(old_import, new_import):
    """Issue a deprecation warning for old imports."""
    warnings.warn(
        f"Importing from '{old_import}' is deprecated. "
        f"Please use '{new_import}' instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy registration function for backward compatibility
def register_mcp_tools(mcp_app):
    """Register MCP tools using the new modular system.
    
    This function is deprecated. Use tools.register_tools() instead.
    """
    deprecation_warning("mcp_tools.register_mcp_tools", "tools.register_tools")
    return register_tools(mcp_app)

# Add any other functions that might be imported from mcp_tools
__all__ = [
    # Core functions
    'health_check',
    'register_mcp_tools',
    
    # Indexing functions
    'index_directory',
    'search',
    'get_chunking_metrics',
    'reset_chunking_metrics',
    'diagnose_parser_health',
    'get_indexing_progress',
    'check_index_status',
    
    # Project functions
    'get_project_info',
    'list_indexed_projects',
    'clear_project_data',
    'clear_project_collections',  # alias
    'get_current_project',
    'get_collection_name',
    
    # File functions
    'get_file_metadata',
    'clear_file_metadata',
    'reindex_file',
    
    # Database functions
    'get_qdrant_client',
    'ensure_collection',
    'check_qdrant_health',
    'retry_qdrant_operation',
    'log_database_metrics',
    
    # Memory functions
    'get_memory_usage_mb',
    'log_memory_usage',
    'force_memory_cleanup',
    'should_cleanup_memory',
    'get_adaptive_batch_size',
    'clear_processing_variables',
    
    # Utility functions
    'retry_operation',
    
    # Error classes (all errors from tools.core.errors)
]