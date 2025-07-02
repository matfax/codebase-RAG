"""Core tools for health checks and memory management.

This module contains fundamental tools for system health and resource management.
"""

from tools.core.error_utils import (
    chain_exceptions,
    create_error_context,
    format_error_details,
    handle_service_error,
    log_and_raise_error,
    safe_execute,
    validate_field_types,
    validate_required_fields,
)
from tools.core.errors import (
    ChunkingError,
    CollectionError,
    ConfigurationError,
    EmbeddingError,
    IndexingError,
    MCPToolError,
    MetadataError,
    ParsingError,
    ProjectError,
    QdrantConnectionError,
    SearchError,
    ServiceError,
    ValidationError,
)
from tools.core.health import basic_health_check, health_check
from tools.core.memory_utils import (
    check_memory_usage,
    clear_processing_variables,
    force_memory_cleanup,
    get_adaptive_batch_size,
    get_memory_stats,
    get_memory_usage_mb,
    log_memory_usage,
    should_cleanup_memory,
)
from tools.core.retry_utils import retry_operation, retry_with_context

__all__ = [
    "health_check",
    "basic_health_check",
    "retry_operation",
    "retry_with_context",
    "get_memory_stats",
    "check_memory_usage",
    "force_memory_cleanup",
    "should_cleanup_memory",
    "get_adaptive_batch_size",
    "clear_processing_variables",
    "get_memory_usage_mb",
    "log_memory_usage",
    "MCPToolError",
    "QdrantConnectionError",
    "IndexingError",
    "SearchError",
    "CollectionError",
    "ProjectError",
    "EmbeddingError",
    "ParsingError",
    "ChunkingError",
    "MetadataError",
    "ServiceError",
    "ConfigurationError",
    "ValidationError",
    "log_and_raise_error",
    "handle_service_error",
    "validate_required_fields",
    "validate_field_types",
    "safe_execute",
    "create_error_context",
    "format_error_details",
    "chain_exceptions",
]
