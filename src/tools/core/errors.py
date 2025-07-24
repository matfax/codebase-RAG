"""Custom error types for MCP tools.

This module defines custom exceptions for better error handling.
"""

from typing import Union


class MCPToolError(Exception):
    """Base exception for all MCP tool errors."""

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.details = details


class QdrantConnectionError(MCPToolError):
    """Raised when connection to Qdrant fails."""

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message, details)


class IndexingError(MCPToolError):
    """Raised when file indexing fails."""

    def __init__(self, message: str, file_path: str, details: str | None = None):
        super().__init__(message, details)
        self.file_path = file_path


class SearchError(MCPToolError):
    """Raised when search operation fails."""

    def __init__(self, message: str, query: str, details: str | None = None):
        super().__init__(message, details)
        self.query = query


class CollectionError(MCPToolError):
    """Raised when collection operations fail."""

    def __init__(self, message: str, collection_name: str, details: str | None = None):
        super().__init__(message, details)
        self.collection_name = collection_name


class ProjectError(MCPToolError):
    """Raised when project operations fail."""

    def __init__(self, message: str, project_path: str, details: str | None = None):
        super().__init__(message, details)
        self.project_path = project_path


class EmbeddingError(MCPToolError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, model_name: str = None, details: str | None = None):
        super().__init__(message, details)
        self.model_name = model_name


class ParsingError(MCPToolError):
    """Raised when code parsing fails."""

    def __init__(
        self,
        message: str,
        file_path: str = None,
        language: str = None,
        details: str | None = None,
    ):
        super().__init__(message, details)
        self.file_path = file_path
        self.language = language


class ChunkingError(MCPToolError):
    """Raised when code chunking fails."""

    def __init__(
        self,
        message: str,
        file_path: str = None,
        chunk_type: str = None,
        details: str | None = None,
    ):
        super().__init__(message, details)
        self.file_path = file_path
        self.chunk_type = chunk_type


class MetadataError(MCPToolError):
    """Raised when metadata operations fail."""

    def __init__(self, message: str, metadata_type: str = None, details: str | None = None):
        super().__init__(message, details)
        self.metadata_type = metadata_type


class ServiceError(MCPToolError):
    """Raised when service operations fail."""

    def __init__(self, message: str, service_name: str = None, details: str | None = None):
        super().__init__(message, details)
        self.service_name = service_name


class ConfigurationError(MCPToolError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str = None, details: str | None = None):
        super().__init__(message, details)
        self.config_key = config_key


class ValidationError(MCPToolError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str = None,
        value: str = None,
        details: str | None = None,
    ):
        super().__init__(message, details)
        self.field_name = field_name
        self.value = value


class ParserError(MCPToolError):
    """Raised when Tree-sitter parser operations fail."""

    def __init__(
        self,
        message: str,
        language: str = None,
        parser_name: str = None,
        details: str | None = None,
    ):
        super().__init__(message, details)
        self.language = language
        self.parser_name = parser_name


class FileOperationError(MCPToolError):
    """Raised when file operations fail."""

    def __init__(
        self,
        message: str,
        file_path: str = None,
        operation: str = None,
        details: str | None = None,
    ):
        super().__init__(message, details)
        self.file_path = file_path
        self.operation = operation


class CacheError(MCPToolError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        cache_name: str = None,
        operation: str = None,
        details: str | None = None,
    ):
        super().__init__(message, details)
        self.cache_name = cache_name
        self.operation = operation
