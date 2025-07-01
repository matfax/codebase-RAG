"""Custom error types for MCP tools.

This module defines custom exceptions for better error handling.
"""

from typing import Optional


class MCPToolError(Exception):
    """Base exception for all MCP tool errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details


class QdrantConnectionError(MCPToolError):
    """Raised when connection to Qdrant fails."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, details)


class IndexingError(MCPToolError):
    """Raised when file indexing fails."""
    
    def __init__(self, message: str, file_path: str, details: Optional[str] = None):
        super().__init__(message, details)
        self.file_path = file_path


class SearchError(MCPToolError):
    """Raised when search operation fails."""
    
    def __init__(self, message: str, query: str, details: Optional[str] = None):
        super().__init__(message, details)
        self.query = query


class CollectionError(MCPToolError):
    """Raised when collection operations fail."""
    
    def __init__(self, message: str, collection_name: str, details: Optional[str] = None):
        super().__init__(message, details)
        self.collection_name = collection_name


class ProjectError(MCPToolError):
    """Raised when project operations fail."""
    
    def __init__(self, message: str, project_path: str, details: Optional[str] = None):
        super().__init__(message, details)
        self.project_path = project_path