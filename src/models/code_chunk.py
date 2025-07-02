"""
Data models for code chunks and intelligent chunking functionality.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ChunkType(Enum):
    """Enumeration of different code chunk types for intelligent parsing."""

    # Function-level chunks
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"  # C++ destructors
    ASYNC_FUNCTION = "async_function"

    # Class-level chunks
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    IMPL = "impl"  # Rust implementation blocks
    NAMESPACE = "namespace"  # C++ namespace blocks

    # Variable-level chunks
    CONSTANT = "constant"
    VARIABLE = "variable"
    PROPERTY = "property"

    # Module-level chunks
    IMPORT = "import"
    EXPORT = "export"
    MODULE_DOCSTRING = "module_docstring"

    # Configuration and data chunks
    CONFIG_BLOCK = "config_block"  # JSON, YAML, TOML sections
    DATA_STRUCTURE = "data_structure"  # Complex data definitions

    # Documentation chunks
    DOCSTRING = "docstring"
    COMMENT_BLOCK = "comment_block"

    # Type definition chunks (TypeScript, etc.)
    TYPE_ALIAS = "type_alias"
    TYPE_DEFINITION = "type_definition"
    TEMPLATE = "template"  # C++ template definitions

    # Fallback for unrecognized or whole-file content
    RAW_CODE = "raw_code"
    WHOLE_FILE = "whole_file"


@dataclass
class CodeChunk:
    """
    Represents a semantically meaningful chunk of code extracted through intelligent parsing.

    This model extends the existing Chunk format with enhanced metadata for code understanding.
    """

    # Core identification
    chunk_id: str  # Unique identifier for this chunk
    file_path: str  # Full path to the source file
    content: str  # The actual code content

    # Chunk classification
    chunk_type: ChunkType  # Type of code construct this chunk represents
    language: str  # Programming language (python, javascript, etc.)

    # Position and context information
    start_line: int  # Starting line number in source file
    end_line: int  # Ending line number in source file
    start_byte: int  # Starting byte offset in source file
    end_byte: int  # Ending byte offset in source file

    # Semantic metadata
    name: str | None = None  # Function name, class name, variable name, etc.
    parent_name: str | None = None  # Parent class or module name for context
    signature: str | None = None  # Function signature, class definition, etc.
    docstring: str | None = None  # Associated documentation string

    # Context enhancement
    breadcrumb: str | None = None  # Full hierarchical path (module.class.method)
    context_before: str | None = None  # 5 lines of code before this chunk
    context_after: str | None = None  # 5 lines of code after this chunk

    # Processing metadata
    content_hash: str | None = None  # SHA256 hash of the content for change detection
    embedding_text: str | None = None  # Optimized text for embedding generation
    indexed_at: datetime | None = None  # When this chunk was processed

    # Additional metadata for search and retrieval
    tags: list[str] = None  # Additional tags for categorization
    complexity_score: float | None = None  # Estimated complexity (0.0-1.0)
    dependencies: list[str] = None  # Referenced functions, classes, modules

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []

    @property
    def line_count(self) -> int:
        """Calculate the number of lines in this chunk."""
        return self.end_line - self.start_line + 1

    @property
    def char_count(self) -> int:
        """Calculate the number of characters in this chunk."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CodeChunk to a dictionary for serialization.

        This is used for storing in vector databases and API responses.
        """
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "name": self.name,
            "parent_name": self.parent_name,
            "signature": self.signature,
            "docstring": self.docstring,
            "breadcrumb": self.breadcrumb,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "content_hash": self.content_hash,
            "embedding_text": self.embedding_text,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "tags": self.tags,
            "complexity_score": self.complexity_score,
            "dependencies": self.dependencies,
            "line_count": self.line_count,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk":
        """
        Create a CodeChunk instance from a dictionary.

        This is used for deserializing from vector databases and API requests.
        """
        # Convert string chunk_type back to enum
        chunk_type = ChunkType(data["chunk_type"])

        # Convert ISO datetime string back to datetime object
        indexed_at = None
        if data.get("indexed_at"):
            indexed_at = datetime.fromisoformat(data["indexed_at"])

        return cls(
            chunk_id=data["chunk_id"],
            file_path=data["file_path"],
            content=data["content"],
            chunk_type=chunk_type,
            language=data["language"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            start_byte=data["start_byte"],
            end_byte=data["end_byte"],
            name=data.get("name"),
            parent_name=data.get("parent_name"),
            signature=data.get("signature"),
            docstring=data.get("docstring"),
            breadcrumb=data.get("breadcrumb"),
            context_before=data.get("context_before"),
            context_after=data.get("context_after"),
            content_hash=data.get("content_hash"),
            embedding_text=data.get("embedding_text"),
            indexed_at=indexed_at,
            tags=data.get("tags", []),
            complexity_score=data.get("complexity_score"),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class CodeSyntaxError:
    """Information about a syntax error found during parsing."""

    start_line: int  # Line where error starts (1-based)
    end_line: int  # Line where error ends (1-based)
    start_column: int  # Column where error starts (0-based)
    end_column: int  # Column where error ends (0-based)
    error_type: str  # Type of syntax error (e.g., 'missing_semicolon', 'unexpected_token')
    context: str  # Surrounding code context
    severity: str = "error"  # Severity level: 'error', 'warning', 'info'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "error_type": self.error_type,
            "context": self.context,
            "severity": self.severity,
        }


@dataclass
class ParseResult:
    """
    Result of parsing a source file into intelligent chunks.

    This contains both successful chunks and any parsing errors encountered.
    """

    chunks: list[CodeChunk]  # Successfully parsed chunks
    file_path: str  # Source file path
    language: str  # Detected programming language
    parse_success: bool  # Whether parsing completed successfully
    error_count: int = 0  # Number of syntax errors encountered
    fallback_used: bool = False  # Whether fallback to whole-file chunking was used
    processing_time_ms: float = 0.0  # Time taken to parse this file
    syntax_errors: list[CodeSyntaxError] = None  # Detailed syntax error information
    error_recovery_used: bool = False  # Whether error recovery was used
    valid_sections_count: int = 0  # Number of valid code sections extracted during error recovery

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.syntax_errors is None:
            self.syntax_errors = []

    def to_dict(self) -> dict[str, Any]:
        """Convert ParseResult to dictionary for logging and debugging."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "parse_success": self.parse_success,
            "chunk_count": len(self.chunks),
            "error_count": self.error_count,
            "fallback_used": self.fallback_used,
            "processing_time_ms": self.processing_time_ms,
            "chunk_types": [chunk.chunk_type.value for chunk in self.chunks],
            "syntax_errors": [error.to_dict() for error in self.syntax_errors],
            "error_recovery_used": self.error_recovery_used,
            "valid_sections_count": self.valid_sections_count,
        }
