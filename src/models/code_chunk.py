"""
Data models for code chunks and intelligent chunking functionality.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Union


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

    # Function call chunks (for relationship detection)
    FUNCTION_CALL = "function_call"
    METHOD_CALL = "method_call"
    ASYNC_CALL = "async_call"
    ATTRIBUTE_ACCESS = "attribute_access"

    # Configuration and data chunks
    CONFIG_BLOCK = "config_block"  # JSON, YAML, TOML sections
    DATA_STRUCTURE = "data_structure"  # Complex data definitions
    OBJECT = "object"  # JSON/YAML object chunks
    SECTION = "section"  # Markdown section chunks

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
    parent_name: str | None = None  # Direct parent class/module name (e.g., "MyClass", "mymodule")
    signature: str | None = None  # Function signature, class definition, etc.
    docstring: str | None = None  # Associated documentation string

    # Context enhancement for Graph RAG
    breadcrumb: str | None = None  # Full hierarchical path (e.g., "module.class.method", "package::namespace::function")
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

    # PRD-defined metadata fields (FR-5.1)
    access_modifier: str | None = None  # public/private/protected visibility
    imports_used: list[str] = None  # Dependencies from static analysis
    has_syntax_errors: bool = False  # Whether chunk contains syntax errors
    error_details: str | None = None  # Syntax error descriptions if any

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.imports_used is None:
            self.imports_used = []

    @property
    def line_count(self) -> int:
        """Calculate the number of lines in this chunk."""
        return self.end_line - self.start_line + 1

    @property
    def char_count(self) -> int:
        """Calculate the number of characters in this chunk."""
        return len(self.content)

    @property
    def is_nested(self) -> bool:
        """Check if this chunk is nested within another structure."""
        return self.parent_name is not None

    @property
    def breadcrumb_depth(self) -> int:
        """Calculate the depth of the breadcrumb hierarchy."""
        if not self.breadcrumb:
            return 0
        # Count separators to determine depth
        # Support both dot notation (Python/JS) and :: notation (C++/Rust)
        dot_count = self.breadcrumb.count(".")
        double_colon_count = self.breadcrumb.count("::")
        return max(dot_count, double_colon_count) + 1

    def get_breadcrumb_components(self) -> list[str]:
        """
        Split breadcrumb into its component parts.

        Returns:
            List of breadcrumb components from root to current chunk

        Examples:
            "module.class.method" -> ["module", "class", "method"]
            "namespace::class::function" -> ["namespace", "class", "function"]
        """
        if not self.breadcrumb:
            return []

        # Support both dot notation and double colon notation
        if "::" in self.breadcrumb:
            return self.breadcrumb.split("::")
        else:
            return self.breadcrumb.split(".")

    def get_parent_breadcrumb(self) -> str | None:
        """
        Get the breadcrumb of the parent (one level up).

        Returns:
            Parent breadcrumb or None if at root level

        Examples:
            "module.class.method" -> "module.class"
            "namespace::class::function" -> "namespace::class"
        """
        if not self.breadcrumb:
            return None

        components = self.get_breadcrumb_components()
        if len(components) <= 1:
            return None

        separator = "::" if "::" in self.breadcrumb else "."
        return separator.join(components[:-1])

    def build_breadcrumb(self, components: list[str], separator: str = ".") -> str:
        """
        Build a breadcrumb from components.

        Args:
            components: List of breadcrumb components
            separator: Separator to use ('.' for Python/JS, '::' for C++/Rust)

        Returns:
            Formatted breadcrumb string
        """
        if not components:
            return ""
        return separator.join(components)

    def validate_structure_fields(self) -> tuple[bool, list[str]]:
        """
        Validate the breadcrumb and parent_name fields for consistency.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if parent_name is consistent with breadcrumb
        if self.parent_name and self.breadcrumb:
            components = self.get_breadcrumb_components()
            if len(components) >= 2:
                # Parent name should match the second-to-last component
                expected_parent = components[-2]
                if self.parent_name != expected_parent:
                    errors.append(f"parent_name '{self.parent_name}' does not match breadcrumb parent '{expected_parent}'")

        # Check if breadcrumb contains current name
        if self.name and self.breadcrumb:
            components = self.get_breadcrumb_components()
            if components and components[-1] != self.name:
                errors.append(f"chunk name '{self.name}' does not match breadcrumb tail '{components[-1]}'")

        # Check for invalid characters in breadcrumb
        if self.breadcrumb:
            # Breadcrumb should not contain spaces (use _ instead)
            if " " in self.breadcrumb:
                errors.append("breadcrumb contains spaces - use underscores or camelCase")

            # Should not start or end with separators
            if self.breadcrumb.startswith((".", "::")):
                errors.append("breadcrumb should not start with separator")
            if self.breadcrumb.endswith((".", "::")):
                errors.append("breadcrumb should not end with separator")

            # Should not contain mixed separators
            if "." in self.breadcrumb and "::" in self.breadcrumb:
                errors.append("breadcrumb should not mix '.' and '::' separators")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CodeChunk to a dictionary for serialization.

        This is used for storing in vector databases and API responses.
        Includes enhanced Graph RAG metadata and computed properties.
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
            "access_modifier": self.access_modifier,
            "imports_used": self.imports_used,
            "has_syntax_errors": self.has_syntax_errors,
            "error_details": self.error_details,
            "line_count": self.line_count,
            "char_count": self.char_count,
            # Enhanced Graph RAG metadata
            "is_nested": self.is_nested,
            "breadcrumb_depth": self.breadcrumb_depth,
            "breadcrumb_components": self.get_breadcrumb_components(),
            "parent_breadcrumb": self.get_parent_breadcrumb(),
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
            access_modifier=data.get("access_modifier"),
            imports_used=data.get("imports_used", []),
            has_syntax_errors=data.get("has_syntax_errors", False),
            error_details=data.get("error_details"),
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
