"""
Language Registry utility for centralized language support information.

This module provides a centralized registry for programming language support,
including metadata, capabilities, and configuration information for each language.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.models.code_chunk import ChunkType


class LanguageComplexity(Enum):
    """Enumeration of language parsing complexity levels."""

    SIMPLE = "simple"  # Basic syntax, straightforward parsing
    MODERATE = "moderate"  # Standard complexity, most common languages
    COMPLEX = "complex"  # Advanced features, complex syntax
    EXPERIMENTAL = "experimental"  # Newly added, may have limitations


class LanguageParadigm(Enum):
    """Programming paradigms supported by languages."""

    OBJECT_ORIENTED = "object_oriented"
    FUNCTIONAL = "functional"
    PROCEDURAL = "procedural"
    IMPERATIVE = "imperative"
    DECLARATIVE = "declarative"
    SCRIPTING = "scripting"


@dataclass
class LanguageCapabilities:
    """Defines what features are supported for a programming language."""

    # Core parsing capabilities
    functions: bool = True
    classes: bool = False
    interfaces: bool = False
    namespaces: bool = False
    modules: bool = False

    # Advanced features
    templates: bool = False
    generics: bool = False
    macros: bool = False
    annotations: bool = False
    decorators: bool = False

    # Type system features
    static_typing: bool = False
    dynamic_typing: bool = True
    type_inference: bool = False
    type_hints: bool = False

    # Documentation features
    docstrings: bool = False
    inline_docs: bool = False
    comments: bool = True

    # Error handling
    exceptions: bool = True
    error_propagation: bool = False

    # Memory management
    automatic_memory: bool = True
    manual_memory: bool = False
    garbage_collection: bool = True


@dataclass
class LanguageMetadata:
    """Comprehensive metadata for a programming language."""

    # Basic information
    name: str
    display_name: str
    paradigms: list[LanguageParadigm]
    complexity: LanguageComplexity

    # File associations
    extensions: list[str]
    mime_types: list[str] = field(default_factory=list)

    # Tree-sitter configuration
    tree_sitter_module: str = ""
    tree_sitter_function: str = "language"

    # Parsing capabilities
    capabilities: LanguageCapabilities = field(default_factory=LanguageCapabilities)

    # Supported chunk types for this language
    supported_chunk_types: set[ChunkType] = field(default_factory=set)

    # Language-specific configuration
    comment_styles: dict[str, str] = field(default_factory=dict)
    string_delimiters: list[str] = field(default_factory=lambda: ['"', "'"])

    # Quality metrics
    parser_stability: float = 1.0  # 0.0 to 1.0, higher is more stable
    chunking_accuracy: float = 1.0  # 0.0 to 1.0, higher is more accurate

    # Documentation and links
    documentation_url: str = ""
    tree_sitter_repo: str = ""

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if not self.supported_chunk_types:
            # Set default chunk types based on capabilities
            chunk_types = {
                ChunkType.FUNCTION,
                ChunkType.IMPORT,
                ChunkType.CONSTANT,
                ChunkType.VARIABLE,
            }

            if self.capabilities.classes:
                chunk_types.add(ChunkType.CLASS)
            if self.capabilities.interfaces:
                chunk_types.add(ChunkType.INTERFACE)
            if self.capabilities.namespaces:
                chunk_types.add(ChunkType.NAMESPACE)
            if self.capabilities.templates:
                chunk_types.add(ChunkType.TEMPLATE)
            if self.capabilities.docstrings:
                chunk_types.add(ChunkType.DOCSTRING)

            self.supported_chunk_types = chunk_types


class LanguageRegistry:
    """
    Centralized registry for programming language support information.

    This class maintains comprehensive metadata about supported programming languages,
    their capabilities, and configuration details for parsing and chunking.
    """

    def __init__(self):
        """Initialize the language registry with built-in language definitions."""
        self.logger = logging.getLogger(__name__)
        self._languages: dict[str, LanguageMetadata] = {}
        self._extension_map: dict[str, str] = {}

        # Initialize with built-in language definitions
        self._initialize_builtin_languages()
        self._rebuild_extension_map()

    def _initialize_builtin_languages(self) -> None:
        """Initialize the registry with built-in language definitions."""

        # Python
        python_caps = LanguageCapabilities(
            functions=True,
            classes=True,
            decorators=True,
            modules=True,
            static_typing=False,
            dynamic_typing=True,
            type_hints=True,
            docstrings=True,
            inline_docs=True,
            exceptions=True,
            automatic_memory=True,
            garbage_collection=True,
        )
        self.register_language(
            LanguageMetadata(
                name="python",
                display_name="Python",
                paradigms=[
                    LanguageParadigm.OBJECT_ORIENTED,
                    LanguageParadigm.SCRIPTING,
                ],
                complexity=LanguageComplexity.MODERATE,
                extensions=[".py", ".pyw", ".pyi"],
                mime_types=["text/x-python", "application/x-python"],
                tree_sitter_module="tree_sitter_python",
                capabilities=python_caps,
                comment_styles={"line": "#", "block": '"""'},
                parser_stability=0.95,
                chunking_accuracy=0.92,
                documentation_url="https://docs.python.org/",
            )
        )

        # JavaScript
        js_caps = LanguageCapabilities(
            functions=True,
            classes=True,
            modules=True,
            static_typing=False,
            dynamic_typing=True,
            docstrings=False,
            inline_docs=True,
            exceptions=True,
            automatic_memory=True,
            garbage_collection=True,
        )
        self.register_language(
            LanguageMetadata(
                name="javascript",
                display_name="JavaScript",
                paradigms=[
                    LanguageParadigm.OBJECT_ORIENTED,
                    LanguageParadigm.FUNCTIONAL,
                    LanguageParadigm.SCRIPTING,
                ],
                complexity=LanguageComplexity.MODERATE,
                extensions=[".js", ".jsx", ".mjs", ".cjs"],
                mime_types=["text/javascript", "application/javascript"],
                tree_sitter_module="tree_sitter_javascript",
                capabilities=js_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.90,
                chunking_accuracy=0.88,
                documentation_url="https://developer.mozilla.org/en-US/docs/Web/JavaScript",
            )
        )

        # TypeScript
        ts_caps = LanguageCapabilities(
            functions=True,
            classes=True,
            interfaces=True,
            modules=True,
            static_typing=True,
            dynamic_typing=False,
            type_inference=True,
            annotations=True,
            docstrings=False,
            inline_docs=True,
            exceptions=True,
            automatic_memory=True,
            garbage_collection=True,
        )
        self.register_language(
            LanguageMetadata(
                name="typescript",
                display_name="TypeScript",
                paradigms=[
                    LanguageParadigm.OBJECT_ORIENTED,
                    LanguageParadigm.FUNCTIONAL,
                ],
                complexity=LanguageComplexity.MODERATE,
                extensions=[".ts"],
                mime_types=["text/typescript"],
                tree_sitter_module="tree_sitter_typescript",
                tree_sitter_function="language_typescript",
                capabilities=ts_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.90,
                chunking_accuracy=0.88,
                documentation_url="https://www.typescriptlang.org/docs/",
            )
        )

        # TSX (TypeScript React)
        tsx_caps = ts_caps  # Same capabilities as TypeScript
        self.register_language(
            LanguageMetadata(
                name="tsx",
                display_name="TypeScript JSX",
                paradigms=[
                    LanguageParadigm.OBJECT_ORIENTED,
                    LanguageParadigm.FUNCTIONAL,
                ],
                complexity=LanguageComplexity.MODERATE,
                extensions=[".tsx"],
                mime_types=["text/typescript"],
                tree_sitter_module="tree_sitter_typescript",
                tree_sitter_function="language_tsx",
                capabilities=tsx_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.88,
                chunking_accuracy=0.85,
                documentation_url="https://www.typescriptlang.org/docs/",
            )
        )

        # Go
        go_caps = LanguageCapabilities(
            functions=True,
            classes=False,
            interfaces=True,
            modules=True,
            static_typing=True,
            type_inference=True,
            docstrings=False,
            inline_docs=True,
            exceptions=False,
            error_propagation=True,
            automatic_memory=True,
            garbage_collection=True,
        )
        self.register_language(
            LanguageMetadata(
                name="go",
                display_name="Go",
                paradigms=[LanguageParadigm.PROCEDURAL, LanguageParadigm.IMPERATIVE],
                complexity=LanguageComplexity.MODERATE,
                extensions=[".go"],
                mime_types=["text/x-go"],
                tree_sitter_module="tree_sitter_go",
                capabilities=go_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.92,
                chunking_accuracy=0.90,
                documentation_url="https://golang.org/doc/",
            )
        )

        # Rust
        rust_caps = LanguageCapabilities(
            functions=True,
            classes=False,
            modules=True,
            static_typing=True,
            type_inference=True,
            generics=True,
            macros=True,
            annotations=True,
            docstrings=False,
            inline_docs=True,
            exceptions=False,
            error_propagation=True,
            automatic_memory=False,
            manual_memory=True,
        )
        self.register_language(
            LanguageMetadata(
                name="rust",
                display_name="Rust",
                paradigms=[LanguageParadigm.FUNCTIONAL, LanguageParadigm.IMPERATIVE],
                complexity=LanguageComplexity.COMPLEX,
                extensions=[".rs"],
                mime_types=["text/rust"],
                tree_sitter_module="tree_sitter_rust",
                capabilities=rust_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.88,
                chunking_accuracy=0.85,
                documentation_url="https://doc.rust-lang.org/",
            )
        )

        # Java
        java_caps = LanguageCapabilities(
            functions=True,
            classes=True,
            interfaces=True,
            static_typing=True,
            annotations=True,
            generics=True,
            docstrings=False,
            inline_docs=True,
            exceptions=True,
            automatic_memory=True,
            garbage_collection=True,
        )
        self.register_language(
            LanguageMetadata(
                name="java",
                display_name="Java",
                paradigms=[LanguageParadigm.OBJECT_ORIENTED],
                complexity=LanguageComplexity.MODERATE,
                extensions=[".java"],
                mime_types=["text/x-java-source"],
                tree_sitter_module="tree_sitter_java",
                capabilities=java_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.93,
                chunking_accuracy=0.91,
                documentation_url="https://docs.oracle.com/javase/",
            )
        )

        # C++
        cpp_caps = LanguageCapabilities(
            functions=True,
            classes=True,
            namespaces=True,
            templates=True,
            static_typing=True,
            macros=True,
            inline_docs=True,
            exceptions=True,
            automatic_memory=False,
            manual_memory=True,
        )
        self.register_language(
            LanguageMetadata(
                name="cpp",
                display_name="C++",
                paradigms=[
                    LanguageParadigm.OBJECT_ORIENTED,
                    LanguageParadigm.PROCEDURAL,
                ],
                complexity=LanguageComplexity.COMPLEX,
                extensions=[".cpp", ".cxx", ".cc", ".c", ".hpp", ".hxx", ".hh", ".h"],
                mime_types=["text/x-c++src", "text/x-c++hdr"],
                tree_sitter_module="tree_sitter_cpp",
                capabilities=cpp_caps,
                comment_styles={"line": "//", "block": "/* */"},
                parser_stability=0.85,
                chunking_accuracy=0.82,
                documentation_url="https://en.cppreference.com/",
            )
        )

    def register_language(self, metadata: LanguageMetadata) -> None:
        """
        Register a new language in the registry.

        Args:
            metadata: Complete language metadata
        """
        self._languages[metadata.name] = metadata
        self.logger.info(f"Registered language: {metadata.display_name} ({metadata.name})")

    def get_language(self, name: str) -> LanguageMetadata | None:
        """
        Get language metadata by name.

        Args:
            name: Language name

        Returns:
            Language metadata if found, None otherwise
        """
        return self._languages.get(name)

    def detect_language_from_extension(self, file_path: str) -> str | None:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language name if detected, None otherwise
        """
        from pathlib import Path

        extension = Path(file_path).suffix.lower()
        return self._extension_map.get(extension)

    def get_supported_languages(self) -> list[str]:
        """
        Get list of all supported language names.

        Returns:
            Sorted list of supported language names
        """
        return sorted(self._languages.keys())

    def get_languages_by_paradigm(self, paradigm: LanguageParadigm) -> list[str]:
        """
        Get languages that support a specific programming paradigm.

        Args:
            paradigm: Programming paradigm to filter by

        Returns:
            List of language names supporting the paradigm
        """
        return [name for name, metadata in self._languages.items() if paradigm in metadata.paradigms]

    def get_languages_by_complexity(self, complexity: LanguageComplexity) -> list[str]:
        """
        Get languages at a specific complexity level.

        Args:
            complexity: Complexity level to filter by

        Returns:
            List of language names at the specified complexity
        """
        return [name for name, metadata in self._languages.items() if metadata.complexity == complexity]

    def get_all_extensions(self) -> set[str]:
        """
        Get all supported file extensions.

        Returns:
            Set of file extensions (including the dot)
        """
        extensions = set()
        for metadata in self._languages.values():
            extensions.update(metadata.extensions)
        return extensions

    def get_language_capabilities(self, name: str) -> LanguageCapabilities | None:
        """
        Get language capabilities by name.

        Args:
            name: Language name

        Returns:
            Language capabilities if found, None otherwise
        """
        metadata = self.get_language(name)
        return metadata.capabilities if metadata else None

    def supports_chunk_type(self, language: str, chunk_type: ChunkType) -> bool:
        """
        Check if a language supports a specific chunk type.

        Args:
            language: Language name
            chunk_type: Chunk type to check

        Returns:
            True if the language supports the chunk type, False otherwise
        """
        metadata = self.get_language(language)
        if not metadata:
            return False
        return chunk_type in metadata.supported_chunk_types

    def get_tree_sitter_config(self, name: str) -> dict[str, str] | None:
        """
        Get Tree-sitter configuration for a language.

        Args:
            name: Language name

        Returns:
            Dictionary with module and function names, None if not found
        """
        metadata = self.get_language(name)
        if not metadata:
            return None

        return {
            "module": metadata.tree_sitter_module,
            "function": metadata.tree_sitter_function,
            "extensions": metadata.extensions,
        }

    def _rebuild_extension_map(self) -> None:
        """Rebuild the extension to language mapping."""
        self._extension_map.clear()
        for name, metadata in self._languages.items():
            for ext in metadata.extensions:
                self._extension_map[ext] = name

    def get_registry_stats(self) -> dict[str, Any]:
        """
        Get statistics about the language registry.

        Returns:
            Dictionary with registry statistics
        """
        stats = {
            "total_languages": len(self._languages),
            "total_extensions": len(self._extension_map),
            "by_complexity": {},
            "by_paradigm": {},
            "average_stability": 0.0,
            "average_accuracy": 0.0,
        }

        # Count by complexity
        for complexity in LanguageComplexity:
            count = len(self.get_languages_by_complexity(complexity))
            if count > 0:
                stats["by_complexity"][complexity.value] = count

        # Count by paradigm
        for paradigm in LanguageParadigm:
            count = len(self.get_languages_by_paradigm(paradigm))
            if count > 0:
                stats["by_paradigm"][paradigm.value] = count

        # Calculate averages
        if self._languages:
            total_stability = sum(m.parser_stability for m in self._languages.values())
            total_accuracy = sum(m.chunking_accuracy for m in self._languages.values())
            stats["average_stability"] = total_stability / len(self._languages)
            stats["average_accuracy"] = total_accuracy / len(self._languages)

        return stats

    def validate_language_config(self, name: str) -> list[str]:
        """
        Validate a language configuration and return any issues found.

        Args:
            name: Language name to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        metadata = self.get_language(name)

        if not metadata:
            issues.append(f"Language '{name}' not found in registry")
            return issues

        # Check required fields
        if not metadata.tree_sitter_module:
            issues.append(f"Missing tree_sitter_module for {name}")

        if not metadata.extensions:
            issues.append(f"No file extensions defined for {name}")

        # Check stability and accuracy ranges
        if not (0.0 <= metadata.parser_stability <= 1.0):
            issues.append(f"Parser stability out of range for {name}: {metadata.parser_stability}")

        if not (0.0 <= metadata.chunking_accuracy <= 1.0):
            issues.append(f"Chunking accuracy out of range for {name}: {metadata.chunking_accuracy}")

        return issues


# Global instance for easy access
language_registry = LanguageRegistry()
