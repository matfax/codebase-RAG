"""
Breadcrumb Extractor utility for extracting hierarchical relationships from code.

This module provides language-aware extraction of code structure hierarchies
to support Graph RAG functionality. It analyzes Tree-sitter AST nodes to
build breadcrumb paths showing the full context of code chunks.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

try:
    from tree_sitter import Node
except ImportError:
    Node = Any  # Fallback for type hints when tree-sitter not available


@dataclass
class BreadcrumbContext:
    """Context information for breadcrumb extraction."""

    file_path: str
    language: str
    content: str
    content_lines: list[str]

    def get_node_text(self, node: Node) -> str:
        """Extract text content from a Tree-sitter node."""
        if not node:
            return ""
        return self.content[node.start_byte : node.end_byte]


@dataclass
class StructureInfo:
    """Information about a code structure for breadcrumb building."""

    name: str
    structure_type: str  # class, function, method, module, namespace, etc.
    parent_name: str | None = None
    breadcrumb_components: list[str] = None
    separator: str = "."  # Language-specific separator

    def __post_init__(self):
        if self.breadcrumb_components is None:
            self.breadcrumb_components = []

    def build_breadcrumb(self) -> str:
        """Build the complete breadcrumb path."""
        if not self.breadcrumb_components:
            return self.name
        return self.separator.join(self.breadcrumb_components + [self.name])


class BaseBreadcrumbExtractor(ABC):
    """Base class for language-specific breadcrumb extractors."""

    def __init__(self, language: str):
        self.language = language
        self.logger = logging.getLogger(f"{__name__}.{language}")

    @abstractmethod
    def extract_structure_info(self, node: Node, context: BreadcrumbContext, parent_path: list[str] = None) -> StructureInfo:
        """
        Extract structure information from a Tree-sitter node.

        Args:
            node: Tree-sitter AST node
            context: Breadcrumb extraction context
            parent_path: List of parent breadcrumb components

        Returns:
            StructureInfo with breadcrumb and parent information
        """
        pass

    @abstractmethod
    def get_separator(self) -> str:
        """Get the breadcrumb separator for this language."""
        pass

    def get_node_name(self, node: Node, context: BreadcrumbContext) -> str:
        """Extract the name from a node (default implementation)."""
        if not node:
            return "unnamed"

        # Try common name extraction patterns
        name_fields = ["name", "identifier", "property_identifier", "type_identifier"]

        for field in name_fields:
            name_node = node.child_by_field_name(field)
            if name_node:
                return context.get_node_text(name_node)

        # Fallback to direct text if small enough
        text = context.get_node_text(node)
        if len(text) < 50 and "\n" not in text:
            return text.strip()

        return "unnamed"

    def find_parent_structures(self, node: Node, context: BreadcrumbContext) -> list[str]:
        """Find parent structures by traversing up the AST."""
        parent_path = []
        current = node.parent

        while current:
            if self._is_structure_node(current):
                parent_name = self.get_node_name(current, context)
                if parent_name and parent_name != "unnamed":
                    parent_path.insert(0, parent_name)
            current = current.parent

        return parent_path

    @abstractmethod
    def _is_structure_node(self, node: Node) -> bool:
        """Check if a node represents a structural element (class, function, etc.)."""
        pass


class PythonBreadcrumbExtractor(BaseBreadcrumbExtractor):
    """Breadcrumb extractor for Python code."""

    def __init__(self):
        super().__init__("python")

    def get_separator(self) -> str:
        return "."

    def extract_structure_info(self, node: Node, context: BreadcrumbContext, parent_path: list[str] = None) -> StructureInfo:
        """Extract Python structure information."""
        if parent_path is None:
            parent_path = self.find_parent_structures(node, context)

        node_name = self.get_node_name(node, context)
        structure_type = self._get_python_structure_type(node)

        # Handle special cases for Python
        if structure_type == "method" and parent_path:
            # For methods, the parent is the class
            parent_name = parent_path[-1] if parent_path else None
        elif parent_path:
            parent_name = parent_path[-1]
        else:
            parent_name = None

        return StructureInfo(
            name=node_name,
            structure_type=structure_type,
            parent_name=parent_name,
            breadcrumb_components=parent_path,
            separator=self.get_separator(),
        )

    def _get_python_structure_type(self, node: Node) -> str:
        """Determine the Python structure type."""
        node_type = node.type

        if node_type == "function_definition":
            # Check if it's a method (inside a class)
            parent = node.parent
            while parent:
                if parent.type == "class_definition":
                    return "method"
                parent = parent.parent
            return "function"
        elif node_type == "class_definition":
            return "class"
        elif node_type == "import_statement" or node_type == "import_from_statement":
            return "import"
        elif node_type == "assignment" or node_type == "expression_statement":
            return "variable"
        else:
            return "unknown"

    def _is_structure_node(self, node: Node) -> bool:
        """Check if node is a Python structural element."""
        return node.type in [
            "function_definition",
            "class_definition",
            "module",
        ]


class JavaScriptBreadcrumbExtractor(BaseBreadcrumbExtractor):
    """Breadcrumb extractor for JavaScript/TypeScript code."""

    def __init__(self):
        super().__init__("javascript")

    def get_separator(self) -> str:
        return "."

    def extract_structure_info(self, node: Node, context: BreadcrumbContext, parent_path: list[str] = None) -> StructureInfo:
        """Extract JavaScript structure information."""
        if parent_path is None:
            parent_path = self.find_parent_structures(node, context)

        node_name = self.get_node_name(node, context)
        structure_type = self._get_javascript_structure_type(node)

        parent_name = parent_path[-1] if parent_path else None

        return StructureInfo(
            name=node_name,
            structure_type=structure_type,
            parent_name=parent_name,
            breadcrumb_components=parent_path,
            separator=self.get_separator(),
        )

    def _get_javascript_structure_type(self, node: Node) -> str:
        """Determine the JavaScript structure type."""
        node_type = node.type

        if node_type in ["function_declaration", "function", "arrow_function", "method_definition"]:
            return "function"
        elif node_type == "class_declaration":
            return "class"
        elif node_type == "import_statement":
            return "import"
        elif node_type in ["variable_declaration", "lexical_declaration"]:
            return "variable"
        else:
            return "unknown"

    def _is_structure_node(self, node: Node) -> bool:
        """Check if node is a JavaScript structural element."""
        return node.type in [
            "function_declaration",
            "function",
            "arrow_function",
            "method_definition",
            "class_declaration",
            "object_expression",
        ]


class CppBreadcrumbExtractor(BaseBreadcrumbExtractor):
    """Breadcrumb extractor for C++ code."""

    def __init__(self):
        super().__init__("cpp")

    def get_separator(self) -> str:
        return "::"

    def extract_structure_info(self, node: Node, context: BreadcrumbContext, parent_path: list[str] = None) -> StructureInfo:
        """Extract C++ structure information."""
        if parent_path is None:
            parent_path = self.find_parent_structures(node, context)

        node_name = self.get_node_name(node, context)
        structure_type = self._get_cpp_structure_type(node)

        parent_name = parent_path[-1] if parent_path else None

        return StructureInfo(
            name=node_name,
            structure_type=structure_type,
            parent_name=parent_name,
            breadcrumb_components=parent_path,
            separator=self.get_separator(),
        )

    def _get_cpp_structure_type(self, node: Node) -> str:
        """Determine the C++ structure type."""
        node_type = node.type

        if node_type == "function_definition":
            return "function"
        elif node_type == "class_specifier":
            return "class"
        elif node_type == "struct_specifier":
            return "struct"
        elif node_type == "namespace_definition":
            return "namespace"
        elif node_type == "template_declaration":
            return "template"
        else:
            return "unknown"

    def _is_structure_node(self, node: Node) -> bool:
        """Check if node is a C++ structural element."""
        return node.type in [
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "template_declaration",
        ]


class RustBreadcrumbExtractor(BaseBreadcrumbExtractor):
    """Breadcrumb extractor for Rust code."""

    def __init__(self):
        super().__init__("rust")

    def get_separator(self) -> str:
        return "::"

    def extract_structure_info(self, node: Node, context: BreadcrumbContext, parent_path: list[str] = None) -> StructureInfo:
        """Extract Rust structure information."""
        if parent_path is None:
            parent_path = self.find_parent_structures(node, context)

        node_name = self.get_node_name(node, context)
        structure_type = self._get_rust_structure_type(node)

        parent_name = parent_path[-1] if parent_path else None

        return StructureInfo(
            name=node_name,
            structure_type=structure_type,
            parent_name=parent_name,
            breadcrumb_components=parent_path,
            separator=self.get_separator(),
        )

    def _get_rust_structure_type(self, node: Node) -> str:
        """Determine the Rust structure type."""
        node_type = node.type

        if node_type == "function_item":
            return "function"
        elif node_type == "struct_item":
            return "struct"
        elif node_type == "enum_item":
            return "enum"
        elif node_type == "impl_item":
            return "impl"
        elif node_type == "trait_item":
            return "trait"
        elif node_type == "mod_item":
            return "module"
        else:
            return "unknown"

    def _is_structure_node(self, node: Node) -> bool:
        """Check if node is a Rust structural element."""
        return node.type in [
            "function_item",
            "struct_item",
            "enum_item",
            "impl_item",
            "trait_item",
            "mod_item",
        ]


class BreadcrumbExtractorFactory:
    """Factory for creating language-specific breadcrumb extractors."""

    _extractors = {
        "python": PythonBreadcrumbExtractor,
        "javascript": JavaScriptBreadcrumbExtractor,
        "typescript": JavaScriptBreadcrumbExtractor,  # TypeScript uses same as JS
        "cpp": CppBreadcrumbExtractor,
        "c": CppBreadcrumbExtractor,  # C uses same as C++
        "rust": RustBreadcrumbExtractor,
    }

    @classmethod
    def create_extractor(cls, language: str) -> BaseBreadcrumbExtractor | None:
        """
        Create a breadcrumb extractor for the specified language.

        Args:
            language: Programming language name

        Returns:
            Language-specific breadcrumb extractor or None if not supported
        """
        extractor_class = cls._extractors.get(language.lower())
        if extractor_class:
            return extractor_class()
        return None

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of supported languages for breadcrumb extraction."""
        return list(cls._extractors.keys())

    @classmethod
    def register_extractor(cls, language: str, extractor_class: type[BaseBreadcrumbExtractor]):
        """Register a new breadcrumb extractor for a language."""
        cls._extractors[language.lower()] = extractor_class


def extract_breadcrumb_info(node: Node, language: str, context: BreadcrumbContext) -> StructureInfo | None:
    """
    Convenience function to extract breadcrumb information from a node.

    Args:
        node: Tree-sitter AST node
        language: Programming language
        context: Breadcrumb extraction context

    Returns:
        StructureInfo with breadcrumb data or None if language not supported
    """
    extractor = BreadcrumbExtractorFactory.create_extractor(language)
    if extractor:
        return extractor.extract_structure_info(node, context)
    return None


def build_module_breadcrumb(file_path: str, language: str) -> str:
    """
    Build a module-level breadcrumb from file path.

    Args:
        file_path: Path to the source file
        language: Programming language

    Returns:
        Module breadcrumb string
    """
    import os

    # Extract filename without extension
    basename = os.path.basename(file_path)
    module_name = os.path.splitext(basename)[0]

    # Language-specific module naming conventions
    if language == "python":
        # Python: convert underscores to dots for package structure
        return module_name.replace("_", ".")
    elif language in ["javascript", "typescript"]:
        # JS/TS: use filename as module name
        return module_name
    elif language in ["cpp", "c", "rust"]:
        # C++/Rust: use namespace-style naming
        return module_name.replace("-", "_")
    else:
        return module_name
