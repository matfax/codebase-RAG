"""
Chunking strategies for language-specific code parsing and chunk extraction.

This module implements the Strategy pattern for handling different programming
languages' specific chunking logic, providing a flexible and extensible way
to support multiple languages with their unique characteristics.
"""

import logging
from abc import ABC, abstractmethod

from tree_sitter import Node

from src.models.code_chunk import ChunkType, CodeChunk

from .ast_extraction_service import AstExtractionService


class BaseChunkingStrategy(ABC):
    """
    Abstract base class for language-specific chunking strategies.

    This class defines the interface that all chunking strategies must implement,
    providing a consistent way to handle different programming languages while
    allowing for language-specific customizations.
    """

    def __init__(self, language: str):
        """
        Initialize the chunking strategy.

        Args:
            language: Programming language this strategy handles
        """
        self.language = language
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.ast_extractor = AstExtractionService()

    @abstractmethod
    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """
        Get language-specific AST node type mappings.

        Returns:
            Dictionary mapping chunk types to AST node types for this language
        """
        pass

    @abstractmethod
    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract code chunks from the AST for this language.

        Args:
            root_node: Root node of the parsed AST
            file_path: Path to the source file
            content: Original file content

        Returns:
            List of extracted code chunks
        """
        pass

    @abstractmethod
    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """
        Determine if a node should be included as a chunk for this language.

        Args:
            node: AST node to evaluate
            chunk_type: Proposed chunk type

        Returns:
            True if the chunk should be included, False otherwise
        """
        pass

    def get_language(self) -> str:
        """Get the language this strategy handles."""
        return self.language

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """
        Extract additional language-specific metadata for a chunk.

        Args:
            node: AST node
            chunk: Code chunk being processed

        Returns:
            Dictionary with additional metadata (empty by default)
        """
        return {}

    def validate_chunk(self, chunk: CodeChunk) -> bool:
        """
        Validate that a chunk meets language-specific requirements.

        Args:
            chunk: Code chunk to validate

        Returns:
            True if chunk is valid, False otherwise
        """
        # Basic validation - can be overridden by subclasses
        return chunk.content and chunk.content.strip() and chunk.start_line > 0 and chunk.end_line >= chunk.start_line

    def post_process_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """
        Post-process extracted chunks for language-specific optimizations.

        Args:
            chunks: List of extracted chunks

        Returns:
            Processed list of chunks
        """
        # Filter out invalid chunks
        valid_chunks = [chunk for chunk in chunks if self.validate_chunk(chunk)]

        # Sort by start line
        valid_chunks.sort(key=lambda c: c.start_line)

        return valid_chunks


class ChunkingStrategyRegistry:
    """
    Registry for managing and discovering chunking strategies.

    This class provides a centralized way to register and retrieve
    chunking strategies for different programming languages.
    """

    def __init__(self):
        """Initialize the strategy registry."""
        self.logger = logging.getLogger(__name__)
        self._strategies: dict[str, BaseChunkingStrategy] = {}

    def register_strategy(self, language: str, strategy: BaseChunkingStrategy) -> None:
        """
        Register a chunking strategy for a language.

        Args:
            language: Programming language name
            strategy: Chunking strategy instance
        """
        self._strategies[language] = strategy
        self.logger.info(f"Registered chunking strategy for {language}")

    def get_strategy(self, language: str) -> BaseChunkingStrategy | None:
        """
        Get the chunking strategy for a language.

        Args:
            language: Programming language name

        Returns:
            Chunking strategy instance or None if not found
        """
        return self._strategies.get(language)

    def has_strategy(self, language: str) -> bool:
        """
        Check if a strategy is registered for a language.

        Args:
            language: Programming language name

        Returns:
            True if strategy exists, False otherwise
        """
        return language in self._strategies

    def get_supported_languages(self) -> list[str]:
        """Get list of languages with registered strategies."""
        return list(self._strategies.keys())

    def unregister_strategy(self, language: str) -> bool:
        """
        Unregister a chunking strategy.

        Args:
            language: Programming language name

        Returns:
            True if strategy was removed, False if not found
        """
        if language in self._strategies:
            del self._strategies[language]
            self.logger.info(f"Unregistered chunking strategy for {language}")
            return True
        return False


# Global registry instance
chunking_strategy_registry = ChunkingStrategyRegistry()


def register_chunking_strategy(language: str):
    """
    Decorator for automatically registering chunking strategies.

    Args:
        language: Programming language name

    Returns:
        Decorator function
    """

    def decorator(strategy_class):
        """Register the strategy class."""
        strategy_instance = strategy_class(language)
        chunking_strategy_registry.register_strategy(language, strategy_instance)
        return strategy_class

    return decorator


class FallbackChunkingStrategy(BaseChunkingStrategy):
    """
    Fallback chunking strategy for unsupported languages.

    This strategy provides basic whole-file chunking when no specific
    language strategy is available.
    """

    def __init__(self, language: str):
        """Initialize the fallback strategy."""
        super().__init__(language)

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Return empty mappings for fallback strategy."""
        return {}

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract a single chunk containing the entire file.

        Args:
            root_node: Root node of the parsed AST
            file_path: Path to the source file
            content: Original file content

        Returns:
            List containing a single whole-file chunk
        """
        import hashlib

        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        content_lines = content.split("\n")

        # Generate chunk_id using file path and content hash
        chunk_id = f"{file_path}:{content_hash[:8]}"

        chunk = CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=self.language,
            start_line=1,
            end_line=len(content_lines),
            start_byte=0,
            end_byte=len(content.encode("utf-8")),
            name=f"file_{self.language}",
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )

        return [chunk]

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Always include chunks in fallback strategy."""
        return True


class StructuredFileChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy for structured files (JSON, YAML, etc.).

    This strategy handles non-code files that have structured content
    that can be meaningfully chunked.
    """

    def __init__(self, language: str):
        """Initialize the structured file strategy."""
        super().__init__(language)

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Return empty mappings as structured files don't use AST."""
        return {}

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract chunks from structured files.

        Args:
            root_node: Root node (not used for structured files)
            file_path: Path to the source file
            content: Original file content

        Returns:
            List of extracted chunks based on file structure
        """
        if self.language == "json":
            return self._extract_json_chunks(file_path, content)
        elif self.language == "yaml":
            return self._extract_yaml_chunks(file_path, content)
        elif self.language == "markdown":
            return self._extract_markdown_chunks(file_path, content)

        # Fallback to single chunk
        return self._extract_single_chunk(file_path, content)

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Include all chunks for structured files."""
        return True

    def _extract_json_chunks(self, file_path: str, content: str) -> list[CodeChunk]:
        """Extract chunks from JSON files."""
        try:
            import json

            data = json.loads(content)
            chunks = []

            if isinstance(data, dict):
                for i, (key, value) in enumerate(data.items()):
                    chunk_content = json.dumps({key: value}, indent=2)
                    chunk = self._create_structured_chunk(
                        file_path,
                        chunk_content,
                        ChunkType.OBJECT,
                        name=key,
                        start_line=i + 1,
                        end_line=i + 1,
                    )
                    chunks.append(chunk)

            return chunks if chunks else [self._extract_single_chunk(file_path, content)]

        except json.JSONDecodeError:
            return [self._extract_single_chunk(file_path, content)]

    def _extract_yaml_chunks(self, file_path: str, content: str) -> list[CodeChunk]:
        """Extract chunks from YAML files."""
        try:
            import yaml

            data = yaml.safe_load(content)
            chunks = []

            if isinstance(data, dict):
                for i, (key, value) in enumerate(data.items()):
                    chunk_content = yaml.dump({key: value}, default_flow_style=False)
                    chunk = self._create_structured_chunk(
                        file_path,
                        chunk_content,
                        ChunkType.OBJECT,
                        name=key,
                        start_line=i + 1,
                        end_line=i + 1,
                    )
                    chunks.append(chunk)

            return chunks if chunks else [self._extract_single_chunk(file_path, content)]

        except yaml.YAMLError:
            return [self._extract_single_chunk(file_path, content)]

    def _extract_markdown_chunks(self, file_path: str, content: str) -> list[CodeChunk]:
        """Extract chunks from Markdown files based on headers."""
        chunks = []
        lines = content.split("\n")
        current_section = []
        current_header = None
        section_start = 1

        for i, line in enumerate(lines):
            if line.startswith("#"):
                # New header found
                if current_section and current_header:
                    # Save previous section
                    section_content = "\n".join(current_section)
                    chunk = self._create_structured_chunk(
                        file_path,
                        section_content,
                        ChunkType.SECTION,
                        name=current_header.strip("#").strip(),
                        start_line=section_start,
                        end_line=i,
                    )
                    chunks.append(chunk)

                # Start new section
                current_header = line
                current_section = [line]
                section_start = i + 1
            else:
                current_section.append(line)

        # Add final section
        if current_section and current_header:
            section_content = "\n".join(current_section)
            chunk = self._create_structured_chunk(
                file_path,
                section_content,
                ChunkType.SECTION,
                name=current_header.strip("#").strip(),
                start_line=section_start,
                end_line=len(lines),
            )
            chunks.append(chunk)

        return chunks if chunks else [self._extract_single_chunk(file_path, content)]

    def _create_structured_chunk(
        self,
        file_path: str,
        content: str,
        chunk_type: ChunkType,
        name: str,
        start_line: int,
        end_line: int,
    ) -> CodeChunk:
        """Create a structured file chunk."""
        import hashlib

        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

        # Generate chunk_id using file path and content hash
        chunk_id = f"{file_path}:{content_hash[:8]}"

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=chunk_type,
            language=self.language,
            start_line=start_line,
            end_line=end_line,
            start_byte=0,  # Structured files start at beginning
            end_byte=len(content.encode("utf-8")),
            name=name,
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )

    def _extract_single_chunk(self, file_path: str, content: str) -> CodeChunk:
        """Create a single chunk for the entire file."""
        import hashlib

        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        content_lines = content.split("\n")

        # Generate chunk_id using file path and content hash
        chunk_id = f"{file_path}:{content_hash[:8]}"

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=self.language,
            start_line=1,
            end_line=len(content_lines),
            start_byte=0,
            end_byte=len(content.encode("utf-8")),
            name=f"file_{self.language}",
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )


# Register structured file strategies
structured_json_strategy = StructuredFileChunkingStrategy("json")
structured_yaml_strategy = StructuredFileChunkingStrategy("yaml")
structured_markdown_strategy = StructuredFileChunkingStrategy("markdown")

chunking_strategy_registry.register_strategy("json", structured_json_strategy)
chunking_strategy_registry.register_strategy("yaml", structured_yaml_strategy)
chunking_strategy_registry.register_strategy("markdown", structured_markdown_strategy)


@register_chunking_strategy("python")
class PythonChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy specifically designed for Python code.

    This strategy handles Python-specific constructs like functions, classes,
    decorators, async functions, and module-level constants.
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get Python-specific AST node type mappings."""
        return {
            # Core code structure chunks
            ChunkType.FUNCTION: ["function_definition"],
            ChunkType.CLASS: ["class_definition"],
            ChunkType.CONSTANT: ["assignment"],  # Filtered by context
            ChunkType.VARIABLE: ["assignment"],
            ChunkType.IMPORT: ["import_statement", "import_from_statement"],
            # Function call and relationship detection chunks
            ChunkType.FUNCTION_CALL: ["call"],
            ChunkType.METHOD_CALL: ["call"],  # Filtered by attribute context
            ChunkType.ASYNC_CALL: ["await"],
            ChunkType.ATTRIBUTE_ACCESS: ["attribute"],
        }

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """Extract Python-specific chunks from the AST."""
        node_mappings = self.get_node_mappings()
        chunks = self.ast_extractor.extract_chunks(root_node, file_path, content, self.language, node_mappings)

        # Python-specific post-processing
        processed_chunks = []
        for chunk in chunks:
            # Enhanced validation for Python
            if self._is_valid_python_chunk(chunk):
                # Add Python-specific metadata
                additional_metadata = self.extract_additional_metadata(root_node, chunk)
                if additional_metadata:
                    chunk.metadata = getattr(chunk, "metadata", {})
                    chunk.metadata.update(additional_metadata)

                processed_chunks.append(chunk)

        return self.post_process_chunks(processed_chunks)

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Determine if a Python node should be included as a chunk."""
        if chunk_type == ChunkType.CONSTANT:
            # Only include module-level assignments that look like constants
            return self._is_python_constant(node)

        elif chunk_type == ChunkType.FUNCTION:
            # Include all function definitions, including async
            return True

        elif chunk_type == ChunkType.CLASS:
            # Include all class definitions
            return True

        elif chunk_type == ChunkType.IMPORT:
            # Include import statements for dependency tracking
            return True

        elif chunk_type == ChunkType.FUNCTION_CALL:
            # Include function calls for relationship detection
            return self._is_significant_function_call(node)

        elif chunk_type == ChunkType.METHOD_CALL:
            # Include method calls for relationship detection
            return self._is_method_call(node)

        elif chunk_type == ChunkType.ASYNC_CALL:
            # Include async calls for relationship detection
            return True

        elif chunk_type == ChunkType.ATTRIBUTE_ACCESS:
            # Include significant attribute access for relationship detection
            return self._is_significant_attribute_access(node)

        return True

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """Extract Python-specific metadata."""
        metadata = {}

        # Check for decorators
        decorators = self._extract_decorators(node)
        if decorators:
            metadata["decorators"] = decorators

        # Check for async functions
        if chunk.chunk_type == ChunkType.FUNCTION:
            metadata["is_async"] = self.ast_extractor.is_async_function(node)

        # Check for class inheritance
        if chunk.chunk_type == ChunkType.CLASS:
            inheritance = self._extract_class_inheritance(node)
            if inheritance:
                metadata["inheritance"] = inheritance

        # Extract type hints if present
        type_hints = self._extract_type_hints(node)
        if type_hints:
            metadata["type_hints"] = type_hints

        # Handle function call metadata
        if chunk.chunk_type in [ChunkType.FUNCTION_CALL, ChunkType.METHOD_CALL]:
            call_metadata = self._extract_call_metadata(node)
            if call_metadata:
                metadata.update(call_metadata)

        # Handle async call metadata
        elif chunk.chunk_type == ChunkType.ASYNC_CALL:
            async_metadata = self._extract_async_call_metadata(node)
            if async_metadata:
                metadata.update(async_metadata)

        # Handle attribute access metadata
        elif chunk.chunk_type == ChunkType.ATTRIBUTE_ACCESS:
            attr_metadata = self._extract_attribute_metadata(node)
            if attr_metadata:
                metadata.update(attr_metadata)

        return metadata

    def _is_valid_python_chunk(self, chunk: CodeChunk) -> bool:
        """Validate Python-specific chunk requirements."""
        if not self.validate_chunk(chunk):
            return False

        # Python-specific validations
        if chunk.chunk_type == ChunkType.FUNCTION:
            # Function should have a name and some content
            return chunk.name and chunk.name != "unnamed_function"

        elif chunk.chunk_type == ChunkType.CLASS:
            # Class should have a name
            return chunk.name and chunk.name != "unnamed_class"

        elif chunk.chunk_type == ChunkType.CONSTANT:
            # Constants should be uppercase with underscores
            return chunk.name and chunk.name.isupper() and "_" in chunk.name and len(chunk.content.strip()) < 200  # Avoid huge constants

        return True

    def _is_python_constant(self, node: Node) -> bool:
        """Check if a Python assignment represents a constant."""
        if node.type != "assignment":
            return False

        # Check if assignment target is an identifier
        if not node.children or node.children[0].type != "identifier":
            return False

        # Get the variable name
        name = node.children[0].text.decode("utf-8")

        # Consider uppercase names with underscores as constants
        return name.isupper() and "_" in name and not name.startswith("_")

    def _extract_decorators(self, node: Node) -> list[str]:
        """Extract decorator names from a Python function or class."""
        decorators = []

        # Look for decorator nodes before the function/class
        for child in node.children:
            if child.type == "decorator":
                # Extract decorator name
                for subchild in child.children:
                    if subchild.type == "identifier":
                        decorators.append(f"@{subchild.text.decode('utf-8')}")
                        break
                    elif subchild.type == "attribute":
                        # Handle complex decorators like @dataclass.decorator
                        decorators.append(f"@{subchild.text.decode('utf-8')}")
                        break

        return decorators

    def _extract_class_inheritance(self, node: Node) -> list[str]:
        """Extract base classes from a Python class definition."""
        inheritance = []

        if node.type == "class_definition":
            # Look for argument_list which contains base classes
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            inheritance.append(arg.text.decode("utf-8"))

        return inheritance

    def _extract_type_hints(self, node: Node) -> dict[str, str]:
        """Extract type hints from Python functions."""
        type_hints = {}

        if node.type == "function_definition":
            # Look for parameters with type annotations
            for child in node.children:
                if child.type == "parameters":
                    # Extract parameter type hints
                    # This is a simplified implementation
                    pass

        return type_hints

    def _is_significant_function_call(self, node: Node) -> bool:
        """Check if a function call is significant enough to include as a chunk."""
        if node.type != "call":
            return False

        # Get the function being called
        function_node = node.child_by_field_name("function")
        if not function_node:
            return False

        # Include direct function calls (identifier)
        if function_node.type == "identifier":
            function_name = function_node.text.decode("utf-8")
            # Filter out very common built-in functions that add noise
            common_builtins = {"print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple"}
            return function_name not in common_builtins

        # Include module function calls (attribute access)
        elif function_node.type == "attribute":
            return True

        return True

    def _is_method_call(self, node: Node) -> bool:
        """Check if a call node represents a method call (obj.method())."""
        if node.type != "call":
            return False

        function_node = node.child_by_field_name("function")
        if not function_node:
            return False

        # Method calls have attribute access as the function
        return function_node.type == "attribute"

    def _is_significant_attribute_access(self, node: Node) -> bool:
        """Check if attribute access is significant for relationship detection."""
        if node.type != "attribute":
            return False

        # Always include attribute access that could be method calls or property access
        # We can filter further during processing based on usage context
        object_node = node.child_by_field_name("object")
        attribute_node = node.child_by_field_name("attribute")

        if not object_node or not attribute_node:
            return False

        # Get attribute name
        attribute_name = attribute_node.text.decode("utf-8")

        # Filter out some very common attributes that might add noise
        common_attrs = {"__dict__", "__class__", "__module__"}
        return attribute_name not in common_attrs

    def _extract_call_metadata(self, node: Node) -> dict[str, any]:
        """Extract metadata for function/method call nodes."""
        metadata = {}

        if node.type != "call":
            return metadata

        # Get function being called
        function_node = node.child_by_field_name("function")
        if function_node:
            if function_node.type == "identifier":
                # Direct function call
                metadata["call_type"] = "function"
                metadata["function_name"] = function_node.text.decode("utf-8")
            elif function_node.type == "attribute":
                # Method call
                metadata["call_type"] = "method"

                # Extract object and method name
                object_node = function_node.child_by_field_name("object")
                attribute_node = function_node.child_by_field_name("attribute")

                if object_node:
                    metadata["object_name"] = object_node.text.decode("utf-8")
                if attribute_node:
                    metadata["method_name"] = attribute_node.text.decode("utf-8")

        # Count arguments
        arguments_node = node.child_by_field_name("arguments")
        if arguments_node:
            arg_count = len([child for child in arguments_node.children if child.type != "," and child.text.decode("utf-8").strip()])
            metadata["argument_count"] = arg_count

        return metadata

    def _extract_async_call_metadata(self, node: Node) -> dict[str, any]:
        """Extract metadata for async call nodes (await expressions)."""
        metadata = {"is_async": True}

        if node.type != "await":
            return metadata

        # Get the expression being awaited
        for child in node.children:
            if child.type == "call":
                # Extract call metadata from the awaited call
                call_metadata = self._extract_call_metadata(child)
                metadata.update(call_metadata)
                metadata["call_type"] = f"async_{call_metadata.get('call_type', 'unknown')}"
                break

        return metadata

    def _extract_attribute_metadata(self, node: Node) -> dict[str, any]:
        """Extract metadata for attribute access nodes."""
        metadata = {}

        if node.type != "attribute":
            return metadata

        # Extract object and attribute names
        object_node = node.child_by_field_name("object")
        attribute_node = node.child_by_field_name("attribute")

        if object_node:
            metadata["object_name"] = object_node.text.decode("utf-8")
            metadata["object_type"] = object_node.type

        if attribute_node:
            metadata["attribute_name"] = attribute_node.text.decode("utf-8")

        # Determine if this is a chained access
        if object_node and object_node.type == "attribute":
            metadata["is_chained"] = True
            metadata["chain_depth"] = self._calculate_attribute_chain_depth(node)
        else:
            metadata["is_chained"] = False
            metadata["chain_depth"] = 1

        return metadata

    def _calculate_attribute_chain_depth(self, node: Node) -> int:
        """Calculate the depth of attribute access chain (e.g., a.b.c = depth 3)."""
        depth = 1
        current = node

        while current and current.type == "attribute":
            object_node = current.child_by_field_name("object")
            if object_node and object_node.type == "attribute":
                depth += 1
                current = object_node
            else:
                break

        return depth


@register_chunking_strategy("javascript")
class JavaScriptChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy specifically designed for JavaScript/TypeScript code.

    This strategy handles JavaScript-specific constructs like functions, classes,
    arrow functions, async functions, and ES6+ features.
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get JavaScript-specific AST node type mappings."""
        return {
            ChunkType.FUNCTION: [
                "function_declaration",
                "arrow_function",
                "method_definition",
            ],
            ChunkType.ASYNC_FUNCTION: ["async_function_declaration"],
            ChunkType.CLASS: ["class_declaration"],
            ChunkType.CONSTANT: ["lexical_declaration"],  # const declarations
            ChunkType.VARIABLE: ["variable_declaration"],
            ChunkType.IMPORT: ["import_statement"],
            ChunkType.EXPORT: ["export_statement"],
        }

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """Extract JavaScript-specific chunks from the AST."""
        node_mappings = self.get_node_mappings()
        chunks = self.ast_extractor.extract_chunks(root_node, file_path, content, self.language, node_mappings)

        # JavaScript-specific post-processing
        processed_chunks = []
        for chunk in chunks:
            if self._is_valid_javascript_chunk(chunk):
                # Add JavaScript-specific metadata
                additional_metadata = self.extract_additional_metadata(root_node, chunk)
                if additional_metadata:
                    chunk.metadata = getattr(chunk, "metadata", {})
                    chunk.metadata.update(additional_metadata)

                processed_chunks.append(chunk)

        return self.post_process_chunks(processed_chunks)

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Determine if a JavaScript node should be included as a chunk."""
        if chunk_type == ChunkType.CONSTANT:
            # Only include const declarations at module level
            return self._is_javascript_constant(node)

        elif chunk_type == ChunkType.FUNCTION:
            # Include all function types
            return True

        elif chunk_type == ChunkType.CLASS:
            # Include class declarations
            return True

        elif chunk_type in [ChunkType.IMPORT, ChunkType.EXPORT]:
            # Include import/export statements
            return True

        return True

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """Extract JavaScript-specific metadata."""
        metadata = {}

        # Check for arrow functions
        if chunk.chunk_type == ChunkType.FUNCTION:
            metadata["is_arrow_function"] = node.type == "arrow_function"
            metadata["is_async"] = self.ast_extractor.is_async_function(node)
            metadata["is_method"] = node.type == "method_definition"

        # Check for class features
        if chunk.chunk_type == ChunkType.CLASS:
            class_features = self._extract_class_features(node)
            metadata.update(class_features)

        # Check for ES6+ features
        es6_features = self._extract_es6_features(node)
        if es6_features:
            metadata["es6_features"] = es6_features

        return metadata

    def _is_valid_javascript_chunk(self, chunk: CodeChunk) -> bool:
        """Validate JavaScript-specific chunk requirements."""
        if not self.validate_chunk(chunk):
            return False

        # JavaScript-specific validations
        if chunk.chunk_type == ChunkType.FUNCTION:
            # Function should have meaningful content
            return len(chunk.content.strip()) > 10

        elif chunk.chunk_type == ChunkType.CLASS:
            # Class should have a name and body
            return chunk.name and "{" in chunk.content

        elif chunk.chunk_type == ChunkType.CONSTANT:
            # Constants should be const declarations
            return "const" in chunk.content and "=" in chunk.content

        return True

    def _is_javascript_constant(self, node: Node) -> bool:
        """Check if a JavaScript declaration represents a constant."""
        if node.type != "lexical_declaration":
            return False

        # Check for 'const' keyword
        for child in node.children:
            if child.type == "const" or (child.text and child.text.decode("utf-8") == "const"):
                return True

        return False

    def _extract_class_features(self, node: Node) -> dict[str, any]:
        """Extract JavaScript class features."""
        features = {}

        if node.type == "class_declaration":
            # Check for extends clause
            for child in node.children:
                if child.type == "class_heritage":
                    # Has inheritance
                    features["has_inheritance"] = True
                    # Extract parent class name
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            features["extends"] = subchild.text.decode("utf-8")
                            break

        return features

    def _extract_es6_features(self, node: Node) -> list[str]:
        """Extract ES6+ features used in the code."""
        features = []

        # Check for various ES6+ features
        node_text = node.text.decode("utf-8") if node.text else ""

        if "=>" in node_text:
            features.append("arrow_functions")

        if "async" in node_text:
            features.append("async_await")

        if "const " in node_text or "let " in node_text:
            features.append("block_scoping")

        if "`" in node_text:
            features.append("template_literals")

        if "class " in node_text:
            features.append("classes")

        return features


@register_chunking_strategy("typescript")
class TypeScriptChunkingStrategy(JavaScriptChunkingStrategy):
    """
    Chunking strategy for TypeScript code.

    Extends JavaScript strategy with TypeScript-specific features like
    interfaces, type aliases, and enhanced type annotations.
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get TypeScript-specific AST node type mappings."""
        base_mappings = super().get_node_mappings()

        # Add TypeScript-specific mappings
        typescript_mappings = {
            ChunkType.INTERFACE: ["interface_declaration"],
            ChunkType.TYPE_ALIAS: ["type_alias_declaration"],
            ChunkType.ENUM: ["enum_declaration"],
        }

        base_mappings.update(typescript_mappings)
        return base_mappings

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """TypeScript-specific chunk inclusion logic."""
        if chunk_type in [ChunkType.INTERFACE, ChunkType.TYPE_ALIAS, ChunkType.ENUM]:
            # Always include TypeScript type definitions
            return True

        # Fall back to JavaScript logic for other types
        return super().should_include_chunk(node, chunk_type)

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """Extract TypeScript-specific metadata."""
        metadata = super().extract_additional_metadata(node, chunk)

        # Add TypeScript-specific metadata
        if chunk.chunk_type == ChunkType.INTERFACE:
            interface_features = self._extract_interface_features(node)
            metadata.update(interface_features)

        elif chunk.chunk_type == ChunkType.TYPE_ALIAS:
            type_features = self._extract_type_alias_features(node)
            metadata.update(type_features)

        # Extract generic type parameters
        generic_params = self._extract_generic_parameters(node)
        if generic_params:
            metadata["generic_parameters"] = generic_params

        return metadata

    def _extract_interface_features(self, node: Node) -> dict[str, any]:
        """Extract TypeScript interface features."""
        features = {}

        if node.type == "interface_declaration":
            # Check for interface inheritance
            for child in node.children:
                if child.type == "extends_clause":
                    features["has_inheritance"] = True
                    # Extract parent interfaces
                    extends_list = []
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            extends_list.append(subchild.text.decode("utf-8"))
                    features["extends"] = extends_list

        return features

    def _extract_type_alias_features(self, node: Node) -> dict[str, any]:
        """Extract TypeScript type alias features."""
        features = {}

        if node.type == "type_alias_declaration":
            # Analyze the type being aliased
            node_text = node.text.decode("utf-8") if node.text else ""

            if "union" in node_text or "|" in node_text:
                features["is_union_type"] = True

            if "intersection" in node_text or "&" in node_text:
                features["is_intersection_type"] = True

            if "Record<" in node_text or "Partial<" in node_text:
                features["uses_utility_types"] = True

        return features

    def _extract_generic_parameters(self, node: Node) -> list[str]:
        """Extract generic type parameters from TypeScript constructs."""
        generics = []

        # Look for type_parameters child
        for child in node.children:
            if child.type == "type_parameters":
                for param in child.children:
                    if param.type == "type_parameter":
                        for subchild in param.children:
                            if subchild.type == "type_identifier":
                                generics.append(subchild.text.decode("utf-8"))

        return generics
