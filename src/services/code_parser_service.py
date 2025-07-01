"""
CodeParser service for intelligent code chunking using Tree-sitter.

This service provides semantic code parsing capabilities to break down source files
into meaningful chunks based on code structure rather than simple text splitting.
"""

import logging
import time
import json
import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import hashlib
from datetime import datetime

try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")

from models.code_chunk import CodeChunk, ChunkType, ParseResult, CodeSyntaxError
from utils.file_system_utils import get_file_size, get_file_mtime
from utils.tree_sitter_manager import TreeSitterManager
from utils.chunking_metrics_tracker import chunking_metrics_tracker


class CodeParserService:
    """
    Service for parsing source code into intelligent semantic chunks.
    
    This service uses Tree-sitter to parse code into an Abstract Syntax Tree (AST)
    and then extracts meaningful code constructs like functions, classes, and constants.
    """
    
    def __init__(self):
        """Initialize the CodeParser service with language support."""
        self.logger = logging.getLogger(__name__)
        
        # Use TreeSitterManager for robust parser management
        self._tree_sitter_manager = TreeSitterManager()
        
        # Get initialization summary for logging
        summary = self._tree_sitter_manager.get_initialization_summary()
        self.logger.info(f"Initialized Tree-sitter parsers: {summary['successful_languages']}/{summary['total_languages']} languages successful")
        
        if summary['failed_languages']:
            self.logger.warning(f"Failed to initialize: {', '.join(summary['failed_languages'])}")
        
        # Legacy attributes for backward compatibility
        self._parsers: Dict[str, Parser] = {}
        self._languages: Dict[str, Language] = {}
        for lang in summary['supported_languages']:
            self._parsers[lang] = self._tree_sitter_manager.get_parser(lang)
            self._languages[lang] = self._tree_sitter_manager.get_language(lang)
        
        # Language-specific node types for different code constructs
        self._node_mappings = {
            'python': {
                ChunkType.FUNCTION: ['function_definition'],
                ChunkType.CLASS: ['class_definition'],
                ChunkType.CONSTANT: ['assignment'],  # We'll filter these by context
                ChunkType.VARIABLE: ['assignment'],
                ChunkType.IMPORT: ['import_statement', 'import_from_statement']
                # Note: ASYNC_FUNCTION will be detected by checking for 'async' child in function_definition
                # Note: DOCSTRING detection will be handled specially to avoid over-extraction
            },
            'javascript': {
                ChunkType.FUNCTION: ['function_declaration', 'arrow_function', 'method_definition'],
                ChunkType.ASYNC_FUNCTION: ['async_function_declaration'],
                ChunkType.CLASS: ['class_declaration'],
                ChunkType.CONSTANT: ['lexical_declaration'],  # const declarations
                ChunkType.VARIABLE: ['variable_declaration'],
                ChunkType.IMPORT: ['import_statement'],
                ChunkType.EXPORT: ['export_statement']
            },
            'typescript': {
                ChunkType.FUNCTION: ['function_declaration', 'arrow_function', 'method_definition', 'method_signature'],
                ChunkType.ASYNC_FUNCTION: ['async_function_declaration'],
                ChunkType.CLASS: ['class_declaration'],
                ChunkType.INTERFACE: ['interface_declaration'],
                ChunkType.TYPE_ALIAS: ['type_alias_declaration'],
                ChunkType.CONSTANT: ['lexical_declaration'],
                ChunkType.VARIABLE: ['variable_declaration'],
                ChunkType.IMPORT: ['import_statement'],
                ChunkType.EXPORT: ['export_statement']
            },
            'go': {
                ChunkType.FUNCTION: ['function_declaration', 'method_declaration'],
                ChunkType.STRUCT: ['type_declaration'],  # Go structs and interfaces (distinguished by special handling)
                ChunkType.CONSTANT: ['const_declaration'],
                ChunkType.VARIABLE: ['var_declaration'],
                ChunkType.IMPORT: ['import_declaration']
            },
            'rust': {
                ChunkType.FUNCTION: ['function_item'],
                ChunkType.STRUCT: ['struct_item'],
                ChunkType.ENUM: ['enum_item'],
                ChunkType.IMPL: ['impl_item'],
                ChunkType.CONSTANT: ['const_item'],
                ChunkType.VARIABLE: ['let_declaration'],
                ChunkType.IMPORT: ['use_declaration']
            },
            'java': {
                ChunkType.FUNCTION: ['method_declaration'],
                ChunkType.CONSTRUCTOR: ['constructor_declaration'],
                ChunkType.CLASS: ['class_declaration'],
                ChunkType.INTERFACE: ['interface_declaration'],
                ChunkType.ENUM: ['enum_declaration'],
                ChunkType.CONSTANT: ['field_declaration'],  # Static final fields
                ChunkType.VARIABLE: ['field_declaration'],
                ChunkType.IMPORT: ['import_declaration']
            },
            'cpp': {
                ChunkType.FUNCTION: ['function_definition', 'function_declarator'],
                ChunkType.CLASS: ['class_specifier'],
                ChunkType.STRUCT: ['struct_specifier'],
                ChunkType.NAMESPACE: ['namespace_definition'],
                ChunkType.CONSTANT: ['declaration'],  # const declarations (will filter by context)
                ChunkType.VARIABLE: ['declaration'],  # variable declarations
                ChunkType.IMPORT: ['preproc_include'],  # #include statements
                ChunkType.CONSTRUCTOR: ['function_definition'],  # constructors (special handling needed)
                ChunkType.DESTRUCTOR: ['function_definition'],  # destructors (special handling needed)
                ChunkType.TEMPLATE: ['template_declaration']  # template definitions
            }
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return list(self._parsers.keys())
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect programming language from file extension.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Language name if supported, None otherwise
        """
        # Use TreeSitterManager for language detection
        detected = self._tree_sitter_manager.detect_language_from_extension(file_path)
        
        # Special handling for TSX files - TreeSitterManager returns 'tsx' but we also support 'typescript'
        if detected is None:
            path = Path(file_path)
            extension = path.suffix.lower()
            # Handle .tsx files that could be parsed as either tsx or typescript
            if extension == '.tsx' and self._tree_sitter_manager.is_language_supported('tsx'):
                return 'tsx'
            # Handle JSON/YAML files for structured chunking
            elif extension in ['.json', '.jsonl']:
                return 'json'
            elif extension in ['.yaml', '.yml']:
                return 'yaml'
            # Handle Markdown files for hierarchical chunking
            elif extension in ['.md', '.markdown']:
                return 'markdown'
        
        return detected
    
    def parse_file(self, file_path: str, content: Optional[str] = None) -> ParseResult:
        """
        Parse a source file into intelligent code chunks.
        
        Args:
            file_path: Path to the source file
            content: Optional file content (if None, will read from file)
            
        Returns:
            ParseResult containing extracted chunks and metadata
        """
        start_time = time.time()
        
        # Detect language
        language = self.detect_language(file_path)
        if not language:
            return self._create_fallback_result(file_path, content, start_time)
        
        # Handle JSON/YAML/Markdown files with structured parsing
        if language in ['json', 'yaml', 'markdown']:
            return self._parse_structured_file(file_path, content, language, start_time)
        
        # Handle Tree-sitter parseable languages
        if language not in self._parsers:
            return self._create_fallback_result(file_path, content, start_time)
        
        # Read file content if not provided
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                self.logger.error(f"Failed to read file {file_path}: {e}")
                return self._create_fallback_result(file_path, "", start_time, error=True)
        
        # Parse the content with enhanced error recovery
        try:
            parser = self._parsers[language]
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Validate parser state and tree structure
            if not self._validate_parse_tree(tree, file_path):
                self.logger.warning(f"Invalid parse tree for {file_path}, attempting recovery")
                # Try alternative parsing approaches
                tree = self._attempt_parse_recovery(parser, content, file_path, language)
            
            # Extract chunks from the AST with enhanced error handling
            chunks = self._extract_chunks(tree.root_node, file_path, content, language)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Check for parsing errors and collect detailed error information
            error_count = self._count_errors(tree.root_node)
            parse_success = error_count == 0
            syntax_errors = self._collect_detailed_errors(tree.root_node, content.split('\n'), language)
            
            # Enhanced error classification and recovery assessment
            error_recovery_used = error_count > 0 and len(chunks) > 0
            valid_sections_count = len(chunks) if error_recovery_used else 0
            
            # Log detailed recovery statistics
            if error_recovery_used:
                recovery_rate = len(chunks) / max(1, error_count + len(chunks))
                self.logger.info(f"Error recovery for {file_path}: {len(chunks)} chunks recovered "
                               f"from {error_count} errors (recovery rate: {recovery_rate:.2%})")
            
            # Comprehensive chunk quality validation
            validated_chunks, quality_issues = self._validate_chunk_quality(chunks, content.split('\n'), language)
            
            # Log quality issues if any
            if quality_issues:
                self.logger.warning(f"Chunk quality issues found in {file_path}:")
                for issue in quality_issues:
                    self.logger.warning(f"  - {issue}")
            
            # Create parse result
            parse_result = ParseResult(
                chunks=validated_chunks,
                file_path=file_path,
                language=language,
                parse_success=parse_success,
                error_count=error_count,
                fallback_used=False,
                processing_time_ms=processing_time,
                syntax_errors=syntax_errors,
                error_recovery_used=error_recovery_used,
                valid_sections_count=valid_sections_count
            )
            
            # Enhanced logging with detailed error analysis
            self._enhance_error_logging_in_parse(file_path, language, parse_result)
            
            # Record metrics for performance tracking
            quality_issues_count = len(quality_issues) if 'quality_issues' in locals() else 0
            self._record_parsing_metrics(parse_result, file_path, quality_issues_count)
            
            return parse_result
            
        except Exception as e:
            self.logger.error(f"Critical parsing failure for {file_path}: {e}")
            # Enhanced fallback with error context
            fallback_result = self._create_fallback_result(file_path, content, start_time, error=True, 
                                                          exception_context=str(e))
            
            # Record metrics for fallback case
            self._record_parsing_metrics(fallback_result, file_path, 0)
            
            return fallback_result
    
    def _extract_chunks(self, root_node: Node, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """
        Extract code chunks from the parsed AST with intelligent error handling.
        
        Args:
            root_node: Root node of the parsed AST
            file_path: Path to the source file
            content: File content as string
            language: Programming language
            
        Returns:
            List of extracted CodeChunk objects
        """
        content_lines = content.split('\n')
        
        # Check for syntax errors and handle intelligently
        error_count = self._count_errors(root_node)
        
        if error_count > 0:
            # Use intelligent error handling
            chunks = self._handle_syntax_errors(root_node, file_path, content, content_lines, language)
        else:
            # Normal parsing without errors
            chunks = []
            node_mappings = self._node_mappings.get(language, {})
            self._traverse_ast(root_node, chunks, file_path, content, content_lines, language, node_mappings)
        
        # Enhanced fallback logic - only use whole-file chunking when absolutely necessary
        if not chunks:
            if self._should_use_whole_file_fallback(file_path, content, language, error_count):
                chunks.append(self._create_whole_file_chunk(file_path, content, language))
            else:
                # Try alternative chunking strategies before falling back to whole-file
                chunks = self._attempt_alternative_chunking(file_path, content, language, content_lines)
        
        return chunks
    
    def _traverse_ast(self, node: Node, chunks: List[CodeChunk], file_path: str, 
                     content: str, content_lines: List[str], language: str, 
                     node_mappings: Dict[ChunkType, List[str]]) -> None:
        """
        Recursively traverse the AST and extract code chunks.
        
        Args:
            node: Current AST node
            chunks: List to append extracted chunks
            file_path: Path to the source file
            content: File content as string
            content_lines: File content split by lines
            language: Programming language
            node_mappings: Mapping of chunk types to AST node types
        """
        # Check if this node represents a chunk we want to extract
        chunk_type = self._get_chunk_type(node, node_mappings, language)
        
        if chunk_type:
            chunk = self._create_chunk_from_node(
                node, chunk_type, file_path, content, content_lines, language
            )
            if chunk:
                chunks.append(chunk)
        
        # Recursively process child nodes
        for child in node.children:
            self._traverse_ast(child, chunks, file_path, content, content_lines, language, node_mappings)
    
    def _get_chunk_type(self, node: Node, node_mappings: Dict[ChunkType, List[str]], language: str) -> Optional[ChunkType]:
        """
        Determine the chunk type for a given AST node.
        
        Args:
            node: AST node to classify
            node_mappings: Mapping of chunk types to node types
            language: Programming language for intelligent classification
            
        Returns:
            ChunkType if the node should be extracted, None otherwise
        """
        node_type = node.type
        
        # Special handling for Python
        if language == 'python':
            if node_type == 'function_definition':
                # Check if it's an async function
                if self._is_async_function(node):
                    return ChunkType.ASYNC_FUNCTION
                else:
                    return ChunkType.FUNCTION
            elif node_type == 'assignment':
                # Only include module-level assignments, skip those inside functions/classes
                if self._is_module_level_assignment(node):
                    return self._classify_python_assignment(node)
                else:
                    return None  # Skip assignments inside functions/classes
        
        # Special handling for JavaScript/TypeScript
        elif language in ['javascript', 'typescript', 'tsx']:
            if node_type == 'method_definition':
                # Check if it's an async method
                if self._is_async_function(node):
                    return ChunkType.ASYNC_FUNCTION
                else:
                    return ChunkType.FUNCTION
            elif node_type == 'function_declaration':
                # Check if it's an async function
                if self._is_async_function(node):
                    return ChunkType.ASYNC_FUNCTION
                else:
                    return ChunkType.FUNCTION
            elif node_type == 'arrow_function':
                # Only extract arrow functions that are not part of variable declarations
                # (those will be handled by the lexical_declaration extraction)
                parent = node.parent
                if parent and parent.type == 'variable_declarator':
                    # This arrow function is part of a variable assignment, skip it
                    # The variable declaration will handle both the name and function
                    return None
                
                # Standalone arrow functions can also be async
                if self._is_async_function(node):
                    return ChunkType.ASYNC_FUNCTION
                else:
                    return ChunkType.FUNCTION
            elif node_type in ['lexical_declaration', 'variable_declaration']:
                # Only include module-level declarations, skip those inside functions/classes
                if self._is_module_level_js_declaration(node):
                    return self._classify_js_declaration(node)
                else:
                    return None  # Skip declarations inside functions/classes
        
        # Special handling for Java
        elif language == 'java':
            if node_type == 'method_declaration':
                # Check if it's a constructor (has class name as return type, no actual return type)
                if self._is_java_constructor(node):
                    return ChunkType.CONSTRUCTOR
                else:
                    return ChunkType.FUNCTION
        
        # Special handling for Go
        elif language == 'go':
            if node_type == 'type_declaration':
                # Check if it's a struct or interface within the type_declaration
                for child in node.children:
                    if child.type == 'type_spec':
                        for spec_child in child.children:
                            if spec_child.type == 'struct_type':
                                return ChunkType.STRUCT
                            elif spec_child.type == 'interface_type':
                                return ChunkType.INTERFACE
                # Default to STRUCT if we can't determine
                return ChunkType.STRUCT
        
        # Special handling for C++
        elif language == 'cpp':
            if node_type == 'function_definition':
                # Check if it's a constructor, destructor, or regular function
                func_name = self._extract_cpp_function_name(node)
                if func_name:
                    if func_name.startswith('~'):
                        return ChunkType.DESTRUCTOR
                    elif self._is_cpp_constructor(node, func_name):
                        return ChunkType.CONSTRUCTOR
                    else:
                        return ChunkType.FUNCTION
                return ChunkType.FUNCTION
            elif node_type == 'declaration':
                # Distinguish between constants and variables
                if self._is_cpp_const_declaration(node):
                    return ChunkType.CONSTANT
                else:
                    return ChunkType.VARIABLE
        
        # Standard mapping lookup for other cases
        for chunk_type, node_types in node_mappings.items():
            if node_type in node_types:
                # Special handling for assignments to distinguish constants vs variables
                if language == 'python' and node_type == 'assignment':
                    # This is now handled above for better context filtering
                    continue
                elif language in ['javascript', 'typescript'] and node_type == 'lexical_declaration':
                    return self._classify_js_declaration(node)
                return chunk_type
        
        return None
    
    def _is_async_function(self, node: Node) -> bool:
        """Check if a function_definition node represents an async function."""
        for child in node.children:
            if child.type == 'async':
                return True
        return False
    
    def _is_java_constructor(self, node: Node) -> bool:
        """Check if a method_declaration node represents a Java constructor."""
        # Java constructors have:
        # 1. type_identifier (class name) followed by empty identifier
        # 2. Regular methods have type_identifier (return type) + non-empty identifier (method name)
        
        children_types = [child.type for child in node.children]
        
        # Must have both type_identifier and identifier
        if not ('type_identifier' in children_types and 'identifier' in children_types):
            return False
        
        # Check if the identifier (method name) is empty
        # In constructors, the identifier node exists but has empty text
        for child in node.children:
            if child.type == 'identifier':
                identifier_text = child.text.decode('utf-8').strip()
                # If identifier is empty or just whitespace, it's a constructor
                return len(identifier_text) == 0
        
        return False
    
    def _is_module_level_assignment(self, node: Node) -> bool:
        """
        Check if an assignment is at module level (not inside a function or class).
        
        This is a heuristic approach - we check the parent hierarchy to see
        if we're inside a function or class definition.
        """
        # For this implementation, we'll use a simple approach:
        # If the assignment is directly under a module or under a simple block,
        # consider it module-level. This may need refinement based on actual usage.
        
        parent = node.parent
        while parent:
            if parent.type in ['function_definition', 'class_definition']:
                return False
            elif parent.type == 'module':
                return True
            parent = parent.parent
        
        # If we can't determine, err on the side of inclusion
        return True
    
    def _is_module_level_js_declaration(self, node: Node) -> bool:
        """
        Check if a JavaScript declaration is at module level (not inside a function or class).
        
        Args:
            node: The lexical_declaration or variable_declaration node
            
        Returns:
            True if at module level, False if inside function/class/method
        """
        parent = node.parent
        while parent:
            if parent.type in ['function_declaration', 'method_definition', 'arrow_function', 'class_declaration']:
                return False
            elif parent.type == 'program':  # JavaScript uses 'program' instead of 'module'
                return True
            # Also check for statement blocks that might be inside functions
            elif parent.type == 'statement_block':
                # Look at the grandparent to see what contains this statement block
                grandparent = parent.parent
                if grandparent and grandparent.type in ['function_declaration', 'method_definition', 'arrow_function']:
                    return False
            parent = parent.parent
        
        # If we can't determine, err on the side of inclusion
        return True
    
    def _classify_python_assignment(self, node: Node) -> ChunkType:
        """
        Classify Python assignment as constant or variable based on naming convention.
        
        Args:
            node: Assignment AST node
            
        Returns:
            ChunkType.CONSTANT for UPPERCASE names, ChunkType.VARIABLE otherwise
        """
        # Extract the variable name
        for child in node.children:
            if child.type == 'identifier':
                name = child.text.decode('utf-8')
                # Python convention: UPPERCASE names are constants
                if name.isupper() and len(name) > 1:
                    return ChunkType.CONSTANT
                break
        
        return ChunkType.VARIABLE
    
    def _classify_js_declaration(self, node: Node) -> ChunkType:
        """
        Classify JavaScript/TypeScript lexical declaration as constant, variable, or function.
        
        Args:
            node: Lexical declaration AST node
            
        Returns:
            ChunkType.FUNCTION/ASYNC_FUNCTION for arrow functions, 
            ChunkType.CONSTANT for const declarations, 
            ChunkType.VARIABLE for let declarations
        """
        # First check if this declaration contains an arrow function
        arrow_function_node = None
        for child in node.children:
            if child.type == 'variable_declarator':
                for declarator_child in child.children:
                    if declarator_child.type == 'arrow_function':
                        arrow_function_node = declarator_child
                        break
        
        # If it contains an arrow function, classify based on async status
        if arrow_function_node:
            if self._is_async_function(arrow_function_node):
                return ChunkType.ASYNC_FUNCTION
            else:
                return ChunkType.FUNCTION
        
        # Otherwise, classify based on declaration type (const/let/var)
        for child in node.children:
            if child.type == 'const' or child.text.decode('utf-8') == 'const':
                return ChunkType.CONSTANT
            elif child.type == 'let' or child.text.decode('utf-8') == 'let':
                return ChunkType.VARIABLE
        
        # Default to variable if we can't determine
        return ChunkType.VARIABLE
    
    def _create_chunk_from_node(self, node: Node, chunk_type: ChunkType, file_path: str,
                              content: str, content_lines: List[str], language: str) -> Optional[CodeChunk]:
        """
        Create a CodeChunk from an AST node.
        
        Args:
            node: AST node to convert
            chunk_type: Type of chunk to create
            file_path: Path to the source file
            content: File content as string
            content_lines: File content split by lines
            language: Programming language
            
        Returns:
            CodeChunk object or None if creation failed
        """
        try:
            # Extract basic position information
            start_line = node.start_point[0] + 1  # Tree-sitter uses 0-based line numbers
            end_line = node.end_point[0] + 1
            start_byte = node.start_byte
            end_byte = node.end_byte
            
            # Extract content
            chunk_content = content[start_byte:end_byte]
            
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_type)
            
            # Extract semantic information
            name = self._extract_name(node, language)
            signature = self._extract_signature(node, language)
            docstring = self._extract_docstring(node, content_lines, language)
            
            # Create breadcrumb path
            breadcrumb = self._create_breadcrumb(file_path, name)
            
            # Extract context (surrounding lines)
            context_before = self._extract_context_before(content_lines, start_line - 1, 5)
            context_after = self._extract_context_after(content_lines, end_line - 1, 5)
            
            # Generate content hash
            content_hash = hashlib.sha256(chunk_content.encode('utf-8')).hexdigest()
            
            # Create embedding text (optimized for search)
            embedding_text = self._create_embedding_text(chunk_content, name, signature, docstring)
            
            return CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=chunk_content,
                chunk_type=chunk_type,
                language=language,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                name=name,
                signature=signature,
                docstring=docstring,
                breadcrumb=breadcrumb,
                context_before=context_before,
                context_after=context_after,
                content_hash=content_hash,
                embedding_text=embedding_text,
                indexed_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create chunk from node: {e}")
            return None
    
    def _extract_name(self, node: Node, language: str) -> Optional[str]:
        """Extract the name of a code construct (function, class, etc.)."""
        if language == 'python':
            return self._extract_python_name(node)
        elif language in ['javascript', 'typescript']:
            return self._extract_js_name(node)
        elif language == 'go':
            return self._extract_go_name(node)
        elif language == 'rust':
            return self._extract_rust_name(node)
        elif language == 'java':
            return self._extract_java_name(node)
        elif language == 'cpp':
            return self._extract_cpp_name(node)
        
        return None
    
    def _extract_signature(self, node: Node, language: str) -> Optional[str]:
        """Extract the signature of a code construct."""
        if language == 'python':
            return self._extract_python_signature(node)
        elif language in ['javascript', 'typescript']:
            return self._extract_js_signature(node)
        elif language == 'go':
            return self._extract_go_signature(node)
        elif language == 'rust':
            return self._extract_rust_signature(node)
        elif language == 'java':
            return self._extract_java_signature(node)
        elif language == 'cpp':
            return self._extract_cpp_signature(node)
        
        return None
    
    def _extract_docstring(self, node: Node, content_lines: List[str], language: str) -> Optional[str]:
        """Extract associated documentation string."""
        if language == 'python':
            return self._extract_python_docstring(node, content_lines)
        elif language in ['javascript', 'typescript']:
            return self._extract_js_docstring(node, content_lines)
        # Other languages don't have standard docstring conventions yet
        return None
    
    def _create_breadcrumb(self, file_path: str, name: Optional[str]) -> str:
        """Create a breadcrumb path for the code chunk."""
        path = Path(file_path)
        if name:
            return f"{path.stem}.{name}"
        return path.stem
    
    def _extract_context_before(self, lines: List[str], start_idx: int, num_lines: int) -> Optional[str]:
        """Extract context lines before the chunk."""
        if start_idx <= 0:
            return None
        
        context_start = max(0, start_idx - num_lines)
        context_lines = lines[context_start:start_idx]
        return '\n'.join(context_lines) if context_lines else None
    
    def _extract_context_after(self, lines: List[str], end_idx: int, num_lines: int) -> Optional[str]:
        """Extract context lines after the chunk."""
        if end_idx >= len(lines) - 1:
            return None
        
        context_end = min(len(lines), end_idx + 1 + num_lines)
        context_lines = lines[end_idx + 1:context_end]
        return '\n'.join(context_lines) if context_lines else None
    
    def _create_embedding_text(self, content: str, name: Optional[str], 
                             signature: Optional[str], docstring: Optional[str]) -> str:
        """Create optimized text for embedding generation."""
        parts = []
        
        # Add name and signature for better search
        if name:
            parts.append(f"Name: {name}")
        if signature:
            parts.append(f"Signature: {signature}")
        if docstring:
            parts.append(f"Description: {docstring}")
        
        # Add the actual content
        parts.append(content)
        
        return '\n'.join(parts)
    
    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int, chunk_type: ChunkType) -> str:
        """Generate a unique identifier for a code chunk."""
        path = Path(file_path)
        base = f"{path.stem}_{chunk_type.value}_{start_line}_{end_line}"
        return hashlib.md5(base.encode('utf-8')).hexdigest()[:12]
    
    def _count_errors(self, node: Node) -> int:
        """Count the number of ERROR nodes in the AST."""
        error_count = 0
        
        if node.type == 'ERROR':
            error_count += 1
        
        for child in node.children:
            error_count += self._count_errors(child)
        
        return error_count
    
    def _handle_syntax_errors(self, node: Node, file_path: str, content: str, 
                            content_lines: List[str], language: str) -> List[CodeChunk]:
        """
        Enhanced intelligent syntax error handling with improved recovery strategies.
        
        This method uses multiple recovery techniques:
        1. Valid section extraction between errors
        2. Partial AST traversal for recoverable nodes  
        3. Heuristic-based code block detection
        4. Language-specific error recovery patterns
        
        Args:
            node: Root AST node that may contain errors
            file_path: Path to the source file
            content: File content as string
            content_lines: File content split by lines
            language: Programming language
            
        Returns:
            List of chunks extracted from valid code sections
        """
        chunks = []
        error_locations = []
        
        # First pass: collect all error locations with severity analysis
        self._collect_error_locations_enhanced(node, error_locations, content_lines)
        
        if not error_locations:
            # No errors found, proceed with normal parsing
            return self._extract_chunks_normal(node, file_path, content, content_lines, language)
        
        # Classify errors by severity and impact
        critical_errors, recoverable_errors = self._classify_errors_by_severity(error_locations, content_lines, language)
        
        self.logger.warning(f"Found {len(error_locations)} syntax errors in {file_path} ({len(critical_errors)} critical, {len(recoverable_errors)} recoverable)")
        
        # Strategy 1: Try partial AST traversal for recoverable sections
        partial_chunks = self._extract_from_partial_ast(node, file_path, content, content_lines, language, error_locations)
        chunks.extend(partial_chunks)
        
        # Strategy 2: Extract valid sections between errors using enhanced detection
        valid_sections = self._identify_valid_sections_enhanced(error_locations, content_lines, language)
        
        for section_start, section_end in valid_sections:
            # Skip sections that were already processed by partial AST
            if self._section_already_processed(section_start, section_end, chunks):
                continue
                
            section_content = '\n'.join(content_lines[section_start:section_end + 1])
            
            # Enhanced validation for section content
            if self._is_valid_code_section(section_content, language):
                chunk = self._create_valid_section_chunk(
                    section_content, file_path, language, 
                    section_start + 1, section_end + 1
                )
                if chunk:
                    chunks.append(chunk)
        
        # Strategy 3: Language-specific heuristic recovery
        heuristic_chunks = self._recover_using_heuristics(content_lines, file_path, language, error_locations)
        chunks.extend(heuristic_chunks)
        
        # Enhanced fallback for error recovery - avoid whole-file when possible
        if not chunks:
            if self._should_use_whole_file_fallback(file_path, content, language, len(error_locations)):
                chunks.append(self._create_error_annotated_chunk(file_path, content, language, error_locations))
            else:
                # Try alternative chunking even with errors
                alt_chunks = self._attempt_alternative_chunking(file_path, content, language, content_lines)
                if alt_chunks:
                    chunks = alt_chunks
                else:
                    # Last resort - create error-annotated whole-file chunk
                    chunks.append(self._create_error_annotated_chunk(file_path, content, language, error_locations))
        else:
            # Sort chunks by line number for consistency
            chunks.sort(key=lambda c: c.start_line)
        
        return chunks
    
    def _collect_error_locations(self, node: Node, error_locations: List[Tuple[int, int]]) -> None:
        """Recursively collect all ERROR node locations."""
        if node.type == 'ERROR':
            error_locations.append((node.start_point[0], node.end_point[0]))
        
        for child in node.children:
            self._collect_error_locations(child, error_locations)
    
    def _collect_detailed_errors(self, node: Node, content_lines: List[str], 
                               language: str) -> List[CodeSyntaxError]:
        """Collect detailed syntax error information."""
        errors = []
        self._traverse_for_errors(node, content_lines, language, errors)
        return errors
    
    def _traverse_for_errors(self, node: Node, content_lines: List[str], 
                           language: str, errors: List[CodeSyntaxError]) -> None:
        """Recursively traverse AST to find and classify errors."""
        if node.type == 'ERROR':
            error = self._classify_syntax_error(node, content_lines, language)
            if error:
                errors.append(error)
        
        for child in node.children:
            self._traverse_for_errors(child, content_lines, language, errors)
    
    def _classify_syntax_error(self, error_node: Node, content_lines: List[str], 
                             language: str) -> Optional[CodeSyntaxError]:
        """Classify and create detailed information about a syntax error."""
        start_line = error_node.start_point[0] + 1  # Convert to 1-based
        end_line = error_node.end_point[0] + 1
        start_column = error_node.start_point[1]
        end_column = error_node.end_point[1]
        
        # Extract context around the error
        context_lines = []
        context_start = max(0, start_line - 3)  # 2 lines before
        context_end = min(len(content_lines), end_line + 2)  # 2 lines after
        
        for i in range(context_start, context_end):
            if i < len(content_lines):
                line_marker = ">>> " if i + 1 == start_line else "    "
                context_lines.append(f"{line_marker}{i + 1}: {content_lines[i]}")
        
        context = '\n'.join(context_lines)
        
        # Classify error type based on content and language
        error_type = self._determine_error_type(error_node, content_lines, language, start_line - 1)
        
        return CodeSyntaxError(
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            error_type=error_type,
            context=context,
            severity="error"
        )
    
    def _determine_error_type(self, error_node: Node, content_lines: List[str], 
                            language: str, line_index: int) -> str:
        """Determine the specific type of syntax error."""
        if line_index >= len(content_lines):
            return "unknown_error"
        
        error_line = content_lines[line_index].strip()
        error_text = error_node.text.decode('utf-8') if error_node.text else ""
        
        # Language-specific error classification
        if language == 'python':
            return self._classify_python_error(error_line, error_text)
        elif language in ['javascript', 'typescript']:
            return self._classify_js_error(error_line, error_text)
        else:
            return self._classify_generic_error(error_line, error_text)
    
    def _classify_python_error(self, error_line: str, error_text: str) -> str:
        """Classify Python-specific syntax errors."""
        if ':' not in error_line and ('def ' in error_line or 'class ' in error_line or 'if ' in error_line):
            return "missing_colon"
        elif error_line.count('(') != error_line.count(')'):
            return "unmatched_parentheses"
        elif error_line.count('[') != error_line.count(']'):
            return "unmatched_brackets"
        elif error_line.count('{') != error_line.count('}'):
            return "unmatched_braces"
        elif 'IndentationError' in error_text:
            return "indentation_error"
        elif error_line.startswith('import ') or error_line.startswith('from '):
            return "import_error"
        else:
            return "python_syntax_error"
    
    def _classify_js_error(self, error_line: str, error_text: str) -> str:
        """Classify JavaScript/TypeScript-specific syntax errors."""
        if not error_line.endswith(';') and any(keyword in error_line for keyword in ['var ', 'let ', 'const ', 'return']):
            return "missing_semicolon"
        elif error_line.count('(') != error_line.count(')'):
            return "unmatched_parentheses"
        elif error_line.count('{') != error_line.count('}'):
            return "unmatched_braces"
        elif error_line.count('[') != error_line.count(']'):
            return "unmatched_brackets"
        elif 'function' in error_line:
            return "function_declaration_error"
        elif 'import' in error_line or 'export' in error_line:
            return "module_error"
        else:
            return "javascript_syntax_error"
    
    def _classify_generic_error(self, error_line: str, error_text: str) -> str:
        """Classify generic syntax errors for other languages."""
        if error_line.count('(') != error_line.count(')'):
            return "unmatched_parentheses"
        elif error_line.count('{') != error_line.count('}'):
            return "unmatched_braces"
        elif error_line.count('[') != error_line.count(']'):
            return "unmatched_brackets"
        else:
            return "syntax_error"
    
    def _identify_valid_sections(self, error_locations: List[Tuple[int, int]], 
                               content_lines: List[str]) -> List[Tuple[int, int]]:
        """
        Identify continuous valid code sections between syntax errors.
        
        Args:
            error_locations: List of (start_line, end_line) tuples for errors
            content_lines: File content split by lines
            
        Returns:
            List of (start_line, end_line) tuples for valid sections
        """
        if not error_locations:
            return [(0, len(content_lines) - 1)]
        
        # Sort error locations by start line
        sorted_errors = sorted(error_locations, key=lambda x: x[0])
        valid_sections = []
        
        # Check for valid section before first error
        first_error_start = sorted_errors[0][0]
        if first_error_start > 0:
            # Look for meaningful content before the first error
            section_end = first_error_start - 1
            section_start = self._find_section_start(content_lines, 0, section_end)
            if section_start <= section_end:
                valid_sections.append((section_start, section_end))
        
        # Check for valid sections between errors
        for i in range(len(sorted_errors) - 1):
            current_error_end = sorted_errors[i][1]
            next_error_start = sorted_errors[i + 1][0]
            
            if current_error_end < next_error_start - 1:
                section_start = self._find_section_start(content_lines, current_error_end + 1, next_error_start - 1)
                section_end = next_error_start - 1
                if section_start <= section_end:
                    valid_sections.append((section_start, section_end))
        
        # Check for valid section after last error
        last_error_end = sorted_errors[-1][1]
        if last_error_end < len(content_lines) - 1:
            section_start = self._find_section_start(content_lines, last_error_end + 1, len(content_lines) - 1)
            if section_start < len(content_lines):
                valid_sections.append((section_start, len(content_lines) - 1))
        
        return valid_sections
    
    def _find_section_start(self, content_lines: List[str], min_line: int, max_line: int) -> int:
        """Find the first meaningful line of code in a range."""
        for i in range(min_line, min(max_line + 1, len(content_lines))):
            line = content_lines[i].strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                return i
        return min_line
    
    def _extract_chunks_normal(self, node: Node, file_path: str, content: str, 
                             content_lines: List[str], language: str) -> List[CodeChunk]:
        """Extract chunks using normal parsing when no errors are present."""
        chunks = []
        node_mappings = self._node_mappings.get(language, {})
        self._traverse_ast(node, chunks, file_path, content, content_lines, language, node_mappings)
        return chunks
    
    def _create_valid_section_chunk(self, content: str, file_path: str, language: str,
                                  start_line: int, end_line: int) -> CodeChunk:
        """Create a chunk for a valid code section around syntax errors."""
        chunk_id = self._generate_chunk_id(file_path, start_line, end_line, ChunkType.RAW_CODE)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.RAW_CODE,
            language=language,
            start_line=start_line,
            end_line=end_line,
            start_byte=0,  # Approximate for section chunks
            end_byte=len(content),
            name=f"valid_section_{start_line}_{end_line}",
            breadcrumb=f"{Path(file_path).stem}.valid_section_{start_line}_{end_line}",
            content_hash=content_hash,
            embedding_text=content,
            indexed_at=datetime.now(),
            tags=["syntax_error_recovery", "valid_section"]
        )
    
    def _create_error_annotated_chunk(self, file_path: str, content: str, language: str,
                                    error_locations: List[Tuple[int, int]]) -> CodeChunk:
        """Create a whole-file chunk with error annotations when no valid sections found."""
        chunk_id = self._generate_chunk_id(file_path, 1, len(content.split('\n')), ChunkType.WHOLE_FILE)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        error_summary = f"File contains {len(error_locations)} syntax errors"
        embedding_text = f"{error_summary}\n\n{content}"
        
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=language,
            start_line=1,
            end_line=len(content.split('\n')),
            start_byte=0,
            end_byte=len(content),
            name="error_file",
            breadcrumb=f"{Path(file_path).stem}.error_file",
            content_hash=content_hash,
            embedding_text=embedding_text,
            indexed_at=datetime.now(),
            tags=["syntax_errors", "whole_file", "parse_failed"],
            complexity_score=1.0  # Mark as high complexity due to errors
        )
    
    def _create_fallback_result(self, file_path: str, content: Optional[str], 
                              start_time: float, error: bool = False, 
                              exception_context: Optional[str] = None) -> ParseResult:
        """Create a fallback ParseResult with whole-file chunk."""
        if content is None:
            content = ""
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create a single whole-file chunk with enhanced error context
        chunks = []
        if content:
            language = self.detect_language(file_path) or "unknown"
            chunks.append(self._create_whole_file_chunk(file_path, content, language, exception_context))
        
        # Create syntax error details if we have exception context
        syntax_errors = []
        if exception_context:
            syntax_errors.append(CodeSyntaxError(
                start_line=1,
                end_line=len(content.split('\n')) if content else 1,
                start_column=0,
                end_column=0,
                error_type="parsing_exception",
                context=f"Exception during parsing: {exception_context}",
                severity="critical"
            ))
        
        return ParseResult(
            chunks=chunks,
            file_path=file_path,
            language=language if content else "unknown",
            parse_success=not error,
            error_count=1 if error else 0,
            fallback_used=True,
            processing_time_ms=processing_time,
            syntax_errors=syntax_errors
        )
    
    def _create_whole_file_chunk(self, file_path: str, content: str, language: str, 
                               exception_context: Optional[str] = None) -> CodeChunk:
        """Create a whole-file chunk as fallback."""
        lines = content.split('\n')
        chunk_id = self._generate_chunk_id(file_path, 1, len(lines), ChunkType.WHOLE_FILE)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Create enhanced tags with exception context
        tags = ["whole_file", "fallback"]
        if exception_context:
            tags.extend(["parsing_exception", "critical_error"])
        
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=language,
            start_line=1,
            end_line=len(lines),
            start_byte=0,
            end_byte=len(content),
            breadcrumb=Path(file_path).stem,
            content_hash=content_hash,
            embedding_text=content,
            indexed_at=datetime.now(),
            tags=tags
        )
    
    # Python-specific parsing methods
    def _extract_python_name(self, node: Node) -> Optional[str]:
        """Extract name from Python AST node."""
        node_type = node.type
        
        if node_type in ['function_definition', 'async_function_definition']:
            # Look for identifier child node
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'class_definition':
            # Look for identifier child node after 'class' keyword
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'assignment':
            # Look for the assigned variable name
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type in ['import_statement', 'import_from_statement']:
            # Extract import module names
            names = []
            for child in node.children:
                if child.type in ['dotted_name', 'identifier']:
                    names.append(child.text.decode('utf-8'))
            return ', '.join(names) if names else None
        
        return None
    
    def _extract_python_signature(self, node: Node) -> Optional[str]:
        """Extract function/class signature from Python AST node."""
        node_type = node.type
        
        if node_type == 'function_definition':
            # Get the function signature (name + parameters)
            signature_parts = []
            
            # Check if this is an async function by looking for async child
            is_async = self._is_async_function(node)
            if is_async:
                signature_parts.append('async')
            
            signature_parts.append('def')
            
            # Extract function name and parameters
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'parameters':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'type' and child.text.decode('utf-8').startswith('->'):
                    # Return type annotation
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'class_definition':
            # Get class signature with inheritance
            signature_parts = ['class']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'argument_list':
                    # Base classes
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        return None
    
    def _extract_python_docstring(self, node: Node, content_lines: List[str]) -> Optional[str]:
        """Extract docstring from Python function or class."""
        # Look for the first string literal in the body
        for child in node.children:
            if child.type == 'block':
                # Look for expression_statement containing a string
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'string':
                                # Extract and clean the docstring
                                docstring = expr.text.decode('utf-8')
                                # Remove quotes and clean up formatting
                                if docstring.startswith('"""') or docstring.startswith("'''"):
                                    docstring = docstring[3:-3]
                                elif docstring.startswith('"') or docstring.startswith("'"):
                                    docstring = docstring[1:-1]
                                return docstring.strip()
        return None
    
    # Placeholder methods for other languages (to be implemented)
    def _extract_js_name(self, node: Node) -> Optional[str]:
        """Extract name from JavaScript/TypeScript AST node."""
        node_type = node.type
        
        if node_type in ['function_declaration', 'async_function_declaration']:
            # Look for identifier child node
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'method_definition':
            # Look for property_identifier for method name
            for child in node.children:
                if child.type == 'property_identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'arrow_function':
            # For arrow functions, we need to look at the parent context
            # to find the variable name it's assigned to
            parent = node.parent
            if parent and parent.type == 'variable_declarator':
                for child in parent.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf-8')
            return 'arrow_function'  # Fallback if no context found
        
        elif node_type == 'class_declaration':
            # Look for identifier child node after 'class' keyword
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type in ['lexical_declaration', 'variable_declaration']:
            # Look for variable declarator
            for child in node.children:
                if child.type == 'variable_declarator':
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            return declarator_child.text.decode('utf-8')
        
        elif node_type in ['import_statement']:
            # Extract import specifiers
            names = []
            for child in node.children:
                if child.type == 'import_clause':
                    for clause_child in child.children:
                        if clause_child.type == 'identifier':
                            names.append(clause_child.text.decode('utf-8'))
                        elif clause_child.type == 'named_imports':
                            # Extract named imports
                            for import_child in clause_child.children:
                                if import_child.type == 'import_specifier':
                                    for spec_child in import_child.children:
                                        if spec_child.type == 'identifier':
                                            names.append(spec_child.text.decode('utf-8'))
            return ', '.join(names) if names else None
        
        elif node_type == 'export_statement':
            # Handle export statements
            for child in node.children:
                if child.type in ['function_declaration', 'class_declaration']:
                    return self._extract_js_name(child)
                elif child.type == 'lexical_declaration':
                    return self._extract_js_name(child)
        
        return None
    
    def _extract_js_signature(self, node: Node) -> Optional[str]:
        """Extract signature from JavaScript/TypeScript AST node."""
        node_type = node.type
        
        if node_type == 'function_declaration':
            # Get the function signature
            signature_parts = []
            
            # Check for async keyword
            is_async = self._is_async_function(node)
            if is_async:
                signature_parts.append('async')
            
            signature_parts.append('function')
            
            # Extract function name and parameters
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'formal_parameters':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'type_annotation':
                    # TypeScript return type annotation
                    signature_parts.append(':')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'method_definition':
            # Method definition signature
            signature_parts = []
            
            # Check for async keyword
            is_async = self._is_async_function(node)
            if is_async:
                signature_parts.append('async')
            
            # Extract method name and parameters
            for child in node.children:
                if child.type == 'property_identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'formal_parameters':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'type_annotation':
                    # TypeScript return type annotation
                    signature_parts.append(':')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'arrow_function':
            # Arrow function signature
            signature_parts = []
            
            # Check for async keyword
            is_async = self._is_async_function(node)
            if is_async:
                signature_parts.append('async')
            
            for child in node.children:
                if child.type == 'formal_parameters':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'type_annotation':
                    # TypeScript return type
                    signature_parts.append(':')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            signature_parts.append('=>')
            return ' '.join(signature_parts)
        
        elif node_type == 'class_declaration':
            # Get class signature with extends clause
            signature_parts = ['class']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'class_heritage':
                    # extends/implements clauses
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type in ['lexical_declaration', 'variable_declaration']:
            # Variable/const declaration signature
            signature_parts = []
            
            # Check if this contains an arrow function
            arrow_function_node = None
            variable_name = None
            
            for child in node.children:
                if child.type in ['const', 'let', 'var']:
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'variable_declarator':
                    # Get variable name and check for arrow function
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            variable_name = declarator_child.text.decode('utf-8')
                            signature_parts.append(variable_name)
                        elif declarator_child.type == 'arrow_function':
                            arrow_function_node = declarator_child
                        elif declarator_child.type == 'type_annotation':
                            signature_parts.append(':')
                            signature_parts.append(declarator_child.text.decode('utf-8'))
                    break
            
            # If this contains an arrow function, create a function-style signature
            if arrow_function_node:
                # Replace the const/let with async if needed
                if self._is_async_function(arrow_function_node):
                    signature_parts[0] = 'async'
                else:
                    # Remove the const/let for function-style signature
                    signature_parts = signature_parts[1:]
                
                # Add arrow function parameters
                for child in arrow_function_node.children:
                    if child.type == 'formal_parameters':
                        signature_parts.append(child.text.decode('utf-8'))
                    elif child.type == 'type_annotation':
                        signature_parts.append(':')
                        signature_parts.append(child.text.decode('utf-8'))
                        break
                
                signature_parts.append('=>')
            
            return ' '.join(signature_parts)
        
        return None
    
    def _extract_js_docstring(self, node: Node, content_lines: List[str]) -> Optional[str]:
        """Extract JSDoc comment from JavaScript/TypeScript function or class."""
        # JSDoc comments appear before the function/class declaration
        start_line = node.start_point[0]  # 0-based line number
        
        # Look for JSDoc comment in the lines before the node
        for i in range(start_line - 1, max(0, start_line - 10), -1):  # Check up to 10 lines before
            line = content_lines[i].strip()
            
            # Check if this line ends a JSDoc comment
            if line.endswith('*/'):
                # Found end of JSDoc, collect the whole comment
                jsdoc_lines = []
                
                # Collect lines backwards until we find the start
                for j in range(i, max(0, i - 20), -1):  # Check up to 20 lines for the start
                    comment_line = content_lines[j].strip()
                    jsdoc_lines.insert(0, comment_line)
                    
                    if comment_line.startswith('/**'):
                        # Found the start of JSDoc comment
                        # Clean up the JSDoc content
                        cleaned_lines = []
                        for doc_line in jsdoc_lines:
                            # Remove /** and */ and leading * characters
                            cleaned = doc_line.replace('/**', '').replace('*/', '').strip()
                            if cleaned.startswith('*'):
                                cleaned = cleaned[1:].strip()
                            if cleaned:  # Only add non-empty lines
                                cleaned_lines.append(cleaned)
                        
                        return '\n'.join(cleaned_lines) if cleaned_lines else None
                
                break
        
        return None
    
    def _extract_go_name(self, node: Node) -> Optional[str]:
        """Extract name from Go AST node."""
        node_type = node.type
        
        if node_type == 'function_declaration':
            # Look for identifier child node
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'method_declaration':
            # Look for field_identifier (method name) child node
            for child in node.children:
                if child.type == 'field_identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'type_declaration':
            # Go struct/interface/type declarations
            for child in node.children:
                if child.type == 'type_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'type_identifier':
                            return spec_child.text.decode('utf-8')
        
        elif node_type == 'interface_type':
            # Interface definitions
            # Interfaces are typically nested in type_declaration
            return "interface"  # Will be refined by parent context
        
        elif node_type == 'const_declaration':
            # Look for constant identifier
            for child in node.children:
                if child.type == 'const_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'identifier':
                            return spec_child.text.decode('utf-8')
        
        elif node_type == 'var_declaration':
            # Look for variable identifier
            for child in node.children:
                if child.type == 'var_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'identifier':
                            return spec_child.text.decode('utf-8')
        
        elif node_type == 'import_declaration':
            # Extract imported package names
            names = []
            
            def extract_import_from_spec(spec_node):
                for spec_child in spec_node.children:
                    if spec_child.type in ['interpreted_string_literal', 'raw_string_literal']:
                        # Extract package path from quotes
                        package_path = spec_child.text.decode('utf-8').strip('"\'`')
                        package_name = package_path.split('/')[-1]  # Get last part
                        return package_name
                return None
            
            for child in node.children:
                if child.type == 'import_spec':
                    # Single import
                    name = extract_import_from_spec(child)
                    if name:
                        names.append(name)
                elif child.type == 'import_spec_list':
                    # Multiple imports in parentheses
                    for spec_child in child.children:
                        if spec_child.type == 'import_spec':
                            name = extract_import_from_spec(spec_child)
                            if name:
                                names.append(name)
            
            return ', '.join(names) if names else None
        
        return None
    
    def _extract_go_signature(self, node: Node) -> Optional[str]:
        """Extract signature from Go AST node."""
        node_type = node.type
        
        if node_type == 'function_declaration':
            # func name(params) return_type
            signature_parts = ['func']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'parameter_list':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'parenthesized_type', 'pointer_type']:
                    # Return type
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'method_declaration':
            # func (receiver) name(params) return_type
            signature_parts = ['func']
            
            for child in node.children:
                if child.type == 'parameter_list':
                    # This could be receiver or parameters
                    param_text = child.text.decode('utf-8')
                    if len(signature_parts) == 1:  # First parameter list is receiver
                        signature_parts.append(param_text)
                    else:  # Second parameter list is parameters
                        signature_parts.append(param_text)
                elif child.type == 'field_identifier':
                    # Method name
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'parenthesized_type', 'pointer_type']:
                    # Return type
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'type_declaration':
            # type Name struct/interface/underlying_type
            signature_parts = ['type']
            
            for child in node.children:
                if child.type == 'type_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'type_identifier':
                            signature_parts.append(spec_child.text.decode('utf-8'))
                        elif spec_child.type in ['struct_type', 'interface_type', 'type_identifier']:
                            # The actual type definition
                            if spec_child.type == 'struct_type':
                                signature_parts.append('struct')
                            elif spec_child.type == 'interface_type':
                                signature_parts.append('interface')
                            else:
                                signature_parts.append(spec_child.text.decode('utf-8'))
                            break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'const_declaration':
            # const Name = value or const Name Type = value
            signature_parts = ['const']
            
            for child in node.children:
                if child.type == 'const_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'identifier':
                            signature_parts.append(spec_child.text.decode('utf-8'))
                        elif spec_child.type in ['type_identifier', 'pointer_type']:
                            signature_parts.append(spec_child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'var_declaration':
            # var Name Type or var Name = value
            signature_parts = ['var']
            
            for child in node.children:
                if child.type == 'var_spec':
                    for spec_child in child.children:
                        if spec_child.type == 'identifier':
                            signature_parts.append(spec_child.text.decode('utf-8'))
                        elif spec_child.type in ['type_identifier', 'pointer_type', 'array_type', 'slice_type']:
                            signature_parts.append(spec_child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'import_declaration':
            # import "package" or import ( ... )
            signature_parts = ['import']
            
            # Get all the imported packages for the signature
            names = []
            
            def extract_import_from_spec(spec_node):
                for spec_child in spec_node.children:
                    if spec_child.type in ['interpreted_string_literal', 'raw_string_literal']:
                        return spec_child.text.decode('utf-8')
                return None
            
            for child in node.children:
                if child.type == 'import_spec':
                    # Single import
                    name = extract_import_from_spec(child)
                    if name:
                        names.append(name)
                elif child.type == 'import_spec_list':
                    # Multiple imports in parentheses
                    for spec_child in child.children:
                        if spec_child.type == 'import_spec':
                            name = extract_import_from_spec(spec_child)
                            if name:
                                names.append(name)
            
            if names:
                if len(names) == 1:
                    signature_parts.append(names[0])
                else:
                    signature_parts.append(f"({', '.join(names)})")
            
            return ' '.join(signature_parts)
        
        return None
    
    def _extract_rust_name(self, node: Node) -> Optional[str]:
        """Extract name from Rust AST node."""
        node_type = node.type
        
        if node_type == 'function_item':
            # fn name(params) -> return_type
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'struct_item':
            # struct Name { ... } or struct Name(...)
            for child in node.children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'enum_item':
            # enum Name { ... }
            for child in node.children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'const_item':
            # const NAME: Type = value
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'let_declaration':
            # let name: Type = value
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'use_declaration':
            # use path::to::module
            names = []
            self._extract_rust_use_paths(node, names)
            return ', '.join(names) if names else None
        
        elif node_type == 'impl_item':
            # impl Type or impl Trait for Type
            # Return the last type (the one being implemented for)
            last_type = None
            for child in node.children:
                if child.type == 'type_identifier':
                    last_type = child.text.decode('utf-8')
                elif child.type == 'scoped_type_identifier':
                    # For qualified types, extract just the last part
                    scoped_text = child.text.decode('utf-8')
                    if '::' in scoped_text:
                        last_type = scoped_text.split('::')[-1]
                    else:
                        last_type = scoped_text
            return last_type
        
        return None
    
    def _extract_rust_use_paths(self, node: Node, names: List[str]) -> None:
        """Recursively extract use paths from Rust use declarations."""
        if node.type == 'identifier':
            names.append(node.text.decode('utf-8'))
        elif node.type == 'scoped_identifier':
            # Extract the last part of the scoped identifier
            for child in reversed(node.children):
                if child.type == 'identifier':
                    names.append(child.text.decode('utf-8'))
                    break
        
        for child in node.children:
            self._extract_rust_use_paths(child, names)
    
    def _extract_rust_signature(self, node: Node) -> Optional[str]:
        """Extract signature from Rust AST node."""
        node_type = node.type
        
        if node_type == 'function_item':
            # fn name(params) -> return_type
            signature_parts = ['fn']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'parameters':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'primitive_type', 'reference_type', 'generic_type']:
                    # Return type (after ->)
                    signature_parts.append('->')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'struct_item':
            # struct Name { ... } or struct Name(fields)
            signature_parts = ['struct']
            
            for child in node.children:
                if child.type == 'type_identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'field_declaration_list':
                    # Struct with named fields
                    signature_parts.append('{...}')
                    break
                elif child.type == 'ordered_field_declaration_list':
                    # Tuple struct
                    signature_parts.append('(...)')
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'enum_item':
            # enum Name { variants }
            signature_parts = ['enum']
            
            for child in node.children:
                if child.type == 'type_identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'const_item':
            # const NAME: Type = value
            signature_parts = ['const']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'primitive_type', 'reference_type']:
                    signature_parts.append(':')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'let_declaration':
            # let name: Type = value
            signature_parts = ['let']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'primitive_type', 'reference_type']:
                    signature_parts.append(':')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'impl_item':
            # impl Type or impl Trait for Type
            signature_parts = ['impl']
            
            found_for = False
            
            for child in node.children:
                if child.type == 'type_identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'scoped_type_identifier':
                    # Handle qualified types like fmt::Display
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'for':
                    signature_parts.append('for')
                    found_for = True
            
            return ' '.join(signature_parts)
        
        elif node_type == 'use_declaration':
            # use path::to::module;
            signature_parts = ['use']
            
            # Extract the full path for the signature
            for child in node.children:
                if child.type in ['identifier', 'scoped_identifier', 'use_list']:
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        return None
    
    def _extract_java_name(self, node: Node) -> Optional[str]:
        """Extract name from Java AST node."""
        node_type = node.type
        
        if node_type == 'method_declaration':
            # Check if this is a constructor first
            if self._is_java_constructor(node):
                # For constructors, get the class name from type_identifier
                for child in node.children:
                    if child.type == 'type_identifier':
                        return child.text.decode('utf-8')
            else:
                # For regular methods, look for method name identifier
                for child in node.children:
                    if child.type == 'identifier':
                        return child.text.decode('utf-8')
        
        elif node_type == 'constructor_declaration':
            # For constructor_declaration nodes, get the class name from identifier
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type in ['class_declaration', 'interface_declaration', 'enum_declaration']:
            # Look for class/interface/enum name
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'field_declaration':
            # Look for field variable declarator
            for child in node.children:
                if child.type == 'variable_declarator':
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            return declarator_child.text.decode('utf-8')
        
        elif node_type == 'import_declaration':
            # Extract imported class/package names
            names = []
            for child in node.children:
                if child.type == 'scoped_identifier':
                    # Get the last part of the scoped identifier
                    identifier_parts = child.text.decode('utf-8').split('.')
                    names.append(identifier_parts[-1])
                elif child.type == 'identifier':
                    names.append(child.text.decode('utf-8'))
            return ', '.join(names) if names else None
        
        return None
    
    def _extract_java_signature(self, node: Node) -> Optional[str]:
        """Extract signature from Java AST node."""
        node_type = node.type
        
        if node_type == 'method_declaration':
            # Check if this is a constructor first
            if self._is_java_constructor(node):
                # For constructors: [modifiers] ClassName(parameters)
                signature_parts = []
                
                for child in node.children:
                    if child.type == 'modifiers':
                        # public, private, static, etc.
                        signature_parts.append(child.text.decode('utf-8'))
                    elif child.type == 'type_identifier':
                        # Constructor name (class name)
                        signature_parts.append(child.text.decode('utf-8'))
                    elif child.type == 'formal_parameters':
                        # Parameters
                        signature_parts.append(child.text.decode('utf-8'))
                        break
                
                return ' '.join(signature_parts)
            else:
                # For regular methods: [modifiers] return_type name(parameters)
                signature_parts = []
                
                for child in node.children:
                    if child.type == 'modifiers':
                        # public, private, static, etc.
                        signature_parts.append(child.text.decode('utf-8'))
                    elif child.type in ['type_identifier', 'primitive_type', 'void_type', 'generic_type', 'integral_type', 'floating_point_type', 'boolean_type']:
                        # Return type
                        signature_parts.append(child.text.decode('utf-8'))
                    elif child.type == 'identifier':
                        # Method name
                        signature_parts.append(child.text.decode('utf-8'))
                    elif child.type == 'formal_parameters':
                        # Parameters
                        signature_parts.append(child.text.decode('utf-8'))
                        break
                
                return ' '.join(signature_parts)
        
        elif node_type == 'constructor_declaration':
            # For constructor_declaration: [modifiers] ClassName(parameters)
            signature_parts = []
            
            for child in node.children:
                if child.type == 'modifiers':
                    # public, private, static, etc.
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'identifier':
                    # Constructor name (class name)
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'formal_parameters':
                    # Parameters
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type in ['class_declaration', 'interface_declaration', 'enum_declaration']:
            # [modifiers] class/interface/enum Name [extends/implements]
            signature_parts = []
            
            for child in node.children:
                if child.type == 'modifiers':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['class', 'interface', 'enum']:
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['superclass', 'super_interfaces']:
                    # extends/implements clauses
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'class_body':
                    # Stop at body
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'field_declaration':
            # [modifiers] type name [= value]
            signature_parts = []
            
            for child in node.children:
                if child.type == 'modifiers':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'primitive_type', 'generic_type', 'array_type']:
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'variable_declarator':
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            signature_parts.append(declarator_child.text.decode('utf-8'))
                            break
                    break
            
            return ' '.join(signature_parts)
        
        return None
    
    # C++ specific parsing methods
    def _extract_cpp_function_name(self, node: Node) -> Optional[str]:
        """Extract function name from C++ function_definition node."""
        # Look for function_declarator child which contains the function name
        for child in node.children:
            if child.type == 'function_declarator':
                for declarator_child in child.children:
                    if declarator_child.type == 'identifier':
                        return declarator_child.text.decode('utf-8')
                    elif declarator_child.type == 'destructor_name':
                        # Destructors have special naming
                        return declarator_child.text.decode('utf-8')
        return None
    
    def _is_cpp_constructor(self, node: Node, func_name: str) -> bool:
        """Check if a C++ function is a constructor."""
        # In C++, constructors have the same name as their class
        # We need to look at the context to determine the class name
        # For now, we'll use a heuristic: if the function has no return type, it's likely a constructor
        
        # Look for a return type - constructors don't have explicit return types
        for child in node.children:
            if child.type in ['primitive_type', 'type_identifier', 'qualified_identifier']:
                # Has return type, so not a constructor
                return False
        
        # Additional check: constructor names typically match class names (uppercase)
        return func_name and func_name[0].isupper()
    
    def _is_cpp_const_declaration(self, node: Node) -> bool:
        """Check if a C++ declaration is a const declaration."""
        # Look for 'const' keyword in the declaration
        for child in node.children:
            if child.type == 'storage_class_specifier':
                if child.text.decode('utf-8') == 'const':
                    return True
            elif child.type == 'type_qualifier':
                if child.text.decode('utf-8') == 'const':
                    return True
        return False
    
    def _extract_cpp_name(self, node: Node) -> Optional[str]:
        """Extract name from C++ AST node."""
        node_type = node.type
        
        if node_type == 'function_definition':
            return self._extract_cpp_function_name(node)
        
        elif node_type in ['class_specifier', 'struct_specifier']:
            # Look for type_identifier child
            for child in node.children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'namespace_definition':
            # Look for identifier child
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'template_declaration':
            # Template declarations wrap other declarations
            for child in node.children:
                if child.type in ['class_specifier', 'struct_specifier', 'function_definition']:
                    return self._extract_cpp_name(child)
        
        elif node_type == 'declaration':
            # Look for init_declarator and then identifier
            for child in node.children:
                if child.type == 'init_declarator':
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            return declarator_child.text.decode('utf-8')
        
        elif node_type == 'preproc_include':
            # Extract include file name
            for child in node.children:
                if child.type in ['string_literal', 'system_lib_string']:
                    include_path = child.text.decode('utf-8').strip('"<>')
                    # Return just the filename, not the full path
                    return include_path.split('/')[-1]
        
        return None
    
    def _extract_cpp_signature(self, node: Node) -> Optional[str]:
        """Extract signature from C++ AST node."""
        node_type = node.type
        
        if node_type == 'function_definition':
            # Extract full function signature including return type
            signature_parts = []
            
            # Look for return type, function name, and parameters
            for child in node.children:
                if child.type in ['primitive_type', 'type_identifier', 'qualified_identifier']:
                    # Return type
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'function_declarator':
                    # Function name and parameters
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type in ['class_specifier', 'struct_specifier']:
            # class/struct Name [: base_classes]
            signature_parts = [node_type.replace('_specifier', '')]
            
            for child in node.children:
                if child.type == 'type_identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'base_class_clause':
                    # Inheritance information
                    signature_parts.append(':')
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'namespace_definition':
            # namespace Name
            signature_parts = ['namespace']
            
            for child in node.children:
                if child.type == 'identifier':
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'template_declaration':
            # template<...> declaration
            signature_parts = []
            
            # Add template parameters
            for child in node.children:
                if child.type == 'template_parameter_list':
                    signature_parts.append('template')
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['class_specifier', 'struct_specifier', 'function_definition']:
                    # Add the templated declaration
                    inner_signature = self._extract_cpp_signature(child)
                    if inner_signature:
                        signature_parts.append(inner_signature)
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'declaration':
            # Variable/constant declaration
            signature_parts = []
            
            for child in node.children:
                if child.type in ['storage_class_specifier', 'type_qualifier']:
                    # const, static, etc.
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['primitive_type', 'type_identifier']:
                    # Type
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'init_declarator':
                    # Variable name
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            signature_parts.append(declarator_child.text.decode('utf-8'))
                            break
                    break
            
            return ' '.join(signature_parts)
        
        elif node_type == 'preproc_include':
            # #include <file> or #include "file"
            signature_parts = ['#include']
            
            for child in node.children:
                if child.type in ['string_literal', 'system_lib_string']:
                    signature_parts.append(child.text.decode('utf-8'))
                    break
            
            return ' '.join(signature_parts)
        
        return None
    
    # Enhanced error recovery methods
    def _collect_error_locations_enhanced(self, node: Node, error_locations: List[Tuple[int, int, str]], content_lines: List[str]) -> None:
        """Collect error locations with enhanced context information."""
        if node.type == 'ERROR':
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            # Get error context for better classification
            error_context = ""
            if start_line < len(content_lines):
                error_context = content_lines[start_line].strip()
            
            error_locations.append((start_line, end_line, error_context))
        
        for child in node.children:
            self._collect_error_locations_enhanced(child, error_locations, content_lines)
    
    def _classify_errors_by_severity(self, error_locations: List[Tuple[int, int, str]], 
                                   content_lines: List[str], language: str) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """Classify errors into critical and recoverable categories."""
        critical_errors = []
        recoverable_errors = []
        
        for start_line, end_line, context in error_locations:
            if self._is_critical_error(context, language):
                critical_errors.append((start_line, end_line, context))
            else:
                recoverable_errors.append((start_line, end_line, context))
        
        return critical_errors, recoverable_errors
    
    def _is_critical_error(self, error_context: str, language: str) -> bool:
        """Determine if an error is critical and likely to break parsing."""
        if not error_context:
            return True
        
        # Language-specific critical error patterns
        critical_patterns = {
            'python': ['class ', 'def ', 'import ', 'from '],
            'javascript': ['function ', 'class ', 'import ', 'export '],
            'typescript': ['function ', 'class ', 'interface ', 'import ', 'export '],
            'java': ['class ', 'public class ', 'interface ', 'import '],
            'cpp': ['class ', 'namespace ', '#include ', 'template'],
            'go': ['func ', 'type ', 'package ', 'import'],
            'rust': ['fn ', 'struct ', 'impl ', 'use ']
        }
        
        patterns = critical_patterns.get(language, [])
        return any(pattern in error_context.lower() for pattern in patterns)
    
    def _extract_from_partial_ast(self, node: Node, file_path: str, content: str, 
                                content_lines: List[str], language: str, 
                                error_locations: List[Tuple[int, int, str]]) -> List[CodeChunk]:
        """Extract chunks from non-error parts of the AST."""
        chunks = []
        
        # Get error line numbers for filtering
        error_lines = set()
        for start_line, end_line, _ in error_locations:
            error_lines.update(range(start_line, end_line + 1))
        
        # Traverse AST and extract valid nodes that don't overlap with errors
        node_mappings = self._node_mappings.get(language, {})
        self._traverse_ast_with_error_filtering(node, chunks, file_path, content, content_lines, 
                                              language, node_mappings, error_lines)
        
        return chunks
    
    def _traverse_ast_with_error_filtering(self, node: Node, chunks: List[CodeChunk], 
                                         file_path: str, content: str, content_lines: List[str], 
                                         language: str, node_mappings: Dict[ChunkType, List[str]], 
                                         error_lines: Set[int]) -> None:
        """Traverse AST while filtering out nodes that overlap with errors."""
        # Skip if this node or its subtree contains errors
        if node.type == 'ERROR':
            return
        
        # Check if this node overlaps with error locations
        node_start = node.start_point[0]
        node_end = node.end_point[0]
        
        # If node overlaps with error lines, skip it
        if any(line in error_lines for line in range(node_start, node_end + 1)):
            # Still traverse children in case some parts are valid
            for child in node.children:
                self._traverse_ast_with_error_filtering(child, chunks, file_path, content, 
                                                       content_lines, language, node_mappings, error_lines)
            return
        
        # Check if this node represents a chunk we want to extract
        chunk_type = self._get_chunk_type(node, node_mappings, language)
        
        if chunk_type:
            chunk = self._create_chunk_from_node(
                node, chunk_type, file_path, content, content_lines, language
            )
            if chunk:
                chunks.append(chunk)
        
        # Recursively process child nodes
        for child in node.children:
            self._traverse_ast_with_error_filtering(child, chunks, file_path, content, 
                                                   content_lines, language, node_mappings, error_lines)
    
    def _identify_valid_sections_enhanced(self, error_locations: List[Tuple[int, int, str]], 
                                        content_lines: List[str], language: str) -> List[Tuple[int, int]]:
        """Enhanced valid section identification with language-specific heuristics."""
        if not error_locations:
            return [(0, len(content_lines) - 1)]
        
        # Sort error locations by start line
        sorted_errors = sorted(error_locations, key=lambda x: x[0])
        valid_sections = []
        
        # Check for valid section before first error
        first_error_start = sorted_errors[0][0]
        if first_error_start > 0:
            section_end = first_error_start - 1
            section_start = self._find_section_start_enhanced(content_lines, 0, section_end, language)
            if section_start <= section_end:
                valid_sections.append((section_start, section_end))
        
        # Check for valid sections between errors
        for i in range(len(sorted_errors) - 1):
            current_error_end = sorted_errors[i][1]
            next_error_start = sorted_errors[i + 1][0]
            
            if current_error_end < next_error_start - 1:
                section_start = self._find_section_start_enhanced(content_lines, current_error_end + 1, 
                                                                next_error_start - 1, language)
                section_end = next_error_start - 1
                if section_start <= section_end:
                    valid_sections.append((section_start, section_end))
        
        # Check for valid section after last error
        last_error_end = sorted_errors[-1][1]
        if last_error_end < len(content_lines) - 1:
            section_start = self._find_section_start_enhanced(content_lines, last_error_end + 1, 
                                                            len(content_lines) - 1, language)
            if section_start < len(content_lines):
                valid_sections.append((section_start, len(content_lines) - 1))
        
        return valid_sections
    
    def _find_section_start_enhanced(self, content_lines: List[str], min_line: int, 
                                   max_line: int, language: str) -> int:
        """Find the first meaningful line of code with language-specific patterns."""
        # Language-specific meaningful line patterns
        meaningful_patterns = {
            'python': ['def ', 'class ', 'import ', 'from ', '@'],
            'javascript': ['function ', 'class ', 'const ', 'let ', 'var ', 'import ', 'export '],
            'typescript': ['function ', 'class ', 'interface ', 'const ', 'let ', 'var ', 'import ', 'export '],
            'java': ['public ', 'private ', 'protected ', 'class ', 'interface ', 'import '],
            'cpp': ['class ', 'struct ', 'namespace ', 'template', 'int ', 'void ', 'double '],
            'go': ['func ', 'type ', 'var ', 'const ', 'package ', 'import '],
            'rust': ['fn ', 'struct ', 'impl ', 'pub ', 'use ', 'mod ']
        }
        
        patterns = meaningful_patterns.get(language, [])
        
        for i in range(min_line, min(max_line + 1, len(content_lines))):
            line = content_lines[i].strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Check for language-specific meaningful patterns
                if any(pattern in line for pattern in patterns):
                    return i
                # Also accept non-comment, non-empty lines
                if len(line) > 5:  # Reasonable minimum line length
                    return i
        
        return min_line
    
    def _section_already_processed(self, section_start: int, section_end: int, 
                                 existing_chunks: List[CodeChunk]) -> bool:
        """Check if a section has already been processed by looking at existing chunks."""
        for chunk in existing_chunks:
            # Check for overlap
            chunk_start = chunk.start_line - 1  # Convert to 0-based
            chunk_end = chunk.end_line - 1
            
            if (section_start <= chunk_end and section_end >= chunk_start):
                return True
        
        return False
    
    def _is_valid_code_section(self, section_content: str, language: str) -> bool:
        """Enhanced validation for code section content."""
        lines = section_content.strip().split('\n')
        
        if len(lines) == 0:
            return False
        
        # Filter out empty lines and comments
        meaningful_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                meaningful_lines.append(stripped)
        
        # Must have at least one meaningful line
        if len(meaningful_lines) == 0:
            return False
        
        # Check for minimum complexity
        total_chars = sum(len(line) for line in meaningful_lines)
        if total_chars < 20:  # Too short to be meaningful
            return False
        
        # Language-specific validation
        if language == 'python':
            # Check for valid Python constructs
            return any(':' in line or '=' in line or 'def ' in line or 'class ' in line 
                     for line in meaningful_lines)
        elif language in ['javascript', 'typescript']:
            # Check for valid JS/TS constructs
            return any('{' in line or '}' in line or '=' in line or 'function' in line 
                     or 'class' in line for line in meaningful_lines)
        elif language in ['java', 'cpp']:
            # Check for valid Java/C++ constructs
            return any('{' in line or '}' in line or ';' in line 
                     for line in meaningful_lines)
        
        return True  # Default to valid for other languages
    
    def _recover_using_heuristics(self, content_lines: List[str], file_path: str, 
                                language: str, error_locations: List[Tuple[int, int, str]]) -> List[CodeChunk]:
        """Use language-specific heuristics to recover code chunks."""
        chunks = []
        
        # Language-specific heuristic patterns
        if language == 'python':
            chunks.extend(self._recover_python_heuristics(content_lines, file_path))
        elif language in ['javascript', 'typescript']:
            chunks.extend(self._recover_js_heuristics(content_lines, file_path, language))
        elif language == 'java':
            chunks.extend(self._recover_java_heuristics(content_lines, file_path))
        elif language == 'cpp':
            chunks.extend(self._recover_cpp_heuristics(content_lines, file_path))
        
        return chunks
    
    def _recover_python_heuristics(self, content_lines: List[str], file_path: str) -> List[CodeChunk]:
        """Python-specific heuristic recovery."""
        chunks = []
        
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            
            # Look for function definitions
            if stripped.startswith('def ') and ':' in stripped:
                # Find the end of the function
                end_line = self._find_python_block_end(content_lines, i)
                if end_line > i:
                    func_content = '\n'.join(content_lines[i:end_line + 1])
                    chunks.append(self._create_heuristic_chunk(
                        func_content, file_path, "python", i + 1, end_line + 1, ChunkType.FUNCTION
                    ))
            
            # Look for class definitions
            elif stripped.startswith('class ') and ':' in stripped:
                end_line = self._find_python_block_end(content_lines, i)
                if end_line > i:
                    class_content = '\n'.join(content_lines[i:end_line + 1])
                    chunks.append(self._create_heuristic_chunk(
                        class_content, file_path, "python", i + 1, end_line + 1, ChunkType.CLASS
                    ))
        
        return chunks
    
    def _find_python_block_end(self, content_lines: List[str], start_line: int) -> int:
        """Find the end of a Python code block using indentation."""
        if start_line >= len(content_lines):
            return start_line
        
        base_indent = len(content_lines[start_line]) - len(content_lines[start_line].lstrip())
        
        for i in range(start_line + 1, len(content_lines)):
            line = content_lines[i]
            if line.strip() == '':
                continue  # Skip empty lines
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                return i - 1
        
        return len(content_lines) - 1
    
    def _recover_js_heuristics(self, content_lines: List[str], file_path: str, language: str) -> List[CodeChunk]:
        """JavaScript/TypeScript-specific heuristic recovery."""
        chunks = []
        
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            
            # Look for function definitions
            if 'function ' in stripped and '{' in stripped:
                end_line = self._find_brace_block_end(content_lines, i)
                if end_line > i:
                    func_content = '\n'.join(content_lines[i:end_line + 1])
                    chunks.append(self._create_heuristic_chunk(
                        func_content, file_path, language, i + 1, end_line + 1, ChunkType.FUNCTION
                    ))
            
            # Look for class definitions
            elif 'class ' in stripped and '{' in stripped:
                end_line = self._find_brace_block_end(content_lines, i)
                if end_line > i:
                    class_content = '\n'.join(content_lines[i:end_line + 1])
                    chunks.append(self._create_heuristic_chunk(
                        class_content, file_path, language, i + 1, end_line + 1, ChunkType.CLASS
                    ))
        
        return chunks
    
    def _recover_java_heuristics(self, content_lines: List[str], file_path: str) -> List[CodeChunk]:
        """Java-specific heuristic recovery."""
        return self._recover_js_heuristics(content_lines, file_path, "java")  # Similar brace-based structure
    
    def _recover_cpp_heuristics(self, content_lines: List[str], file_path: str) -> List[CodeChunk]:
        """C++-specific heuristic recovery."""
        return self._recover_js_heuristics(content_lines, file_path, "cpp")  # Similar brace-based structure
    
    def _find_brace_block_end(self, content_lines: List[str], start_line: int) -> int:
        """Find the end of a brace-delimited block."""
        if start_line >= len(content_lines):
            return start_line
        
        brace_count = 0
        found_opening = False
        
        for i in range(start_line, len(content_lines)):
            line = content_lines[i]
            
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    
                    if found_opening and brace_count == 0:
                        return i
        
        return len(content_lines) - 1
    
    def _create_heuristic_chunk(self, content: str, file_path: str, language: str,
                              start_line: int, end_line: int, chunk_type: ChunkType) -> CodeChunk:
        """Create a chunk using heuristic recovery."""
        chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_type)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=chunk_type,
            language=language,
            start_line=start_line,
            end_line=end_line,
            start_byte=0,  # Approximate for heuristic chunks
            end_byte=len(content),
            name=f"heuristic_{chunk_type.value}_{start_line}_{end_line}",
            breadcrumb=f"{Path(file_path).stem}.heuristic_{chunk_type.value}_{start_line}_{end_line}",
            content_hash=content_hash,
            embedding_text=content,
            indexed_at=datetime.now(),
            tags=["heuristic_recovery", chunk_type.value]
        )
    
    def _validate_parse_tree(self, tree, file_path: str) -> bool:
        """
        Validate that the parse tree is well-formed and usable.
        
        Args:
            tree: Tree-sitter parse tree
            file_path: Path to the source file for logging
            
        Returns:
            True if tree is valid, False if recovery is needed
        """
        if not tree or not tree.root_node:
            self.logger.warning(f"Parse tree missing root node for {file_path}")
            return False
        
        # Check for completely malformed trees (all ERROR nodes)
        if tree.root_node.type == 'ERROR' and not tree.root_node.children:
            self.logger.warning(f"Parse tree is entirely error nodes for {file_path}")
            return False
        
        # Check for minimal structure - ensure we have some valid nodes
        valid_nodes = self._count_valid_nodes(tree.root_node)
        error_nodes = self._count_errors(tree.root_node)
        
        # If error ratio is too high, consider recovery
        if error_nodes > 0 and valid_nodes > 0:
            error_ratio = error_nodes / (valid_nodes + error_nodes)
            if error_ratio > 0.8:  # More than 80% errors
                self.logger.warning(f"High error ratio ({error_ratio:.2%}) in parse tree for {file_path}")
                return False
        
        return True
    
    def _count_valid_nodes(self, node: Node) -> int:
        """Count non-error nodes in the AST."""
        count = 0 if node.type == 'ERROR' else 1
        for child in node.children:
            count += self._count_valid_nodes(child)
        return count
    
    def _attempt_parse_recovery(self, parser: Parser, content: str, file_path: str, language: str):
        """
        Attempt alternative parsing strategies for recovery.
        
        Args:
            parser: Tree-sitter parser
            content: File content
            file_path: Path to source file
            language: Programming language
            
        Returns:
            Recovered parse tree or original tree if recovery fails
        """
        self.logger.info(f"Attempting parse recovery for {file_path}")
        
        # Strategy 1: Try parsing with relaxed encoding
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    encoded_content = content.encode(encoding, errors='replace')
                    recovery_tree = parser.parse(encoded_content)
                    if self._validate_parse_tree(recovery_tree, file_path):
                        self.logger.info(f"Parse recovery successful with {encoding} encoding for {file_path}")
                        return recovery_tree
                except Exception:
                    continue
        except Exception as e:
            self.logger.debug(f"Encoding recovery failed for {file_path}: {e}")
        
        # Strategy 2: Try parsing with comment removal (may help with malformed comments)
        try:
            cleaned_content = self._remove_problematic_content(content, language)
            recovery_tree = parser.parse(bytes(cleaned_content, 'utf8'))
            if self._validate_parse_tree(recovery_tree, file_path):
                self.logger.info(f"Parse recovery successful with content cleaning for {file_path}")
                return recovery_tree
        except Exception as e:
            self.logger.debug(f"Content cleaning recovery failed for {file_path}: {e}")
        
        # Strategy 3: Return original tree and let error handling deal with it
        self.logger.warning(f"Parse recovery failed for {file_path}, using original tree")
        return parser.parse(bytes(content, 'utf8'))
    
    def _remove_problematic_content(self, content: str, language: str) -> str:
        """
        Remove potentially problematic content that might cause parsing issues.
        
        Args:
            content: Original file content
            language: Programming language
            
        Returns:
            Cleaned content with problematic elements removed/replaced
        """
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line
            
            # Language-specific cleaning
            if language == 'python':
                # Remove potentially malformed string literals
                if line.strip().startswith('#'):
                    cleaned_line = line  # Keep comments as-is
                elif '"""' in line or "'''" in line:
                    # Handle malformed docstrings
                    if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                        cleaned_line = line.replace('"""', '"').replace("'''", "'")
            
            elif language in ['javascript', 'typescript']:
                # Handle malformed template literals
                if '`' in line and line.count('`') % 2 == 1:
                    cleaned_line = line.replace('`', '"')
            
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _validate_chunk_boundaries(self, chunks: List[CodeChunk], content_lines: List[str]) -> List[CodeChunk]:
        """
        Validate and fix chunk boundaries to ensure they don't exceed file bounds.
        
        Args:
            chunks: List of extracted chunks
            content_lines: File content split by lines
            
        Returns:
            List of validated chunks with corrected boundaries
        """
        validated_chunks = []
        max_line = len(content_lines)
        
        for chunk in chunks:
            # Validate line boundaries
            start_line = max(1, min(chunk.start_line, max_line))
            end_line = max(start_line, min(chunk.end_line, max_line))
            
            # Create corrected chunk if boundaries changed
            if start_line != chunk.start_line or end_line != chunk.end_line:
                self.logger.debug(f"Corrected chunk boundaries from {chunk.start_line}-{chunk.end_line} "
                                f"to {start_line}-{end_line} for {chunk.file_path}")
                
                # Recalculate content based on corrected boundaries
                corrected_content = '\n'.join(content_lines[start_line-1:end_line])
                corrected_chunk = CodeChunk(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    content=corrected_content,
                    chunk_type=chunk.chunk_type,
                    language=chunk.language,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=chunk.start_byte,
                    end_byte=chunk.end_byte,
                    name=chunk.name,
                    breadcrumb=chunk.breadcrumb,
                    content_hash=hashlib.sha256(corrected_content.encode('utf-8')).hexdigest(),
                    embedding_text=corrected_content,
                    indexed_at=chunk.indexed_at,
                    tags=chunk.tags + ["boundary_corrected"]
                )
                validated_chunks.append(corrected_chunk)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _validate_chunk_quality(self, chunks: List[CodeChunk], content_lines: List[str], 
                               language: str) -> Tuple[List[CodeChunk], List[str]]:
        """
        Comprehensive chunk quality validation with detailed quality metrics.
        
        Args:
            chunks: List of extracted chunks
            content_lines: File content split by lines
            language: Programming language
            
        Returns:
            Tuple of (validated_chunks, quality_issues)
        """
        validated_chunks = []
        quality_issues = []
        
        for i, chunk in enumerate(chunks):
            issues = []
            
            # 1. Boundary validation
            if chunk.start_line < 1 or chunk.end_line > len(content_lines):
                issues.append(f"Invalid line boundaries: {chunk.start_line}-{chunk.end_line}")
            
            # 2. Content consistency validation
            if chunk.start_line <= len(content_lines) and chunk.end_line <= len(content_lines):
                expected_content = '\n'.join(content_lines[chunk.start_line-1:chunk.end_line])
                if chunk.content != expected_content:
                    issues.append("Content mismatch with file lines")
            
            # 3. Language-specific validation
            lang_issues = self._validate_chunk_language_specific(chunk, language)
            issues.extend(lang_issues)
            
            # 4. Chunk overlap detection
            overlap_issues = self._detect_chunk_overlaps(chunk, chunks[:i], chunks[i+1:])
            issues.extend(overlap_issues)
            
            # 5. Content quality validation
            content_issues = self._validate_chunk_content_quality(chunk, language)
            issues.extend(content_issues)
            
            if issues:
                quality_issues.extend([f"Chunk {i+1} ({chunk.chunk_type.value}): {issue}" for issue in issues])
                
                # Try to fix fixable issues
                fixed_chunk = self._attempt_chunk_repair(chunk, content_lines, issues)
                if fixed_chunk:
                    validated_chunks.append(fixed_chunk)
                else:
                    self.logger.warning(f"Chunk {i+1} has unfixable quality issues: {issues}")
                    # Include with quality warning tags
                    chunk.tags = (chunk.tags or []) + ["quality_issues"]
                    validated_chunks.append(chunk)
            else:
                validated_chunks.append(chunk)
        
        return validated_chunks, quality_issues
    
    def _validate_chunk_language_specific(self, chunk: CodeChunk, language: str) -> List[str]:
        """Validate chunk based on language-specific rules."""
        issues = []
        
        if language == 'python':
            # Check Python indentation consistency
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS]:
                lines = chunk.content.split('\n')
                if len(lines) > 1:
                    # Check if first line has proper definition structure
                    first_line = lines[0].strip()
                    if chunk.chunk_type == ChunkType.FUNCTION and not first_line.startswith('def '):
                        issues.append("Python function chunk doesn't start with 'def'")
                    elif chunk.chunk_type == ChunkType.CLASS and not first_line.startswith('class '):
                        issues.append("Python class chunk doesn't start with 'class'")
        
        elif language in ['javascript', 'typescript']:
            # Check JavaScript/TypeScript brace balance
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS]:
                brace_count = chunk.content.count('{') - chunk.content.count('}')
                if abs(brace_count) > 1:  # Allow for small imbalances due to chunking
                    issues.append(f"Unbalanced braces (difference: {brace_count})")
        
        elif language == 'java':
            # Check Java class/method structure
            if chunk.chunk_type == ChunkType.CLASS:
                if 'class ' not in chunk.content and 'interface ' not in chunk.content:
                    issues.append("Java class chunk missing class/interface declaration")
        
        return issues
    
    def _detect_chunk_overlaps(self, chunk: CodeChunk, before_chunks: List[CodeChunk], 
                             after_chunks: List[CodeChunk]) -> List[str]:
        """Detect overlapping chunks."""
        issues = []
        
        for other_chunk in before_chunks + after_chunks:
            # Check line overlap
            if (chunk.start_line <= other_chunk.end_line and 
                chunk.end_line >= other_chunk.start_line):
                overlap_start = max(chunk.start_line, other_chunk.start_line)
                overlap_end = min(chunk.end_line, other_chunk.end_line)
                issues.append(f"Overlaps with {other_chunk.chunk_type.value} chunk at lines {overlap_start}-{overlap_end}")
        
        return issues
    
    def _validate_chunk_content_quality(self, chunk: CodeChunk, language: str) -> List[str]:
        """Validate the quality of chunk content."""
        issues = []
        
        # 1. Check for empty or whitespace-only content
        if not chunk.content.strip():
            issues.append("Empty or whitespace-only content")
        
        # 2. Check for reasonable size bounds
        lines = chunk.content.split('\n')
        if len(lines) > 1000:  # Very large chunk
            issues.append(f"Unusually large chunk ({len(lines)} lines)")
        elif len(lines) == 1 and chunk.chunk_type not in [ChunkType.IMPORT, ChunkType.CONSTANT]:
            issues.append("Single-line chunk for multi-line construct")
        
        # 3. Check for incomplete constructs based on chunk type
        if chunk.chunk_type == ChunkType.FUNCTION:
            if not self._validate_function_completeness(chunk.content, language):
                issues.append("Incomplete function definition")
        elif chunk.chunk_type == ChunkType.CLASS:
            if not self._validate_class_completeness(chunk.content, language):
                issues.append("Incomplete class definition")
        
        return issues
    
    def _validate_function_completeness(self, content: str, language: str) -> bool:
        """Check if function chunk appears to be complete."""
        if language == 'python':
            # Check for def keyword and basic structure
            return 'def ' in content and ':' in content
        elif language in ['javascript', 'typescript']:
            # Check for function keyword or arrow function
            return ('function' in content or '=>' in content) and '{' in content
        elif language == 'java':
            # Check for method signature with braces
            return '{' in content and '}' in content
        return True  # Default to valid for unknown patterns
    
    def _validate_class_completeness(self, content: str, language: str) -> bool:
        """Check if class chunk appears to be complete."""
        if language == 'python':
            return 'class ' in content and ':' in content
        elif language in ['javascript', 'typescript']:
            return 'class ' in content and '{' in content
        elif language == 'java':
            return ('class ' in content or 'interface ' in content) and '{' in content
        return True
    
    def _attempt_chunk_repair(self, chunk: CodeChunk, content_lines: List[str], 
                            issues: List[str]) -> Optional[CodeChunk]:
        """Attempt to repair fixable chunk issues."""
        # For now, only fix boundary issues
        for issue in issues:
            if "Invalid line boundaries" in issue:
                # Fix boundary issues
                corrected_start = max(1, min(chunk.start_line, len(content_lines)))
                corrected_end = max(corrected_start, min(chunk.end_line, len(content_lines)))
                
                if corrected_start != chunk.start_line or corrected_end != chunk.end_line:
                    corrected_content = '\n'.join(content_lines[corrected_start-1:corrected_end])
                    
                    # Create repaired chunk
                    repaired_chunk = CodeChunk(
                        chunk_id=chunk.chunk_id,
                        file_path=chunk.file_path,
                        content=corrected_content,
                        chunk_type=chunk.chunk_type,
                        language=chunk.language,
                        start_line=corrected_start,
                        end_line=corrected_end,
                        start_byte=chunk.start_byte,
                        end_byte=chunk.end_byte,
                        name=chunk.name,
                        breadcrumb=chunk.breadcrumb,
                        content_hash=hashlib.sha256(corrected_content.encode('utf-8')).hexdigest(),
                        embedding_text=corrected_content,
                        indexed_at=chunk.indexed_at,
                        tags=(chunk.tags or []) + ["boundary_repaired"]
                    )
                    return repaired_chunk
        
        return None
    
    def _log_detailed_parse_failure(self, file_path: str, language: str, 
                                   syntax_errors: List[CodeSyntaxError], 
                                   processing_time_ms: float) -> None:
        """
        Log detailed information about parse failures for debugging and monitoring.
        
        Args:
            file_path: Path to the failed file
            language: Programming language
            syntax_errors: List of detected syntax errors
            processing_time_ms: Time taken for parsing attempt
        """
        # Classify errors by type and severity
        error_types = {}
        critical_count = 0
        for error in syntax_errors:
            error_type = error.error_type
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
            
            if error.severity == "critical":
                critical_count += 1
        
        # Log structured error information
        self.logger.error(
            f"Parse failure analysis for {file_path}:\n"
            f"  Language: {language}\n"
            f"  Total errors: {len(syntax_errors)}\n"
            f"  Critical errors: {critical_count}\n"
            f"  Processing time: {processing_time_ms:.2f}ms\n"
            f"  Error types: {dict(error_types)}\n"
            f"  File size: {self._get_file_size_info(file_path)}"
        )
        
        # Log detailed error locations for the first few errors
        max_detailed_errors = 3
        for i, error in enumerate(syntax_errors[:max_detailed_errors]):
            self.logger.debug(
                f"Error {i+1} at line {error.start_line}: {error.error_type}\n"
                f"Context:\n{error.context}"
            )
        
        if len(syntax_errors) > max_detailed_errors:
            self.logger.debug(f"... and {len(syntax_errors) - max_detailed_errors} more errors")
    
    def _get_file_size_info(self, file_path: str) -> str:
        """Get human-readable file size information."""
        try:
            import os
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            return "unknown size"
    
    def _log_parse_success_metrics(self, file_path: str, language: str, 
                                  chunks: List[CodeChunk], processing_time_ms: float,
                                  error_recovery_used: bool) -> None:
        """
        Log metrics for successful parsing operations.
        
        Args:
            file_path: Path to the successfully parsed file
            language: Programming language
            chunks: List of extracted chunks
            processing_time_ms: Time taken for parsing
            error_recovery_used: Whether error recovery was needed
        """
        # Analyze chunk types
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type.value
            if chunk_type not in chunk_types:
                chunk_types[chunk_type] = 0
            chunk_types[chunk_type] += 1
        
        # Log success metrics
        recovery_status = " (with error recovery)" if error_recovery_used else ""
        self.logger.info(
            f"Parse success for {file_path}{recovery_status}:\n"
            f"  Language: {language}\n"
            f"  Chunks extracted: {len(chunks)}\n"
            f"  Chunk types: {dict(chunk_types)}\n"
            f"  Processing time: {processing_time_ms:.2f}ms\n"
            f"  Performance: {len(chunks) / max(1, processing_time_ms / 1000):.1f} chunks/sec"
        )
    
    def _enhance_error_logging_in_parse(self, file_path: str, language: str, 
                                       parse_result: ParseResult) -> None:
        """
        Enhanced logging for parse results with detailed error analysis.
        
        Args:
            file_path: Path to the parsed file
            language: Programming language
            parse_result: Result of parsing operation
        """
        if not parse_result.parse_success:
            # Log detailed failure analysis
            self._log_detailed_parse_failure(
                file_path, language, parse_result.syntax_errors or [], 
                parse_result.processing_time_ms
            )
        elif parse_result.error_recovery_used:
            # Log warning for files that needed error recovery
            self.logger.warning(
                f"Error recovery used for {file_path}: "
                f"{parse_result.valid_sections_count} valid sections recovered "
                f"from {parse_result.error_count} errors"
            )
            # Also log success metrics
            self._log_parse_success_metrics(
                file_path, language, parse_result.chunks, 
                parse_result.processing_time_ms, True
            )
        else:
            # Log success metrics for clean parses
            self._log_parse_success_metrics(
                file_path, language, parse_result.chunks, 
                parse_result.processing_time_ms, False
            )
    
    def _record_parsing_metrics(self, parse_result: ParseResult, file_path: str, 
                              quality_issues_count: int) -> None:
        """
        Record comprehensive parsing metrics for performance monitoring.
        
        Args:
            parse_result: Result of the parsing operation
            file_path: Path to the parsed file
            quality_issues_count: Number of quality issues found during validation
        """
        try:
            # Get file size
            file_size_bytes = 0
            try:
                import os
                file_size_bytes = os.path.getsize(file_path)
            except Exception:
                pass  # File size optional
            
            # Count repaired chunks
            repaired_chunks = 0
            for chunk in parse_result.chunks:
                if chunk.tags and any("repaired" in tag for tag in chunk.tags):
                    repaired_chunks += 1
            
            # Record the metrics
            chunking_metrics_tracker.record_parsing_operation(
                parse_result=parse_result,
                file_size_bytes=file_size_bytes,
                quality_issues=quality_issues_count,
                repaired_chunks=repaired_chunks
            )
            
        except Exception as e:
            self.logger.debug(f"Failed to record metrics for {file_path}: {e}")
    
    def get_performance_summary(self) -> str:
        """
        Get a performance summary report for the current session.
        
        Returns:
            Human-readable performance report
        """
        return chunking_metrics_tracker.get_performance_report()
    
    def get_language_performance(self, language: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a specific language.
        
        Args:
            language: Programming language name
            
        Returns:
            Dictionary with language-specific metrics, or None if not found
        """
        lang_metrics = chunking_metrics_tracker.get_language_metrics(language)
        if not lang_metrics:
            return None
        
        return {
            "language": language,
            "total_files": lang_metrics.total_files,
            "success_rate": lang_metrics.success_rate(),
            "recent_success_rate": lang_metrics.recent_success_rate(),
            "average_chunks_per_file": lang_metrics.average_chunks_per_file(),
            "average_processing_time_ms": lang_metrics.average_processing_time_ms(),
            "chunks_per_second": lang_metrics.chunks_per_second(),
            "fallback_rate": (lang_metrics.fallback_files / max(1, lang_metrics.total_files)) * 100,
            "error_recovery_rate": (lang_metrics.error_recovery_files / max(1, lang_metrics.total_files)) * 100,
            "total_chunks": lang_metrics.total_chunks,
            "total_errors": lang_metrics.total_errors,
            "chunk_types_distribution": dict(lang_metrics.chunk_types),
            "quality_issues": lang_metrics.quality_issues,
            "repaired_chunks": lang_metrics.repaired_chunks
        }
    
    def export_performance_metrics(self, export_path: str) -> None:
        """
        Export performance metrics to a file.
        
        Args:
            export_path: Path to export the metrics file
        """
        chunking_metrics_tracker.export_metrics(export_path)
        self.logger.info(f"Performance metrics exported to {export_path}")
    
    def reset_session_metrics(self) -> None:
        """Reset session-specific performance metrics."""
        chunking_metrics_tracker.reset_session_metrics()
        self.logger.info("Session performance metrics reset")
    
    def _should_use_whole_file_fallback(self, file_path: str, content: str, 
                                       language: str, error_count: int) -> bool:
        """
        Determine if whole-file chunking should be used as a last resort.
        
        This method implements intelligent fallback logic to minimize the use
        of whole-file chunks, which are less useful for semantic search.
        
        Args:
            file_path: Path to the source file
            content: File content
            language: Programming language
            error_count: Number of parse errors detected
            
        Returns:
            True if whole-file chunking should be used, False to try alternatives
        """
        lines = content.split('\n')
        line_count = len(lines)
        
        # Always use whole-file for very small files (< 10 lines)
        if line_count < 10:
            self.logger.debug(f"Using whole-file for small file: {file_path} ({line_count} lines)")
            return True
        
        # Use whole-file for very large files that might overwhelm alternative chunking
        if line_count > 5000:
            self.logger.debug(f"Using whole-file for very large file: {file_path} ({line_count} lines)")
            return True
        
        # Check if content is mostly non-code (comments, whitespace, etc.)
        code_lines = self._count_code_lines(content, language)
        if code_lines < line_count * 0.1:  # Less than 10% actual code
            self.logger.debug(f"Using whole-file for mostly non-code file: {file_path} "
                            f"({code_lines}/{line_count} code lines)")
            return True
        
        # Use whole-file if error rate is extremely high (> 50% of lines have errors)
        if error_count > line_count * 0.5:
            self.logger.debug(f"Using whole-file for heavily corrupted file: {file_path} "
                            f"({error_count} errors in {line_count} lines)")
            return True
        
        # Check for specific file patterns that are better as whole-file
        file_name = Path(file_path).name.lower()
        
        # Configuration files, data files, etc.
        if any(pattern in file_name for pattern in [
            'config', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini',
            'package.json', 'requirements.txt', 'dockerfile', 'makefile'
        ]):
            self.logger.debug(f"Using whole-file for configuration file: {file_path}")
            return True
        
        # Generated files or minified code
        if any(pattern in file_name for pattern in [
            '.min.', '.bundle.', '.generated.', '.auto.', '_pb2.py'
        ]):
            self.logger.debug(f"Using whole-file for generated/minified file: {file_path}")
            return True
        
        # For unknown or unsupported languages, be more conservative
        if language == "unknown" or language not in self._node_mappings:
            if line_count > 100:  # Use whole-file for larger unknown files
                self.logger.debug(f"Using whole-file for unknown language file: {file_path}")
                return True
        
        # Default: try alternatives before whole-file
        self.logger.debug(f"Attempting alternatives before whole-file for: {file_path}")
        return False
    
    def _count_code_lines(self, content: str, language: str) -> int:
        """
        Count lines that appear to contain actual code (not just comments/whitespace).
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            Number of lines that appear to contain code
        """
        lines = content.split('\n')
        code_lines = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Language-specific comment detection
            is_comment = False
            if language == 'python':
                is_comment = stripped.startswith('#')
            elif language in ['javascript', 'typescript', 'java', 'cpp', 'go', 'rust']:
                is_comment = stripped.startswith('//') or stripped.startswith('/*')
            elif language == 'html':
                is_comment = stripped.startswith('<!--')
            elif language == 'css':
                is_comment = stripped.startswith('/*')
            
            # Count as code if it's not a comment and has some substance
            if not is_comment and len(stripped) > 2:
                code_lines += 1
        
        return code_lines
    
    def _attempt_alternative_chunking(self, file_path: str, content: str, 
                                    language: str, content_lines: List[str]) -> List[CodeChunk]:
        """
        Attempt alternative chunking strategies before falling back to whole-file.
        
        Args:
            file_path: Path to the source file
            content: File content
            language: Programming language  
            content_lines: File content split by lines
            
        Returns:
            List of alternative chunks, empty if no alternatives found
        """
        chunks = []
        
        self.logger.info(f"Attempting alternative chunking for {file_path}")
        
        # Strategy 1: Line-based chunking for structured content
        line_chunks = self._create_line_based_chunks(file_path, content_lines, language)
        if line_chunks:
            chunks.extend(line_chunks)
        
        # Strategy 2: Pattern-based chunking for known structures
        pattern_chunks = self._create_pattern_based_chunks(file_path, content, language)
        if pattern_chunks:
            chunks.extend(pattern_chunks)
        
        # Strategy 3: Paragraph-based chunking for documentation
        if self._is_documentation_file(file_path):
            para_chunks = self._create_paragraph_chunks(file_path, content, language)
            if para_chunks:
                chunks.extend(para_chunks)
        
        # Strategy 4: Import/export chunking for module files
        if language in ['javascript', 'typescript', 'python']:
            import_chunks = self._create_import_export_chunks(file_path, content_lines, language)
            if import_chunks:
                chunks.extend(import_chunks)
        
        # Filter out very small chunks (< 3 lines) that aren't meaningful
        meaningful_chunks = [
            chunk for chunk in chunks 
            if chunk.end_line - chunk.start_line + 1 >= 3 or 
            chunk.chunk_type in [ChunkType.IMPORT, ChunkType.EXPORT, ChunkType.CONSTANT]
        ]
        
        if meaningful_chunks:
            self.logger.info(f"Created {len(meaningful_chunks)} alternative chunks for {file_path}")
            return meaningful_chunks
        
        self.logger.warning(f"No meaningful alternative chunks found for {file_path}")
        return []
    
    def _create_line_based_chunks(self, file_path: str, content_lines: List[str], 
                                language: str) -> List[CodeChunk]:
        """Create chunks based on logical line groupings."""
        chunks = []
        chunk_size = 50  # Lines per chunk
        
        if len(content_lines) < chunk_size * 2:  # Don't chunk small files
            return []
        
        for i in range(0, len(content_lines), chunk_size):
            end_idx = min(i + chunk_size, len(content_lines))
            chunk_content = '\n'.join(content_lines[i:end_idx])
            
            if chunk_content.strip():  # Only create non-empty chunks
                chunk = self._create_alternative_chunk(
                    chunk_content, file_path, language, i + 1, end_idx,
                    ChunkType.VARIABLE, f"lines_{i+1}_{end_idx}"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_pattern_based_chunks(self, file_path: str, content: str, 
                                   language: str) -> List[CodeChunk]:
        """Create chunks based on common code patterns."""
        chunks = []
        lines = content.split('\n')
        
        # Look for function-like patterns even without proper AST parsing
        if language == 'python':
            patterns = [r'^\s*def\s+\w+', r'^\s*class\s+\w+', r'^\s*@\w+']
        elif language in ['javascript', 'typescript']:
            patterns = [r'^\s*function\s+\w+', r'^\s*const\s+\w+\s*=', r'^\s*class\s+\w+']
        elif language == 'java':
            patterns = [r'^\s*public\s+\w+', r'^\s*private\s+\w+', r'^\s*class\s+\w+']
        else:
            return []  # No patterns for this language
        
        import re
        
        current_chunk_start = None
        current_chunk_lines = []
        
        for i, line in enumerate(lines):
            is_pattern_start = any(re.match(pattern, line) for pattern in patterns)
            
            if is_pattern_start:
                # Save previous chunk if exists
                if current_chunk_start is not None and current_chunk_lines:
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunk = self._create_alternative_chunk(
                        chunk_content, file_path, language,
                        current_chunk_start + 1, current_chunk_start + len(current_chunk_lines),
                        ChunkType.FUNCTION, f"pattern_{current_chunk_start+1}"
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk_start = i
                current_chunk_lines = [line]
            elif current_chunk_start is not None:
                current_chunk_lines.append(line)
                
                # End chunk if we hit a blank line and have some content
                if not line.strip() and len(current_chunk_lines) > 5:
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunk = self._create_alternative_chunk(
                        chunk_content, file_path, language,
                        current_chunk_start + 1, current_chunk_start + len(current_chunk_lines),
                        ChunkType.FUNCTION, f"pattern_{current_chunk_start+1}"
                    )
                    chunks.append(chunk)
                    current_chunk_start = None
                    current_chunk_lines = []
        
        # Handle final chunk
        if current_chunk_start is not None and current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk = self._create_alternative_chunk(
                chunk_content, file_path, language,
                current_chunk_start + 1, current_chunk_start + len(current_chunk_lines),
                ChunkType.FUNCTION, f"pattern_{current_chunk_start+1}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_paragraph_chunks(self, file_path: str, content: str, 
                               language: str) -> List[CodeChunk]:
        """Create chunks based on paragraph breaks for documentation."""
        chunks = []
        paragraphs = content.split('\n\n')
        
        current_line = 1
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                lines_in_para = paragraph.count('\n') + 1
                chunk = self._create_alternative_chunk(
                    paragraph, file_path, language,
                    current_line, current_line + lines_in_para - 1,
                    ChunkType.DOCSTRING, f"paragraph_{i+1}"
                )
                chunks.append(chunk)
                current_line += lines_in_para + 1  # +1 for the empty line
            else:
                current_line += 1
        
        return chunks
    
    def _create_import_export_chunks(self, file_path: str, content_lines: List[str], 
                                   language: str) -> List[CodeChunk]:
        """Create chunks for import/export statements and module-level code."""
        chunks = []
        
        import_lines = []
        other_lines = []
        
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            
            if language == 'python':
                is_import = stripped.startswith(('import ', 'from '))
            elif language in ['javascript', 'typescript']:
                is_import = stripped.startswith(('import ', 'export ', 'const ', 'let ', 'var '))
            else:
                is_import = False
            
            if is_import:
                import_lines.append((i, line))
            else:
                other_lines.append((i, line))
        
        # Create import chunk if we have imports
        if import_lines:
            import_content = '\n'.join(line for _, line in import_lines)
            first_line = import_lines[0][0] + 1
            last_line = import_lines[-1][0] + 1
            
            chunk = self._create_alternative_chunk(
                import_content, file_path, language, first_line, last_line,
                ChunkType.IMPORT, "imports"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _is_documentation_file(self, file_path: str) -> bool:
        """Check if file appears to be documentation."""
        file_name = Path(file_path).name.lower()
        return any(ext in file_name for ext in ['.md', '.rst', '.txt', 'readme', 'license'])
    
    def _create_alternative_chunk(self, content: str, file_path: str, language: str,
                                start_line: int, end_line: int, chunk_type: ChunkType,
                                name_suffix: str) -> CodeChunk:
        """Create an alternative chunk with proper metadata."""
        chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_type)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=chunk_type,
            language=language,
            start_line=start_line,
            end_line=end_line,
            start_byte=0,  # Approximate
            end_byte=len(content),
            name=f"alt_{chunk_type.value}_{name_suffix}",
            breadcrumb=f"{Path(file_path).stem}.alt_{chunk_type.value}_{name_suffix}",
            content_hash=content_hash,
            embedding_text=content,
            indexed_at=datetime.now(),
            tags=["alternative_chunking", chunk_type.value, "smart_fallback"]
        )
    
    def _parse_structured_file(self, file_path: str, content: Optional[str], 
                             language: str, start_time: float) -> ParseResult:
        """
        Parse JSON/YAML files into structured chunks.
        
        Args:
            file_path: Path to the file
            content: File content (if None, will read from file)
            language: File type ('json' or 'yaml')
            start_time: Processing start time
            
        Returns:
            ParseResult with structured chunks
        """
        # Read file content if not provided
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                self.logger.error(f"Failed to read {language} file {file_path}: {e}")
                return self._create_fallback_result(file_path, "", start_time, error=True)
        
        # Parse the structured content
        try:
            if language == 'json':
                parsed_data = json.loads(content)
                chunks = self._extract_structured_chunks(parsed_data, file_path, content, language)
            elif language == 'yaml':
                parsed_data = yaml.safe_load(content)
                chunks = self._extract_structured_chunks(parsed_data, file_path, content, language)
            elif language == 'markdown':
                chunks = self._extract_markdown_chunks(file_path, content)
            else:
                return self._create_fallback_result(file_path, content, start_time)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create ParseResult
            result = ParseResult(
                chunks=chunks,
                file_path=file_path,
                language=language,
                parse_success=True,
                processing_time_ms=processing_time,
                error_count=0,
                fallback_used=False
            )
            
            self._log_parse_success_metrics(file_path, language, chunks, processing_time, False)
            return result
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.warning(f"Failed to parse {language} file {file_path}: {e}")
            return self._create_fallback_result(file_path, content, start_time, error=True)
        except Exception as e:
            self.logger.error(f"Unexpected error parsing {language} file {file_path}: {e}")
            return self._create_fallback_result(file_path, content, start_time, error=True)
    
    def _extract_structured_chunks(self, data: Union[dict, list], file_path: str, 
                                 content: str, language: str) -> List[CodeChunk]:
        """
        Extract meaningful chunks from parsed JSON/YAML data.
        
        Args:
            data: Parsed JSON/YAML data
            file_path: Source file path
            content: Original file content
            language: File type ('json' or 'yaml')
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        content_lines = content.split('\n')
        
        if isinstance(data, dict):
            chunks.extend(self._extract_dict_chunks(data, file_path, content_lines, language))
        elif isinstance(data, list):
            chunks.extend(self._extract_list_chunks(data, file_path, content_lines, language))
        else:
            # Single value file - create one chunk
            chunks.append(self._create_structured_chunk(
                content, file_path, language, 1, len(content_lines),
                ChunkType.CONSTANT, "root_value"
            ))
        
        return chunks
    
    def _extract_dict_chunks(self, data: dict, file_path: str, content_lines: List[str], 
                           language: str) -> List[CodeChunk]:
        """Extract chunks from dictionary data."""
        chunks = []
        
        for key, value in data.items():
            # Find the approximate line range for this key-value pair
            start_line, end_line = self._find_key_lines(key, value, content_lines, language)
            
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                # Large nested structure - create a separate chunk
                value_content = self._format_structured_content(key, value, language)
                chunk_type = ChunkType.CLASS if isinstance(value, dict) else ChunkType.CONSTANT
                chunks.append(self._create_structured_chunk(
                    value_content, file_path, language, start_line, end_line,
                    chunk_type, key
                ))
            else:
                # Simple value - group with other simple values
                simple_content = self._format_structured_content(key, value, language)
                chunks.append(self._create_structured_chunk(
                    simple_content, file_path, language, start_line, end_line,
                    ChunkType.VARIABLE, key
                ))
        
        return chunks
    
    def _extract_list_chunks(self, data: list, file_path: str, content_lines: List[str], 
                           language: str) -> List[CodeChunk]:
        """Extract chunks from list data."""
        chunks = []
        
        # If it's a list of objects, treat each as a separate chunk
        for i, item in enumerate(data):
            start_line, end_line = self._find_list_item_lines(i, item, content_lines, language)
            
            if isinstance(item, (dict, list)):
                item_content = self._format_structured_content(f"item_{i}", item, language)
                chunk_type = ChunkType.CLASS if isinstance(item, dict) else ChunkType.CONSTANT
            else:
                item_content = self._format_structured_content(f"item_{i}", item, language)
                chunk_type = ChunkType.VARIABLE
            
            chunks.append(self._create_structured_chunk(
                item_content, file_path, language, start_line, end_line,
                chunk_type, f"item_{i}"
            ))
        
        return chunks
    
    def _find_key_lines(self, key: str, value: Any, content_lines: List[str], 
                       language: str) -> Tuple[int, int]:
        """Find the line range for a key-value pair."""
        # Simple heuristic - find the key in the content
        for i, line in enumerate(content_lines):
            if key in line:
                # Estimate the end line based on value complexity
                if isinstance(value, (dict, list)):
                    # Complex value - estimate multiple lines
                    value_str = str(value)
                    estimated_lines = min(max(value_str.count('\n'), 1), 20)
                    return i + 1, i + estimated_lines + 1
                else:
                    # Simple value - probably one line
                    return i + 1, i + 1
        
        # Fallback if key not found
        return 1, len(content_lines)
    
    def _find_list_item_lines(self, index: int, item: Any, content_lines: List[str], 
                            language: str) -> Tuple[int, int]:
        """Find the line range for a list item."""
        # Estimate based on item position and complexity
        total_lines = len(content_lines)
        estimated_start = max(1, (index * total_lines) // max(len(content_lines), 1))
        
        if isinstance(item, (dict, list)):
            estimated_lines = min(max(str(item).count('\n'), 1), 10)
        else:
            estimated_lines = 1
        
        return estimated_start, min(estimated_start + estimated_lines, total_lines)
    
    def _format_structured_content(self, key: str, value: Any, language: str) -> str:
        """Format structured content for embedding."""
        if language == 'json':
            if isinstance(value, (dict, list)):
                return f'"{key}": {json.dumps(value, indent=2)}'
            else:
                return f'"{key}": {json.dumps(value)}'
        else:  # yaml
            if isinstance(value, (dict, list)):
                return f'{key}:\n{yaml.dump(value, indent=2)}'
            else:
                return f'{key}: {yaml.dump(value).strip()}'
    
    def _create_structured_chunk(self, content: str, file_path: str, language: str,
                               start_line: int, end_line: int, chunk_type: ChunkType,
                               name: str) -> CodeChunk:
        """Create a structured chunk for JSON/YAML content."""
        chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_type)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=chunk_type,
            language=language,
            start_line=start_line,
            end_line=end_line,
            start_byte=0,  # Approximate
            end_byte=len(content),
            name=name,
            breadcrumb=f"{Path(file_path).stem}.{name}",
            content_hash=content_hash,
            embedding_text=f"Name: {name}\nType: {chunk_type.value}\nContent: {content}",
            indexed_at=datetime.now(),
            tags=["structured_data", language, chunk_type.value]
        )
    
    def _extract_markdown_chunks(self, file_path: str, content: str) -> List[CodeChunk]:
        """
        Extract hierarchical chunks from Markdown content based on header levels.
        
        Args:
            file_path: Path to the Markdown file
            content: File content
            
        Returns:
            List of CodeChunk objects representing sections
        """
        chunks = []
        content_lines = content.split('\n')
        
        # Find all headers with their levels and positions
        headers = self._find_markdown_headers(content_lines)
        
        if not headers:
            # No headers found - create a single chunk for the entire content
            chunks.append(self._create_structured_chunk(
                content, file_path, 'markdown', 1, len(content_lines),
                ChunkType.CONSTANT, "document"
            ))
            return chunks
        
        # Create chunks for each section
        for i, (level, title, start_line) in enumerate(headers):
            # Determine end line for this section
            if i + 1 < len(headers):
                end_line = headers[i + 1][2] - 1  # End before next header
            else:
                end_line = len(content_lines)
            
            # Extract section content
            section_lines = content_lines[start_line - 1:end_line]
            section_content = '\n'.join(section_lines)
            
            # Determine chunk type based on header level
            chunk_type = self._get_markdown_chunk_type(level)
            
            # Create breadcrumb path
            breadcrumb = self._create_markdown_breadcrumb(file_path, title, level, headers[:i + 1])
            
            chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, start_line, end_line, chunk_type),
                file_path=file_path,
                content=section_content,
                chunk_type=chunk_type,
                language='markdown',
                start_line=start_line,
                end_line=end_line,
                start_byte=0,  # Approximate
                end_byte=len(section_content),
                name=title,
                breadcrumb=breadcrumb,
                content_hash=hashlib.sha256(section_content.encode('utf-8')).hexdigest(),
                embedding_text=f"Section: {title}\nLevel: {level}\nContent: {section_content}",
                indexed_at=datetime.now(),
                tags=["markdown", f"level_{level}", chunk_type.value]
            )
            chunks.append(chunk)
        
        return chunks
    
    def _find_markdown_headers(self, content_lines: List[str]) -> List[Tuple[int, str, int]]:
        """
        Find all Markdown headers and their levels.
        
        Args:
            content_lines: List of content lines
            
        Returns:
            List of tuples: (level, title, line_number)
        """
        headers = []
        
        for i, line in enumerate(content_lines):
            line = line.strip()
            
            # ATX-style headers (# ## ### etc.)
            if line.startswith('#'):
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                if level <= 6 and line[level:].strip():  # Valid header
                    title = line[level:].strip()
                    headers.append((level, title, i + 1))
            
            # Setext-style headers (underlined with = or -)
            elif i > 0 and line and all(c in '=-' for c in line):
                prev_line = content_lines[i - 1].strip()
                if prev_line:  # Previous line has content
                    level = 1 if line[0] == '=' else 2
                    headers.append((level, prev_line, i))  # Use previous line number
        
        return headers
    
    def _get_markdown_chunk_type(self, level: int) -> ChunkType:
        """Get appropriate chunk type for header level."""
        if level == 1:
            return ChunkType.CLASS      # Main sections
        elif level == 2:
            return ChunkType.FUNCTION   # Subsections
        elif level == 3:
            return ChunkType.CONSTANT   # Sub-subsections
        else:
            return ChunkType.VARIABLE   # Deeper levels
    
    def _create_markdown_breadcrumb(self, file_path: str, title: str, level: int,
                                  header_hierarchy: List[Tuple[int, str, int]]) -> str:
        """
        Create a hierarchical breadcrumb for Markdown sections.
        
        Args:
            file_path: Path to the file
            title: Current section title
            level: Current header level
            header_hierarchy: List of headers up to and including current
            
        Returns:
            Breadcrumb string showing hierarchical path
        """
        file_stem = Path(file_path).stem
        
        # Build hierarchical path
        path_parts = [file_stem]
        
        # Add parent headers to build proper hierarchy
        current_level = level
        for header_level, header_title, _ in reversed(header_hierarchy[:-1]):  # Exclude current header
            if header_level < current_level:
                # Clean title for breadcrumb
                clean_title = re.sub(r'[^\w\s-]', '', header_title).strip()
                clean_title = re.sub(r'\s+', '_', clean_title)
                path_parts.insert(-1, clean_title)  # Insert before current title
                current_level = header_level
        
        # Add current title
        clean_current = re.sub(r'[^\w\s-]', '', title).strip()
        clean_current = re.sub(r'\s+', '_', clean_current)
        path_parts.append(clean_current)
        
        return '.'.join(path_parts)