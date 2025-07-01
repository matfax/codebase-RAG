"""
CodeParser service for intelligent code chunking using Tree-sitter.

This service provides semantic code parsing capabilities to break down source files
into meaningful chunks based on code structure rather than simple text splitting.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
from datetime import datetime

try:
    import tree_sitter
    from tree_sitter import Language, Parser, Node
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")

from ..models.code_chunk import CodeChunk, ChunkType, ParseResult
from ..utils.file_system_utils import get_file_size, get_file_mtime


class CodeParserService:
    """
    Service for parsing source code into intelligent semantic chunks.
    
    This service uses Tree-sitter to parse code into an Abstract Syntax Tree (AST)
    and then extracts meaningful code constructs like functions, classes, and constants.
    """
    
    def __init__(self):
        """Initialize the CodeParser service with language support."""
        self.logger = logging.getLogger(__name__)
        self._parsers: Dict[str, Parser] = {}
        self._languages: Dict[str, Language] = {}
        self._supported_languages = {
            'python': 'tree_sitter_python',
            'javascript': 'tree_sitter_javascript', 
            'typescript': 'tree_sitter_typescript.tsx',  # Supports both TS and TSX
            'go': 'tree_sitter_go',
            'rust': 'tree_sitter_rust',
            'java': 'tree_sitter_java',
            'json': 'tree_sitter_json'
        }
        
        # Language-specific node types for different code constructs
        self._node_mappings = {
            'python': {
                ChunkType.FUNCTION: ['function_definition'],
                ChunkType.ASYNC_FUNCTION: ['async_function_definition'],
                ChunkType.CLASS: ['class_definition'],
                ChunkType.CONSTANT: ['assignment'],  # We'll filter these by context
                ChunkType.VARIABLE: ['assignment'],
                ChunkType.IMPORT: ['import_statement', 'import_from_statement'],
                ChunkType.DOCSTRING: ['expression_statement']  # String literals at module/class/function level
            },
            'javascript': {
                ChunkType.FUNCTION: ['function_declaration', 'arrow_function'],
                ChunkType.ASYNC_FUNCTION: ['async_function_declaration'],
                ChunkType.CLASS: ['class_declaration'],
                ChunkType.CONSTANT: ['lexical_declaration'],  # const declarations
                ChunkType.VARIABLE: ['variable_declaration'],
                ChunkType.IMPORT: ['import_statement'],
                ChunkType.EXPORT: ['export_statement']
            },
            'typescript': {
                ChunkType.FUNCTION: ['function_declaration', 'arrow_function'],
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
                ChunkType.STRUCT: ['type_declaration'],  # Go structs
                ChunkType.INTERFACE: ['interface_type'],
                ChunkType.CONSTANT: ['const_declaration'],
                ChunkType.VARIABLE: ['var_declaration'],
                ChunkType.IMPORT: ['import_declaration']
            },
            'rust': {
                ChunkType.FUNCTION: ['function_item'],
                ChunkType.STRUCT: ['struct_item'],
                ChunkType.ENUM: ['enum_item'],
                ChunkType.CONSTANT: ['const_item'],
                ChunkType.VARIABLE: ['let_declaration'],
                ChunkType.IMPORT: ['use_declaration']
            },
            'java': {
                ChunkType.FUNCTION: ['method_declaration'],
                ChunkType.CLASS: ['class_declaration'],
                ChunkType.INTERFACE: ['interface_declaration'],
                ChunkType.ENUM: ['enum_declaration'],
                ChunkType.CONSTANT: ['field_declaration'],  # Static final fields
                ChunkType.VARIABLE: ['field_declaration'],
                ChunkType.IMPORT: ['import_declaration']
            },
            'json': {
                ChunkType.CONFIG_BLOCK: ['object', 'array'],
                ChunkType.DATA_STRUCTURE: ['object']
            }
        }
        
        self._initialize_parsers()
    
    def _initialize_parsers(self) -> None:
        """Initialize Tree-sitter parsers for supported languages."""
        for lang_name, module_name in self._supported_languages.items():
            try:
                # Import the language module dynamically
                if '.' in module_name:
                    # Handle TypeScript which has submodules
                    parts = module_name.split('.')
                    module = __import__(parts[0], fromlist=[parts[1]])
                    language_func = getattr(module, parts[1])
                else:
                    module = __import__(module_name)
                    language_func = getattr(module, 'language')
                
                # Create language and parser
                language = language_func()
                parser = Parser()
                parser.set_language(language)
                
                self._languages[lang_name] = language
                self._parsers[lang_name] = parser
                
                self.logger.info(f"Initialized {lang_name} parser")
                
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Failed to initialize {lang_name} parser: {e}")
    
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
        path = Path(file_path)
        extension = path.suffix.lower()
        
        # Extension to language mapping
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.json': 'json'
        }
        
        return ext_map.get(extension)
    
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
        if not language or language not in self._parsers:
            return self._create_fallback_result(file_path, content, start_time)
        
        # Read file content if not provided
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                self.logger.error(f"Failed to read file {file_path}: {e}")
                return self._create_fallback_result(file_path, "", start_time, error=True)
        
        # Parse the content
        try:
            parser = self._parsers[language]
            tree = parser.parse(bytes(content, 'utf8'))
            
            # Extract chunks from the AST
            chunks = self._extract_chunks(tree.root_node, file_path, content, language)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Check for parsing errors
            error_count = self._count_errors(tree.root_node)
            parse_success = error_count == 0
            
            return ParseResult(
                chunks=chunks,
                file_path=file_path,
                language=language,
                parse_success=parse_success,
                error_count=error_count,
                fallback_used=False,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return self._create_fallback_result(file_path, content, start_time, error=True)
    
    def _extract_chunks(self, root_node: Node, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """
        Extract code chunks from the parsed AST.
        
        Args:
            root_node: Root node of the parsed AST
            file_path: Path to the source file
            content: File content as string
            language: Programming language
            
        Returns:
            List of extracted CodeChunk objects
        """
        chunks = []
        content_lines = content.split('\n')
        node_mappings = self._node_mappings.get(language, {})
        
        # Traverse the AST and extract relevant nodes
        self._traverse_ast(root_node, chunks, file_path, content, content_lines, language, node_mappings)
        
        # If no chunks were extracted, create a whole-file chunk
        if not chunks:
            chunks.append(self._create_whole_file_chunk(file_path, content, language))
        
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
        
        for chunk_type, node_types in node_mappings.items():
            if node_type in node_types:
                # Special handling for Python assignments to distinguish constants vs variables
                if language == 'python' and node_type == 'assignment':
                    return self._classify_python_assignment(node)
                return chunk_type
        
        return None
    
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
    
    def _create_fallback_result(self, file_path: str, content: Optional[str], 
                              start_time: float, error: bool = False) -> ParseResult:
        """Create a fallback ParseResult with whole-file chunk."""
        if content is None:
            content = ""
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create a single whole-file chunk
        chunks = []
        if content:
            language = self.detect_language(file_path) or "unknown"
            chunks.append(self._create_whole_file_chunk(file_path, content, language))
        
        return ParseResult(
            chunks=chunks,
            file_path=file_path,
            language=language if content else "unknown",
            parse_success=not error,
            error_count=1 if error else 0,
            fallback_used=True,
            processing_time_ms=processing_time
        )
    
    def _create_whole_file_chunk(self, file_path: str, content: str, language: str) -> CodeChunk:
        """Create a whole-file chunk as fallback."""
        lines = content.split('\n')
        chunk_id = self._generate_chunk_id(file_path, 1, len(lines), ChunkType.WHOLE_FILE)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
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
            indexed_at=datetime.now()
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
        
        if node_type in ['function_definition', 'async_function_definition']:
            # Get the function signature (name + parameters)
            signature_parts = []
            
            # Add async keyword if present
            if node_type == 'async_function_definition':
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
        # TODO: Implement JavaScript name extraction
        return None
    
    def _extract_js_signature(self, node: Node) -> Optional[str]:
        """Extract signature from JavaScript/TypeScript AST node."""
        # TODO: Implement JavaScript signature extraction
        return None
    
    def _extract_js_docstring(self, node: Node, content_lines: List[str]) -> Optional[str]:
        """Extract JSDoc comment from JavaScript/TypeScript function or class."""
        # TODO: Implement JavaScript docstring extraction
        return None
    
    def _extract_go_name(self, node: Node) -> Optional[str]:
        """Extract name from Go AST node."""
        # TODO: Implement Go name extraction
        return None
    
    def _extract_go_signature(self, node: Node) -> Optional[str]:
        """Extract signature from Go AST node."""
        # TODO: Implement Go signature extraction
        return None
    
    def _extract_rust_name(self, node: Node) -> Optional[str]:
        """Extract name from Rust AST node."""
        # TODO: Implement Rust name extraction
        return None
    
    def _extract_rust_signature(self, node: Node) -> Optional[str]:
        """Extract signature from Rust AST node."""
        # TODO: Implement Rust signature extraction
        return None
    
    def _extract_java_name(self, node: Node) -> Optional[str]:
        """Extract name from Java AST node."""
        # TODO: Implement Java name extraction
        return None
    
    def _extract_java_signature(self, node: Node) -> Optional[str]:
        """Extract signature from Java AST node."""
        # TODO: Implement Java signature extraction
        return None