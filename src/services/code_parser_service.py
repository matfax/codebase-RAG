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

from models.code_chunk import CodeChunk, ChunkType, ParseResult, CodeSyntaxError
from utils.file_system_utils import get_file_size, get_file_mtime


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
            'java': 'tree_sitter_java'
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
            '.java': 'java'
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
            
            # Check for parsing errors and collect detailed error information
            error_count = self._count_errors(tree.root_node)
            parse_success = error_count == 0
            syntax_errors = self._collect_detailed_errors(tree.root_node, content.split('\n'), language)
            
            # Determine if error recovery was used
            error_recovery_used = error_count > 0 and len(chunks) > 0
            valid_sections_count = len(chunks) if error_recovery_used else 0
            
            return ParseResult(
                chunks=chunks,
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
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return self._create_fallback_result(file_path, content, start_time, error=True)
    
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
                # Special handling for assignments to distinguish constants vs variables
                if language == 'python' and node_type == 'assignment':
                    return self._classify_python_assignment(node)
                elif language in ['javascript', 'typescript'] and node_type == 'lexical_declaration':
                    return self._classify_js_declaration(node)
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
    
    def _classify_js_declaration(self, node: Node) -> ChunkType:
        """
        Classify JavaScript/TypeScript lexical declaration as constant or variable.
        
        Args:
            node: Lexical declaration AST node
            
        Returns:
            ChunkType.CONSTANT for const declarations, ChunkType.VARIABLE for let
        """
        # Check if this is a const declaration
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
    
    def _handle_syntax_errors(self, node: Node, file_path: str, content: str, 
                            content_lines: List[str], language: str) -> List[CodeChunk]:
        """
        Intelligently handle syntax errors by extracting valid code around errors.
        
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
        
        # First pass: collect all error locations
        self._collect_error_locations(node, error_locations)
        
        if not error_locations:
            # No errors found, proceed with normal parsing
            return self._extract_chunks_normal(node, file_path, content, content_lines, language)
        
        self.logger.warning(f"Found {len(error_locations)} syntax errors in {file_path}")
        
        # Second pass: extract valid code sections between errors
        valid_sections = self._identify_valid_sections(error_locations, content_lines)
        
        for section_start, section_end in valid_sections:
            # Extract content for this valid section
            section_content = '\n'.join(content_lines[section_start:section_end + 1])
            
            if len(section_content.strip()) > 10:  # Only process substantial sections
                # Create a chunk for this valid section
                chunk = self._create_valid_section_chunk(
                    section_content, file_path, language, 
                    section_start + 1, section_end + 1  # Convert to 1-based line numbers
                )
                if chunk:
                    chunks.append(chunk)
        
        # If no valid sections found, create a whole-file chunk with error annotation
        if not chunks:
            chunks.append(self._create_error_annotated_chunk(file_path, content, language, error_locations))
        
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
        node_type = node.type
        
        if node_type in ['function_declaration', 'async_function_declaration']:
            # Look for identifier child node
            for child in node.children:
                if child.type == 'identifier':
                    return child.text.decode('utf-8')
        
        elif node_type == 'arrow_function':
            # Arrow functions might not have explicit names, look for assignment context
            # This would need parent context analysis for proper naming
            return 'arrow_function'  # Placeholder
        
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
        
        if node_type in ['function_declaration', 'async_function_declaration']:
            # Get the function signature
            signature_parts = []
            
            # Add async keyword if present
            if node_type == 'async_function_declaration':
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
        
        elif node_type == 'arrow_function':
            # Arrow function signature
            signature_parts = []
            
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
            
            # Add declaration keyword (const, let, var)
            for child in node.children:
                if child.type in ['const', 'let', 'var']:
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type == 'variable_declarator':
                    # Get variable name and type
                    for declarator_child in child.children:
                        if declarator_child.type == 'identifier':
                            signature_parts.append(declarator_child.text.decode('utf-8'))
                        elif declarator_child.type == 'type_annotation':
                            signature_parts.append(':')
                            signature_parts.append(declarator_child.text.decode('utf-8'))
                    break
            
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
        
        if node_type in ['function_declaration', 'method_declaration']:
            # Look for identifier child node
            for child in node.children:
                if child.type == 'identifier':
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
            for child in node.children:
                if child.type == 'import_spec':
                    for spec_child in child.children:
                        if spec_child.type in ['interpreted_string_literal', 'raw_string_literal']:
                            # Extract package path from quotes
                            package_path = spec_child.text.decode('utf-8').strip('"\'`')
                            package_name = package_path.split('/')[-1]  # Get last part
                            names.append(package_name)
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
                elif child.type == 'identifier':
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
        
        return None
    
    def _extract_java_name(self, node: Node) -> Optional[str]:
        """Extract name from Java AST node."""
        node_type = node.type
        
        if node_type == 'method_declaration':
            # Look for method name identifier
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
            # [modifiers] return_type name(parameters)
            signature_parts = []
            
            for child in node.children:
                if child.type == 'modifiers':
                    # public, private, static, etc.
                    signature_parts.append(child.text.decode('utf-8'))
                elif child.type in ['type_identifier', 'primitive_type', 'void_type', 'generic_type']:
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