"""
Comprehensive unit tests for the CodeParser service.

Tests cover all major functionality including:
- Language detection and parser loading
- AST parsing and intelligent chunking
- Syntax error handling and recovery
- Metadata extraction and breadcrumb generation
- Multi-language support
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.code_parser_service import CodeParserService
from models.code_chunk import CodeChunk, ChunkType, ParseResult, CodeSyntaxError
from utils.language_detector import detect_language


class TestCodeParserService:
    """Test suite for CodeParserService functionality."""
    
    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance for testing."""
        return CodeParserService()
    
    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''"""Module docstring."""

import os
import sys

CONSTANT_VALUE = 42

class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        """Return the name."""
        return self.name
    
    async def async_method(self) -> None:
        """An async method."""
        await some_operation()

def global_function(x: int, y: int) -> int:
    """A global function."""
    return x + y

if __name__ == "__main__":
    main()
'''
    
    @pytest.fixture
    def sample_javascript_code(self):
        """Sample JavaScript code for testing."""
        return '''// JavaScript module
import { Component } from 'react';

const CONSTANT = 'value';

class MyComponent extends Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }
    
    handleClick = () => {
        this.setState({ count: this.state.count + 1 });
    }
    
    render() {
        return <div>{this.state.count}</div>;
    }
}

function utilityFunction(a, b) {
    return a + b;
}

const arrowFunction = (x) => x * 2;

export default MyComponent;
'''
    
    @pytest.fixture
    def sample_typescript_code(self):
        """Sample TypeScript code for testing."""
        return '''interface User {
    id: number;
    name: string;
    email?: string;
}

type Status = 'active' | 'inactive';

class UserService {
    private users: User[] = [];
    
    constructor(private apiClient: ApiClient) {}
    
    async getUser(id: number): Promise<User | null> {
        return this.users.find(user => user.id === id) || null;
    }
    
    addUser(user: User): void {
        this.users.push(user);
    }
}

function processUser(user: User): string {
    return `Processing ${user.name}`;
}
'''
    
    @pytest.fixture
    def sample_code_with_errors(self):
        """Sample code with syntax errors for testing error handling."""
        return '''def broken_function(
    # Missing closing parenthesis
    pass

class IncompleteClass
    # Missing colon
    def method(self):
        return "test"
        
# Unclosed string
invalid = "unclosed string
'''

    def test_initialization(self, parser_service):
        """Test CodeParserService initialization."""
        assert parser_service is not None
        assert hasattr(parser_service, '_parsers')
        assert hasattr(parser_service, '_languages')
        assert hasattr(parser_service, '_supported_languages')
        assert hasattr(parser_service, '_node_mappings')
    
    def test_supported_languages(self, parser_service):
        """Test that all expected languages are supported."""
        expected_languages = {'python', 'javascript', 'typescript', 'go', 'rust', 'java'}
        assert set(parser_service._supported_languages.keys()) == expected_languages
    
    def test_language_detection(self, parser_service):
        """Test language detection from file content and paths."""
        # Test Python detection
        py_content = "def function(): pass"
        assert detect_language("test.py", py_content) == "python"
        
        # Test JavaScript detection
        js_content = "function test() { return 'js'; }"
        assert detect_language("test.js", js_content) == "javascript"
        
        # Test TypeScript detection
        ts_content = "interface Test { name: string; }"
        assert detect_language("test.ts", ts_content) == "typescript"
    
    @patch('tree_sitter.Language')
    @patch('tree_sitter.Parser')
    def test_parser_loading_success(self, mock_parser_class, mock_language_class, parser_service):
        """Test successful parser loading for supported languages."""
        # Mock successful language and parser creation
        mock_language = MagicMock()
        mock_parser = MagicMock()
        mock_language_class.return_value = mock_language
        mock_parser_class.return_value = mock_parser
        
        # Test parser loading
        parser = parser_service._get_parser("python")
        
        assert parser is not None
        assert mock_language_class.called
        assert mock_parser_class.called
        assert mock_parser.set_language.called
    
    def test_parser_loading_failure(self, parser_service):
        """Test parser loading failure for unsupported languages."""
        # Test unsupported language
        parser = parser_service._get_parser("unsupported_language")
        assert parser is None
    
    @patch.object(CodeParserService, '_get_parser')
    def test_parse_code_success(self, mock_get_parser, parser_service, sample_python_code):
        """Test successful code parsing and chunk generation."""
        # Mock parser and tree
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_root_node = MagicMock()
        
        mock_get_parser.return_value = mock_parser
        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = mock_root_node
        mock_tree.root_node.has_error = False
        
        # Mock node structure for Python function
        mock_function_node = MagicMock()
        mock_function_node.type = 'function_definition'
        mock_function_node.start_point = (10, 0)
        mock_function_node.end_point = (15, 0)
        mock_function_node.start_byte = 100
        mock_function_node.end_byte = 200
        mock_function_node.text = b'def test_function(): pass'
        
        # Mock children traversal
        mock_root_node.children = [mock_function_node]
        
        # Test parsing
        result = parser_service.parse_code(sample_python_code, "test.py", "python")
        
        assert isinstance(result, ParseResult)
        assert result.parse_success is True
        assert result.language == "python"
        assert result.file_path == "test.py"
    
    @patch.object(CodeParserService, '_get_parser')
    def test_parse_code_with_syntax_errors(self, mock_get_parser, parser_service, sample_code_with_errors):
        """Test parsing code with syntax errors."""
        # Mock parser with error nodes
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_root_node = MagicMock()
        
        mock_get_parser.return_value = mock_parser
        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = mock_root_node
        mock_tree.root_node.has_error = True
        
        # Mock error node
        mock_error_node = MagicMock()
        mock_error_node.type = 'ERROR'
        mock_error_node.start_point = (1, 0)
        mock_error_node.end_point = (1, 10)
        mock_error_node.text = b'broken code'
        
        mock_root_node.children = [mock_error_node]
        
        # Test parsing with errors
        result = parser_service.parse_code(sample_code_with_errors, "broken.py", "python")
        
        assert isinstance(result, ParseResult)
        assert result.error_count > 0
        assert len(result.syntax_errors) > 0
        assert result.error_recovery_used is True
    
    def test_chunk_metadata_extraction(self, parser_service):
        """Test extraction of metadata from code chunks."""
        # Create a sample chunk with mock AST node
        mock_node = MagicMock()
        mock_node.type = 'function_definition'
        mock_node.start_point = (5, 0)
        mock_node.end_point = (10, 0)
        mock_node.start_byte = 50
        mock_node.end_byte = 150
        mock_node.text = b'def test_function(arg1, arg2):\n    return arg1 + arg2'
        
        # Mock child nodes for function name and parameters
        mock_name_node = MagicMock()
        mock_name_node.type = 'identifier'
        mock_name_node.text = b'test_function'
        
        mock_parameters_node = MagicMock()
        mock_parameters_node.type = 'parameters'
        mock_parameters_node.text = b'(arg1, arg2)'
        
        mock_node.children = [mock_name_node, mock_parameters_node]
        
        # Test metadata extraction
        metadata = parser_service._extract_node_metadata(mock_node, "python")
        
        assert 'name' in metadata
        assert 'signature' in metadata
        assert 'chunk_type' in metadata
    
    def test_breadcrumb_generation(self, parser_service):
        """Test breadcrumb generation for nested code structures."""
        # Mock class and method hierarchy
        class_context = {'name': 'TestClass', 'type': 'class'}
        method_name = 'test_method'
        
        breadcrumb = parser_service._generate_breadcrumb(
            method_name, 
            parent_context=class_context,
            file_path="test.py"
        )
        
        assert 'TestClass' in breadcrumb
        assert 'test_method' in breadcrumb
        assert breadcrumb == 'test.TestClass.test_method'
    
    def test_context_extraction(self, parser_service, sample_python_code):
        """Test extraction of surrounding code context."""
        lines = sample_python_code.split('\n')
        
        # Test context extraction for a chunk in the middle
        context_before, context_after = parser_service._extract_context(
            lines, start_line=10, end_line=12, context_lines=2
        )
        
        assert isinstance(context_before, str)
        assert isinstance(context_after, str)
        assert len(context_before.split('\n')) <= 2
        assert len(context_after.split('\n')) <= 2
    
    def test_error_classification(self, parser_service):
        """Test syntax error classification and categorization."""
        # Mock error node with different error types
        mock_error_node = MagicMock()
        mock_error_node.type = 'ERROR'
        mock_error_node.start_point = (5, 10)
        mock_error_node.end_point = (5, 20)
        mock_error_node.text = b'invalid syntax'
        
        error = parser_service._classify_syntax_error(mock_error_node, "python")
        
        assert isinstance(error, CodeSyntaxError)
        assert error.start_line == 6  # 1-based line numbering
        assert error.error_type in ['syntax_error', 'unexpected_token', 'missing_token']
    
    def test_fallback_chunking(self, parser_service, sample_python_code):
        """Test fallback to whole-file chunking when parsing fails."""
        # Test with parsing disabled/failed
        with patch.object(parser_service, '_get_parser', return_value=None):
            result = parser_service.parse_code(sample_python_code, "test.py", "python")
            
            assert isinstance(result, ParseResult)
            assert result.fallback_used is True
            assert len(result.chunks) == 1
            assert result.chunks[0].chunk_type == ChunkType.WHOLE_FILE
    
    @pytest.mark.parametrize("language,chunk_types", [
        ("python", [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.CONSTANT]),
        ("javascript", [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.CONSTANT]),
        ("typescript", [ChunkType.INTERFACE, ChunkType.TYPE_ALIAS, ChunkType.CLASS]),
        ("java", [ChunkType.CLASS, ChunkType.METHOD, ChunkType.INTERFACE]),
        ("go", [ChunkType.FUNCTION, ChunkType.STRUCT, ChunkType.INTERFACE]),
        ("rust", [ChunkType.FUNCTION, ChunkType.STRUCT, ChunkType.ENUM])
    ])
    def test_language_specific_chunking(self, parser_service, language, chunk_types):
        """Test that each language produces appropriate chunk types."""
        # This would require more detailed mocking of language-specific AST structures
        # For now, test that the node mappings exist
        assert language in parser_service._node_mappings
        mappings = parser_service._node_mappings[language]
        
        for chunk_type in chunk_types:
            assert chunk_type in mappings
            assert isinstance(mappings[chunk_type], list)
            assert len(mappings[chunk_type]) > 0
    
    def test_performance_monitoring(self, parser_service, sample_python_code):
        """Test that parsing performance is tracked."""
        # Parse code and check that timing is recorded
        result = parser_service.parse_code(sample_python_code, "test.py", "python")
        
        assert hasattr(result, 'processing_time_ms')
        assert result.processing_time_ms >= 0
    
    def test_memory_management(self, parser_service):
        """Test that memory is properly managed during parsing."""
        # Test with multiple large code samples
        large_code = "def function_{}(): pass\n" * 1000
        
        results = []
        for i in range(10):
            result = parser_service.parse_code(large_code, f"test_{i}.py", "python")
            results.append(result)
        
        # Should not crash or consume excessive memory
        assert len(results) == 10
        assert all(isinstance(r, ParseResult) for r in results)
    
    def test_thread_safety(self, parser_service):
        """Test that the parser service is thread-safe."""
        import threading
        import concurrent.futures
        
        def parse_worker(code, file_id):
            return parser_service.parse_code(code, f"test_{file_id}.py", "python")
        
        test_code = "def test(): pass"
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(parse_worker, test_code, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 20
        assert all(isinstance(r, ParseResult) for r in results)


class TestCodeChunkModel:
    """Test suite for CodeChunk data model."""
    
    def test_code_chunk_creation(self):
        """Test CodeChunk instantiation and basic properties."""
        chunk = CodeChunk(
            chunk_id="test_chunk_1",
            file_path="/test/file.py",
            content="def test(): pass",
            chunk_type=ChunkType.FUNCTION,
            language="python",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
            name="test",
            breadcrumb="file.test"
        )
        
        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.file_path == "/test/file.py"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.language == "python"
        assert chunk.name == "test"
        assert chunk.breadcrumb == "file.test"
        assert chunk.line_count == 2
        assert chunk.char_count == 17
    
    def test_code_chunk_serialization(self):
        """Test CodeChunk to_dict and from_dict methods."""
        original_chunk = CodeChunk(
            chunk_id="test_chunk_1",
            file_path="/test/file.py",
            content="def test(): pass",
            chunk_type=ChunkType.FUNCTION,
            language="python",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
            name="test",
            tags=["utility", "test"]
        )
        
        # Test serialization
        chunk_dict = original_chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict['chunk_id'] == "test_chunk_1"
        assert chunk_dict['chunk_type'] == "function"
        assert chunk_dict['tags'] == ["utility", "test"]
        
        # Test deserialization
        restored_chunk = CodeChunk.from_dict(chunk_dict)
        assert restored_chunk.chunk_id == original_chunk.chunk_id
        assert restored_chunk.chunk_type == original_chunk.chunk_type
        assert restored_chunk.content == original_chunk.content
        assert restored_chunk.tags == original_chunk.tags


class TestIntegrationScenarios:
    """Integration tests for real-world parsing scenarios."""
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            yield f
        os.unlink(f.name)
    
    def test_end_to_end_parsing(self, temp_file):
        """Test complete parsing workflow from file to chunks."""
        # Write sample code to temp file
        sample_code = '''"""Module for testing."""

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        """Add two numbers."""
        return x + y
    
    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

def main():
    """Main function."""
    calc = Calculator()
    print(calc.add(2, 3))

if __name__ == "__main__":
    main()
'''
        
        temp_file.write(sample_code)
        temp_file.flush()
        
        # Parse the file
        parser_service = CodeParserService()
        with open(temp_file.name, 'r') as f:
            content = f.read()
        
        result = parser_service.parse_code(content, temp_file.name, "python")
        
        # Verify results
        assert isinstance(result, ParseResult)
        assert result.parse_success is True
        assert len(result.chunks) > 0
        
        # Check for expected chunk types
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.CLASS in chunk_types or ChunkType.FUNCTION in chunk_types
    
    def test_error_recovery_integration(self):
        """Test error recovery in realistic scenarios."""
        # Code with recoverable syntax errors
        problematic_code = '''
class BrokenClass:
    def good_method(self):
        return "works"
    
    def broken_method(
        # Missing closing paren and colon
        pass
        
    def another_good_method(self):
        return "also works"
'''
        
        parser_service = CodeParserService()
        result = parser_service.parse_code(problematic_code, "broken.py", "python")
        
        # Should still extract some valid chunks despite errors
        assert isinstance(result, ParseResult)
        assert result.error_count > 0
        assert result.error_recovery_used is True
        
        # Should have found at least some valid methods
        assert result.valid_sections_count > 0 or len(result.chunks) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])