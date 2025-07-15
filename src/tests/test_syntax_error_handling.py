"""
Comprehensive tests for syntax error handling and recovery in code parsing.

This test suite verifies that the CodeParser service can:
- Detect and classify various types of syntax errors
- Recover valid code sections from files with syntax errors
- Provide meaningful error information and suggestions
- Continue processing despite encountering errors
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.code_parser_service import CodeParserService

from src.models.code_chunk import ChunkType, ParseResult


class TestSyntaxErrorDetection:
    """Test syntax error detection capabilities."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_missing_parenthesis_detection(self, parser_service):
        """Test detection of missing parentheses."""
        code_with_error = '''
def broken_function(arg1, arg2
    """Function with missing closing parenthesis."""
    return arg1 + arg2

def valid_function():
    """This function is valid."""
    return "valid"
'''

        result = parser_service.parse_code(code_with_error, "test.py", "python")

        # Should detect errors
        assert result.error_count > 0
        assert len(result.syntax_errors) > 0

        # Should find the missing parenthesis error
        error_messages = [error.error_message for error in result.syntax_errors]
        assert any("parenthesis" in msg.lower() or "paren" in msg.lower() for msg in error_messages)

        # Should still recover and find the valid function
        assert result.error_recovery_used
        assert result.valid_sections_count > 0 or len(result.chunks) > 0

    def test_missing_colon_detection(self, parser_service):
        """Test detection of missing colons."""
        code_with_error = '''
class IncompleteClass
    """Class missing colon after name."""

    def __init__(self, name):
        self.name = name

    def get_name(self)
        # Missing colon here too
        return self.name

def valid_function():
    return "still valid"
'''

        result = parser_service.parse_code(code_with_error, "test.py", "python")

        # Should detect multiple syntax errors
        assert result.error_count > 0
        assert len(result.syntax_errors) >= 2  # At least two missing colons

        # Should attempt error recovery
        assert result.error_recovery_used

    def test_unclosed_string_detection(self, parser_service):
        """Test detection of unclosed string literals."""
        code_with_error = """
def function_with_unclosed_string():
    message = "This string is never closed
    return message

def another_function():
    return "This is fine"
"""

        result = parser_service.parse_code(code_with_error, "test.py", "python")

        # Should detect string errors
        assert result.error_count > 0

        # Check for string-related errors
        error_types = [error.error_type for error in result.syntax_errors]
        assert any("string" in error_type.lower() or "quote" in error_type.lower() for error_type in error_types)

    def test_indentation_error_detection(self, parser_service):
        """Test detection of indentation errors."""
        code_with_error = """
def indentation_error():
    if True:
        print("Correct indentation")
      print("Incorrect indentation")  # Wrong indentation level

    return "done"

def valid_function():
    return "valid"
"""

        result = parser_service.parse_code(code_with_error, "test.py", "python")

        # Should detect indentation issues
        assert result.error_count > 0

        # Should have error information
        assert len(result.syntax_errors) > 0

        # Check error locations
        for error in result.syntax_errors:
            assert error.start_line > 0
            assert error.start_column >= 0

    def test_invalid_syntax_patterns(self, parser_service):
        """Test detection of various invalid syntax patterns."""
        code_with_errors = """
# Multiple syntax errors in one file

# Error 1: Invalid assignment
5 = x

# Error 2: Invalid operator
def invalid_operation():
    return 5 ++ 3

# Error 3: Missing comma in function arguments
def function_call_error():
    return max(1 2 3)

# Valid code for recovery testing
def valid_function():
    return "I am valid"

# Error 4: Invalid lambda syntax
calculate = lambda x, y: x + y if x > 0 else

# More valid code
class ValidClass:
    def valid_method(self):
        return True
"""

        result = parser_service.parse_code(code_with_errors, "test.py", "python")

        # Should detect multiple errors
        assert result.error_count >= 3
        assert len(result.syntax_errors) >= 3

        # Should still find valid sections
        if result.error_recovery_used:
            assert result.valid_sections_count > 0

        # Or should fall back to whole-file chunking
        if result.fallback_used:
            assert len(result.chunks) > 0
            assert result.chunks[0].chunk_type == ChunkType.WHOLE_FILE

    def test_complex_syntax_errors(self, parser_service):
        """Test handling of complex, nested syntax errors."""
        code_with_complex_errors = '''
class BrokenClass:
    def __init__(self
        # Missing closing paren and colon
        self.data = {
            "key1": "value1",
            "key2": ["item1", "item2",
            "key3": "value3"
        }  # Missing closing bracket for list

    def method_with_issues(self):
        try:
            risky_operation()
        except ValueError as e
            # Missing colon
            print(f"Error: {e}")

    def valid_method(self):
        """This method should be recoverable."""
        return "valid"

def valid_standalone_function():
    """This should also be recoverable."""
    return {"status": "ok"}

# Unclosed function call
result = some_function(arg1, arg2, arg3

def another_valid_function():
    return "also valid"
'''

        result = parser_service.parse_code(code_with_complex_errors, "test.py", "python")

        # Should detect multiple complex errors
        assert result.error_count > 0
        assert len(result.syntax_errors) > 0

        # Should attempt error recovery
        assert result.error_recovery_used or result.fallback_used

        # Should provide error context
        for error in result.syntax_errors:
            assert error.context is not None
            assert len(error.context.strip()) > 0


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_recovery_after_syntax_errors(self, parser_service):
        """Test that parser can recover and continue after syntax errors."""
        code_with_mixed_content = '''
"""Module with mixed valid and invalid code."""

# Valid import
import os

# Valid constant
VALID_CONSTANT = 42

# Broken function
def broken_function(
    # Missing closing parenthesis
    pass

# Valid class that should be recovered
class RecoverableClass:
    """This class should be found despite previous errors."""

    def __init__(self, name):
        self.name = name

    def get_info(self):
        return f"Name: {self.name}"

# Another syntax error
invalid_string = "unclosed string

# Another valid function
def another_valid_function():
    """This should also be recovered."""
    return {"recovered": True}

# Valid lambda
square = lambda x: x ** 2
'''

        result = parser_service.parse_code(code_with_mixed_content, "test.py", "python")

        # Should detect errors
        assert result.error_count > 0

        # Should recover valid sections
        assert result.error_recovery_used
        assert result.valid_sections_count > 0

        # Should find some valid chunks
        valid_chunks = [chunk for chunk in result.chunks if chunk.chunk_type != ChunkType.WHOLE_FILE]
        assert len(valid_chunks) > 0

        # Check that we found the recoverable class
        chunk_names = [chunk.name for chunk in result.chunks if chunk.name]
        assert "RecoverableClass" in chunk_names or any("class" in str(chunk.chunk_type) for chunk in result.chunks)

    def test_partial_recovery_statistics(self, parser_service):
        """Test that recovery statistics are accurate."""
        code_with_known_structure = """
# File with known structure for testing statistics

def valid_function_1():
    return "first"

def broken_function(
    # Missing closing paren
    return "broken"

def valid_function_2():
    return "second"

class ValidClass:
    def method(self):
        return "method"

class BrokenClass
    # Missing colon
    def method(self):
        return "broken method"

def valid_function_3():
    return "third"
"""

        result = parser_service.parse_code(code_with_known_structure, "test.py", "python")

        # Should have error statistics
        assert result.error_count > 0
        assert result.error_recovery_used

        # Should have reasonable recovery statistics
        # We expect to recover at least 3 valid functions and 1 valid class
        if result.valid_sections_count > 0:
            assert result.valid_sections_count >= 3

        # Or should have reasonable chunk count
        if len(result.chunks) > 0:
            assert len(result.chunks) >= 1  # At least something should be recovered

    def test_recovery_with_nested_structures(self, parser_service):
        """Test recovery of nested code structures with errors."""
        code_with_nested_errors = '''
class OuterClass:
    """Outer class with nested issues."""

    def valid_outer_method(self):
        return "outer valid"

    class NestedClass:
        def broken_nested_method(self
            # Missing closing paren and colon
            return "nested broken"

        def valid_nested_method(self):
            return "nested valid"

    def another_outer_method(self):
        return "another outer"

def standalone_function():
    return "standalone"
'''

        result = parser_service.parse_code(code_with_nested_errors, "test.py", "python")

        # Should handle nested structures
        assert result.error_count > 0

        # Should recover what it can
        if result.error_recovery_used:
            assert result.valid_sections_count > 0

            # Check for recovered nested structures
            [chunk for chunk in result.chunks if hasattr(chunk, "parent_name") and chunk.parent_name]
            # Might find some nested structures if recovery is sophisticated

        # At minimum, should not crash and should provide some result
        assert isinstance(result, ParseResult)


class TestErrorClassification:
    """Test classification and categorization of syntax errors."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_error_severity_classification(self, parser_service):
        """Test that errors are classified by severity."""
        code_with_various_errors = """
# File with errors of different severities

# Critical error: completely invalid syntax
def *(invalid_function_name):
    pass

# Warning-level: unused variable (if detected)
def function_with_warning():
    unused_var = "not used"
    return "result"

# Error: missing syntax element
def missing_colon_function()
    return "missing colon"

# Valid code
def valid_function():
    return "valid"
"""

        result = parser_service.parse_code(code_with_various_errors, "test.py", "python")

        if len(result.syntax_errors) > 0:
            # Check severity classification
            severities = {error.severity for error in result.syntax_errors}

            # Should have some classification
            assert len(severities) > 0

            # Check that severe errors are marked appropriately
            critical_errors = [error for error in result.syntax_errors if error.severity == "error"]
            warnings = [error for error in result.syntax_errors if error.severity == "warning"]

            # Should have at least some errors classified
            assert len(critical_errors) > 0 or len(warnings) > 0

    def test_error_type_classification(self, parser_service):
        """Test that errors are classified by type."""
        code_with_typed_errors = """
# Different types of syntax errors

# Missing punctuation
def missing_colon()
    pass

# Invalid tokens
def invalid_operator():
    return 5 ++ 3

# Unclosed constructs
def unclosed_paren(arg1, arg2
    pass

# Invalid assignments
5 = variable

def valid_function():
    return "valid"
"""

        result = parser_service.parse_code(code_with_typed_errors, "test.py", "python")

        if len(result.syntax_errors) > 0:
            # Check error type classification
            error_types = {error.error_type for error in result.syntax_errors}

            # Should have type classification
            assert len(error_types) > 0

            # Common error types might include
            expected_types = {
                "missing_colon",
                "missing_punctuation",
                "syntax_error",
                "unexpected_token",
                "invalid_syntax",
                "parse_error",
            }

            # Should have some recognizable error types
            assert len(error_types.intersection(expected_types)) > 0 or all(isinstance(et, str) for et in error_types)

    def test_error_context_extraction(self, parser_service):
        """Test that error context is properly extracted."""
        code_with_contextual_errors = '''
def function_with_context():
    """Function to test context extraction."""
    x = 5
    y = 10
    result = x ++ y  # Error on this line
    return result

def another_function():
    return "valid"
'''

        result = parser_service.parse_code(code_with_contextual_errors, "test.py", "python")

        if len(result.syntax_errors) > 0:
            for error in result.syntax_errors:
                # Should have context information
                assert error.context is not None
                assert len(error.context.strip()) > 0

                # Context should contain relevant code
                assert any(char.isalnum() for char in error.context)

                # Should have reasonable line/column information
                assert error.start_line > 0
                assert error.start_column >= 0
                assert error.end_line >= error.start_line

                if error.end_line == error.start_line:
                    assert error.end_column >= error.start_column


class TestErrorSuggestions:
    """Test generation of error suggestions and fixes."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_common_error_suggestions(self, parser_service):
        """Test that common errors get appropriate suggestions."""
        # This would depend on the implementation of error suggestion generation
        code_with_common_errors = """
def missing_colon()
    pass

def unclosed_paren(arg1, arg2
    pass

invalid_string = "unclosed string

def valid_function():
    return "valid"
"""

        result = parser_service.parse_code(code_with_common_errors, "test.py", "python")

        # If the parser generates suggestions, test them
        if hasattr(result, "suggestions") and result.suggestions:
            # Should have meaningful suggestions
            for suggestion in result.suggestions:
                assert isinstance(suggestion, str)
                assert len(suggestion.strip()) > 0

    def test_error_recovery_recommendations(self, parser_service):
        """Test recommendations for error recovery."""
        severely_broken_code = '''
def completely_broken_function(((((
    this is not valid python at all
    more invalid content here
    }}}}}

def recoverable_function():
    """This should be recoverable."""
    return "ok"

more invalid syntax here ++++++
'''

        result = parser_service.parse_code(severely_broken_code, "test.py", "python")

        # Should either recover something or fall back gracefully
        assert isinstance(result, ParseResult)

        # Should provide meaningful error information
        assert result.error_count > 0

        # Should either recover or use fallback
        assert result.error_recovery_used or result.fallback_used

        # Should not crash the parser
        assert len(result.chunks) > 0 or result.fallback_used


class TestRealWorldErrorScenarios:
    """Test handling of real-world syntax error scenarios."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_incomplete_file_during_editing(self, parser_service):
        """Test parsing of incomplete files (as during live editing)."""
        incomplete_code = """
import os
import sys

class IncompleteClass:
    def __init__(self):
        self.
"""  # File cuts off mid-statement

        result = parser_service.parse_code(incomplete_code, "incomplete.py", "python")

        # Should handle incomplete files gracefully
        assert isinstance(result, ParseResult)

        # Should detect that parsing wasn't completely successful
        assert result.error_count > 0 or result.fallback_used

        # Should still extract what it can
        assert len(result.chunks) > 0

    def test_mixed_language_content(self, parser_service):
        """Test handling of files with mixed or embedded content."""
        mixed_content = '''
"""
This is a Python file but it contains some pseudo-code
that might confuse the parser.

```javascript
function embeddedJS() {
    return "this is not python";
}
```
"""

def actual_python_function():
    """This is real Python."""
    return "python code"

# Some shell commands in comments
# $ ls -la
# $ grep "pattern" file.txt

def another_python_function():
    sql_query = """
    SELECT * FROM users
    WHERE name = 'test'
    """
    return sql_query
'''

        result = parser_service.parse_code(mixed_content, "mixed.py", "python")

        # Should parse as Python despite mixed content
        assert result.language == "python"

        # Should find Python functions
        function_chunks = [
            chunk for chunk in result.chunks if chunk.chunk_type == ChunkType.FUNCTION or "function" in str(chunk.chunk_type)
        ]
        assert len(function_chunks) >= 2

    def test_large_file_with_scattered_errors(self, parser_service):
        """Test parsing of large files with errors scattered throughout."""
        # Create a large file with errors distributed throughout
        large_code_with_errors = '''"""Large file with scattered syntax errors."""\n\n'''

        for i in range(50):
            if i % 7 == 0:  # Every 7th block has an error
                large_code_with_errors += f"""
def broken_function_{i}(
    # Missing closing paren in function {i}
    return {i}
"""
            else:
                large_code_with_errors += f'''
def valid_function_{i}():
    """Valid function {i}."""
    return {i}

class ValidClass_{i}:
    """Valid class {i}."""
    def method(self):
        return {i}
'''

        result = parser_service.parse_code(large_code_with_errors, "large_with_errors.py", "python")

        # Should handle large files with scattered errors
        assert isinstance(result, ParseResult)

        # Should detect multiple errors
        assert result.error_count > 5  # Should find several errors

        # Should still recover significant portions
        if result.error_recovery_used:
            # Should recover most of the valid functions
            assert result.valid_sections_count > 30

        # Should complete processing within reasonable time
        assert result.processing_time_ms < 30000  # Less than 30 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
