"""
Tests for TreeSitterManager utility class.

This module contains unit tests for the TreeSitterManager class,
verifying parser initialization, language detection, and error handling.
"""

from unittest.mock import patch

import pytest
from tree_sitter_manager import TreeSitterManager


class TestTreeSitterManager:
    """Test cases for TreeSitterManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_initialization(self):
        """Test that the manager initializes successfully."""
        assert isinstance(self.manager, TreeSitterManager)
        assert hasattr(self.manager, "_languages")
        assert hasattr(self.manager, "_parsers")
        assert hasattr(self.manager, "_failed_languages")

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = self.manager.get_supported_languages()
        assert isinstance(languages, list)
        # Should include common languages if they initialized successfully
        expected_languages = {
            "python",
            "javascript",
            "typescript",
            "tsx",
            "go",
            "rust",
            "java",
        }
        supported_set = set(languages)
        # At least some languages should be supported
        assert len(supported_set.intersection(expected_languages)) > 0

    def test_get_failed_languages(self):
        """Test getting list of failed languages."""
        failed = self.manager.get_failed_languages()
        assert isinstance(failed, list)
        # Failed languages should be subset of all configured languages
        all_configured = set(self.manager._language_configs.keys())
        failed_set = set(failed)
        assert failed_set.issubset(all_configured)

    def test_detect_language_from_extension(self):
        """Test language detection from file extensions."""
        test_cases = [
            ("test.py", "python"),
            ("script.js", "javascript"),
            ("component.jsx", "javascript"),
            ("app.ts", "typescript"),
            ("component.tsx", "tsx"),
            ("main.go", "go"),
            ("lib.rs", "rust"),
            ("Main.java", "java"),
            ("unknown.xyz", None),
        ]

        for file_path, expected_lang in test_cases:
            detected = self.manager.detect_language_from_extension(file_path)
            if expected_lang is None:
                assert detected is None, f"Expected None for {file_path}, got {detected}"
            else:
                # Only assert if the language was successfully initialized
                if expected_lang in self.manager.get_supported_languages():
                    assert detected == expected_lang, f"Expected {expected_lang} for {file_path}, got {detected}"

    def test_get_parser(self):
        """Test getting parser instances."""
        supported_languages = self.manager.get_supported_languages()

        for lang in supported_languages:
            parser = self.manager.get_parser(lang)
            assert parser is not None, f"Parser for {lang} should not be None"
            # Verify it's actually a parser-like object
            assert hasattr(parser, "parse"), f"Parser for {lang} should have parse method"

        # Test unsupported language
        unsupported_parser = self.manager.get_parser("nonexistent")
        assert unsupported_parser is None

    def test_get_language(self):
        """Test getting Language instances."""
        supported_languages = self.manager.get_supported_languages()

        for lang in supported_languages:
            language = self.manager.get_language(lang)
            assert language is not None, f"Language for {lang} should not be None"
            # Verify it's actually a Language object
            assert hasattr(language, "version"), f"Language for {lang} should have version"

        # Test unsupported language
        unsupported_language = self.manager.get_language("nonexistent")
        assert unsupported_language is None

    def test_is_language_supported(self):
        """Test checking if languages are supported."""
        supported_languages = self.manager.get_supported_languages()

        for lang in supported_languages:
            assert self.manager.is_language_supported(lang), f"{lang} should be supported"

        assert not self.manager.is_language_supported("nonexistent")
        assert not self.manager.is_language_supported("")

    def test_get_language_info(self):
        """Test getting detailed language information."""
        supported_languages = self.manager.get_supported_languages()

        for lang in supported_languages:
            info = self.manager.get_language_info(lang)
            assert info is not None, f"Info for {lang} should not be None"
            assert "name" in info
            assert "version" in info
            assert "extensions" in info
            assert "module" in info
            assert "function" in info
            assert "parser_available" in info
            assert info["name"] == lang
            assert info["parser_available"] is True
            assert isinstance(info["extensions"], list)
            assert len(info["extensions"]) > 0

        # Test unsupported language
        unsupported_info = self.manager.get_language_info("nonexistent")
        assert unsupported_info is None

    def test_get_all_extensions(self):
        """Test getting all supported file extensions."""
        extensions = self.manager.get_all_extensions()
        assert isinstance(extensions, set)

        # Should include common extensions for supported languages
        supported_languages = self.manager.get_supported_languages()

        # Check that extensions for supported languages are included
        for lang in supported_languages:
            config = self.manager._language_configs.get(lang)
            if config:
                for ext in config["extensions"]:
                    assert ext in extensions, f"Extension {ext} for {lang} should be in supported extensions"

    def test_get_initialization_summary(self):
        """Test getting initialization summary."""
        summary = self.manager.get_initialization_summary()

        assert isinstance(summary, dict)
        required_keys = [
            "total_languages",
            "successful_languages",
            "failed_languages",
            "success_rate",
            "supported_languages",
            "failed_languages",
            "supported_extensions",
        ]

        for key in required_keys:
            assert key in summary, f"Summary should contain {key}"

        # Verify data consistency
        assert summary["total_languages"] > 0
        assert summary["successful_languages"] >= 0
        assert summary["failed_languages"] >= 0
        assert summary["successful_languages"] + summary["failed_languages"] == summary["total_languages"]
        assert 0 <= summary["success_rate"] <= 1
        assert isinstance(summary["supported_languages"], list)
        assert isinstance(summary["failed_languages"], list)
        assert isinstance(summary["supported_extensions"], list)

    def test_parser_functionality(self):
        """Test that parsers can actually parse code."""
        test_cases = [
            ("python", "def hello(): pass"),
            ("javascript", "function hello() {}"),
            ("go", "func hello() {}"),
            ("java", "public class Test {}"),
        ]

        for lang, code in test_cases:
            if self.manager.is_language_supported(lang):
                parser = self.manager.get_parser(lang)
                assert parser is not None

                # Test parsing
                tree = parser.parse(bytes(code, "utf8"))
                assert tree is not None
                assert tree.root_node is not None
                assert tree.root_node.type is not None

    @patch("tree_sitter_manager.__import__")
    def test_failed_language_initialization(self, mock_import):
        """Test handling of failed language initialization."""
        # Mock import to raise ImportError
        mock_import.side_effect = ImportError("Module not found")

        # Create new manager to trigger initialization
        manager = TreeSitterManager()

        # All languages should be in failed set
        failed = manager.get_failed_languages()
        assert len(failed) > 0

        # No languages should be supported
        supported = manager.get_supported_languages()
        assert len(supported) == 0

    def test_reinitialize_language(self):
        """Test reinitializing a language."""
        # Test reinitializing an existing language
        if "python" in self.manager.get_supported_languages():
            result = self.manager.reinitialize_language("python")
            assert result is True
            assert "python" in self.manager.get_supported_languages()

        # Test reinitializing a nonexistent language
        result = self.manager.reinitialize_language("nonexistent")
        assert result is False


# Integration tests
class TestTreeSitterManagerIntegration:
    """Integration tests for TreeSitterManager with actual files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TreeSitterManager()

    def test_parse_real_python_code(self):
        """Test parsing actual Python code."""
        if not self.manager.is_language_supported("python"):
            pytest.skip("Python parser not available")

        python_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
'''

        parser = self.manager.get_parser("python")
        tree = parser.parse(bytes(python_code, "utf8"))

        assert tree.root_node.type == "module"

        # Find function and class definitions
        functions = []
        classes = []

        def traverse(node):
            if node.type == "function_definition":
                functions.append(node)
            elif node.type == "class_definition":
                classes.append(node)
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)

        assert len(functions) >= 2  # fibonacci and add methods
        assert len(classes) == 1  # Calculator class

    def test_parse_real_javascript_code(self):
        """Test parsing actual JavaScript code."""
        if not self.manager.is_language_supported("javascript"):
            pytest.skip("JavaScript parser not available")

        js_code = """
function greet(name) {
    return `Hello, ${name}!`;
}

const add = (a, b) => a + b;

class Person {
    constructor(name) {
        this.name = name;
    }

    sayHello() {
        return greet(this.name);
    }
}
"""

        parser = self.manager.get_parser("javascript")
        tree = parser.parse(bytes(js_code, "utf8"))

        assert tree.root_node.type == "program"

        # Verify we can parse complex JavaScript constructs
        assert tree.root_node.text.decode("utf8") == js_code


if __name__ == "__main__":
    # Run basic smoke test
    manager = TreeSitterManager()
    summary = manager.get_initialization_summary()

    print("TreeSitterManager Smoke Test")
    print("=" * 30)
    print(f"Total languages: {summary['total_languages']}")
    print(f"Successful: {summary['successful_languages']}")
    print(f"Failed: {summary['failed_languages']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Supported languages: {', '.join(summary['supported_languages'])}")
    if summary["failed_languages"]:
        print(f"Failed languages: {', '.join(summary['failed_languages'])}")
    print(f"Supported extensions: {', '.join(summary['supported_extensions'])}")

    # Test parsing a simple example for each supported language
    test_code = {
        "python": "def test(): pass",
        "javascript": "function test() {}",
        "typescript": "function test(): void {}",
        "tsx": "const element = <div></div>;",
        "go": "func test() {}",
        "rust": "fn test() {}",
        "java": "public class Test {}",
    }

    print("\nParsing Tests:")
    print("-" * 15)
    for lang in summary["supported_languages"]:
        if lang in test_code:
            parser = manager.get_parser(lang)
            try:
                tree = parser.parse(bytes(test_code[lang], "utf8"))
                print(f"✅ {lang}: {tree.root_node.type}")
            except Exception as e:
                print(f"❌ {lang}: {e}")
