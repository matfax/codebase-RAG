"""
Tree-sitter manager utility for robust parser initialization and language management.

This module provides a centralized way to manage Tree-sitter parsers and languages,
handling version compatibility and initialization errors gracefully.
"""

import logging
from pathlib import Path
from typing import Union

try:
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")


class TreeSitterManager:
    """
    Centralized manager for Tree-sitter parsers and languages.

    This class handles the initialization of Tree-sitter parsers for different
    programming languages, manages version compatibility, and provides error recovery.
    """

    def __init__(self):
        """Initialize the TreeSitter manager."""
        self.logger = logging.getLogger(__name__)
        self._languages: dict[str, Language] = {}
        self._parsers: dict[str, Parser] = {}
        self._failed_languages: set[str] = set()

        # Language module mappings with their specific function names
        self._language_configs = {
            "python": {
                "module": "tree_sitter_python",
                "function": "language",
                "extensions": [".py", ".pyi"],
            },
            "javascript": {
                "module": "tree_sitter_javascript",
                "function": "language",
                "extensions": [".js", ".jsx", ".mjs", ".cjs"],
            },
            "typescript": {
                "module": "tree_sitter_typescript",
                "function": "language_typescript",
                "extensions": [".ts"],
            },
            "tsx": {
                "module": "tree_sitter_typescript",
                "function": "language_tsx",
                "extensions": [".tsx"],
            },
            "go": {
                "module": "tree_sitter_go",
                "function": "language",
                "extensions": [".go"],
            },
            "rust": {
                "module": "tree_sitter_rust",
                "function": "language",
                "extensions": [".rs"],
            },
            "java": {
                "module": "tree_sitter_java",
                "function": "language",
                "extensions": [".java"],
            },
            "cpp": {
                "module": "tree_sitter_cpp",
                "function": "language",
                "extensions": [
                    ".cpp",
                    ".cxx",
                    ".cc",
                    ".c",
                    ".hpp",
                    ".hxx",
                    ".hh",
                    ".h",
                ],
            },
        }

        # Initialize all available languages
        self._initialize_all_languages()

    def _initialize_all_languages(self) -> None:
        """Initialize all supported language parsers."""
        for lang_name, config in self._language_configs.items():
            try:
                self._initialize_language(lang_name, config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize {lang_name} parser: {e}")
                self._failed_languages.add(lang_name)

    def _initialize_language(self, lang_name: str, config: dict[str, str]) -> None:
        """
        Initialize a specific language parser.

        Args:
            lang_name: Name of the language (e.g., 'python', 'javascript')
            config: Configuration dict with module and function names
        """
        module_name = config["module"]
        function_name = config["function"]

        try:
            # Import the language module dynamically
            module = __import__(module_name)

            # Get the language function
            if not hasattr(module, function_name):
                raise AttributeError(f"Module {module_name} has no function {function_name}")

            language_func = getattr(module, function_name)

            # Create language object from PyCapsule
            capsule = language_func()
            language = Language(capsule)

            # Validate language object
            if not isinstance(language, Language):
                raise TypeError(f"Expected Language object, got {type(language)}")

            # Create parser with language
            parser = Parser(language)

            # Store successful initialization
            self._languages[lang_name] = language
            self._parsers[lang_name] = parser

            self.logger.info(f"Successfully initialized {lang_name} parser (version {language.version})")

        except ImportError as e:
            raise ImportError(f"Failed to import {module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {lang_name} parser: {e}")

    def get_supported_languages(self) -> list[str]:
        """
        Get list of successfully initialized languages.

        Returns:
            List of language names that are available for parsing
        """
        return list(self._parsers.keys())

    def get_failed_languages(self) -> list[str]:
        """
        Get list of languages that failed to initialize.

        Returns:
            List of language names that failed initialization
        """
        return list(self._failed_languages)

    def get_parser(self, language: str) -> Parser | None:
        """
        Get a parser for the specified language.

        Args:
            language: Language name (e.g., 'python', 'javascript')

        Returns:
            Parser instance if available, None otherwise
        """
        return self._parsers.get(language)

    def get_language(self, language: str) -> Language | None:
        """
        Get a Language object for the specified language.

        Args:
            language: Language name (e.g., 'python', 'javascript')

        Returns:
            Language instance if available, None otherwise
        """
        return self._languages.get(language)

    def detect_language_from_extension(self, file_path: str) -> str | None:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language name if supported, None otherwise
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        # Find language by extension
        for lang_name, config in self._language_configs.items():
            if extension in config["extensions"]:
                # Only return if the language was successfully initialized
                if lang_name in self._parsers:
                    return lang_name

        return None

    def is_language_supported(self, language: str) -> bool:
        """
        Check if a language is supported and available.

        Args:
            language: Language name to check

        Returns:
            True if the language parser is available, False otherwise
        """
        return language in self._parsers

    def get_language_info(self, language: str) -> dict[str, any] | None:
        """
        Get detailed information about a language.

        Args:
            language: Language name

        Returns:
            Dictionary with language information or None if not supported
        """
        if language not in self._parsers:
            return None

        lang_obj = self._languages[language]
        config = self._language_configs[language]

        return {
            "name": language,
            "version": lang_obj.version,
            "extensions": config["extensions"],
            "module": config["module"],
            "function": config["function"],
            "parser_available": True,
        }

    def get_all_extensions(self) -> set[str]:
        """
        Get all supported file extensions.

        Returns:
            Set of file extensions that can be parsed
        """
        extensions = set()
        for lang_name, config in self._language_configs.items():
            if lang_name in self._parsers:  # Only include successfully initialized languages
                extensions.update(config["extensions"])
        return extensions

    def reinitialize_language(self, language: str) -> bool:
        """
        Attempt to reinitialize a specific language parser.

        Args:
            language: Language name to reinitialize

        Returns:
            True if reinitialization succeeded, False otherwise
        """
        if language not in self._language_configs:
            self.logger.warning(f"Unknown language: {language}")
            return False

        # Remove from failed set if present
        self._failed_languages.discard(language)

        # Remove existing parser/language if present
        self._parsers.pop(language, None)
        self._languages.pop(language, None)

        try:
            config = self._language_configs[language]
            self._initialize_language(language, config)
            return True
        except Exception as e:
            self.logger.error(f"Failed to reinitialize {language}: {e}")
            self._failed_languages.add(language)
            return False

    def get_initialization_summary(self) -> dict[str, any]:
        """
        Get a summary of the initialization status.

        Returns:
            Dictionary with initialization statistics and details
        """
        total_languages = len(self._language_configs)
        successful_languages = len(self._parsers)
        failed_languages = len(self._failed_languages)

        return {
            "total_languages": total_languages,
            "successful_languages": successful_languages,
            "failed_languages_count": failed_languages,
            "success_rate": (successful_languages / total_languages if total_languages > 0 else 0),
            "supported_languages": list(self._parsers.keys()),
            "failed_languages": list(self._failed_languages),
            "supported_extensions": sorted(self.get_all_extensions()),
        }
