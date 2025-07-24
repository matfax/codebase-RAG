"""Language detection utility.

from typing import Union
This module provides language detection functionality for source code files.
"""

from .language_registry import LanguageRegistry


def detect_language(file_path: str) -> Union[str, None]:
    """Detect the programming language of a source file.

    Args:
        file_path: Path to the source file

    Returns:
        The detected language name or None if not detected
    """
    registry = LanguageRegistry()
    return registry.detect_language_from_extension(file_path)


def is_supported_language(file_path: str) -> bool:
    """Check if the file's language is supported for parsing.

    Args:
        file_path: Path to the source file

    Returns:
        True if the language is supported, False otherwise
    """
    return detect_language(file_path) is not None


def get_supported_extensions() -> list[str]:
    """Get all supported file extensions.

    Returns:
        List of supported file extensions
    """
    registry = LanguageRegistry()
    return registry.get_supported_extensions()
