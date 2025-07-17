"""
Structure Validator utility for validating and normalizing breadcrumb and parent_name data.

This module provides comprehensive validation and normalization mechanisms for
Graph RAG structure fields, ensuring data consistency and quality across the
codebase indexing and retrieval pipeline.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from src.models.code_chunk import ChunkType, CodeChunk


@dataclass
class ValidationResult:
    """Result of structure validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    normalized_data: dict[str, Any] | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    @property
    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation found any warnings."""
        return len(self.warnings) > 0

    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)


class StructureValidator:
    """
    Comprehensive validator for code structure fields.

    This class provides validation and normalization for breadcrumb and parent_name
    fields, ensuring consistency and quality for Graph RAG functionality.
    """

    # Language-specific separator patterns
    SEPARATORS = {
        "python": ".",
        "javascript": ".",
        "typescript": ".",
        "java": ".",
        "cpp": "::",
        "c": "::",
        "rust": "::",
        "go": ".",
    }

    # Valid identifier patterns by language
    IDENTIFIER_PATTERNS = {
        "python": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        "javascript": r"^[a-zA-Z_$][a-zA-Z0-9_$]*$",
        "typescript": r"^[a-zA-Z_$][a-zA-Z0-9_$]*$",
        "java": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        "cpp": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        "c": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        "rust": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        "go": r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    }

    # Reserved keywords that shouldn't appear as names
    RESERVED_KEYWORDS = {
        "python": {"def", "class", "if", "else", "for", "while", "import", "from", "as", "with", "try", "except"},
        "javascript": {"function", "class", "if", "else", "for", "while", "import", "export", "const", "let", "var"},
        "typescript": {"function", "class", "if", "else", "for", "while", "import", "export", "const", "let", "var", "interface", "type"},
        "java": {"class", "interface", "if", "else", "for", "while", "import", "package", "public", "private", "protected"},
        "cpp": {"class", "struct", "if", "else", "for", "while", "namespace", "public", "private", "protected"},
        "c": {"struct", "if", "else", "for", "while", "static", "extern", "typedef"},
        "rust": {"fn", "struct", "enum", "impl", "trait", "if", "else", "for", "while", "mod", "use", "pub"},
        "go": {"func", "struct", "interface", "if", "else", "for", "range", "package", "import", "type"},
    }

    def __init__(self):
        """Initialize the structure validator."""
        self.logger = logging.getLogger(__name__)

        # Validation statistics
        self._validation_stats = {
            "chunks_validated": 0,
            "validation_errors": 0,
            "validation_warnings": 0,
            "normalizations_applied": 0,
            "language_breakdown": {},
        }

    def validate_chunk_structure(self, chunk: CodeChunk, strict: bool = False) -> ValidationResult:
        """
        Validate the structure fields of a CodeChunk.

        Args:
            chunk: CodeChunk to validate
            strict: If True, apply stricter validation rules

        Returns:
            ValidationResult with validation outcome and normalized data
        """
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        try:
            # Update statistics
            self._validation_stats["chunks_validated"] += 1
            language = chunk.language.lower() if chunk.language else "unknown"

            if language not in self._validation_stats["language_breakdown"]:
                self._validation_stats["language_breakdown"][language] = 0
            self._validation_stats["language_breakdown"][language] += 1

            # Initialize normalized data
            normalized_data = {}

            # Validate breadcrumb
            breadcrumb_result = self._validate_breadcrumb(chunk.breadcrumb, language, strict)
            if breadcrumb_result["errors"]:
                result.errors.extend(breadcrumb_result["errors"])
            if breadcrumb_result["warnings"]:
                result.warnings.extend(breadcrumb_result["warnings"])
            if breadcrumb_result["normalized"] != chunk.breadcrumb:
                normalized_data["breadcrumb"] = breadcrumb_result["normalized"]
                self._validation_stats["normalizations_applied"] += 1

            # Validate parent_name
            parent_result = self._validate_parent_name(chunk.parent_name, language, strict)
            if parent_result["errors"]:
                result.errors.extend(parent_result["errors"])
            if parent_result["warnings"]:
                result.warnings.extend(parent_result["warnings"])
            if parent_result["normalized"] != chunk.parent_name:
                normalized_data["parent_name"] = parent_result["normalized"]
                self._validation_stats["normalizations_applied"] += 1

            # Validate consistency between breadcrumb and parent_name
            consistency_result = self._validate_consistency(chunk, strict)
            if consistency_result["errors"]:
                result.errors.extend(consistency_result["errors"])
            if consistency_result["warnings"]:
                result.warnings.extend(consistency_result["warnings"])

            # Validate chunk name consistency
            name_result = self._validate_name_consistency(chunk, strict)
            if name_result["errors"]:
                result.errors.extend(name_result["errors"])
            if name_result["warnings"]:
                result.warnings.extend(name_result["warnings"])

            # Set final validation state
            result.is_valid = len(result.errors) == 0
            result.normalized_data = normalized_data if normalized_data else None

            # Update statistics
            if result.errors:
                self._validation_stats["validation_errors"] += len(result.errors)
            if result.warnings:
                self._validation_stats["validation_warnings"] += len(result.warnings)

            return result

        except Exception as e:
            self.logger.error(f"Error validating chunk structure: {e}")
            result.add_error(f"Validation error: {str(e)}")
            return result

    def normalize_breadcrumb(self, breadcrumb: str, language: str) -> str:
        """
        Normalize a breadcrumb string according to language conventions.

        Args:
            breadcrumb: Original breadcrumb string
            language: Programming language

        Returns:
            Normalized breadcrumb string
        """
        if not breadcrumb:
            return breadcrumb

        normalized = breadcrumb.strip()

        # Remove leading/trailing separators
        separators = [".", "::"]
        for sep in separators:
            normalized = normalized.strip(sep)

        # Replace spaces with underscores
        normalized = re.sub(r"\s+", "_", normalized)

        # Ensure consistent separator usage
        expected_separator = self.SEPARATORS.get(language.lower(), ".")

        if expected_separator == "::":
            # Convert dots to double colons for C++/Rust
            normalized = normalized.replace(".", "::")
        else:
            # Convert double colons to dots for other languages
            normalized = normalized.replace("::", ".")

        # Remove consecutive separators
        if expected_separator == "::":
            normalized = re.sub(r"::+", "::", normalized)
        else:
            normalized = re.sub(r"\.+", ".", normalized)

        return normalized

    def normalize_parent_name(self, parent_name: str, language: str) -> str:
        """
        Normalize a parent name according to language conventions.

        Args:
            parent_name: Original parent name
            language: Programming language

        Returns:
            Normalized parent name
        """
        if not parent_name:
            return parent_name

        normalized = parent_name.strip()

        # Replace spaces with underscores
        normalized = re.sub(r"\s+", "_", normalized)

        # Remove any separator characters (parent names should be simple identifiers)
        normalized = normalized.replace(".", "_").replace("::", "_")

        return normalized

    def apply_normalizations(self, chunk: CodeChunk, validation_result: ValidationResult) -> CodeChunk:
        """
        Apply normalizations to a CodeChunk based on validation results.

        Args:
            chunk: CodeChunk to normalize
            validation_result: Validation result with normalization data

        Returns:
            Normalized CodeChunk
        """
        if not validation_result.normalized_data:
            return chunk

        # Apply normalizations
        for field, normalized_value in validation_result.normalized_data.items():
            if hasattr(chunk, field):
                setattr(chunk, field, normalized_value)
                self.logger.debug(f"Applied normalization: {field} = {normalized_value}")

        return chunk

    def _validate_breadcrumb(self, breadcrumb: str, language: str, strict: bool) -> dict[str, Any]:
        """Validate a breadcrumb string."""
        result = {"errors": [], "warnings": [], "normalized": breadcrumb}

        if not breadcrumb:
            return result

        # Normalize the breadcrumb
        normalized = self.normalize_breadcrumb(breadcrumb, language)
        result["normalized"] = normalized

        # Check for empty components
        separator = self.SEPARATORS.get(language.lower(), ".")
        components = normalized.split(separator)

        if any(not comp.strip() for comp in components):
            result["errors"].append("Breadcrumb contains empty components")

        # Validate each component as a valid identifier
        identifier_pattern = self.IDENTIFIER_PATTERNS.get(language.lower())
        if identifier_pattern:
            pattern = re.compile(identifier_pattern)
            for component in components:
                if component and not pattern.match(component):
                    if strict:
                        result["errors"].append(f"Invalid identifier in breadcrumb: '{component}'")
                    else:
                        result["warnings"].append(f"Potentially invalid identifier: '{component}'")

        # Check for reserved keywords
        reserved = self.RESERVED_KEYWORDS.get(language.lower(), set())
        for component in components:
            if component.lower() in reserved:
                result["warnings"].append(f"Reserved keyword in breadcrumb: '{component}'")

        # Check length limits
        if len(normalized) > 500:  # Arbitrary but reasonable limit
            result["warnings"].append("Breadcrumb is very long (>500 characters)")

        if len(components) > 10:  # Deep nesting warning
            result["warnings"].append(f"Deep nesting detected ({len(components)} levels)")

        return result

    def _validate_parent_name(self, parent_name: str, language: str, strict: bool) -> dict[str, Any]:
        """Validate a parent name."""
        result = {"errors": [], "warnings": [], "normalized": parent_name}

        if not parent_name:
            return result

        # Normalize the parent name
        normalized = self.normalize_parent_name(parent_name, language)
        result["normalized"] = normalized

        # Check if it's a valid identifier
        identifier_pattern = self.IDENTIFIER_PATTERNS.get(language.lower())
        if identifier_pattern:
            pattern = re.compile(identifier_pattern)
            if not pattern.match(normalized):
                if strict:
                    result["errors"].append(f"Invalid parent name identifier: '{normalized}'")
                else:
                    result["warnings"].append(f"Potentially invalid parent name: '{normalized}'")

        # Check for reserved keywords
        reserved = self.RESERVED_KEYWORDS.get(language.lower(), set())
        if normalized.lower() in reserved:
            result["warnings"].append(f"Reserved keyword used as parent name: '{normalized}'")

        # Check length
        if len(normalized) > 100:  # Reasonable limit for single identifier
            result["warnings"].append("Parent name is very long (>100 characters)")

        return result

    def _validate_consistency(self, chunk: CodeChunk, strict: bool) -> dict[str, Any]:
        """Validate consistency between breadcrumb and parent_name."""
        result = {"errors": [], "warnings": []}

        if not chunk.breadcrumb or not chunk.parent_name:
            return result

        # Extract parent from breadcrumb
        language = chunk.language.lower() if chunk.language else "unknown"
        separator = self.SEPARATORS.get(language, ".")
        breadcrumb_components = chunk.breadcrumb.split(separator)

        if len(breadcrumb_components) >= 2:
            expected_parent = breadcrumb_components[-2]
            if chunk.parent_name != expected_parent:
                if strict:
                    result["errors"].append(f"Parent name '{chunk.parent_name}' doesn't match breadcrumb parent '{expected_parent}'")
                else:
                    result["warnings"].append(f"Parent name mismatch: '{chunk.parent_name}' vs breadcrumb '{expected_parent}'")

        return result

    def _validate_name_consistency(self, chunk: CodeChunk, strict: bool) -> dict[str, Any]:
        """Validate consistency between chunk name and breadcrumb."""
        result = {"errors": [], "warnings": []}

        if not chunk.name or not chunk.breadcrumb:
            return result

        # Extract name from breadcrumb
        language = chunk.language.lower() if chunk.language else "unknown"
        separator = self.SEPARATORS.get(language, ".")
        breadcrumb_components = chunk.breadcrumb.split(separator)

        if breadcrumb_components:
            breadcrumb_name = breadcrumb_components[-1]
            if chunk.name != breadcrumb_name:
                if strict:
                    result["errors"].append(f"Chunk name '{chunk.name}' doesn't match breadcrumb tail '{breadcrumb_name}'")
                else:
                    result["warnings"].append(f"Name mismatch: chunk '{chunk.name}' vs breadcrumb '{breadcrumb_name}'")

        return result

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation statistics."""
        stats = self._validation_stats.copy()

        # Calculate rates
        if stats["chunks_validated"] > 0:
            stats["error_rate"] = stats["validation_errors"] / stats["chunks_validated"]
            stats["warning_rate"] = stats["validation_warnings"] / stats["chunks_validated"]
            stats["normalization_rate"] = stats["normalizations_applied"] / stats["chunks_validated"]
        else:
            stats["error_rate"] = 0.0
            stats["warning_rate"] = 0.0
            stats["normalization_rate"] = 0.0

        return stats

    def reset_statistics(self):
        """Reset validation statistics."""
        self._validation_stats = {
            "chunks_validated": 0,
            "validation_errors": 0,
            "validation_warnings": 0,
            "normalizations_applied": 0,
            "language_breakdown": {},
        }


# Singleton instance for global access
_structure_validator_instance: StructureValidator | None = None


def get_structure_validator() -> StructureValidator:
    """Get the global structure validator instance."""
    global _structure_validator_instance
    if _structure_validator_instance is None:
        _structure_validator_instance = StructureValidator()
    return _structure_validator_instance


def validate_and_normalize_chunk(chunk: CodeChunk, strict: bool = False) -> tuple[CodeChunk, ValidationResult]:
    """
    Convenience function to validate and normalize a chunk.

    Args:
        chunk: CodeChunk to validate and normalize
        strict: Whether to apply strict validation rules

    Returns:
        Tuple of (normalized_chunk, validation_result)
    """
    validator = get_structure_validator()
    validation_result = validator.validate_chunk_structure(chunk, strict)
    normalized_chunk = validator.apply_normalizations(chunk, validation_result)
    return normalized_chunk, validation_result
