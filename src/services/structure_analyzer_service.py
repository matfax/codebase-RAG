"""
Structure Analyzer Service for Graph RAG enhancement.

This service integrates breadcrumb extraction and parent_name identification
to provide comprehensive code structure analysis. It works with the existing
code parsing pipeline to enhance CodeChunk objects with hierarchical
relationship metadata.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

try:
    from tree_sitter import Node
except ImportError:
    Node = Any  # Fallback for type hints when tree-sitter not available

from src.models.code_chunk import ChunkType, CodeChunk
from src.utils.breadcrumb_extractor import (
    BreadcrumbContext,
    BreadcrumbExtractorFactory,
    StructureInfo,
    build_module_breadcrumb,
)
from src.utils.structure_validator import get_structure_validator


@dataclass
class StructureAnalysisResult:
    """Result of structure analysis for a code chunk."""

    breadcrumb: str | None = None
    parent_name: str | None = None
    structure_type: str | None = None
    breadcrumb_depth: int = 0
    breadcrumb_components: list[str] = None
    is_nested: bool = False
    validation_errors: list[str] = None

    def __post_init__(self):
        if self.breadcrumb_components is None:
            self.breadcrumb_components = []
        if self.validation_errors is None:
            self.validation_errors = []


@dataclass
class FileStructureContext:
    """Context information for analyzing file structure."""

    file_path: str
    language: str
    content: str
    content_lines: list[str]
    module_breadcrumb: str | None = None

    def __post_init__(self):
        if not self.content_lines:
            self.content_lines = self.content.splitlines()
        if self.module_breadcrumb is None:
            self.module_breadcrumb = build_module_breadcrumb(self.file_path, self.language)


class StructureAnalyzerService:
    """
    Service for analyzing code structure and extracting hierarchical relationships.

    This service integrates with the breadcrumb extraction system to provide
    comprehensive structure analysis for CodeChunk objects, supporting Graph RAG
    functionality with accurate parent-child relationships and breadcrumb paths.
    """

    def __init__(self):
        """Initialize the structure analyzer service."""
        self.logger = logging.getLogger(__name__)
        self._supported_languages = BreadcrumbExtractorFactory.get_supported_languages()
        self._validator = get_structure_validator()

        # Statistics tracking
        self._analysis_stats = {
            "chunks_analyzed": 0,
            "breadcrumbs_extracted": 0,
            "parent_names_identified": 0,
            "validation_errors": 0,
            "language_breakdown": {},
        }

        self.logger.info(f"StructureAnalyzer initialized with support for {len(self._supported_languages)} languages")

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported for structure analysis."""
        return language.lower() in self._supported_languages

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        return self._supported_languages.copy()

    def analyze_chunk_structure(
        self, chunk: CodeChunk, ast_node: Node | None = None, file_context: FileStructureContext | None = None
    ) -> StructureAnalysisResult:
        """
        Analyze the structure of a code chunk to extract breadcrumb and parent information.

        Args:
            chunk: CodeChunk to analyze
            ast_node: Optional Tree-sitter AST node for the chunk
            file_context: Optional file context information

        Returns:
            StructureAnalysisResult with extracted structure information
        """
        try:
            # Update statistics
            self._analysis_stats["chunks_analyzed"] += 1
            language = chunk.language.lower()

            if language not in self._analysis_stats["language_breakdown"]:
                self._analysis_stats["language_breakdown"][language] = 0
            self._analysis_stats["language_breakdown"][language] += 1

            # Check if language is supported
            if not self.is_language_supported(language):
                self.logger.debug(f"Language {language} not supported for structure analysis")
                return StructureAnalysisResult()

            # Create or validate file context
            if file_context is None:
                file_context = FileStructureContext(
                    file_path=chunk.file_path, language=chunk.language, content=chunk.content, content_lines=chunk.content.splitlines()
                )

            # Extract structure information using breadcrumb extractor
            if ast_node is not None:
                structure_info = self._extract_from_ast_node(ast_node, file_context)
            else:
                structure_info = self._extract_from_chunk_content(chunk, file_context)

            # Build analysis result
            result = self._build_analysis_result(chunk, structure_info, file_context)

            # Validate and normalize the result
            self._validate_and_normalize_result(result, chunk)

            # Update statistics
            if result.breadcrumb:
                self._analysis_stats["breadcrumbs_extracted"] += 1
            if result.parent_name:
                self._analysis_stats["parent_names_identified"] += 1
            if result.validation_errors:
                self._analysis_stats["validation_errors"] += len(result.validation_errors)

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing chunk structure for {chunk.file_path}:{chunk.start_line}: {e}")
            return StructureAnalysisResult(validation_errors=[f"Analysis error: {str(e)}"])

    def analyze_file_structure(self, file_path: str, language: str, content: str) -> FileStructureContext:
        """
        Analyze the overall structure of a file to provide context for chunk analysis.

        Args:
            file_path: Path to the source file
            language: Programming language
            content: File content

        Returns:
            FileStructureContext with file-level structure information
        """
        try:
            context = FileStructureContext(file_path=file_path, language=language, content=content, content_lines=content.splitlines())

            self.logger.debug(f"Analyzed file structure: {file_path} -> module: {context.module_breadcrumb}")
            return context

        except Exception as e:
            self.logger.error(f"Error analyzing file structure for {file_path}: {e}")
            return FileStructureContext(
                file_path=file_path, language=language, content=content, content_lines=content.splitlines() if content else []
            )

    def enhance_chunk_with_structure(
        self, chunk: CodeChunk, ast_node: Node | None = None, file_context: FileStructureContext | None = None
    ) -> CodeChunk:
        """
        Enhance a CodeChunk with structure analysis results.

        Args:
            chunk: CodeChunk to enhance
            ast_node: Optional Tree-sitter AST node
            file_context: Optional file context

        Returns:
            Enhanced CodeChunk with updated structure fields
        """
        analysis_result = self.analyze_chunk_structure(chunk, ast_node, file_context)

        # Update chunk with analysis results
        if analysis_result.breadcrumb:
            chunk.breadcrumb = analysis_result.breadcrumb

        if analysis_result.parent_name:
            chunk.parent_name = analysis_result.parent_name

        # Apply validation and normalization
        validation_result = self._validator.validate_chunk_structure(chunk, strict=False)
        if validation_result.normalized_data:
            chunk = self._validator.apply_normalizations(chunk, validation_result)

        # Log any validation issues
        if validation_result.has_errors:
            self.logger.warning(f"Validation errors for chunk {chunk.name} in {chunk.file_path}: " f"{', '.join(validation_result.errors)}")

        if validation_result.has_warnings:
            self.logger.debug(
                f"Validation warnings for chunk {chunk.name} in {chunk.file_path}: " f"{', '.join(validation_result.warnings)}"
            )

        # Log any original analysis validation errors
        if analysis_result.validation_errors:
            self.logger.debug(
                f"Analysis validation errors for chunk {chunk.name} in {chunk.file_path}: "
                f"{', '.join(analysis_result.validation_errors)}"
            )

        return chunk

    def batch_analyze_chunks(self, chunks: list[CodeChunk], file_context: FileStructureContext | None = None) -> list[CodeChunk]:
        """
        Analyze structure for multiple chunks from the same file.

        Args:
            chunks: List of CodeChunk objects to analyze
            file_context: Optional shared file context

        Returns:
            List of enhanced CodeChunk objects
        """
        if not chunks:
            return []

        # Create file context if not provided
        if file_context is None and chunks:
            first_chunk = chunks[0]
            file_context = self.analyze_file_structure(first_chunk.file_path, first_chunk.language, first_chunk.content)

        # Enhance each chunk
        enhanced_chunks = []
        for chunk in chunks:
            try:
                enhanced_chunk = self.enhance_chunk_with_structure(chunk, None, file_context)
                enhanced_chunks.append(enhanced_chunk)
            except Exception as e:
                self.logger.error(f"Error enhancing chunk {chunk.name}: {e}")
                enhanced_chunks.append(chunk)  # Return original chunk on error

        return enhanced_chunks

    def _extract_from_ast_node(self, ast_node: Node, file_context: FileStructureContext) -> StructureInfo | None:
        """Extract structure information from a Tree-sitter AST node."""
        extractor = BreadcrumbExtractorFactory.create_extractor(file_context.language)
        if not extractor:
            return None

        # Create breadcrumb context
        breadcrumb_context = BreadcrumbContext(
            file_path=file_context.file_path,
            language=file_context.language,
            content=file_context.content,
            content_lines=file_context.content_lines,
        )

        return extractor.extract_structure_info(ast_node, breadcrumb_context)

    def _extract_from_chunk_content(self, chunk: CodeChunk, file_context: FileStructureContext) -> StructureInfo | None:
        """Extract structure information from chunk content (fallback method)."""
        # For chunks without AST nodes, try to infer structure from chunk metadata
        if chunk.name and chunk.chunk_type:
            # Build basic structure info from chunk metadata
            separator = "::" if file_context.language in ["cpp", "c", "rust"] else "."

            structure_info = StructureInfo(name=chunk.name, structure_type=chunk.chunk_type.value, separator=separator)

            # Try to build breadcrumb from existing parent_name
            if chunk.parent_name:
                structure_info.parent_name = chunk.parent_name
                structure_info.breadcrumb_components = [chunk.parent_name]

            # Add module context if available
            if file_context.module_breadcrumb and not structure_info.breadcrumb_components:
                structure_info.breadcrumb_components = [file_context.module_breadcrumb]

            return structure_info

        return None

    def _build_analysis_result(
        self, chunk: CodeChunk, structure_info: StructureInfo | None, file_context: FileStructureContext
    ) -> StructureAnalysisResult:
        """Build a StructureAnalysisResult from extracted information."""
        if not structure_info:
            return StructureAnalysisResult()

        # Build breadcrumb
        breadcrumb = structure_info.build_breadcrumb()

        # Include module context if needed
        if file_context.module_breadcrumb and not structure_info.breadcrumb_components:
            breadcrumb = f"{file_context.module_breadcrumb}.{breadcrumb}"

        return StructureAnalysisResult(
            breadcrumb=breadcrumb,
            parent_name=structure_info.parent_name,
            structure_type=structure_info.structure_type,
            breadcrumb_depth=len(structure_info.breadcrumb_components) + 1,
            breadcrumb_components=structure_info.breadcrumb_components + [structure_info.name],
            is_nested=bool(structure_info.parent_name),
        )

    def _validate_and_normalize_result(self, result: StructureAnalysisResult, chunk: CodeChunk):
        """Validate and normalize the analysis result."""
        errors = []

        # Validate breadcrumb format
        if result.breadcrumb:
            # Check for invalid characters
            if " " in result.breadcrumb:
                errors.append("Breadcrumb contains spaces")
                # Normalize: replace spaces with underscores
                result.breadcrumb = result.breadcrumb.replace(" ", "_")

            # Check separator consistency
            if "." in result.breadcrumb and "::" in result.breadcrumb:
                errors.append("Mixed separators in breadcrumb")

            # Ensure breadcrumb ends with chunk name
            if chunk.name and result.breadcrumb and not result.breadcrumb.endswith(chunk.name):
                # Try to fix by appending chunk name
                separator = "::" if "::" in result.breadcrumb else "."
                result.breadcrumb = f"{result.breadcrumb}{separator}{chunk.name}"

        # Validate parent_name consistency
        if result.parent_name and result.breadcrumb_components:
            if len(result.breadcrumb_components) >= 2:
                expected_parent = result.breadcrumb_components[-2]
                if result.parent_name != expected_parent:
                    errors.append(f"Parent name mismatch: {result.parent_name} vs {expected_parent}")

        result.validation_errors = errors

    def get_analysis_statistics(self) -> dict[str, Any]:
        """Get structure analysis statistics."""
        stats = self._analysis_stats.copy()

        # Calculate success rates
        if stats["chunks_analyzed"] > 0:
            stats["breadcrumb_success_rate"] = stats["breadcrumbs_extracted"] / stats["chunks_analyzed"]
            stats["parent_name_success_rate"] = stats["parent_names_identified"] / stats["chunks_analyzed"]
        else:
            stats["breadcrumb_success_rate"] = 0.0
            stats["parent_name_success_rate"] = 0.0

        # Include validation statistics
        validation_stats = self._validator.get_validation_statistics()
        stats["validation"] = validation_stats

        return stats

    def reset_statistics(self):
        """Reset analysis statistics."""
        self._analysis_stats = {
            "chunks_analyzed": 0,
            "breadcrumbs_extracted": 0,
            "parent_names_identified": 0,
            "validation_errors": 0,
            "language_breakdown": {},
        }
        # Also reset validator statistics
        self._validator.reset_statistics()
        self.logger.info("Structure analysis statistics reset")


# Singleton instance for global access
_structure_analyzer_instance: StructureAnalyzerService | None = None


def get_structure_analyzer() -> StructureAnalyzerService:
    """Get the global structure analyzer service instance."""
    global _structure_analyzer_instance
    if _structure_analyzer_instance is None:
        _structure_analyzer_instance = StructureAnalyzerService()
    return _structure_analyzer_instance
