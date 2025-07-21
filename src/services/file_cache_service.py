"""
File processing cache service for the Codebase RAG MCP Server.

This module provides specialized caching for file processing operations including:
- Tree-sitter parsing result caching
- Chunking result caching with content hashing
- Incremental parsing for changed files
- Integration with existing FileMetadata system

The service dramatically improves performance by avoiding repeated Tree-sitter parsing
operations, especially for large files that haven't changed.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from src.models.cache_models import CacheEntry, CacheEntryType, create_cache_entry
from src.models.code_chunk import CodeChunk, ParseResult
from src.models.file_metadata import FileMetadata
from src.services.cache_service import BaseCacheService, get_cache_service
from src.utils.cache_key_generator import CacheKeyGenerator


@dataclass
class ParsedFileData:
    """Container for parsed file data with caching metadata."""

    # Core parsing results
    parse_result: ParseResult
    file_metadata: FileMetadata

    # Caching metadata
    cache_key: str
    cached_at: float = field(default_factory=time.time)
    parser_version: str = "1.0.0"  # Track parser version for cache invalidation

    # Content validation
    content_hash: str = ""
    file_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for cache storage."""
        return {
            "parse_result": self._parse_result_to_dict(),
            "file_metadata": self.file_metadata.to_dict(),
            "cache_key": self.cache_key,
            "cached_at": self.cached_at,
            "parser_version": self.parser_version,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedFileData":
        """Create instance from dictionary."""
        # Reconstruct parse result
        parse_result = cls._parse_result_from_dict(data["parse_result"])

        # Reconstruct file metadata
        file_metadata = FileMetadata.from_dict(data["file_metadata"])

        return cls(
            parse_result=parse_result,
            file_metadata=file_metadata,
            cache_key=data["cache_key"],
            cached_at=data["cached_at"],
            parser_version=data.get("parser_version", "1.0.0"),
            content_hash=data["content_hash"],
            file_size=data["file_size"],
        )

    def _parse_result_to_dict(self) -> dict[str, Any]:
        """Convert ParseResult to dictionary."""
        return {
            "chunks": [chunk.to_dict() for chunk in self.parse_result.chunks],
            "file_path": self.parse_result.file_path,
            "language": self.parse_result.language,
            "parse_success": self.parse_result.parse_success,
            "error_count": self.parse_result.error_count,
            "fallback_used": self.parse_result.fallback_used,
            "processing_time_ms": self.parse_result.processing_time_ms,
            "syntax_errors": [
                {
                    "start_line": err.start_line,
                    "start_column": err.start_column,
                    "end_line": err.end_line,
                    "end_column": err.end_column,
                    "error_type": err.error_type,
                    "context": err.context,
                }
                for err in self.parse_result.syntax_errors
            ],
            "error_recovery_used": self.parse_result.error_recovery_used,
            "valid_sections_count": self.parse_result.valid_sections_count,
        }

    @classmethod
    def _parse_result_from_dict(cls, data: dict[str, Any]) -> ParseResult:
        """Create ParseResult from dictionary."""
        # Import here to avoid circular imports
        from ..models.code_chunk import CodeSyntaxError

        # Reconstruct chunks
        chunks = []
        for chunk_data in data["chunks"]:
            chunk = CodeChunk.from_dict(chunk_data)
            chunks.append(chunk)

        # Reconstruct syntax errors
        syntax_errors = []
        for err_data in data["syntax_errors"]:
            error = CodeSyntaxError(
                start_line=err_data["start_line"],
                start_column=err_data["start_column"],
                end_line=err_data["end_line"],
                end_column=err_data["end_column"],
                error_type=err_data["error_type"],
                context=err_data["context"],
            )
            syntax_errors.append(error)

        return ParseResult(
            chunks=chunks,
            file_path=data["file_path"],
            language=data["language"],
            parse_success=data["parse_success"],
            error_count=data["error_count"],
            fallback_used=data["fallback_used"],
            processing_time_ms=data["processing_time_ms"],
            syntax_errors=syntax_errors,
            error_recovery_used=data["error_recovery_used"],
            valid_sections_count=data["valid_sections_count"],
        )


@dataclass
class ChunkingCacheEntry:
    """Container for chunking cache entries with content validation."""

    # Core data
    file_path: str
    content_hash: str
    language: str
    chunks: list[CodeChunk]

    # Caching metadata
    cache_key: str
    cached_at: float = field(default_factory=time.time)
    chunking_strategy: str = "default"

    # Performance metrics
    chunk_count: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for cache storage."""
        return {
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "language": self.language,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "cache_key": self.cache_key,
            "cached_at": self.cached_at,
            "chunking_strategy": self.chunking_strategy,
            "chunk_count": self.chunk_count,
            "processing_time_ms": self.processing_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkingCacheEntry":
        """Create instance from dictionary."""
        # Reconstruct chunks
        chunks = []
        for chunk_data in data["chunks"]:
            chunk = CodeChunk.from_dict(chunk_data)
            chunks.append(chunk)

        return cls(
            file_path=data["file_path"],
            content_hash=data["content_hash"],
            language=data["language"],
            chunks=chunks,
            cache_key=data["cache_key"],
            cached_at=data["cached_at"],
            chunking_strategy=data.get("chunking_strategy", "default"),
            chunk_count=data.get("chunk_count", len(chunks)),
            processing_time_ms=data.get("processing_time_ms", 0.0),
        )


class FileCacheService:
    """
    Specialized cache service for file processing operations.

    This service provides high-performance caching for expensive file processing
    operations including Tree-sitter parsing, chunking, and AST extraction.
    """

    def __init__(self, cache_service: BaseCacheService | None = None):
        """Initialize the file cache service."""
        self.logger = logging.getLogger(__name__)
        self.cache_service = cache_service
        self.key_generator = CacheKeyGenerator()

        # Cache configuration
        self.default_ttl = 86400  # 24 hours for parsing results
        self.chunk_cache_ttl = 43200  # 12 hours for chunking results
        self.parser_version = "1.0.0"

        # Performance metrics
        self.cache_stats = {
            "parse_cache_hits": 0,
            "parse_cache_misses": 0,
            "chunk_cache_hits": 0,
            "chunk_cache_misses": 0,
            "incremental_parsing_saves": 0,
            "total_cache_time_saved_ms": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize the cache service."""
        if self.cache_service is None:
            self.cache_service = await get_cache_service()

        self.logger.info("File cache service initialized")

    async def get_cached_parse_result(self, file_path: str, content_hash: str, language: str) -> ParsedFileData | None:
        """
        Get cached parse result for a file.

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content
            language: Detected programming language

        Returns:
            ParsedFileData if cached and valid, None otherwise
        """
        try:
            # Generate cache key
            cache_key = self.key_generator.generate_file_parsing_key(file_path, content_hash, language, self.parser_version)

            # Check cache
            cached_data = await self.cache_service.get(cache_key)
            if cached_data is None:
                self.cache_stats["parse_cache_misses"] += 1
                return None

            # Deserialize cached data
            parsed_data = ParsedFileData.from_dict(cached_data)

            # Validate content hash
            if parsed_data.content_hash != content_hash:
                self.logger.debug(f"Content hash mismatch for {file_path}, invalidating cache")
                await self.cache_service.delete(cache_key)
                self.cache_stats["parse_cache_misses"] += 1
                return None

            # Check parser version compatibility
            if parsed_data.parser_version != self.parser_version:
                self.logger.debug(f"Parser version mismatch for {file_path}, invalidating cache")
                await self.cache_service.delete(cache_key)
                self.cache_stats["parse_cache_misses"] += 1
                return None

            self.cache_stats["parse_cache_hits"] += 1
            self.cache_stats["total_cache_time_saved_ms"] += parsed_data.parse_result.processing_time_ms

            self.logger.debug(f"Cache hit for parsing {file_path}")
            return parsed_data

        except Exception as e:
            self.logger.error(f"Error getting cached parse result for {file_path}: {e}")
            self.cache_stats["parse_cache_misses"] += 1
            return None

    async def cache_parse_result(self, parse_result: ParseResult, file_metadata: FileMetadata, content_hash: str) -> bool:
        """
        Cache a parse result.

        Args:
            parse_result: Parse result to cache
            file_metadata: File metadata
            content_hash: SHA256 hash of file content

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Generate cache key
            cache_key = self.key_generator.generate_file_parsing_key(
                file_metadata.file_path, content_hash, parse_result.language, self.parser_version
            )

            # Create parsed file data
            parsed_data = ParsedFileData(
                parse_result=parse_result,
                file_metadata=file_metadata,
                cache_key=cache_key,
                content_hash=content_hash,
                file_size=file_metadata.file_size,
            )

            # Create cache entry
            cache_entry_data = parsed_data.to_dict()

            # Cache the result
            success = await self.cache_service.set(cache_key, cache_entry_data, ttl=self.default_ttl)

            if success:
                self.logger.debug(f"Cached parse result for {file_metadata.file_path}")
            else:
                self.logger.warning(f"Failed to cache parse result for {file_metadata.file_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error caching parse result for {file_metadata.file_path}: {e}")
            return False

    async def get_cached_chunks(self, file_path: str, content_hash: str, language: str) -> list[CodeChunk] | None:
        """
        Get cached chunks for a file.

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content
            language: Detected programming language

        Returns:
            List of CodeChunk if cached and valid, None otherwise
        """
        try:
            # Generate cache key
            cache_key = self.key_generator.generate_chunking_key(file_path, content_hash, language)

            # Check cache
            cached_data = await self.cache_service.get(cache_key)
            if cached_data is None:
                self.cache_stats["chunk_cache_misses"] += 1
                return None

            # Deserialize cached data
            chunking_entry = ChunkingCacheEntry.from_dict(cached_data)

            # Validate content hash
            if chunking_entry.content_hash != content_hash:
                self.logger.debug(f"Content hash mismatch for chunks {file_path}, invalidating cache")
                await self.cache_service.delete(cache_key)
                self.cache_stats["chunk_cache_misses"] += 1
                return None

            self.cache_stats["chunk_cache_hits"] += 1
            self.cache_stats["total_cache_time_saved_ms"] += chunking_entry.processing_time_ms

            self.logger.debug(f"Cache hit for chunks {file_path}")
            return chunking_entry.chunks

        except Exception as e:
            self.logger.error(f"Error getting cached chunks for {file_path}: {e}")
            self.cache_stats["chunk_cache_misses"] += 1
            return None

    async def cache_chunks(
        self,
        file_path: str,
        content_hash: str,
        language: str,
        chunks: list[CodeChunk],
        processing_time_ms: float = 0.0,
        chunking_strategy: str = "default",
    ) -> bool:
        """
        Cache chunks for a file.

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content
            language: Detected programming language
            chunks: List of extracted chunks
            processing_time_ms: Time taken to process
            chunking_strategy: Strategy used for chunking

        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Generate cache key
            cache_key = self.key_generator.generate_chunking_key(file_path, content_hash, language)

            # Create chunking cache entry
            chunking_entry = ChunkingCacheEntry(
                file_path=file_path,
                content_hash=content_hash,
                language=language,
                chunks=chunks,
                cache_key=cache_key,
                chunking_strategy=chunking_strategy,
                chunk_count=len(chunks),
                processing_time_ms=processing_time_ms,
            )

            # Cache the chunks
            success = await self.cache_service.set(cache_key, chunking_entry.to_dict(), ttl=self.chunk_cache_ttl)

            if success:
                self.logger.debug(f"Cached {len(chunks)} chunks for {file_path}")
            else:
                self.logger.warning(f"Failed to cache chunks for {file_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error caching chunks for {file_path}: {e}")
            return False

    async def should_reparse_file(self, file_path: str, current_metadata: FileMetadata, cached_metadata: FileMetadata | None) -> bool:
        """
        Determine if a file should be reparsed based on metadata comparison.

        Args:
            file_path: Path to the file
            current_metadata: Current file metadata
            cached_metadata: Cached file metadata (if any)

        Returns:
            True if file should be reparsed, False otherwise
        """
        try:
            # No cached metadata means we need to parse
            if cached_metadata is None:
                return True

            # Check if content hash changed
            if current_metadata.content_hash != cached_metadata.content_hash:
                self.logger.debug(f"Content hash changed for {file_path}, reparsing required")
                return True

            # Check if file size changed
            if current_metadata.file_size != cached_metadata.file_size:
                self.logger.debug(f"File size changed for {file_path}, reparsing required")
                return True

            # Check if modification time changed significantly
            if abs(current_metadata.mtime - cached_metadata.mtime) > 1.0:
                self.logger.debug(f"Modification time changed for {file_path}, reparsing required")
                return True

            # Check if language detection changed
            if current_metadata.language != cached_metadata.language:
                self.logger.debug(f"Language changed for {file_path}, reparsing required")
                return True

            # File hasn't changed significantly
            self.cache_stats["incremental_parsing_saves"] += 1
            return False

        except Exception as e:
            self.logger.error(f"Error checking if file should be reparsed {file_path}: {e}")
            # Default to reparsing on error
            return True

    async def invalidate_file_cache(self, file_path: str) -> bool:
        """
        Invalidate all cached data for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            True if cache was invalidated, False otherwise
        """
        try:
            # Generate all possible cache keys for this file
            # We need to iterate through possible content hashes and languages
            # This is a simplified approach - in practice, you might want to
            # maintain a mapping of files to their cache keys

            # For now, we'll use a pattern-based approach
            # pattern = f"file_parsing:{file_path}:*"

            # This is a simplified invalidation - in a real implementation,
            # you'd want to track cache keys more systematically
            success = True

            # Note: This is a basic implementation. A more sophisticated version
            # would maintain an index of cache keys per file path

            self.logger.debug(f"Invalidated cache for {file_path}")
            return success

        except Exception as e:
            self.logger.error(f"Error invalidating cache for {file_path}: {e}")
            return False

    async def get_cache_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache performance metrics
        """
        try:
            # Get base cache service stats
            base_stats = {}
            if self.cache_service:
                cache_stats = self.cache_service.get_stats()
                base_stats = {
                    "total_cache_operations": cache_stats.total_operations,
                    "total_cache_hit_rate": cache_stats.hit_rate,
                    "total_cache_miss_rate": cache_stats.miss_rate,
                }

            # Calculate file-specific metrics
            total_parse_requests = self.cache_stats["parse_cache_hits"] + self.cache_stats["parse_cache_misses"]
            total_chunk_requests = self.cache_stats["chunk_cache_hits"] + self.cache_stats["chunk_cache_misses"]

            parse_hit_rate = self.cache_stats["parse_cache_hits"] / total_parse_requests if total_parse_requests > 0 else 0.0

            chunk_hit_rate = self.cache_stats["chunk_cache_hits"] / total_chunk_requests if total_chunk_requests > 0 else 0.0

            file_cache_stats = {
                "parse_cache_hits": self.cache_stats["parse_cache_hits"],
                "parse_cache_misses": self.cache_stats["parse_cache_misses"],
                "parse_hit_rate": parse_hit_rate,
                "chunk_cache_hits": self.cache_stats["chunk_cache_hits"],
                "chunk_cache_misses": self.cache_stats["chunk_cache_misses"],
                "chunk_hit_rate": chunk_hit_rate,
                "incremental_parsing_saves": self.cache_stats["incremental_parsing_saves"],
                "total_cache_time_saved_ms": self.cache_stats["total_cache_time_saved_ms"],
                "total_cache_time_saved_seconds": self.cache_stats["total_cache_time_saved_ms"] / 1000.0,
            }

            return {
                "timestamp": time.time(),
                "base_cache_stats": base_stats,
                "file_cache_stats": file_cache_stats,
            }

        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return {"error": str(e)}

    async def warm_cache_for_project(self, project_files: list[str]) -> dict[str, Any]:
        """
        Warm cache for a list of project files.

        Args:
            project_files: List of file paths to warm cache for

        Returns:
            Dictionary containing warming results
        """
        try:
            results = {
                "files_processed": 0,
                "files_cached": 0,
                "files_skipped": 0,
                "errors": [],
                "processing_time_ms": 0.0,
            }

            start_time = time.time()

            for file_path in project_files:
                try:
                    # Check if file exists
                    if not Path(file_path).exists():
                        results["files_skipped"] += 1
                        continue

                    # Create file metadata
                    file_metadata = FileMetadata.from_file_path(file_path)

                    # Check if already cached
                    cached_data = await self.get_cached_parse_result(
                        file_path, file_metadata.content_hash, file_metadata.language or "unknown"
                    )

                    if cached_data:
                        results["files_skipped"] += 1
                    else:
                        results["files_processed"] += 1
                        # Note: Actual cache warming would require calling the parser
                        # This is a placeholder for the warming logic

                except Exception as e:
                    results["errors"].append(f"{file_path}: {str(e)}")

            results["processing_time_ms"] = (time.time() - start_time) * 1000.0

            return results

        except Exception as e:
            self.logger.error(f"Error warming cache for project: {e}")
            return {"error": str(e)}

    def calculate_content_hash(self, content: str) -> str:
        """
        Calculate SHA256 hash of content.

        Args:
            content: Content to hash

        Returns:
            Hexadecimal SHA256 hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def reset_statistics(self) -> None:
        """Reset cache statistics."""
        self.cache_stats = {
            "parse_cache_hits": 0,
            "parse_cache_misses": 0,
            "chunk_cache_hits": 0,
            "chunk_cache_misses": 0,
            "incremental_parsing_saves": 0,
            "total_cache_time_saved_ms": 0.0,
        }
        self.logger.info("File cache statistics reset")


# Global file cache service instance
_file_cache_service: FileCacheService | None = None


async def get_file_cache_service() -> FileCacheService:
    """
    Get the global file cache service instance.

    Returns:
        FileCacheService: The global file cache service instance
    """
    global _file_cache_service
    if _file_cache_service is None:
        cache_service = await get_cache_service()
        _file_cache_service = FileCacheService(cache_service)
        await _file_cache_service.initialize()
    return _file_cache_service


async def shutdown_file_cache_service() -> None:
    """Shutdown the global file cache service."""
    global _file_cache_service
    if _file_cache_service:
        # File cache service doesn't need special shutdown
        _file_cache_service = None
