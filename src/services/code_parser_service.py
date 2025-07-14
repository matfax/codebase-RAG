"""
CodeParser service for intelligent code chunking using Tree-sitter.

This refactored service acts as a coordinator that orchestrates the newly created
specialized services and chunking strategies to provide semantic code parsing capabilities.
Enhanced with file processing cache integration for improved performance.
"""

import logging
import time
from datetime import datetime

try:
    from tree_sitter import Parser
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")

from src.models.code_chunk import CodeChunk, CodeSyntaxError, ParseResult
from src.models.file_metadata import FileMetadata
from services.ast_extraction_service import AstExtractionService
from services.chunking_strategies import (
    FallbackChunkingStrategy,
    StructuredFileChunkingStrategy,
    chunking_strategy_registry,
)

# Import file cache service for performance optimization
from services.file_cache_service import get_file_cache_service

# Import the new refactored services
from services.language_support_service import LanguageSupportService
from src.utils.chunking_metrics_tracker import chunking_metrics_tracker
from src.utils.file_system_utils import get_file_mtime, get_file_size


class CodeParserService:
    """
    Refactored coordinator service for parsing source code into intelligent semantic chunks.

    This service orchestrates specialized services and chunking strategies to provide
    semantic code parsing capabilities with enhanced modularity and maintainability.
    """

    def __init__(self):
        """Initialize the CodeParser coordinator with specialized services."""
        self.logger = logging.getLogger(__name__)

        # Initialize specialized services
        self.language_support = LanguageSupportService()
        self.ast_extractor = AstExtractionService()

        # Initialize file cache service (will be lazy-loaded)
        self.file_cache_service = None

        # Get initialization summary for logging
        summary = self.language_support.get_initialization_summary()
        self.logger.info(
            f"CodeParser coordinator initialized: {summary['successful_languages']}/{summary['total_languages']} languages supported"
        )

        if summary["failed_languages"]:
            self.logger.warning(f"Failed language support: {', '.join(summary['failed_languages'])}")

        # Performance metrics
        self._parse_stats = {
            "total_files_processed": 0,
            "total_chunks_extracted": 0,
            "total_processing_time_ms": 0,
            "strategy_usage": {},
            "error_recovery_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    # =================== Public API Methods ===================

    async def _ensure_cache_service(self):
        """Ensure file cache service is initialized."""
        if self.file_cache_service is None:
            self.file_cache_service = await get_file_cache_service()

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return self.language_support.get_supported_languages()

    def detect_language(self, file_path: str) -> str | None:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language name if supported, None otherwise
        """
        return self.language_support.detect_language(file_path)

    async def parse_file(self, file_path: str, content: str | None = None) -> ParseResult:
        """
        Parse a source file into intelligent code chunks using the coordinator approach.

        This method orchestrates the language support service, chunking strategies,
        and AST extraction service to provide sophisticated code parsing capabilities.

        Args:
            file_path: Path to the source file
            content: Optional file content (if None, will read from file)

        Returns:
            ParseResult containing extracted chunks and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Language detection
            language = self.detect_language(file_path)
            if not language:
                return self._create_fallback_result(
                    file_path,
                    content,
                    start_time,
                    error_message="Language not supported",
                )

            # Step 2: Content reading
            if content is None:
                content = self._read_file_content(file_path)
                if content is None:
                    return self._create_fallback_result(file_path, "", start_time, error=True)

            # Step 3: Cache lookup for file parsing results
            await self._ensure_cache_service()
            content_hash = self.file_cache_service.calculate_content_hash(content)

            # Check if we have cached parsing results
            cached_data = await self.file_cache_service.get_cached_parse_result(file_path, content_hash, language)

            if cached_data:
                # Cache hit - return cached parse result
                self._parse_stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for parsing {file_path}")

                # Update statistics for cached result
                cached_result = cached_data.parse_result
                self._update_parsing_statistics(cached_result, language, None)

                return cached_result

            # Cache miss - proceed with parsing
            self._parse_stats["cache_misses"] += 1
            self.logger.debug(f"Cache miss for parsing {file_path}, proceeding with parsing")

            # Step 4: Get appropriate chunking strategy
            strategy = self._get_chunking_strategy(language)

            # Step 4: Execute chunking based on strategy type
            if language in ["json", "yaml", "markdown"]:
                # Use structured file chunking
                chunks = strategy.extract_chunks(None, file_path, content)
                parse_success = True
                error_count = 0
                syntax_errors = []
                error_recovery_used = False
            else:
                # Use Tree-sitter based chunking
                parser = self.language_support.get_parser(language)
                if not parser:
                    return self._create_fallback_result(
                        file_path,
                        content,
                        start_time,
                        error_message=f"No parser available for {language}",
                    )

                # Parse with Tree-sitter
                tree = parser.parse(bytes(content, "utf8"))

                # Extract chunks using strategy
                chunks = strategy.extract_chunks(tree.root_node, file_path, content)

                # Collect error information
                error_count = self.ast_extractor.count_errors(tree.root_node)
                parse_success = error_count == 0
                syntax_errors = self.ast_extractor.traverse_for_errors(tree.root_node, content.split("\n"), language)
                error_recovery_used = error_count > 0 and len(chunks) > 0

                if error_recovery_used:
                    self._parse_stats["error_recovery_count"] += 1

            # Step 5: Post-process and validate chunks
            validated_chunks = self._validate_and_enhance_chunks(chunks, language, strategy)

            # Step 6: Calculate metrics and create result
            processing_time = (time.time() - start_time) * 1000

            parse_result = ParseResult(
                chunks=validated_chunks,
                file_path=file_path,
                language=language,
                parse_success=parse_success,
                error_count=error_count,
                fallback_used=False,
                processing_time_ms=processing_time,
                syntax_errors=syntax_errors,
                error_recovery_used=error_recovery_used,
                valid_sections_count=len(validated_chunks),
            )

            # Step 7: Cache the parsing results
            try:
                # Create file metadata for caching
                file_metadata = FileMetadata.from_file_path(file_path)
                file_metadata.language = language
                file_metadata.chunk_count = len(validated_chunks)

                # Cache the parse result
                await self.file_cache_service.cache_parse_result(parse_result, file_metadata, content_hash)

                # Also cache individual chunks if chunking strategy was used
                if language not in ["json", "yaml", "markdown"]:
                    await self.file_cache_service.cache_chunks(
                        file_path, content_hash, language, validated_chunks, processing_time, strategy.__class__.__name__
                    )

                self.logger.debug(f"Cached parsing results for {file_path}")

            except Exception as cache_error:
                # Don't fail parsing if caching fails
                self.logger.warning(f"Failed to cache parsing results for {file_path}: {cache_error}")

            # Step 8: Update statistics and metrics
            self._update_parsing_statistics(parse_result, language, strategy)

            # Step 9: Log results
            self._log_parsing_results(parse_result, file_path)

            return parse_result

        except Exception as e:
            self.logger.error(f"Critical parsing failure for {file_path}: {e}")
            return self._create_fallback_result(
                file_path,
                content or "",
                start_time,
                error=True,
                exception_context=str(e),
            )

    # =================== Coordinator Helper Methods ===================

    def _read_file_content(self, file_path: str) -> str | None:
        """Read file content with error handling."""
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def _get_chunking_strategy(self, language: str):
        """Get the appropriate chunking strategy for a language."""
        strategy = chunking_strategy_registry.get_strategy(language)
        if strategy:
            return strategy

        # Fallback to language-specific fallback strategy
        if language in ["json", "yaml", "markdown"]:
            return StructuredFileChunkingStrategy(language)
        else:
            return FallbackChunkingStrategy(language)

    def _validate_and_enhance_chunks(self, chunks: list[CodeChunk], language: str, strategy) -> list[CodeChunk]:
        """Validate and enhance chunks with additional metadata."""
        validated_chunks = []

        for chunk in chunks:
            # Validate chunk using strategy
            if strategy.validate_chunk(chunk):
                # Add coordinator-level metadata
                self._enhance_chunk_metadata(chunk, language)
                validated_chunks.append(chunk)
            else:
                self.logger.debug(f"Chunk validation failed for {chunk.name} in {chunk.file_path}")

        return validated_chunks

    def _enhance_chunk_metadata(self, chunk: CodeChunk, language: str):
        """Add coordinator-level metadata to chunks."""
        # Add file-level metadata
        chunk.file_size = get_file_size(chunk.file_path)
        chunk.file_mtime = get_file_mtime(chunk.file_path)

        # Add processing timestamp
        chunk.processed_at = datetime.utcnow().isoformat()

        # Add language features
        if hasattr(chunk, "metadata"):
            chunk.metadata = getattr(chunk, "metadata", {})
        else:
            chunk.metadata = {}

        chunk.metadata["language_features"] = self.language_support.get_language_config(language)

    def _update_parsing_statistics(self, parse_result: ParseResult, language: str, strategy):
        """Update internal parsing statistics."""
        self._parse_stats["total_files_processed"] += 1
        self._parse_stats["total_chunks_extracted"] += len(parse_result.chunks)
        self._parse_stats["total_processing_time_ms"] += parse_result.processing_time_ms

        # Track strategy usage
        strategy_name = strategy.__class__.__name__
        if strategy_name not in self._parse_stats["strategy_usage"]:
            self._parse_stats["strategy_usage"][strategy_name] = 0
        self._parse_stats["strategy_usage"][strategy_name] += 1

        # Record metrics with chunking tracker
        if chunking_metrics_tracker:
            chunking_metrics_tracker.record_file_processed(
                language=language,
                chunk_count=len(parse_result.chunks),
                processing_time_ms=parse_result.processing_time_ms,
                parse_success=parse_result.parse_success,
                error_count=parse_result.error_count,
            )

    def _log_parsing_results(self, parse_result: ParseResult, file_path: str):
        """Log parsing results with appropriate detail level."""
        if parse_result.parse_success:
            self.logger.debug(
                f"Successfully parsed {file_path}: {len(parse_result.chunks)} chunks, " f"{parse_result.processing_time_ms:.1f}ms"
            )
        else:
            self.logger.warning(
                f"Parsed {file_path} with {parse_result.error_count} errors: "
                f"{len(parse_result.chunks)} chunks recovered, "
                f"{parse_result.processing_time_ms:.1f}ms"
            )

            # Log detailed error information
            for error in parse_result.syntax_errors[:5]:  # Limit to first 5 errors
                self.logger.debug(f"  Error at line {error.line}: {error.error_text}")

    async def parse_file_with_cache_optimization(
        self, file_path: str, content: str | None = None, force_reparse: bool = False
    ) -> ParseResult:
        """
        Parse a file with advanced cache optimization for large files.

        Args:
            file_path: Path to the source file
            content: Optional file content (if None, will read from file)
            force_reparse: Force reparsing even if cached results exist

        Returns:
            ParseResult containing extracted chunks and metadata
        """
        if force_reparse:
            # Invalidate existing cache for this file
            await self._ensure_cache_service()
            await self.file_cache_service.invalidate_file_cache(file_path)

        return await self.parse_file(file_path, content)

    async def get_cached_chunks_only(self, file_path: str, content_hash: str, language: str) -> list[CodeChunk] | None:
        """
        Get only cached chunks without parsing if not available.

        Args:
            file_path: Path to the source file
            content_hash: SHA256 hash of file content
            language: Detected programming language

        Returns:
            List of cached chunks or None if not cached
        """
        await self._ensure_cache_service()
        return await self.file_cache_service.get_cached_chunks(file_path, content_hash, language)

    async def handle_incremental_parsing(self, file_paths: list[str], project_root: str | None = None) -> dict[str, ParseResult]:
        """
        Handle incremental parsing for multiple files, using cache for unchanged files.

        Args:
            file_paths: List of file paths to process
            project_root: Optional project root for relative path calculation

        Returns:
            Dictionary mapping file paths to parse results
        """
        await self._ensure_cache_service()
        results = {}

        for file_path in file_paths:
            try:
                # Create current file metadata
                current_metadata = FileMetadata.from_file_path(file_path, project_root)

                # Check if file should be reparsed
                should_reparse = await self.file_cache_service.should_reparse_file(
                    file_path,
                    current_metadata,
                    None,  # We'd need to get cached metadata
                )

                if should_reparse:
                    # Parse the file
                    result = await self.parse_file(file_path)
                    results[file_path] = result
                else:
                    # Try to get cached result
                    cached_data = await self.file_cache_service.get_cached_parse_result(
                        file_path, current_metadata.content_hash, current_metadata.language or "unknown"
                    )

                    if cached_data:
                        results[file_path] = cached_data.parse_result
                        self._parse_stats["cache_hits"] += 1
                    else:
                        # Fallback to parsing if cache miss
                        result = await self.parse_file(file_path)
                        results[file_path] = result

            except Exception as e:
                self.logger.error(f"Error processing {file_path} for incremental parsing: {e}")
                # Create error result
                results[file_path] = self._create_fallback_result(file_path, "", time.time(), error=True, exception_context=str(e))

        return results

    async def optimize_parsing_for_language(self, file_path: str, language: str, content: str) -> ParseResult:
        """
        Language-specific parsing optimizations with caching.

        Args:
            file_path: Path to the source file
            language: Programming language
            content: File content

        Returns:
            Optimized ParseResult
        """
        await self._ensure_cache_service()

        # Check for language-specific cached results
        content_hash = self.file_cache_service.calculate_content_hash(content)

        # Language-specific cache lookup
        cached_chunks = await self.file_cache_service.get_cached_chunks(file_path, content_hash, language)

        if cached_chunks:
            # Create parse result from cached chunks
            self._parse_stats["cache_hits"] += 1
            return ParseResult(
                chunks=cached_chunks,
                file_path=file_path,
                language=language,
                parse_success=True,
                error_count=0,
                fallback_used=False,
                processing_time_ms=0.0,  # Cached, so no processing time
                syntax_errors=[],
                error_recovery_used=False,
                valid_sections_count=len(cached_chunks),
            )

        # Proceed with regular parsing
        return await self.parse_file(file_path, content)

    def get_parsing_statistics(self) -> dict:
        """Get current parsing statistics including cache metrics."""
        cache_total = self._parse_stats["cache_hits"] + self._parse_stats["cache_misses"]
        cache_hit_rate = self._parse_stats["cache_hits"] / cache_total if cache_total > 0 else 0.0

        stats = self._parse_stats.copy()
        stats["cache_hit_rate"] = cache_hit_rate
        stats["cache_total_requests"] = cache_total

        return stats

    def reset_parsing_statistics(self):
        """Reset internal parsing statistics."""
        self._parse_stats = {
            "total_files_processed": 0,
            "total_chunks_extracted": 0,
            "total_processing_time_ms": 0,
            "strategy_usage": {},
            "error_recovery_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _create_fallback_result(
        self,
        file_path: str,
        content: str | None,
        start_time: float,
        error: bool = False,
        error_message: str = None,
        exception_context: str | None = None,
    ) -> ParseResult:
        """Create a simplified fallback ParseResult for the coordinator approach."""
        if content is None:
            content = ""

        processing_time = (time.time() - start_time) * 1000
        language = self.detect_language(file_path) or "unknown"

        # Create a simple whole-file chunk using fallback strategy
        fallback_strategy = FallbackChunkingStrategy(language)
        chunks = []

        if content:
            # Create a dummy root node for structured compatibility
            chunks = fallback_strategy.extract_chunks(None, file_path, content)

        # Create simple error information
        syntax_errors = []
        if error and (error_message or exception_context):
            error_details = error_message or f"Exception: {exception_context}"
            syntax_errors.append(
                CodeSyntaxError(
                    start_line=1,
                    start_column=0,
                    end_line=len(content.split("\n")) if content else 1,
                    end_column=0,
                    error_type="parsing_error",
                    context=error_details,
                )
            )

        return ParseResult(
            chunks=chunks,
            file_path=file_path,
            language=language,
            parse_success=not error,
            error_count=1 if error else 0,
            fallback_used=True,
            processing_time_ms=processing_time,
            syntax_errors=syntax_errors,
            error_recovery_used=False,
            valid_sections_count=len(chunks),
        )
