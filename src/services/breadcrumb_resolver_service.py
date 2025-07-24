"""
BreadcrumbResolver Service for Function Chain MCP Tools.

This service provides natural language to breadcrumb conversion capabilities,
enabling users to use natural language descriptions to find and resolve
precise breadcrumb paths in codebases. It integrates with the existing
search infrastructure to provide semantic conversion with confidence scoring.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from src.models.breadcrumb_cache_models import BreadcrumbCacheConfig
from src.models.code_chunk import CodeChunk
from src.models.function_call import FunctionCall
from src.services.breadcrumb_cache_service import BreadcrumbCacheService
from src.tools.indexing.search_tools import search_async_cached


class BreadcrumbFormat(Enum):
    """Supported breadcrumb formats."""

    DOTTED = "dotted"  # Python-style: module.class.method
    DOUBLE_COLON = "double_colon"  # C++/Rust-style: namespace::class::method
    SLASH = "slash"  # Path-style: module/class/method
    ARROW = "arrow"  # Chain-style: module->class->method


@dataclass
class BreadcrumbCandidate:
    """Represents a candidate breadcrumb resolution with confidence scoring."""

    breadcrumb: str
    confidence_score: float
    source_chunk: CodeChunk
    reasoning: str
    match_type: str  # exact, partial, semantic, fuzzy

    # Additional metadata
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    chunk_type: str = ""
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "breadcrumb": self.breadcrumb,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "match_type": self.match_type,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chunk_type": self.chunk_type,
            "language": self.language,
        }


@dataclass
class BreadcrumbResolutionResult:
    """Result of breadcrumb resolution operation."""

    query: str
    success: bool
    primary_candidate: BreadcrumbCandidate | None = None
    alternative_candidates: list[BreadcrumbCandidate] = None
    error_message: str = ""
    resolution_time_ms: float = 0.0
    search_results_count: int = 0

    def __post_init__(self):
        """Initialize mutable fields."""
        if self.alternative_candidates is None:
            self.alternative_candidates = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "success": self.success,
            "primary_candidate": self.primary_candidate.to_dict() if self.primary_candidate else None,
            "alternative_candidates": [c.to_dict() for c in self.alternative_candidates],
            "error_message": self.error_message,
            "resolution_time_ms": self.resolution_time_ms,
            "search_results_count": self.search_results_count,
        }


class BreadcrumbResolver:
    """
    Service for resolving natural language inputs to breadcrumb paths.

    This service acts as a bridge between natural language function descriptions
    and precise breadcrumb paths, enabling users to find functions and classes
    using descriptive text rather than exact names.
    """

    def __init__(self, cache_enabled: bool = True, cache_config: BreadcrumbCacheConfig | None = None):
        """
        Initialize the BreadcrumbResolver service.

        Args:
            cache_enabled: Whether to enable caching of resolution results
            cache_config: Cache configuration, defaults to environment-based config
        """
        self.logger = logging.getLogger(__name__)
        self.cache_enabled = cache_enabled

        # Initialize enhanced TTL-based cache service
        if cache_enabled:
            self.cache_service = BreadcrumbCacheService(cache_config)
        else:
            self.cache_service = None

        # Legacy simple cache (kept for fallback)
        self._resolution_cache: dict[str, BreadcrumbResolutionResult] = {}

        # Configuration
        self.max_candidates = 5
        self.min_confidence_threshold = 0.3
        self.semantic_search_results = 10

        # Breadcrumb patterns for validation
        self.breadcrumb_patterns = {
            BreadcrumbFormat.DOTTED: re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$"),
            BreadcrumbFormat.DOUBLE_COLON: re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(::([a-zA-Z_][a-zA-Z0-9_]*|<[^>]*>))+$"),
            BreadcrumbFormat.SLASH: re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(/[a-zA-Z_][a-zA-Z0-9_]*)+$"),
            BreadcrumbFormat.ARROW: re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(->[a-zA-Z_][a-zA-Z0-9_]*)+$"),
        }

        self.logger.info("BreadcrumbResolver initialized with cache_enabled=%s", cache_enabled)

    async def start(self):
        """Start the breadcrumb resolver service and its cache."""
        if self.cache_service:
            await self.cache_service.start()
            self.logger.info("BreadcrumbResolver cache service started")

    async def stop(self):
        """Stop the breadcrumb resolver service and its cache."""
        if self.cache_service:
            await self.cache_service.stop()
            self.logger.info("BreadcrumbResolver cache service stopped")

    async def resolve(self, query: str, target_projects: list[str] | None = None) -> BreadcrumbResolutionResult:
        """
        Resolve a natural language query to breadcrumb path(s).

        This is the main entry point for breadcrumb resolution. It handles:
        1. Input validation
        2. Cache lookup
        3. Breadcrumb format detection
        4. Natural language to breadcrumb conversion
        5. Result ranking and filtering

        Args:
            query: Natural language description of the function/class to find
            target_projects: Optional list of specific projects to search in

        Returns:
            BreadcrumbResolutionResult containing the best matches with confidence scores
        """
        start_time = time.time()

        try:
            # Input validation
            if not query or not isinstance(query, str):
                return BreadcrumbResolutionResult(query=query, success=False, error_message="Query must be a non-empty string")

            query = query.strip()
            if not query:
                return BreadcrumbResolutionResult(query=query, success=False, error_message="Query cannot be empty or whitespace only")

            # Check cache first
            if self.cache_enabled:
                cache_key = self._create_cache_key(query, target_projects)

                # Try enhanced TTL cache first
                if self.cache_service:
                    cached_result = await self.cache_service.get(cache_key)
                    if cached_result:
                        self.logger.debug("Enhanced cache hit for query: %s", query[:50])
                        return cached_result
                else:
                    # Fallback to legacy cache
                    cached_result = self._get_from_cache(cache_key)
                    if cached_result:
                        self.logger.debug("Legacy cache hit for query: %s", query[:50])
                        return cached_result

            # Check if input is already a valid breadcrumb
            if self.is_valid_breadcrumb(query):
                self.logger.debug("Input is already a valid breadcrumb: %s", query)
                # Create a high-confidence result for valid breadcrumbs
                primary_candidate = BreadcrumbCandidate(
                    breadcrumb=query,
                    confidence_score=1.0,
                    source_chunk=None,  # No source chunk for direct breadcrumbs
                    reasoning="Input is already a valid breadcrumb format",
                    match_type="exact",
                )

                result = BreadcrumbResolutionResult(
                    query=query, success=True, primary_candidate=primary_candidate, resolution_time_ms=(time.time() - start_time) * 1000
                )

                # Cache the result
                if self.cache_enabled:
                    await self._cache_result_enhanced(cache_key, result, file_dependencies=[], confidence_score=1.0)

                return result

            # Convert natural language to breadcrumb
            result = await self.convert_natural_to_breadcrumb(query, target_projects)

            # Cache the result
            if self.cache_enabled:
                cache_key = self._create_cache_key(query, target_projects)

                # Extract file dependencies and confidence from result
                file_dependencies = []
                confidence_score = 0.0
                if result.success and result.primary_candidate:
                    confidence_score = result.primary_candidate.confidence_score
                    if result.primary_candidate.file_path:
                        file_dependencies.append(result.primary_candidate.file_path)

                await self._cache_result_enhanced(cache_key, result, file_dependencies, confidence_score)

            return result

        except Exception as e:
            self.logger.error("Error resolving breadcrumb for query '%s': %s", query, str(e))
            return BreadcrumbResolutionResult(
                query=query,
                success=False,
                error_message=f"Internal error during resolution: {str(e)}",
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

    def is_valid_breadcrumb(self, input_string: str) -> bool:
        """
        Check if the input string is a valid breadcrumb format.

        This function validates against multiple breadcrumb formats commonly
        used in different programming languages:
        - Dotted notation (Python): module.class.method
        - Double colon notation (C++/Rust): namespace::class::method
        - Slash notation (Path-style): module/class/method
        - Arrow notation (Chain-style): module->class->method

        Args:
            input_string: String to validate as breadcrumb

        Returns:
            True if the input matches a valid breadcrumb format, False otherwise
        """
        if not input_string or not isinstance(input_string, str):
            return False

        input_string = input_string.strip()
        if not input_string:
            return False

        # Check against each supported format
        for format_type, pattern in self.breadcrumb_patterns.items():
            if pattern.match(input_string):
                self.logger.debug("Input '%s' matches %s format", input_string, format_type.value)
                return True

        # Additional validation for edge cases
        return self._validate_breadcrumb_structure(input_string)

    def _validate_breadcrumb_structure(self, input_string: str) -> bool:
        """
        Additional validation for breadcrumb structure.

        This handles edge cases and validates the overall structure
        beyond just pattern matching.

        Args:
            input_string: String to validate

        Returns:
            True if structure is valid, False otherwise
        """
        # Check for minimum length (at least "a.b")
        if len(input_string) < 3:
            return False

        # Check for valid characters and structure
        # Must contain at least one separator
        has_separator = any(sep in input_string for sep in [".", "::", "/", "->"])
        if not has_separator:
            return False

        # Check for invalid patterns
        invalid_patterns = [
            r"^\.",  # starts with separator
            r"\.$",  # ends with separator
            r"\.\.",  # double dots
            r"::$",  # ends with ::
            r"^::",  # starts with ::
            r"/$",  # ends with /
            r"^/",  # starts with /
            r"->$",  # ends with ->
            r"^->",  # starts with ->
            r"\s",  # contains whitespace
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, input_string):
                return False

        return True

    async def convert_natural_to_breadcrumb(self, query: str, target_projects: list[str] | None = None) -> BreadcrumbResolutionResult:
        """
        Convert natural language description to breadcrumb path using semantic search.

        This function uses the existing search infrastructure to find relevant
        code chunks and extract breadcrumb paths from them. It employs multiple
        strategies to improve accuracy:
        1. Semantic search for similar functionality
        2. Pattern matching for function/class names
        3. Context analysis for disambiguation
        4. Confidence scoring for ranking

        Args:
            query: Natural language description
            target_projects: Optional list of projects to search in

        Returns:
            BreadcrumbResolutionResult with candidates sorted by confidence
        """
        start_time = time.time()

        try:
            # Perform semantic search
            search_results = await search_async_cached(
                query=query,
                n_results=self.semantic_search_results,
                search_mode="hybrid",
                include_context=True,
                context_chunks=1,
                target_projects=target_projects,
            )

            if not search_results.get("results"):
                return BreadcrumbResolutionResult(
                    query=query,
                    success=False,
                    error_message="No relevant code found for the query",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                    search_results_count=0,
                )

            # Extract candidates from search results
            candidates = self._extract_candidates_from_search_results(search_results["results"], query)

            if not candidates:
                return BreadcrumbResolutionResult(
                    query=query,
                    success=False,
                    error_message="No valid breadcrumb candidates found",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                    search_results_count=len(search_results["results"]),
                )

            # Filter and rank candidates
            filtered_candidates = self._filter_and_rank_candidates(candidates, query)

            if not filtered_candidates:
                return BreadcrumbResolutionResult(
                    query=query,
                    success=False,
                    error_message="No candidates met the minimum confidence threshold",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                    search_results_count=len(search_results["results"]),
                )

            # Return results
            primary_candidate = filtered_candidates[0]
            alternative_candidates = filtered_candidates[1 : self.max_candidates]

            return BreadcrumbResolutionResult(
                query=query,
                success=True,
                primary_candidate=primary_candidate,
                alternative_candidates=alternative_candidates,
                resolution_time_ms=(time.time() - start_time) * 1000,
                search_results_count=len(search_results["results"]),
            )

        except Exception as e:
            self.logger.error("Error converting natural language to breadcrumb: %s", str(e))
            return BreadcrumbResolutionResult(
                query=query,
                success=False,
                error_message=f"Error during conversion: {str(e)}",
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

    def _extract_candidates_from_search_results(self, search_results: list[dict[str, Any]], query: str) -> list[BreadcrumbCandidate]:
        """
        Extract breadcrumb candidates from search results.

        This function processes search results to identify potential breadcrumb
        paths and creates candidates with appropriate confidence scores.

        Args:
            search_results: List of search results from semantic search
            query: Original query for context

        Returns:
            List of BreadcrumbCandidate objects
        """
        candidates = []

        for result in search_results:
            try:
                # Extract breadcrumb from result
                breadcrumb = self._extract_breadcrumb_from_result(result)
                if not breadcrumb:
                    continue

                # Create candidate
                candidate = BreadcrumbCandidate(
                    breadcrumb=breadcrumb,
                    confidence_score=self._calculate_confidence_score(result, query),
                    source_chunk=self._create_code_chunk_from_result(result),
                    reasoning=self._generate_reasoning(result, query),
                    match_type=self._determine_match_type(result, query),
                    file_path=result.get("file_path", ""),
                    line_start=result.get("line_start", 0),
                    line_end=result.get("line_end", 0),
                    chunk_type=result.get("chunk_type", ""),
                    language=result.get("language", ""),
                )

                candidates.append(candidate)

            except Exception as e:
                self.logger.debug("Error processing search result: %s", str(e))
                continue

        return candidates

    def _extract_breadcrumb_from_result(self, result: dict[str, Any]) -> str | None:
        """
        Extract breadcrumb path from a search result.

        This function attempts to build a breadcrumb path from the available
        metadata in the search result.

        Args:
            result: Search result dictionary

        Returns:
            Breadcrumb string or None if unable to extract
        """
        # Try to get breadcrumb directly
        breadcrumb = result.get("breadcrumb", "")
        if breadcrumb and self.is_valid_breadcrumb(breadcrumb):
            return breadcrumb

        # Build breadcrumb from components
        components = []

        # Add parent component if available
        parent_name = result.get("parent_name", "")
        if parent_name:
            components.append(parent_name)

        # Add current component name
        name = result.get("name", "")
        if name:
            components.append(name)

        # If we have at least 2 components, build breadcrumb
        if len(components) >= 2:
            # Determine separator based on language
            language = result.get("language", "").lower()
            if language in ["cpp", "c", "rust"]:
                separator = "::"
            else:
                separator = "."

            return separator.join(components)

        # Single component - use as is if it's a valid identifier
        if len(components) == 1 and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", components[0]):
            return components[0]

        return None

    def _calculate_confidence_score(self, result: dict[str, Any], query: str) -> float:
        """
        Calculate confidence score for a breadcrumb candidate.

        This function evaluates multiple factors to determine how confident
        we are that this candidate matches the user's intent.

        Args:
            result: Search result dictionary
            query: Original query

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Base score from search relevance
        search_score = result.get("score", 0.0)
        score += search_score * 0.4  # 40% weight for search relevance

        # Content quality score
        content = result.get("content", "")
        if content:
            # Longer, more substantial content gets higher score
            content_score = min(len(content) / 1000, 1.0)  # Normalize to 1.0
            score += content_score * 0.2  # 20% weight for content quality

        # Name matching score
        name = result.get("name", "")
        if name:
            name_score = self._calculate_name_similarity(name, query)
            score += name_score * 0.3  # 30% weight for name similarity

        # Chunk type relevance
        chunk_type = result.get("chunk_type", "")
        if chunk_type in ["function", "method", "class"]:
            score += 0.1  # 10% bonus for relevant chunk types

        # Normalize to 0.0-1.0 range
        return min(score, 1.0)

    def _calculate_name_similarity(self, name: str, query: str) -> float:
        """
        Calculate similarity between a code element name and query.

        This function uses multiple heuristics to determine how similar
        a function/class name is to the natural language query.

        Args:
            name: Name of the code element
            query: Natural language query

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not name or not query:
            return 0.0

        name_lower = name.lower()
        query_lower = query.lower()

        # Exact match
        if name_lower == query_lower:
            return 1.0

        # Substring match
        if name_lower in query_lower or query_lower in name_lower:
            return 0.8

        # Word similarity
        query_words = re.findall(r"\b\w+\b", query_lower)
        name_words = re.findall(r"\b\w+\b", name_lower)

        if not query_words or not name_words:
            return 0.0

        # Calculate word overlap
        common_words = set(query_words) & set(name_words)
        if common_words:
            overlap_score = len(common_words) / max(len(query_words), len(name_words))
            return overlap_score * 0.6

        # Fuzzy similarity for remaining cases
        return self._fuzzy_similarity(name_lower, query_lower)

    def _fuzzy_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate fuzzy similarity between two strings.

        This is a simple implementation of edit distance-based similarity.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 or not str2:
            return 0.0

        if str1 == str2:
            return 1.0

        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0

        # Simple character-based similarity
        common_chars = sum(c1 == c2 for c1, c2 in zip(str1, str2, strict=False))
        return common_chars / max_len

    def _create_code_chunk_from_result(self, result: dict[str, Any]) -> CodeChunk | None:
        """
        Create a CodeChunk object from search result.

        Args:
            result: Search result dictionary

        Returns:
            CodeChunk object or None if unable to create
        """
        try:
            # This is a simplified version - in practice you'd want to
            # fully populate the CodeChunk with all available metadata
            return CodeChunk(
                content=result.get("content", ""),
                file_path=result.get("file_path", ""),
                chunk_type=result.get("chunk_type", ""),
                name=result.get("name", ""),
                line_start=result.get("line_start", 0),
                line_end=result.get("line_end", 0),
                language=result.get("language", ""),
            )
        except Exception as e:
            self.logger.debug("Error creating CodeChunk from result: %s", str(e))
            return None

    def _generate_reasoning(self, result: dict[str, Any], query: str) -> str:
        """
        Generate human-readable reasoning for why this candidate was selected.

        Args:
            result: Search result dictionary
            query: Original query

        Returns:
            Reasoning string
        """
        reasons = []

        # Search relevance
        score = result.get("score", 0.0)
        if score > 0.8:
            reasons.append("high semantic similarity to query")
        elif score > 0.6:
            reasons.append("good semantic similarity to query")
        else:
            reasons.append("moderate semantic similarity to query")

        # Name matching
        name = result.get("name", "")
        if name:
            name_similarity = self._calculate_name_similarity(name, query)
            if name_similarity > 0.8:
                reasons.append(f"function name '{name}' closely matches query")
            elif name_similarity > 0.5:
                reasons.append(f"function name '{name}' partially matches query")

        # Chunk type
        chunk_type = result.get("chunk_type", "")
        if chunk_type in ["function", "method"]:
            reasons.append(f"is a {chunk_type} implementation")

        # File context
        file_path = result.get("file_path", "")
        if file_path:
            reasons.append(f"found in {file_path}")

        return "Selected because: " + ", ".join(reasons)

    def _determine_match_type(self, result: dict[str, Any], query: str) -> str:
        """
        Determine the type of match between result and query.

        Args:
            result: Search result dictionary
            query: Original query

        Returns:
            Match type string
        """
        name = result.get("name", "")
        score = result.get("score", 0.0)

        # Check for exact name match
        if name and name.lower() == query.lower():
            return "exact"

        # Check for partial name match
        if name and (name.lower() in query.lower() or query.lower() in name.lower()):
            return "partial"

        # Check for high semantic similarity
        if score > 0.8:
            return "semantic"

        # Default to fuzzy match
        return "fuzzy"

    def _filter_and_rank_candidates(self, candidates: list[BreadcrumbCandidate], query: str) -> list[BreadcrumbCandidate]:
        """
        Filter and rank candidates based on confidence scores.

        Args:
            candidates: List of candidates to filter and rank
            query: Original query for context

        Returns:
            Filtered and sorted list of candidates
        """
        # Filter by minimum confidence
        filtered = [c for c in candidates if c.confidence_score >= self.min_confidence_threshold]

        # Sort by confidence score (descending)
        filtered.sort(key=lambda c: c.confidence_score, reverse=True)

        # Remove duplicates (same breadcrumb)
        seen_breadcrumbs = set()
        unique_candidates = []
        for candidate in filtered:
            if candidate.breadcrumb not in seen_breadcrumbs:
                seen_breadcrumbs.add(candidate.breadcrumb)
                unique_candidates.append(candidate)

        return unique_candidates

    def _create_cache_key(self, query: str, target_projects: list[str] | None) -> str:
        """
        Create a cache key for the resolution request.

        Args:
            query: Natural language query
            target_projects: Optional list of target projects

        Returns:
            Cache key string
        """
        projects_str = ",".join(sorted(target_projects)) if target_projects else "all"
        return f"{query}|{projects_str}"

    def _get_from_cache(self, cache_key: str) -> BreadcrumbResolutionResult | None:
        """
        Get result from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None
        """
        return self._resolution_cache.get(cache_key)

    async def _cache_result_enhanced(
        self,
        cache_key: str,
        result: BreadcrumbResolutionResult,
        file_dependencies: list[str] | None = None,
        confidence_score: float = 0.0,
    ):
        """
        Cache a resolution result using enhanced TTL-based cache.

        Args:
            cache_key: Cache key
            result: Result to cache
            file_dependencies: List of file paths this result depends on
            confidence_score: Confidence score for TTL calculation
        """
        if self.cache_service:
            success = await self.cache_service.put(
                cache_key=cache_key, result=result, file_dependencies=file_dependencies or [], confidence_score=confidence_score
            )
            if success:
                self.logger.debug(f"Cached result with enhanced service: {cache_key}")
            else:
                # Fallback to legacy cache
                self._cache_result(cache_key, result)
        else:
            # Use legacy cache
            self._cache_result(cache_key, result)

    def _cache_result(self, cache_key: str, result: BreadcrumbResolutionResult):
        """
        Cache a resolution result (legacy method).

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Simple cache implementation - in production you'd want TTL, size limits, etc.
        self._resolution_cache[cache_key] = result

    async def clear_cache(self):
        """Clear the resolution cache."""
        if self.cache_service:
            await self.cache_service.clear()
        self._resolution_cache.clear()
        self.logger.info("BreadcrumbResolver cache cleared")

    async def invalidate_cache_by_file(self, file_path: str) -> int:
        """
        Invalidate cache entries that depend on a specific file.

        Args:
            file_path: Path to the file that was modified

        Returns:
            Number of cache entries invalidated
        """
        if self.cache_service:
            return await self.cache_service.invalidate_by_file(file_path)
        return 0

    async def get_cache_info(self) -> dict[str, Any]:
        """Get comprehensive cache information."""
        if self.cache_service:
            return await self.cache_service.get_cache_info()
        else:
            return {"legacy_cache_size": len(self._resolution_cache), "enhanced_cache_enabled": False}

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "enabled": self.cache_enabled,
            "legacy_cache_size": len(self._resolution_cache),
            "configuration": {
                "max_candidates": self.max_candidates,
                "min_confidence_threshold": self.min_confidence_threshold,
                "semantic_search_results": self.semantic_search_results,
            },
        }

        # Add enhanced cache stats if available
        if self.cache_service:
            enhanced_stats = self.cache_service.get_stats()
            stats.update({"enhanced_cache_stats": enhanced_stats, "cache_type": "enhanced_ttl"})
        else:
            stats["cache_type"] = "legacy"

        return stats

    # =================== Function Call Resolution Methods ===================

    async def resolve_function_call_target(
        self, function_call: FunctionCall, target_projects: list[str] | None = None
    ) -> BreadcrumbResolutionResult:
        """
        Resolve the target breadcrumb for a function call.

        This method takes a FunctionCall object and attempts to resolve its
        target_breadcrumb to a precise, validated breadcrumb path by searching
        the codebase for matching function definitions.

        Args:
            function_call: FunctionCall object with target to resolve
            target_projects: Optional list of specific projects to search in

        Returns:
            BreadcrumbResolutionResult with resolved target breadcrumb
        """
        start_time = time.time()

        try:
            # Extract the call target from the function call
            raw_target = function_call.target_breadcrumb
            call_expression = function_call.call_expression

            # Create resolution query based on call type and expression
            resolution_query = self._create_resolution_query(function_call)

            # Check cache first
            if self.cache_enabled:
                cache_key = self._create_call_cache_key(function_call, target_projects)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.logger.debug("Cache hit for function call resolution: %s", call_expression[:50])
                    return cached_result

            # If the target already looks like a valid breadcrumb, validate it
            if self.is_valid_breadcrumb(raw_target):
                validation_result = await self._validate_breadcrumb_exists(raw_target, target_projects)
                if validation_result.success:
                    result = BreadcrumbResolutionResult(
                        query=resolution_query,
                        success=True,
                        primary_candidate=validation_result.primary_candidate,
                        resolution_time_ms=(time.time() - start_time) * 1000,
                    )
                    if self.cache_enabled:
                        await self._cache_result_enhanced(
                            cache_key, result, file_dependencies=[], confidence_score=validation_result.primary_candidate.confidence_score
                        )
                    return result

            # Perform semantic search to resolve the target
            result = await self._resolve_call_target_with_search(function_call, resolution_query, target_projects)

            # Cache the result
            if self.cache_enabled:
                cache_key = self._create_call_cache_key(function_call, target_projects)

                # Extract file dependencies and confidence
                file_dependencies = [function_call.file_path] if function_call.file_path else []
                confidence_score = 0.0
                if result.success and result.primary_candidate:
                    confidence_score = result.primary_candidate.confidence_score
                    if result.primary_candidate.file_path:
                        file_dependencies.append(result.primary_candidate.file_path)

                await self._cache_result_enhanced(cache_key, result, file_dependencies, confidence_score)

            result.resolution_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            self.logger.error("Error resolving function call target '%s': %s", function_call.call_expression, str(e))
            return BreadcrumbResolutionResult(
                query=resolution_query if "resolution_query" in locals() else function_call.call_expression,
                success=False,
                error_message=f"Internal error during call resolution: {str(e)}",
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

    async def resolve_multiple_function_calls(
        self, function_calls: list[FunctionCall], target_projects: list[str] | None = None
    ) -> dict[str, BreadcrumbResolutionResult]:
        """
        Resolve target breadcrumbs for multiple function calls efficiently.

        Args:
            function_calls: List of FunctionCall objects to resolve
            target_projects: Optional list of specific projects to search in

        Returns:
            Dictionary mapping call expressions to resolution results
        """
        results = {}

        for function_call in function_calls:
            try:
                result = await self.resolve_function_call_target(function_call, target_projects)
                results[function_call.call_expression] = result
            except Exception as e:
                self.logger.error(f"Error resolving call {function_call.call_expression}: {e}")
                results[function_call.call_expression] = BreadcrumbResolutionResult(
                    query=function_call.call_expression, success=False, error_message=f"Resolution failed: {str(e)}"
                )

        return results

    def _create_resolution_query(self, function_call: FunctionCall) -> str:
        """
        Create a search query for resolving a function call target.

        This method analyzes the function call to create an effective search query
        that will help find the target function definition.

        Args:
            function_call: FunctionCall object to create query for

        Returns:
            Search query string optimized for target resolution
        """
        call_expr = function_call.call_expression
        call_type = function_call.call_type

        # Extract the function/method name from the call
        if "(" in call_expr:
            func_part = call_expr.split("(")[0].strip()
        else:
            func_part = call_expr.strip()

        # Create query based on call type
        if call_type.value in ["method", "self_method", "attribute"]:
            # For method calls, extract the method name
            if "." in func_part:
                method_name = func_part.split(".")[-1]
                return f"def {method_name} OR function {method_name} OR method {method_name}"
            else:
                return f"def {func_part} OR function {func_part}"

        elif call_type.value == "direct":
            # For direct function calls
            return f"def {func_part} OR function {func_part}"

        elif call_type.value == "module_function":
            # For module function calls like module.function()
            if "." in func_part:
                parts = func_part.split(".")
                module_name = parts[0]
                function_name = parts[-1]
                return f"{module_name} {function_name} OR def {function_name}"
            else:
                return f"def {func_part} OR function {func_part}"

        elif call_type.value in ["async", "async_method"]:
            # For async calls, include async keyword
            if "." in func_part:
                method_name = func_part.split(".")[-1]
                return f"async def {method_name} OR async {method_name}"
            else:
                return f"async def {func_part} OR async {func_part}"

        elif call_type.value == "asyncio":
            # For asyncio calls, focus on asyncio module
            return f"asyncio {func_part} OR {func_part}"

        else:
            # Default query
            if "." in func_part:
                name = func_part.split(".")[-1]
                return f"def {name} OR function {name}"
            else:
                return f"def {func_part} OR function {func_part}"

    async def _resolve_call_target_with_search(
        self, function_call: FunctionCall, query: str, target_projects: list[str] | None
    ) -> BreadcrumbResolutionResult:
        """
        Resolve function call target using semantic search.

        Args:
            function_call: FunctionCall object to resolve
            query: Search query for target resolution
            target_projects: Optional list of projects to search in

        Returns:
            BreadcrumbResolutionResult with resolution details
        """
        try:
            # Perform semantic search with higher result count for call resolution
            search_results = await search_async_cached(
                query=query,
                n_results=15,  # Higher count for better call resolution
                search_mode="hybrid",
                include_context=True,
                context_chunks=1,
                target_projects=target_projects,
            )

            if not search_results.get("results"):
                return BreadcrumbResolutionResult(
                    query=query,
                    success=False,
                    error_message="No matching function definitions found",
                    search_results_count=0,
                )

            # Extract and rank candidates specifically for function call resolution
            candidates = self._extract_call_target_candidates(search_results["results"], function_call, query)

            if not candidates:
                return BreadcrumbResolutionResult(
                    query=query,
                    success=False,
                    error_message="No valid function call target candidates found",
                    search_results_count=len(search_results["results"]),
                )

            # Filter and rank candidates
            filtered_candidates = self._filter_and_rank_call_candidates(candidates, function_call)

            if not filtered_candidates:
                return BreadcrumbResolutionResult(
                    query=query,
                    success=False,
                    error_message="No call target candidates met the confidence threshold",
                    search_results_count=len(search_results["results"]),
                )

            # Return results
            primary_candidate = filtered_candidates[0]
            alternative_candidates = filtered_candidates[1 : self.max_candidates]

            return BreadcrumbResolutionResult(
                query=query,
                success=True,
                primary_candidate=primary_candidate,
                alternative_candidates=alternative_candidates,
                search_results_count=len(search_results["results"]),
            )

        except Exception as e:
            self.logger.error("Error during call target search resolution: %s", str(e))
            return BreadcrumbResolutionResult(
                query=query,
                success=False,
                error_message=f"Search resolution error: {str(e)}",
            )

    def _extract_call_target_candidates(
        self, search_results: list[dict[str, Any]], function_call: FunctionCall, query: str
    ) -> list[BreadcrumbCandidate]:
        """
        Extract function call target candidates from search results.

        This method specializes in finding function definitions that match
        the function call target, with enhanced scoring for call resolution.

        Args:
            search_results: List of search results
            function_call: Original function call object
            query: Search query used

        Returns:
            List of BreadcrumbCandidate objects for call targets
        """
        candidates = []
        call_expr = function_call.call_expression

        # Extract target function name for matching
        target_name = self._extract_target_function_name(call_expr)

        for result in search_results:
            try:
                # Only consider function/method definitions
                chunk_type = result.get("chunk_type", "")
                if chunk_type not in ["function", "method", "class"]:
                    continue

                # Extract breadcrumb from result
                breadcrumb = self._extract_breadcrumb_from_result(result)
                if not breadcrumb:
                    continue

                # Calculate enhanced confidence for call target matching
                confidence_score = self._calculate_call_target_confidence(result, function_call, target_name, query)

                # Create candidate with enhanced reasoning
                candidate = BreadcrumbCandidate(
                    breadcrumb=breadcrumb,
                    confidence_score=confidence_score,
                    source_chunk=self._create_code_chunk_from_result(result),
                    reasoning=self._generate_call_target_reasoning(result, function_call, target_name),
                    match_type=self._determine_call_target_match_type(result, function_call, target_name),
                    file_path=result.get("file_path", ""),
                    line_start=result.get("line_start", 0),
                    line_end=result.get("line_end", 0),
                    chunk_type=chunk_type,
                    language=result.get("language", ""),
                )

                candidates.append(candidate)

            except Exception as e:
                self.logger.debug("Error processing call target search result: %s", str(e))
                continue

        return candidates

    def _extract_target_function_name(self, call_expression: str) -> str:
        """
        Extract the target function name from a call expression.

        Args:
            call_expression: Full call expression (e.g., "self.helper.process_data()")

        Returns:
            Target function name (e.g., "process_data")
        """
        # Remove arguments first
        if "(" in call_expression:
            func_part = call_expression.split("(")[0].strip()
        else:
            func_part = call_expression.strip()

        # Extract final component after last dot
        if "." in func_part:
            return func_part.split(".")[-1]
        else:
            return func_part

    def _calculate_call_target_confidence(self, result: dict[str, Any], function_call: FunctionCall, target_name: str, query: str) -> float:
        """
        Calculate confidence score specifically for call target matching.

        This is enhanced version of confidence calculation that considers
        function call context and target matching specifics.

        Args:
            result: Search result dictionary
            function_call: Original function call
            target_name: Extracted target function name
            query: Search query

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Base score from search relevance
        search_score = result.get("score", 0.0)
        score += search_score * 0.3  # 30% weight for search relevance

        # Function name matching (higher weight for call resolution)
        result_name = result.get("name", "")
        if result_name:
            name_score = self._calculate_call_target_name_similarity(result_name, target_name)
            score += name_score * 0.4  # 40% weight for name matching

        # Chunk type relevance (prefer functions/methods for call targets)
        chunk_type = result.get("chunk_type", "")
        if chunk_type == "function":
            score += 0.15  # 15% bonus for function definitions
        elif chunk_type == "method":
            score += 0.1  # 10% bonus for method definitions

        # Signature and parameter matching
        signature = result.get("signature", "")
        if signature:
            signature_score = self._calculate_signature_compatibility(signature, function_call.arguments_count)
            score += signature_score * 0.1  # 10% weight for signature compatibility

        # Language and file context matching
        if result.get("language") == "python":  # Assuming Python for now
            score += 0.05  # 5% bonus for language match

        # Normalize to 0.0-1.0 range
        return min(score, 1.0)

    def _calculate_call_target_name_similarity(self, result_name: str, target_name: str) -> float:
        """
        Calculate name similarity specifically for call target matching.

        Args:
            result_name: Name from search result
            target_name: Target function name from call

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not result_name or not target_name:
            return 0.0

        result_lower = result_name.lower()
        target_lower = target_name.lower()

        # Exact match gets highest score
        if result_lower == target_lower:
            return 1.0

        # Check for exact substring match
        if result_lower in target_lower or target_lower in result_lower:
            return 0.9

        # Check for word similarity in camelCase/snake_case
        result_words = re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", result_name)
        target_words = re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", target_name)

        if result_words and target_words:
            common_words = set(w.lower() for w in result_words) & set(w.lower() for w in target_words)
            if common_words:
                overlap_ratio = len(common_words) / max(len(result_words), len(target_words))
                return overlap_ratio * 0.8

        # Fuzzy similarity for remaining cases
        return self._fuzzy_similarity(result_lower, target_lower) * 0.6

    def _calculate_signature_compatibility(self, signature: str, call_arg_count: int) -> float:
        """
        Calculate compatibility between function signature and call arguments.

        Args:
            signature: Function signature from search result
            call_arg_count: Number of arguments in the function call

        Returns:
            Compatibility score between 0.0 and 1.0
        """
        if not signature:
            return 0.0

        try:
            # Count parameters in signature (simplified)
            if "(" in signature and ")" in signature:
                params_part = signature.split("(", 1)[1].split(")", 1)[0]

                # Handle empty parameters
                if not params_part.strip():
                    param_count = 0
                else:
                    # Simple parameter counting (ignores complex cases)
                    param_count = params_part.count(",") + 1

                # Calculate compatibility based on argument count difference
                if param_count == call_arg_count:
                    return 1.0
                elif abs(param_count - call_arg_count) <= 1:
                    return 0.7  # Close match
                elif abs(param_count - call_arg_count) <= 2:
                    return 0.4  # Reasonable match
                else:
                    return 0.1  # Poor match

        except Exception:
            return 0.0

        return 0.0

    def _generate_call_target_reasoning(self, result: dict[str, Any], function_call: FunctionCall, target_name: str) -> str:
        """
        Generate reasoning for call target candidate selection.

        Args:
            result: Search result dictionary
            function_call: Original function call
            target_name: Target function name

        Returns:
            Human-readable reasoning string
        """
        reasons = []

        # Name matching
        result_name = result.get("name", "")
        if result_name:
            name_similarity = self._calculate_call_target_name_similarity(result_name, target_name)
            if name_similarity >= 0.9:
                reasons.append(f"function name '{result_name}' exactly matches call target '{target_name}'")
            elif name_similarity >= 0.7:
                reasons.append(f"function name '{result_name}' closely matches call target '{target_name}'")
            elif name_similarity >= 0.5:
                reasons.append(f"function name '{result_name}' partially matches call target '{target_name}'")

        # Chunk type
        chunk_type = result.get("chunk_type", "")
        if chunk_type in ["function", "method"]:
            reasons.append(f"is a {chunk_type} definition")

        # Signature compatibility
        signature = result.get("signature", "")
        if signature:
            compat_score = self._calculate_signature_compatibility(signature, function_call.arguments_count)
            if compat_score >= 0.7:
                reasons.append(f"signature {signature} is compatible with {function_call.arguments_count} arguments")

        # File and location context
        file_path = result.get("file_path", "")
        if file_path:
            reasons.append(f"found in {file_path}")

        # Search relevance
        score = result.get("score", 0.0)
        if score > 0.8:
            reasons.append("high semantic relevance to call expression")

        if not reasons:
            reasons.append("identified as potential call target")

        return "Selected as call target because: " + ", ".join(reasons)

    def _determine_call_target_match_type(self, result: dict[str, Any], function_call: FunctionCall, target_name: str) -> str:
        """
        Determine the type of match for call target resolution.

        Args:
            result: Search result dictionary
            function_call: Original function call
            target_name: Target function name

        Returns:
            Match type string
        """
        result_name = result.get("name", "")

        if result_name:
            name_similarity = self._calculate_call_target_name_similarity(result_name, target_name)
            if name_similarity >= 0.9:
                return "exact_name"
            elif name_similarity >= 0.7:
                return "close_name"
            elif name_similarity >= 0.5:
                return "partial_name"

        # Check semantic score
        score = result.get("score", 0.0)
        if score > 0.8:
            return "semantic"
        elif score > 0.6:
            return "contextual"
        else:
            return "fuzzy"

    def _filter_and_rank_call_candidates(
        self, candidates: list[BreadcrumbCandidate], function_call: FunctionCall
    ) -> list[BreadcrumbCandidate]:
        """
        Filter and rank call target candidates with call-specific logic.

        Args:
            candidates: List of candidates to filter and rank
            function_call: Original function call for context

        Returns:
            Filtered and sorted list of candidates
        """
        # Use slightly lower threshold for call resolution
        call_min_threshold = max(0.2, self.min_confidence_threshold - 0.1)

        # Filter by minimum confidence
        filtered = [c for c in candidates if c.confidence_score >= call_min_threshold]

        # Sort by confidence score (descending) with tiebreaking
        filtered.sort(
            key=lambda c: (
                c.confidence_score,
                1.0 if c.match_type == "exact_name" else 0.0,
                -len(c.breadcrumb),  # Prefer shorter breadcrumbs for tiebreaking
            ),
            reverse=True,
        )

        # Remove duplicates (same breadcrumb)
        seen_breadcrumbs = set()
        unique_candidates = []
        for candidate in filtered:
            if candidate.breadcrumb not in seen_breadcrumbs:
                seen_breadcrumbs.add(candidate.breadcrumb)
                unique_candidates.append(candidate)

        return unique_candidates

    async def _validate_breadcrumb_exists(self, breadcrumb: str, target_projects: list[str] | None) -> BreadcrumbResolutionResult:
        """
        Validate that a breadcrumb actually exists in the codebase.

        Args:
            breadcrumb: Breadcrumb to validate
            target_projects: Projects to search in

        Returns:
            BreadcrumbResolutionResult indicating if breadcrumb exists
        """
        try:
            # Search for the breadcrumb directly
            search_results = await search_async_cached(
                query=breadcrumb,
                n_results=5,
                search_mode="keyword",  # Use keyword search for exact matching
                include_context=False,
                target_projects=target_projects,
            )

            if search_results.get("results"):
                # Check if any result has matching breadcrumb
                for result in search_results["results"]:
                    result_breadcrumb = result.get("breadcrumb", "")
                    if result_breadcrumb == breadcrumb:
                        # Found exact match
                        candidate = BreadcrumbCandidate(
                            breadcrumb=breadcrumb,
                            confidence_score=1.0,
                            source_chunk=self._create_code_chunk_from_result(result),
                            reasoning="Exact breadcrumb match found in codebase",
                            match_type="exact",
                            file_path=result.get("file_path", ""),
                            line_start=result.get("line_start", 0),
                            line_end=result.get("line_end", 0),
                            chunk_type=result.get("chunk_type", ""),
                            language=result.get("language", ""),
                        )

                        return BreadcrumbResolutionResult(
                            query=breadcrumb, success=True, primary_candidate=candidate, search_results_count=len(search_results["results"])
                        )

            # No exact match found
            return BreadcrumbResolutionResult(
                query=breadcrumb,
                success=False,
                error_message="Breadcrumb not found in codebase",
                search_results_count=len(search_results.get("results", [])),
            )

        except Exception as e:
            return BreadcrumbResolutionResult(query=breadcrumb, success=False, error_message=f"Error validating breadcrumb: {str(e)}")

    def _create_call_cache_key(self, function_call: FunctionCall, target_projects: list[str] | None) -> str:
        """
        Create a cache key for function call resolution.

        Args:
            function_call: FunctionCall object
            target_projects: Optional list of target projects

        Returns:
            Cache key string
        """
        projects_str = ",".join(sorted(target_projects)) if target_projects else "all"
        call_hash = hashlib.md5(function_call.call_expression.encode()).hexdigest()[:8]
        return f"call:{call_hash}|{projects_str}"
