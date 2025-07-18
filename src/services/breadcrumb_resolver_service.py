"""
BreadcrumbResolver Service for Function Chain MCP Tools.

This service provides natural language to breadcrumb conversion capabilities,
enabling users to use natural language descriptions to find and resolve
precise breadcrumb paths in codebases. It integrates with the existing
search infrastructure to provide semantic conversion with confidence scoring.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.models.code_chunk import CodeChunk
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

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the BreadcrumbResolver service.

        Args:
            cache_enabled: Whether to enable caching of resolution results
        """
        self.logger = logging.getLogger(__name__)
        self.cache_enabled = cache_enabled
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
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.logger.debug("Cache hit for query: %s", query[:50])
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
                    self._cache_result(cache_key, result)

                return result

            # Convert natural language to breadcrumb
            result = await self.convert_natural_to_breadcrumb(query, target_projects)

            # Cache the result
            if self.cache_enabled:
                cache_key = self._create_cache_key(query, target_projects)
                self._cache_result(cache_key, result)

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

    def _cache_result(self, cache_key: str, result: BreadcrumbResolutionResult):
        """
        Cache a resolution result.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        # Simple cache implementation - in production you'd want TTL, size limits, etc.
        self._resolution_cache[cache_key] = result

    def clear_cache(self):
        """Clear the resolution cache."""
        self._resolution_cache.clear()
        self.logger.info("BreadcrumbResolver cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self.cache_enabled,
            "size": len(self._resolution_cache),
            "configuration": {
                "max_candidates": self.max_candidates,
                "min_confidence_threshold": self.min_confidence_threshold,
                "semantic_search_results": self.semantic_search_results,
            },
        }
