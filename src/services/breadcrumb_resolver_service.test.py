"""
Unit tests for BreadcrumbResolver Service.

This module provides comprehensive unit tests for the BreadcrumbResolver service,
covering normal operations, edge cases, error conditions, and performance scenarios.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.code_chunk import CodeChunk
from src.services.breadcrumb_resolver_service import BreadcrumbCandidate, BreadcrumbFormat, BreadcrumbResolutionResult, BreadcrumbResolver


class TestBreadcrumbResolver:
    """Test suite for BreadcrumbResolver service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = BreadcrumbResolver(cache_enabled=True)

    def test_init_with_cache_enabled(self):
        """Test initialization with cache enabled."""
        resolver = BreadcrumbResolver(cache_enabled=True)
        assert resolver.cache_enabled is True
        assert resolver.max_candidates == 5
        assert resolver.min_confidence_threshold == 0.3
        assert resolver.semantic_search_results == 10
        assert len(resolver._resolution_cache) == 0

    def test_init_with_cache_disabled(self):
        """Test initialization with cache disabled."""
        resolver = BreadcrumbResolver(cache_enabled=False)
        assert resolver.cache_enabled is False
        assert len(resolver._resolution_cache) == 0

    def test_is_valid_breadcrumb_dotted_format(self):
        """Test validation of dotted format breadcrumbs."""
        # Valid dotted format
        assert self.resolver.is_valid_breadcrumb("module.class.method") is True
        assert self.resolver.is_valid_breadcrumb("pkg.subpkg.module.function") is True
        assert self.resolver.is_valid_breadcrumb("a.b") is True

        # Invalid dotted format
        assert self.resolver.is_valid_breadcrumb(".module.class") is False
        assert self.resolver.is_valid_breadcrumb("module.class.") is False
        assert self.resolver.is_valid_breadcrumb("module..class") is False
        assert self.resolver.is_valid_breadcrumb("module.class method") is False

    def test_is_valid_breadcrumb_double_colon_format(self):
        """Test validation of double colon format breadcrumbs."""
        # Valid double colon format
        assert self.resolver.is_valid_breadcrumb("namespace::class::method") is True
        assert self.resolver.is_valid_breadcrumb("std::vector::push_back") is True
        assert self.resolver.is_valid_breadcrumb("a::b") is True

        # Invalid double colon format
        assert self.resolver.is_valid_breadcrumb("::namespace::class") is False
        assert self.resolver.is_valid_breadcrumb("namespace::class::") is False
        assert self.resolver.is_valid_breadcrumb("namespace class") is False

    def test_is_valid_breadcrumb_slash_format(self):
        """Test validation of slash format breadcrumbs."""
        # Valid slash format
        assert self.resolver.is_valid_breadcrumb("module/class/method") is True
        assert self.resolver.is_valid_breadcrumb("a/b/c") is True

        # Invalid slash format
        assert self.resolver.is_valid_breadcrumb("/module/class") is False
        assert self.resolver.is_valid_breadcrumb("module/class/") is False

    def test_is_valid_breadcrumb_arrow_format(self):
        """Test validation of arrow format breadcrumbs."""
        # Valid arrow format
        assert self.resolver.is_valid_breadcrumb("module->class->method") is True
        assert self.resolver.is_valid_breadcrumb("a->b->c") is True

        # Invalid arrow format
        assert self.resolver.is_valid_breadcrumb("->module->class") is False
        assert self.resolver.is_valid_breadcrumb("module->class->") is False

    def test_is_valid_breadcrumb_edge_cases(self):
        """Test edge cases for breadcrumb validation."""
        # None and empty cases
        assert self.resolver.is_valid_breadcrumb(None) is False
        assert self.resolver.is_valid_breadcrumb("") is False
        assert self.resolver.is_valid_breadcrumb("   ") is False

        # Non-string input
        assert self.resolver.is_valid_breadcrumb(123) is False
        assert self.resolver.is_valid_breadcrumb([]) is False

        # Single component (no separator)
        assert self.resolver.is_valid_breadcrumb("function") is False
        assert self.resolver.is_valid_breadcrumb("class") is False

        # Too short
        assert self.resolver.is_valid_breadcrumb("a") is False
        assert self.resolver.is_valid_breadcrumb("ab") is False

    def test_validate_breadcrumb_structure(self):
        """Test internal structure validation."""
        # Valid structures
        assert self.resolver._validate_breadcrumb_structure("a.b") is True
        assert self.resolver._validate_breadcrumb_structure("namespace::class") is True

        # Invalid structures
        assert self.resolver._validate_breadcrumb_structure("") is False
        assert self.resolver._validate_breadcrumb_structure("a") is False
        assert self.resolver._validate_breadcrumb_structure("a b") is False
        assert self.resolver._validate_breadcrumb_structure(".a") is False
        assert self.resolver._validate_breadcrumb_structure("a.") is False

    @pytest.mark.asyncio
    async def test_resolve_with_valid_breadcrumb(self):
        """Test resolve method with already valid breadcrumb."""
        valid_breadcrumb = "module.class.method"

        result = await self.resolver.resolve(valid_breadcrumb)

        assert result.success is True
        assert result.query == valid_breadcrumb
        assert result.primary_candidate is not None
        assert result.primary_candidate.breadcrumb == valid_breadcrumb
        assert result.primary_candidate.confidence_score == 1.0
        assert result.primary_candidate.match_type == "exact"
        assert result.error_message == ""

    @pytest.mark.asyncio
    async def test_resolve_with_invalid_input(self):
        """Test resolve method with invalid input."""
        # None input
        result = await self.resolver.resolve(None)
        assert result.success is False
        assert "Query must be a non-empty string" in result.error_message

        # Empty string
        result = await self.resolver.resolve("")
        assert result.success is False
        assert "Query cannot be empty" in result.error_message

        # Whitespace only
        result = await self.resolver.resolve("   ")
        assert result.success is False
        assert "Query cannot be empty" in result.error_message

    @pytest.mark.asyncio
    async def test_resolve_with_caching(self):
        """Test resolve method with caching enabled."""
        query = "test query"

        # Mock search results for natural language conversion
        mock_search_results = {
            "results": [
                {
                    "name": "test_function",
                    "parent_name": "TestClass",
                    "breadcrumb": "module.TestClass.test_function",
                    "score": 0.9,
                    "content": "def test_function(): pass",
                    "file_path": "test.py",
                    "line_start": 1,
                    "line_end": 1,
                    "chunk_type": "function",
                    "language": "python",
                }
            ]
        }

        with patch("src.services.breadcrumb_resolver_service.search_async_cached", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_results

            # First call should perform search
            result1 = await self.resolver.resolve(query)
            assert result1.success is True
            assert mock_search.call_count == 1

            # Second call should use cache
            result2 = await self.resolver.resolve(query)
            assert result2.success is True
            assert mock_search.call_count == 1  # Should not increase

            # Results should be identical
            assert result1.primary_candidate.breadcrumb == result2.primary_candidate.breadcrumb

    @pytest.mark.asyncio
    async def test_convert_natural_to_breadcrumb_success(self):
        """Test successful natural language to breadcrumb conversion."""
        query = "find user authentication function"

        mock_search_results = {
            "results": [
                {
                    "name": "authenticate_user",
                    "parent_name": "AuthService",
                    "breadcrumb": "auth.AuthService.authenticate_user",
                    "score": 0.85,
                    "content": "def authenticate_user(username, password): pass",
                    "file_path": "auth.py",
                    "line_start": 10,
                    "line_end": 15,
                    "chunk_type": "method",
                    "language": "python",
                }
            ]
        }

        with patch("src.services.breadcrumb_resolver_service.search_async_cached", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_results

            result = await self.resolver.convert_natural_to_breadcrumb(query)

            assert result.success is True
            assert result.primary_candidate is not None
            assert result.primary_candidate.breadcrumb == "auth.AuthService.authenticate_user"
            assert result.primary_candidate.confidence_score > 0.3
            assert result.search_results_count == 1

    @pytest.mark.asyncio
    async def test_convert_natural_to_breadcrumb_no_results(self):
        """Test conversion when no search results found."""
        query = "nonexistent function"

        mock_search_results = {"results": []}

        with patch("src.services.breadcrumb_resolver_service.search_async_cached", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_results

            result = await self.resolver.convert_natural_to_breadcrumb(query)

            assert result.success is False
            assert "No relevant code found" in result.error_message
            assert result.search_results_count == 0

    @pytest.mark.asyncio
    async def test_convert_natural_to_breadcrumb_low_confidence(self):
        """Test conversion when all candidates have low confidence."""
        query = "test function"

        mock_search_results = {
            "results": [
                {
                    "name": "unrelated_function",
                    "parent_name": "UnrelatedClass",
                    "breadcrumb": "module.UnrelatedClass.unrelated_function",
                    "score": 0.1,  # Very low score
                    "content": "def unrelated_function(): pass",
                    "file_path": "test.py",
                    "line_start": 1,
                    "line_end": 1,
                    "chunk_type": "function",
                    "language": "python",
                }
            ]
        }

        with patch("src.services.breadcrumb_resolver_service.search_async_cached", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_search_results

            result = await self.resolver.convert_natural_to_breadcrumb(query)

            assert result.success is False
            assert "minimum confidence threshold" in result.error_message

    def test_extract_breadcrumb_from_result(self):
        """Test breadcrumb extraction from search result."""
        # Result with direct breadcrumb
        result1 = {"breadcrumb": "module.class.method", "name": "method", "parent_name": "class"}

        breadcrumb = self.resolver._extract_breadcrumb_from_result(result1)
        assert breadcrumb == "module.class.method"

        # Result without breadcrumb, build from components
        result2 = {"name": "method", "parent_name": "class", "language": "python"}

        breadcrumb = self.resolver._extract_breadcrumb_from_result(result2)
        assert breadcrumb == "class.method"

        # Result with C++ language
        result3 = {"name": "method", "parent_name": "class", "language": "cpp"}

        breadcrumb = self.resolver._extract_breadcrumb_from_result(result3)
        assert breadcrumb == "class::method"

        # Result with insufficient data
        result4 = {"name": "function"}

        breadcrumb = self.resolver._extract_breadcrumb_from_result(result4)
        assert breadcrumb == "function"

    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        query = "test function"

        # High quality result
        result1 = {"score": 0.9, "content": "def test_function(): pass", "name": "test_function", "chunk_type": "function"}

        score = self.resolver._calculate_confidence_score(result1, query)
        assert score > 0.7

        # Low quality result
        result2 = {"score": 0.2, "content": "x = 1", "name": "unrelated_var", "chunk_type": "variable"}

        score = self.resolver._calculate_confidence_score(result2, query)
        assert score < 0.5

    def test_calculate_name_similarity(self):
        """Test name similarity calculation."""
        # Exact match
        assert self.resolver._calculate_name_similarity("test_function", "test_function") == 1.0

        # Substring match
        assert self.resolver._calculate_name_similarity("test_function", "test") == 0.8

        # No match
        assert self.resolver._calculate_name_similarity("unrelated", "test") < 0.3

        # Empty inputs
        assert self.resolver._calculate_name_similarity("", "test") == 0.0
        assert self.resolver._calculate_name_similarity("test", "") == 0.0

    def test_fuzzy_similarity(self):
        """Test fuzzy similarity calculation."""
        # Identical strings
        assert self.resolver._fuzzy_similarity("test", "test") == 1.0

        # Similar strings
        similarity = self.resolver._fuzzy_similarity("test", "tests")
        assert 0.5 < similarity < 1.0

        # Different strings
        similarity = self.resolver._fuzzy_similarity("test", "xyz")
        assert similarity < 0.5

        # Empty strings
        assert self.resolver._fuzzy_similarity("", "") == 1.0
        assert self.resolver._fuzzy_similarity("", "test") == 0.0

    def test_filter_and_rank_candidates(self):
        """Test candidate filtering and ranking."""
        candidates = [
            BreadcrumbCandidate(
                breadcrumb="module.class.method1", confidence_score=0.9, source_chunk=None, reasoning="High confidence", match_type="exact"
            ),
            BreadcrumbCandidate(
                breadcrumb="module.class.method2",
                confidence_score=0.7,
                source_chunk=None,
                reasoning="Good confidence",
                match_type="partial",
            ),
            BreadcrumbCandidate(
                breadcrumb="module.class.method3",
                confidence_score=0.2,  # Below threshold
                source_chunk=None,
                reasoning="Low confidence",
                match_type="fuzzy",
            ),
            BreadcrumbCandidate(
                breadcrumb="module.class.method1",  # Duplicate
                confidence_score=0.8,
                source_chunk=None,
                reasoning="Duplicate",
                match_type="exact",
            ),
        ]

        filtered = self.resolver._filter_and_rank_candidates(candidates, "test")

        # Should filter out low confidence and duplicates
        assert len(filtered) == 2
        assert filtered[0].confidence_score == 0.9
        assert filtered[1].confidence_score == 0.7

        # Should be sorted by confidence
        assert filtered[0].confidence_score > filtered[1].confidence_score

    def test_cache_operations(self):
        """Test cache operations."""
        # Test cache key creation
        key = self.resolver._create_cache_key("test query", ["project1", "project2"])
        assert "test query" in key
        assert "project1,project2" in key

        # Test cache get/set
        result = BreadcrumbResolutionResult(
            query="test",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="test.breadcrumb", confidence_score=0.8, source_chunk=None, reasoning="Test", match_type="exact"
            ),
        )

        cache_key = "test_key"
        self.resolver._cache_result(cache_key, result)

        cached_result = self.resolver._get_from_cache(cache_key)
        assert cached_result is not None
        assert cached_result.query == "test"
        assert cached_result.success is True

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.resolver._resolution_cache["test"] = "value"
        assert len(self.resolver._resolution_cache) == 1

        # Clear cache
        self.resolver.clear_cache()
        assert len(self.resolver._resolution_cache) == 0

    def test_get_cache_stats(self):
        """Test cache statistics."""
        # Add some items to cache
        self.resolver._resolution_cache["test1"] = "value1"
        self.resolver._resolution_cache["test2"] = "value2"

        stats = self.resolver.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["size"] == 2
        assert "configuration" in stats
        assert stats["configuration"]["max_candidates"] == 5

    def test_breadcrumb_candidate_to_dict(self):
        """Test BreadcrumbCandidate serialization."""
        candidate = BreadcrumbCandidate(
            breadcrumb="test.breadcrumb",
            confidence_score=0.8,
            source_chunk=None,
            reasoning="Test reasoning",
            match_type="exact",
            file_path="test.py",
            line_start=1,
            line_end=10,
            chunk_type="function",
            language="python",
        )

        result_dict = candidate.to_dict()

        assert result_dict["breadcrumb"] == "test.breadcrumb"
        assert result_dict["confidence_score"] == 0.8
        assert result_dict["match_type"] == "exact"
        assert result_dict["file_path"] == "test.py"
        assert result_dict["language"] == "python"

    def test_breadcrumb_resolution_result_to_dict(self):
        """Test BreadcrumbResolutionResult serialization."""
        candidate = BreadcrumbCandidate(
            breadcrumb="test.breadcrumb", confidence_score=0.8, source_chunk=None, reasoning="Test reasoning", match_type="exact"
        )

        result = BreadcrumbResolutionResult(
            query="test query",
            success=True,
            primary_candidate=candidate,
            alternative_candidates=[],
            resolution_time_ms=100.0,
            search_results_count=5,
        )

        result_dict = result.to_dict()

        assert result_dict["query"] == "test query"
        assert result_dict["success"] is True
        assert result_dict["primary_candidate"]["breadcrumb"] == "test.breadcrumb"
        assert result_dict["resolution_time_ms"] == 100.0
        assert result_dict["search_results_count"] == 5


class TestBreadcrumbCandidate:
    """Test suite for BreadcrumbCandidate dataclass."""

    def test_creation_with_minimal_data(self):
        """Test candidate creation with minimal required data."""
        candidate = BreadcrumbCandidate(
            breadcrumb="test.breadcrumb", confidence_score=0.8, source_chunk=None, reasoning="Test reasoning", match_type="exact"
        )

        assert candidate.breadcrumb == "test.breadcrumb"
        assert candidate.confidence_score == 0.8
        assert candidate.match_type == "exact"
        assert candidate.file_path == ""
        assert candidate.line_start == 0

    def test_creation_with_full_data(self):
        """Test candidate creation with all data."""
        candidate = BreadcrumbCandidate(
            breadcrumb="test.breadcrumb",
            confidence_score=0.8,
            source_chunk=None,
            reasoning="Test reasoning",
            match_type="exact",
            file_path="test.py",
            line_start=1,
            line_end=10,
            chunk_type="function",
            language="python",
        )

        assert candidate.file_path == "test.py"
        assert candidate.line_start == 1
        assert candidate.line_end == 10
        assert candidate.chunk_type == "function"
        assert candidate.language == "python"


class TestBreadcrumbResolutionResult:
    """Test suite for BreadcrumbResolutionResult dataclass."""

    def test_creation_with_success(self):
        """Test result creation for successful resolution."""
        candidate = BreadcrumbCandidate(
            breadcrumb="test.breadcrumb", confidence_score=0.8, source_chunk=None, reasoning="Test reasoning", match_type="exact"
        )

        result = BreadcrumbResolutionResult(query="test query", success=True, primary_candidate=candidate, resolution_time_ms=100.0)

        assert result.query == "test query"
        assert result.success is True
        assert result.primary_candidate == candidate
        assert result.alternative_candidates == []
        assert result.error_message == ""
        assert result.resolution_time_ms == 100.0

    def test_creation_with_failure(self):
        """Test result creation for failed resolution."""
        result = BreadcrumbResolutionResult(query="test query", success=False, error_message="Test error")

        assert result.query == "test query"
        assert result.success is False
        assert result.primary_candidate is None
        assert result.error_message == "Test error"

    def test_post_init_alternative_candidates(self):
        """Test that alternative_candidates is initialized properly."""
        result = BreadcrumbResolutionResult(query="test", success=True)

        assert result.alternative_candidates == []
        assert isinstance(result.alternative_candidates, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
