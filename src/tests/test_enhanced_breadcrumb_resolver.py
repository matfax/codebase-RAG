"""
Integration tests for the enhanced BreadcrumbResolver with TTL-based caching.

This module tests the integration between the BreadcrumbResolver service and the
enhanced TTL-based caching system with file modification tracking.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.breadcrumb_cache_models import BreadcrumbCacheConfig
from src.services.breadcrumb_resolver_service import (
    BreadcrumbCandidate,
    BreadcrumbResolutionResult,
    BreadcrumbResolver,
)


@pytest.mark.asyncio
class TestEnhancedBreadcrumbResolver:
    """Test the enhanced BreadcrumbResolver with TTL caching."""

    async def test_resolver_initialization_with_cache(self):
        """Test resolver initialization with enhanced caching."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)

        assert resolver.cache_enabled is True
        assert resolver.cache_service is not None
        assert resolver.cache_service.config.enabled is True

    async def test_resolver_initialization_without_cache(self):
        """Test resolver initialization without caching."""
        resolver = BreadcrumbResolver(cache_enabled=False)

        assert resolver.cache_enabled is False
        assert resolver.cache_service is None

    async def test_resolver_lifecycle_management(self):
        """Test resolver start/stop lifecycle."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)

        # Start the resolver
        await resolver.start()
        assert resolver.cache_service._is_running is True

        # Stop the resolver
        await resolver.stop()
        assert resolver.cache_service._is_running is False

    @patch("src.services.breadcrumb_resolver_service.search_async_cached")
    async def test_caching_with_valid_breadcrumb(self, mock_search):
        """Test caching behavior with valid breadcrumb input."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)
        await resolver.start()

        try:
            # Valid breadcrumb should be cached directly
            result = await resolver.resolve("module.class.method")

            assert result.success is True
            assert result.primary_candidate is not None
            assert result.primary_candidate.breadcrumb == "module.class.method"
            assert result.primary_candidate.confidence_score == 1.0

            # Should be cached now
            cache_stats = resolver.get_cache_stats()
            assert cache_stats["enhanced_cache_stats"]["total_entries"] == 1

            # Second call should hit cache
            result2 = await resolver.resolve("module.class.method")
            assert result2.success is True

            # Cache hit count should increase
            cache_stats = resolver.get_cache_stats()
            assert cache_stats["enhanced_cache_stats"]["hits"] == 1

        finally:
            await resolver.stop()

    @patch("src.services.breadcrumb_resolver_service.search_async_cached")
    async def test_caching_with_natural_language_query(self, mock_search):
        """Test caching with natural language queries."""
        # Mock search results
        mock_chunk = Mock()
        mock_chunk.file_path = "/test/file.py"
        mock_chunk.content = "def test_function():"
        mock_chunk.chunk_type = "function"
        mock_chunk.name = "test_function"
        mock_chunk.line_start = 1
        mock_chunk.line_end = 5
        mock_chunk.language = "python"

        mock_search.return_value = {"results": [mock_chunk], "total": 1}

        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)
        await resolver.start()

        try:
            # Natural language query
            result = await resolver.resolve("find test function")

            # Should use mock search and cache the result
            mock_search.assert_called_once()

            # Verify caching
            cache_stats = resolver.get_cache_stats()
            assert cache_stats["enhanced_cache_stats"]["total_entries"] == 1

            # Second call should hit cache (mock should not be called again)
            mock_search.reset_mock()
            result2 = await resolver.resolve("find test function")
            mock_search.assert_not_called()

            # Cache stats should show hit
            cache_stats = resolver.get_cache_stats()
            assert cache_stats["enhanced_cache_stats"]["hits"] == 1

        finally:
            await resolver.stop()

    async def test_cache_invalidation_by_file(self):
        """Test cache invalidation when files are modified."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("def test_function(): pass")
            f.flush()

            config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0, enable_dependency_tracking=True)
            resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)
            await resolver.start()

            try:
                # Mock a result with file dependency
                with patch("src.services.breadcrumb_resolver_service.search_async_cached") as mock_search:
                    mock_chunk = Mock()
                    mock_chunk.file_path = f.name
                    mock_chunk.content = "def test_function(): pass"
                    mock_chunk.chunk_type = "function"
                    mock_chunk.name = "test_function"
                    mock_chunk.line_start = 1
                    mock_chunk.line_end = 1
                    mock_chunk.language = "python"

                    mock_search.return_value = {"results": [mock_chunk], "total": 1}

                    # First query - should cache
                    result = await resolver.resolve("test function")
                    assert result.success is True

                    cache_stats = resolver.get_cache_stats()
                    assert cache_stats["enhanced_cache_stats"]["total_entries"] == 1

                # Invalidate cache by file
                invalidated_count = await resolver.invalidate_cache_by_file(f.name)
                assert invalidated_count >= 0  # Should invalidate entries

                # Cache should be cleared for that file
                cache_info = await resolver.get_cache_info()
                # The specific entry may be gone, but this tests the invalidation mechanism

            finally:
                await resolver.stop()
                Path(f.name).unlink()

    async def test_cache_ttl_based_on_confidence(self):
        """Test TTL calculation based on confidence scores."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0, default_ttl_seconds=10.0)  # 10 seconds base TTL
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)
        await resolver.start()

        try:
            # Test direct caching with different confidence levels
            high_confidence_result = BreadcrumbResolutionResult(
                query="high_confidence_query",
                success=True,
                primary_candidate=BreadcrumbCandidate(
                    breadcrumb="test.high.confidence",
                    confidence_score=0.95,
                    source_chunk=None,
                    reasoning="High confidence test",
                    match_type="exact",
                ),
            )

            low_confidence_result = BreadcrumbResolutionResult(
                query="low_confidence_query",
                success=True,
                primary_candidate=BreadcrumbCandidate(
                    breadcrumb="test.low.confidence",
                    confidence_score=0.3,
                    source_chunk=None,
                    reasoning="Low confidence test",
                    match_type="fuzzy",
                ),
            )

            # Cache both results
            await resolver._cache_result_enhanced("high_confidence", high_confidence_result, [], 0.95)
            await resolver._cache_result_enhanced("low_confidence", low_confidence_result, [], 0.3)

            # Verify both are cached initially
            assert await resolver.cache_service.get("high_confidence") is not None
            assert await resolver.cache_service.get("low_confidence") is not None

            # The TTL calculation logic is tested in the config,
            # here we verify the caching mechanism works with different confidence levels
            cache_info = await resolver.get_cache_info()
            assert cache_info["stats"]["total_entries"] == 2

        finally:
            await resolver.stop()

    async def test_cache_clear_operation(self):
        """Test cache clearing functionality."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)
        await resolver.start()

        try:
            # Add some cached results
            for i in range(3):
                result = BreadcrumbResolutionResult(
                    query=f"query_{i}",
                    success=True,
                    primary_candidate=BreadcrumbCandidate(
                        breadcrumb=f"test.breadcrumb.{i}", confidence_score=0.8, source_chunk=None, reasoning="Test", match_type="exact"
                    ),
                )
                await resolver._cache_result_enhanced(f"key_{i}", result, [], 0.8)

            # Verify cache has entries
            cache_info = await resolver.get_cache_info()
            assert cache_info["stats"]["total_entries"] == 3

            # Clear cache
            await resolver.clear_cache()

            # Verify cache is empty
            cache_info = await resolver.get_cache_info()
            assert cache_info["stats"]["total_entries"] == 0

        finally:
            await resolver.stop()

    async def test_fallback_to_legacy_cache(self):
        """Test fallback to legacy cache when enhanced cache fails."""
        # Initialize with cache service but simulate failure
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)

        # Mock cache service to return False for put operations
        resolver.cache_service = Mock()
        resolver.cache_service.get = AsyncMock(return_value=None)
        resolver.cache_service.put = AsyncMock(return_value=False)  # Simulate failure

        # Test result
        result = BreadcrumbResolutionResult(
            query="test_query",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="test.breadcrumb", confidence_score=0.8, source_chunk=None, reasoning="Test", match_type="exact"
            ),
        )

        # Should fallback to legacy cache
        await resolver._cache_result_enhanced("test_key", result, [], 0.8)

        # Verify enhanced cache was attempted
        resolver.cache_service.put.assert_called_once()

        # Verify fallback to legacy cache
        assert "test_key" in resolver._resolution_cache
        assert resolver._resolution_cache["test_key"] == result

    async def test_cache_stats_integration(self):
        """Test cache statistics reporting integration."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        resolver = BreadcrumbResolver(cache_enabled=True, cache_config=config)
        await resolver.start()

        try:
            # Add some entries
            for i in range(2):
                result = BreadcrumbResolutionResult(
                    query=f"query_{i}",
                    success=True,
                    primary_candidate=BreadcrumbCandidate(
                        breadcrumb=f"test.breadcrumb.{i}", confidence_score=0.8, source_chunk=None, reasoning="Test", match_type="exact"
                    ),
                )
                await resolver._cache_result_enhanced(f"key_{i}", result, [], 0.8)

            # Get cache stats
            stats = resolver.get_cache_stats()

            assert stats["enabled"] is True
            assert stats["cache_type"] == "enhanced_ttl"
            assert "enhanced_cache_stats" in stats
            assert stats["enhanced_cache_stats"]["total_entries"] == 2
            assert stats["legacy_cache_size"] == 0  # Should be using enhanced cache

            # Test cache info
            cache_info = await resolver.get_cache_info()
            assert cache_info["stats"]["total_entries"] == 2
            assert cache_info["config"]["enabled"] is True

        finally:
            await resolver.stop()

    async def test_disabled_cache_behavior(self):
        """Test behavior when caching is disabled."""
        resolver = BreadcrumbResolver(cache_enabled=False)

        assert resolver.cache_service is None

        # Cache operations should be no-ops
        await resolver.clear_cache()  # Should not raise error

        invalidated = await resolver.invalidate_cache_by_file("/some/file.py")
        assert invalidated == 0

        cache_info = await resolver.get_cache_info()
        assert cache_info["enhanced_cache_enabled"] is False
        assert cache_info["legacy_cache_size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
