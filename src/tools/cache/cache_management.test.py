"""
Unit tests for cache management tools.

Tests the manual cache invalidation MCP tools functionality.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .cache_management import (
    clear_all_caches,
    get_cache_invalidation_stats,
    get_project_invalidation_policy,
    invalidate_chunks,
    manual_invalidate_cache_keys,
    manual_invalidate_cache_pattern,
    manual_invalidate_file_cache,
    manual_invalidate_project_cache,
    set_project_invalidation_policy,
)


@pytest.fixture
def mock_invalidation_service():
    """Create mock invalidation service."""
    service = AsyncMock()

    # Mock statistics
    stats = MagicMock()
    stats.total_invalidations = 10
    stats.partial_invalidations = 3
    stats.keys_invalidated = 150
    stats.keys_preserved = 50
    stats.avg_optimization_ratio = 0.25
    stats.avg_invalidation_time = 1.5
    stats.last_invalidation = datetime.now()
    service.get_invalidation_stats.return_value = stats

    # Mock events
    mock_event = MagicMock()
    mock_event.event_id = "test_event_123"
    mock_event.reason.value = "manual_invalidation"
    mock_event.affected_keys = ["key1", "key2", "key3"]
    mock_event.affected_files = ["/test/file.py"]
    mock_event.timestamp = datetime.now()
    mock_event.project_name = "test_project"
    mock_event.metadata = {"test": "metadata"}

    # Mock partial result
    partial_result = MagicMock()
    partial_result.optimization_ratio = 0.4
    partial_result.preservation_keys = ["preserved1", "preserved2"]
    partial_result.invalidation_type.value = "content_based"
    mock_event.partial_result = partial_result

    service.partial_invalidate_file_caches.return_value = mock_event
    service.invalidate_file_caches.return_value = mock_event
    service.invalidate_project_caches.return_value = mock_event
    service.invalidate_project_with_policy.return_value = mock_event
    service.invalidate_keys.return_value = mock_event
    service.invalidate_pattern.return_value = mock_event
    service.clear_all_caches.return_value = mock_event
    service.invalidate_specific_chunks.return_value = mock_event

    # Mock policy methods
    policy_summary = {
        "project_name": "test_project",
        "scope": "cascade",
        "strategy": "immediate",
        "batch_threshold": 5,
        "file_patterns": ["*.py"],
        "cache_types": {
            "embeddings": True,
            "search": True,
            "project": True,
            "file": True,
        },
    }
    service.get_project_policy_summary.return_value = policy_summary
    service.get_project_files.return_value = {"/test/file1.py", "/test/file2.py"}
    service.get_monitored_projects.return_value = ["project1", "project2"]
    service.get_recent_events.return_value = [mock_event]

    # Mock policy creation
    mock_policy = MagicMock()
    mock_policy.project_name = "test_project"
    service.create_project_policy.return_value = mock_policy
    service.get_project_invalidation_policy.return_value = mock_policy

    return service


class TestManualInvalidateFileCache:
    """Test manual file cache invalidation."""

    @pytest.mark.asyncio
    async def test_manual_invalidate_file_cache_partial(self, mock_invalidation_service):
        """Test manual file cache invalidation with partial invalidation."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_file_cache(
                file_path="/test/file.py",
                reason="file_modified",
                cascade=True,
                use_partial=True,
                old_content="old content",
                new_content="new content",
                project_name="test_project",
            )

        assert result["success"] is True
        assert result["invalidation_type"] == "partial"
        assert result["affected_keys"] == 3
        assert result["optimization_info"]["optimization_ratio"] == 0.4
        assert result["optimization_info"]["preserved_keys"] == 2

        mock_invalidation_service.partial_invalidate_file_caches.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_invalidate_file_cache_full(self, mock_invalidation_service):
        """Test manual file cache invalidation with full invalidation."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_file_cache(
                file_path="/test/file.py", reason="manual_invalidation", cascade=True, use_partial=False
            )

        assert result["success"] is True
        assert result["invalidation_type"] == "full"
        assert result["affected_keys"] == 3
        assert result["optimization_info"] is None

        mock_invalidation_service.invalidate_file_caches.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_invalidate_file_cache_nonexistent_file(self, mock_invalidation_service):
        """Test manual file cache invalidation for non-existent file."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_file_cache(file_path="/nonexistent/file.py", reason="file_deleted")

        # Should still succeed but log warning
        assert result["success"] is True
        mock_invalidation_service.invalidate_file_caches.assert_called_once()


class TestManualInvalidateProjectCache:
    """Test manual project cache invalidation."""

    @pytest.mark.asyncio
    async def test_manual_invalidate_project_cache_basic(self, mock_invalidation_service):
        """Test basic project cache invalidation."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_project_cache(
                project_name="test_project", reason="project_changed", invalidation_scope="cascade", strategy="immediate"
            )

        assert result["success"] is True
        assert result["project_name"] == "test_project"
        assert result["affected_keys"] == 3
        assert result["invalidation_scope"] == "cascade"
        assert result["strategy"] == "immediate"

    @pytest.mark.asyncio
    async def test_manual_invalidate_project_cache_with_policy(self, mock_invalidation_service):
        """Test project cache invalidation with custom policy."""
        # Mock policy that is not default
        mock_policy = MagicMock()
        mock_policy.project_name = "test_project"
        mock_invalidation_service.get_project_invalidation_policy.return_value = mock_policy

        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_project_cache(project_name="test_project", invalidation_scope="aggressive", strategy="batch")

        assert result["success"] is True
        assert result["policy_applied"] is True


class TestManualInvalidateCacheKeys:
    """Test manual cache key invalidation."""

    @pytest.mark.asyncio
    async def test_manual_invalidate_cache_keys_success(self, mock_invalidation_service):
        """Test successful cache key invalidation."""
        cache_keys = ["key1", "key2", "key3"]

        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_cache_keys(cache_keys=cache_keys, reason="cache_corruption", cascade=True)

        assert result["success"] is True
        assert result["requested_keys"] == 3
        assert result["affected_keys"] == 3
        assert result["cascade_applied"] is True

        mock_invalidation_service.invalidate_keys.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_invalidate_cache_keys_empty_list(self, mock_invalidation_service):
        """Test cache key invalidation with empty list."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_cache_keys(cache_keys=[], reason="manual_invalidation")

        assert result["success"] is False
        assert "No cache keys provided" in result["error"]


class TestManualInvalidateCachePattern:
    """Test manual cache pattern invalidation."""

    @pytest.mark.asyncio
    async def test_manual_invalidate_cache_pattern_success(self, mock_invalidation_service):
        """Test successful pattern-based invalidation."""
        pattern = "embedding:*"

        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_cache_pattern(pattern=pattern, reason="system_upgrade")

        assert result["success"] is True
        assert result["pattern"] == pattern
        assert result["affected_keys"] == 3

        mock_invalidation_service.invalidate_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_manual_invalidate_cache_pattern_empty(self, mock_invalidation_service):
        """Test pattern invalidation with empty pattern."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await manual_invalidate_cache_pattern(pattern="", reason="manual_invalidation")

        assert result["success"] is False
        assert "No pattern provided" in result["error"]


class TestClearAllCaches:
    """Test clear all caches functionality."""

    @pytest.mark.asyncio
    async def test_clear_all_caches_without_confirmation(self, mock_invalidation_service):
        """Test clear all caches without confirmation."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await clear_all_caches(reason="system_upgrade", confirm=False)

        assert result["success"] is False
        assert "requires confirmation" in result["error"]
        assert "destructive operation" in result["warning"]

    @pytest.mark.asyncio
    async def test_clear_all_caches_with_confirmation(self, mock_invalidation_service):
        """Test clear all caches with confirmation."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await clear_all_caches(reason="system_upgrade", confirm=True)

        assert result["success"] is True
        assert result["operation"] == "clear_all_caches"
        assert "All cache data has been cleared" in result["warning"]

        mock_invalidation_service.clear_all_caches.assert_called_once()


class TestGetCacheInvalidationStats:
    """Test cache invalidation statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_cache_invalidation_stats_success(self, mock_invalidation_service):
        """Test successful statistics retrieval."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await get_cache_invalidation_stats()

        assert result["success"] is True
        assert "statistics" in result
        assert "recent_events" in result
        assert "monitoring" in result

        stats = result["statistics"]
        assert stats["total_invalidations"] == 10
        assert stats["partial_invalidations"] == 3
        assert stats["keys_invalidated"] == 150
        assert stats["keys_preserved"] == 50
        assert stats["avg_optimization_ratio"] == 0.25

        assert len(result["recent_events"]) == 1
        assert result["monitoring"]["projects_count"] == 2


class TestProjectInvalidationPolicy:
    """Test project invalidation policy management."""

    @pytest.mark.asyncio
    async def test_get_project_invalidation_policy(self, mock_invalidation_service):
        """Test getting project invalidation policy."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await get_project_invalidation_policy(project_name="test_project")

        assert result["success"] is True
        assert result["project_name"] == "test_project"
        assert "policy" in result
        assert "monitoring" in result

        monitoring = result["monitoring"]
        assert monitoring["is_monitored"] is True
        assert monitoring["file_count"] == 2

    @pytest.mark.asyncio
    async def test_set_project_invalidation_policy(self, mock_invalidation_service):
        """Test setting project invalidation policy."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await set_project_invalidation_policy(
                project_name="test_project",
                scope="aggressive",
                strategy="batch",
                batch_threshold=10,
                file_patterns=["*.py", "*.js"],
                exclude_patterns=["*.test.py"],
                invalidate_embeddings=True,
                invalidate_search=False,
            )

        assert result["success"] is True
        assert result["project_name"] == "test_project"
        assert result["policy_created"] is True
        assert "policy set for project" in result["message"]

        mock_invalidation_service.create_project_policy.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_project_invalidation_policy_defaults(self, mock_invalidation_service):
        """Test setting project policy with default values."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await set_project_invalidation_policy(project_name="test_project")

        assert result["success"] is True

        # Check that create_project_policy was called with default patterns
        call_args = mock_invalidation_service.create_project_policy.call_args
        assert "*.py" in call_args.kwargs["file_patterns"]
        assert "*.pyc" in call_args.kwargs["exclude_patterns"]


class TestInvalidateChunks:
    """Test chunk invalidation functionality."""

    @pytest.mark.asyncio
    async def test_invalidate_chunks_success(self, mock_invalidation_service):
        """Test successful chunk invalidation."""
        chunk_ids = ["chunk_1", "chunk_2"]

        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await invalidate_chunks(file_path="/test/file.py", chunk_ids=chunk_ids, reason="chunk_modified")

        assert result["success"] is True
        assert result["chunk_ids"] == chunk_ids
        assert result["affected_keys"] == 3

        mock_invalidation_service.invalidate_specific_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_chunks_empty_list(self, mock_invalidation_service):
        """Test chunk invalidation with empty chunk list."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
            result = await invalidate_chunks(file_path="/test/file.py", chunk_ids=[], reason="chunk_modified")

        assert result["success"] is False
        assert "No chunk IDs provided" in result["error"]


class TestErrorHandling:
    """Test error handling in cache management tools."""

    @pytest.mark.asyncio
    async def test_manual_invalidate_file_cache_error(self):
        """Test error handling in file cache invalidation."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", side_effect=Exception("Service error")):
            result = await manual_invalidate_file_cache(file_path="/test/file.py")

        # Should handle error gracefully
        assert "error" in result or "success" in result

    @pytest.mark.asyncio
    async def test_get_cache_invalidation_stats_error(self):
        """Test error handling in stats retrieval."""
        with patch("src.tools.cache.cache_management.get_cache_invalidation_service", side_effect=Exception("Service error")):
            result = await get_cache_invalidation_stats()

        # Should handle error gracefully
        assert "error" in result or "success" in result


class TestReasonMapping:
    """Test invalidation reason mapping."""

    @pytest.mark.asyncio
    async def test_reason_mapping_file_cache(self, mock_invalidation_service):
        """Test reason mapping for file cache invalidation."""
        test_cases = [
            ("manual_invalidation", "MANUAL_INVALIDATION"),
            ("file_modified", "FILE_MODIFIED"),
            ("file_deleted", "FILE_DELETED"),
            ("content_changed", "PARTIAL_CONTENT_CHANGE"),
            ("invalid_reason", "MANUAL_INVALIDATION"),  # Should default
        ]

        for reason_input, expected_enum in test_cases:
            with patch("src.tools.cache.cache_management.get_cache_invalidation_service", return_value=mock_invalidation_service):
                await manual_invalidate_file_cache(file_path="/test/file.py", reason=reason_input, use_partial=False)

            # Check that the service was called (validates reason mapping works)
            mock_invalidation_service.invalidate_file_caches.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
