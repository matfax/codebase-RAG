"""
Tests for the enhanced breadcrumb resolution caching service.

This module tests the TTL-based caching system with file modification tracking
for the enhanced function call detection system performance optimization.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.models.breadcrumb_cache_models import (
    BreadcrumbCacheConfig,
    BreadcrumbCacheEntry,
    CacheStats,
    FileModificationTracker,
)
from src.services.breadcrumb_cache_service import BreadcrumbCacheService
from src.services.breadcrumb_resolver_service import BreadcrumbCandidate, BreadcrumbResolutionResult


class TestFileModificationTracker:
    """Test file modification tracking functionality."""

    def test_file_tracker_creation(self):
        """Test creation of file modification tracker."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            tracker = FileModificationTracker(file_path=f.name, last_modified=time.time())

            assert tracker.file_path == f.name
            assert tracker.content_hash is not None
            assert len(tracker.content_hash) == 16  # SHA256 first 16 chars

        Path(f.name).unlink()  # Cleanup

    def test_stale_detection_by_mtime(self):
        """Test stale detection based on modification time."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("initial content")
            f.flush()

            # Create tracker with old timestamp
            tracker = FileModificationTracker(file_path=f.name, last_modified=time.time() - 1000)  # 1000 seconds ago

            assert tracker.is_stale()  # Should be stale due to newer file

        Path(f.name).unlink()

    def test_stale_detection_by_content(self):
        """Test stale detection based on content changes."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("initial content")
            f.flush()

            tracker = FileModificationTracker(file_path=f.name, last_modified=Path(f.name).stat().st_mtime)

            # Modify file content
            with open(f.name, "w") as f2:
                f2.write("modified content")

            assert tracker.is_stale()  # Should be stale due to content change

        Path(f.name).unlink()

    def test_update_tracking(self):
        """Test updating tracking information."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("initial content")
            f.flush()

            tracker = FileModificationTracker(file_path=f.name, last_modified=time.time() - 1000)

            old_hash = tracker.content_hash
            changes_detected = tracker.update_tracking()

            assert changes_detected  # Should detect changes
            assert tracker.content_hash != old_hash  # Hash should change

        Path(f.name).unlink()


class TestBreadcrumbCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test creation of cache entry."""
        result = Mock()
        entry = BreadcrumbCacheEntry(cache_key="test_key", result=result, timestamp=time.time(), ttl_seconds=3600.0, confidence_score=0.8)

        assert entry.cache_key == "test_key"
        assert entry.result == result
        assert entry.confidence_score == 0.8
        assert not entry.is_expired()
        assert entry.is_valid()

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        result = Mock()
        entry = BreadcrumbCacheEntry(
            cache_key="test_key",
            result=result,
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl_seconds=3600.0,  # 1 hour TTL
            confidence_score=0.8,
        )

        assert entry.is_expired()
        assert not entry.is_valid()

    def test_file_dependency_staleness(self):
        """Test staleness detection based on file dependencies."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            tracker = FileModificationTracker(file_path=f.name, last_modified=time.time() - 1000)  # Old timestamp

            result = Mock()
            entry = BreadcrumbCacheEntry(
                cache_key="test_key",
                result=result,
                timestamp=time.time(),
                ttl_seconds=3600.0,
                file_dependencies=[tracker],
                confidence_score=0.8,
            )

            assert entry.is_stale()  # Should be stale due to file changes
            assert not entry.is_valid()

        Path(f.name).unlink()

    def test_access_tracking(self):
        """Test access count and timestamp tracking."""
        result = Mock()
        entry = BreadcrumbCacheEntry(cache_key="test_key", result=result, timestamp=time.time(), ttl_seconds=3600.0, confidence_score=0.8)

        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed

        time.sleep(0.01)  # Small delay
        entry.touch()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestBreadcrumbCacheConfig:
    """Test cache configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = BreadcrumbCacheConfig()

        assert config.enabled is True
        assert config.max_entries == 10000
        assert config.default_ttl_seconds == 3600.0
        assert config.confidence_threshold == 0.7

    def test_ttl_for_confidence(self):
        """Test TTL calculation based on confidence scores."""
        config = BreadcrumbCacheConfig(default_ttl_seconds=1000.0)

        # High confidence -> longer TTL
        assert config.get_ttl_for_confidence(0.95) == 2000.0

        # Normal confidence -> normal TTL
        assert config.get_ttl_for_confidence(0.8) == 1000.0

        # Lower confidence -> shorter TTL
        assert config.get_ttl_for_confidence(0.6) == 500.0

        # Very low confidence -> very short TTL
        assert config.get_ttl_for_confidence(0.3) == 100.0

    def test_config_from_env(self):
        """Test configuration creation from environment variables."""
        with patch.dict(
            "os.environ",
            {"BREADCRUMB_CACHE_ENABLED": "false", "BREADCRUMB_CACHE_MAX_ENTRIES": "5000", "BREADCRUMB_CACHE_TTL_SECONDS": "1800"},
        ):
            config = BreadcrumbCacheConfig.from_env()

            assert config.enabled is False
            assert config.max_entries == 5000
            assert config.default_ttl_seconds == 1800.0


@pytest.mark.asyncio
class TestBreadcrumbCacheService:
    """Test the breadcrumb cache service."""

    async def test_cache_service_initialization(self):
        """Test cache service initialization."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        service = BreadcrumbCacheService(config)

        assert service.config.enabled is True
        assert len(service._cache) == 0
        assert service._stats.hits == 0

    async def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        service = BreadcrumbCacheService(config)
        await service.start()

        try:
            # Create mock result
            result = Mock()

            # Put in cache
            success = await service.put(cache_key="test_key", result=result, confidence_score=0.8)

            assert success is True

            # Get from cache
            cached_result = await service.get("test_key")
            assert cached_result == result

            # Verify stats
            stats = service.get_stats()
            assert stats["hits"] == 1
            assert stats["total_entries"] == 1

        finally:
            await service.stop()

    async def test_cache_miss(self):
        """Test cache miss behavior."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        service = BreadcrumbCacheService(config)
        await service.start()

        try:
            # Attempt to get non-existent key
            result = await service.get("non_existent_key")
            assert result is None

            # Verify stats
            stats = service.get_stats()
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 0.0

        finally:
            await service.stop()

    async def test_file_dependency_tracking(self):
        """Test file dependency tracking and invalidation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("initial content")
            f.flush()

            config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0, enable_dependency_tracking=True)
            service = BreadcrumbCacheService(config)
            await service.start()

            try:
                # Cache with file dependency
                result = Mock()
                await service.put(cache_key="test_key", result=result, file_dependencies=[f.name], confidence_score=0.8)

                # Verify cache hit
                cached_result = await service.get("test_key")
                assert cached_result == result

                # Modify the file
                with open(f.name, "w") as f2:
                    f2.write("modified content")

                # Cache should now be invalid
                cached_result = await service.get("test_key")
                assert cached_result is None  # Should be invalidated

            finally:
                await service.stop()
                Path(f.name).unlink()

    async def test_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0, default_ttl_seconds=0.1)  # Very short TTL
        service = BreadcrumbCacheService(config)
        await service.start()

        try:
            # Cache with short TTL
            result = Mock()
            await service.put(cache_key="test_key", result=result, confidence_score=0.5, custom_ttl=0.1)  # Will use short TTL

            # Should be in cache initially
            cached_result = await service.get("test_key")
            assert cached_result == result

            # Wait for expiration
            await asyncio.sleep(0.2)

            # Should be expired now
            cached_result = await service.get("test_key")
            assert cached_result is None

        finally:
            await service.stop()

    async def test_cache_capacity_management(self):
        """Test cache capacity and eviction."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0, max_entries=3)  # Very small cache
        service = BreadcrumbCacheService(config)
        await service.start()

        try:
            # Fill cache beyond capacity
            for i in range(5):
                await service.put(cache_key=f"key_{i}", result=f"result_{i}", confidence_score=0.8)

            # Should only have max_entries
            stats = service.get_stats()
            assert stats["total_entries"] == 3

            # Oldest entries should be evicted (LRU)
            assert await service.get("key_0") is None
            assert await service.get("key_1") is None
            assert await service.get("key_2") is not None

        finally:
            await service.stop()

    async def test_cache_invalidation_by_file(self):
        """Test cache invalidation by file path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            f.flush()

            config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0, enable_dependency_tracking=True)
            service = BreadcrumbCacheService(config)
            await service.start()

            try:
                # Cache multiple entries with same file dependency
                for i in range(3):
                    await service.put(cache_key=f"key_{i}", result=f"result_{i}", file_dependencies=[f.name], confidence_score=0.8)

                # Verify all are cached
                for i in range(3):
                    assert await service.get(f"key_{i}") is not None

                # Invalidate by file
                invalidated_count = await service.invalidate_by_file(f.name)
                assert invalidated_count == 3

                # All should be invalidated
                for i in range(3):
                    assert await service.get(f"key_{i}") is None

            finally:
                await service.stop()
                Path(f.name).unlink()

    async def test_cache_clear(self):
        """Test cache clearing."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        service = BreadcrumbCacheService(config)
        await service.start()

        try:
            # Add some entries
            for i in range(3):
                await service.put(cache_key=f"key_{i}", result=f"result_{i}", confidence_score=0.8)

            # Verify entries exist
            stats = service.get_stats()
            assert stats["total_entries"] == 3

            # Clear cache
            await service.clear()

            # Verify cache is empty
            stats = service.get_stats()
            assert stats["total_entries"] == 0

        finally:
            await service.stop()

    async def test_cache_info_retrieval(self):
        """Test cache information retrieval."""
        config = BreadcrumbCacheConfig(enabled=True, cleanup_interval_seconds=0)
        service = BreadcrumbCacheService(config)
        await service.start()

        try:
            # Add some entries
            await service.put("key1", "result1", confidence_score=0.8)
            await service.put("key2", "result2", confidence_score=0.9)

            # Get cache info
            info = await service.get_cache_info()

            assert "stats" in info
            assert "config" in info
            assert "cache_details" in info
            assert info["stats"]["total_entries"] == 2
            assert info["config"]["enabled"] is True

        finally:
            await service.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
