"""
Unit tests for cache invalidation service.

Tests the enhanced cache invalidation functionality with:
- Partial invalidation for incremental updates
- Content-based invalidation analysis
- Chunk-level invalidation
- Optimization ratio tracking
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ..config.cache_config import CacheConfig
from ..models.code_chunk import CodeChunk
from ..models.file_metadata import FileMetadata
from .cache_invalidation_service import (
    CacheInvalidationService,
    IncrementalInvalidationType,
    InvalidationEvent,
    InvalidationReason,
    InvalidationStats,
    InvalidationStrategy,
    PartialInvalidationResult,
    ProjectInvalidationPolicy,
    ProjectInvalidationScope,
    ProjectInvalidationTrigger,
    get_cache_invalidation_service,
)


@pytest.fixture
def config():
    """Create test cache configuration."""
    from src.config.cache_config import MemoryCacheConfig, RedisConfig

    return CacheConfig(
        redis=RedisConfig(
            host="localhost",
            port=6379,
            password=None,
            db=0,
        ),
        memory=MemoryCacheConfig(
            max_memory_mb=100,
        ),
        default_ttl=3600,
        debug_mode=True,
    )


@pytest.fixture
def cache_service():
    """Create cache invalidation service for testing."""
    service = CacheInvalidationService()
    return service


@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            content="def function1():\n    return 'hello'",
            file_path="/test/file1.py",
            chunk_type="function",
            name="function1",
            signature="function1()",
            start_line=1,
            end_line=2,
            language="python",
            chunk_id="chunk_1",
            content_hash="hash1",
        ),
        CodeChunk(
            content="def function2():\n    return 'world'",
            file_path="/test/file1.py",
            chunk_type="function",
            name="function2",
            signature="function2()",
            start_line=4,
            end_line=5,
            language="python",
            chunk_id="chunk_2",
            content_hash="hash2",
        ),
    ]


class TestPartialInvalidation:
    """Test partial invalidation functionality."""

    @pytest.mark.asyncio
    async def test_partial_invalidation_analysis_metadata_only(self, cache_service):
        """Test partial invalidation analysis for metadata-only changes."""
        file_path = "/test/file.py"
        old_content = "def test(): pass"
        new_content = "def test(): pass"  # Same content

        # Mock cache key generation
        cache_service._generate_file_cache_keys = AsyncMock(return_value=["file:test", "embedding:test", "search:test"])

        result = await cache_service._analyze_partial_invalidation(file_path, old_content, new_content, "test_project")

        assert result.invalidation_type == IncrementalInvalidationType.METADATA_ONLY
        assert result.optimization_ratio == 1.0
        assert result.file_path == str(Path(file_path).resolve())

    @pytest.mark.asyncio
    async def test_partial_invalidation_analysis_content_based(self, cache_service):
        """Test content-based partial invalidation analysis."""
        file_path = "/test/file.py"
        old_content = "def test(): pass\ndef other(): return 1"
        new_content = "def test(): return 'modified'\ndef other(): return 1"

        # Mock cache key generation
        cache_service._generate_file_cache_keys = AsyncMock(return_value=["file:test", "embedding:test", "search:test", "parsing:test"])

        result = await cache_service._analyze_partial_invalidation(file_path, old_content, new_content, "test_project")

        assert result.invalidation_type == IncrementalInvalidationType.CONTENT_BASED
        assert 0.0 < result.optimization_ratio <= 1.0
        assert len(result.affected_cache_keys) > 0

    @pytest.mark.asyncio
    async def test_partial_invalidation_analysis_chunk_based(self, cache_service, sample_chunks):
        """Test chunk-based partial invalidation analysis."""
        file_path = "/test/file.py"
        old_content = "def function1(): return 'hello'\ndef function2(): return 'world'"
        new_content = "def function1(): return 'modified'\ndef function2(): return 'world'"

        # Mock file cache to return chunks
        mock_file_cache = AsyncMock()
        old_chunks = [sample_chunks[0], sample_chunks[1]]
        new_chunks = [
            CodeChunk(
                content="def function1(): return 'modified'",
                file_path="/test/file1.py",
                chunk_type="function",
                name="function1",
                signature="function1()",
                start_line=1,
                end_line=2,
                language="python",
                chunk_id="chunk_1",
                content_hash="modified_hash1",
            ),
            sample_chunks[1],  # Unchanged
        ]

        mock_file_cache._parse_content_to_chunks = AsyncMock(side_effect=[old_chunks, new_chunks])
        cache_service._file_cache = mock_file_cache

        # Mock chunk cache key generation
        cache_service._generate_chunk_cache_keys = AsyncMock(return_value=["chunk:test", "embedding:chunk:test"])

        # Register the file in chunk mapping
        cache_service._chunk_cache_map[str(Path(file_path).resolve())] = {"chunk_1", "chunk_2"}

        result = await cache_service._analyze_partial_invalidation(file_path, old_content, new_content, "test_project")

        assert result.invalidation_type == IncrementalInvalidationType.CHUNK_BASED
        assert len(result.affected_chunks) == 1  # Only function1 changed
        assert "chunk_1" in result.affected_chunks
        assert result.optimization_ratio > 0.0  # Some chunks preserved

    @pytest.mark.asyncio
    async def test_partial_invalidate_file_caches(self, cache_service):
        """Test full partial invalidation workflow."""
        file_path = "/test/file.py"
        old_content = "def test(): pass"
        new_content = "def test(): return 'modified'"

        # Mock methods
        cache_service._analyze_partial_invalidation = AsyncMock(
            return_value=PartialInvalidationResult(
                file_path=file_path,
                invalidation_type=IncrementalInvalidationType.CONTENT_BASED,
                affected_cache_keys=["file:test", "embedding:test"],
                preservation_keys=["search:test"],
                optimization_ratio=0.33,
            )
        )

        cache_service._perform_targeted_invalidation = AsyncMock(return_value=["file:test", "embedding:test"])
        cache_service._log_invalidation_event = MagicMock()

        event = await cache_service.partial_invalidate_file_caches(file_path, old_content, new_content, "test_project")

        assert isinstance(event, InvalidationEvent)
        assert event.reason == InvalidationReason.PARTIAL_CONTENT_CHANGE
        assert event.partial_result is not None
        assert event.partial_result.optimization_ratio == 0.33
        assert "partial_invalidation" in event.metadata

    def test_content_similarity_calculation(self, cache_service):
        """Test content similarity calculation."""
        old_content = "line1\nline2\nline3"
        new_content = "line1\nmodified_line2\nline3"

        similarity = cache_service._calculate_content_similarity(old_content, new_content)

        # Should have 2/4 lines in common (line1, line3 vs all unique lines)
        assert 0.0 < similarity < 1.0

    def test_chunk_signature_generation(self, cache_service, sample_chunks):
        """Test chunk signature generation for comparison."""
        chunk = sample_chunks[0]
        signature = cache_service._get_chunk_signature(chunk)

        expected = f"{chunk.chunk_type}:{chunk.name}:{chunk.start_line}:{chunk.end_line}"
        assert signature == expected

    def test_identify_changed_chunks(self, cache_service, sample_chunks):
        """Test identification of changed chunks."""
        old_chunks = sample_chunks
        new_chunks = [
            # Modified chunk
            CodeChunk(
                content="def function1(): return 'modified'",
                file_path="/test/file1.py",
                chunk_type="function",
                name="function1",
                signature="function1()",
                start_line=1,
                end_line=2,
                language="python",
                chunk_id="chunk_1",
                content_hash="modified_hash",
            ),
            # Unchanged chunk
            sample_chunks[1],
        ]

        changed = cache_service._identify_changed_chunks(old_chunks, new_chunks)

        assert "chunk_1" in changed  # Modified
        assert "chunk_2" not in changed  # Unchanged

    @pytest.mark.asyncio
    async def test_register_chunk_mapping(self, cache_service, sample_chunks):
        """Test chunk mapping registration."""
        file_path = "/test/file.py"

        # Mock chunk cache key generation
        cache_service._generate_chunk_cache_keys = AsyncMock(return_value=["chunk:test", "embedding:chunk:test"])

        await cache_service.register_chunk_mapping(file_path, sample_chunks)

        abs_path = str(Path(file_path).resolve())
        assert abs_path in cache_service._chunk_cache_map
        assert len(cache_service._chunk_cache_map[abs_path]) == 2
        assert "chunk_1" in cache_service._chunk_cache_map[abs_path]
        assert "chunk_2" in cache_service._chunk_cache_map[abs_path]

    @pytest.mark.asyncio
    async def test_invalidate_specific_chunks(self, cache_service):
        """Test invalidation of specific chunks."""
        file_path = "/test/file.py"
        chunk_ids = ["chunk_1", "chunk_2"]

        # Setup chunk dependency mapping
        cache_service._chunk_dependency_map = {
            "chunk_1": {"embedding:chunk:1", "file:chunk:1"},
            "chunk_2": {"embedding:chunk:2", "file:chunk:2"},
        }

        # Mock invalidation
        cache_service._invalidate_chunk_keys = AsyncMock()
        cache_service._log_invalidation_event = MagicMock()

        event = await cache_service.invalidate_specific_chunks(file_path, chunk_ids)

        assert isinstance(event, InvalidationEvent)
        assert event.reason == InvalidationReason.CHUNK_MODIFIED
        assert len(event.affected_keys) == 4  # 2 keys per chunk
        assert event.metadata["chunk_count"] == 2


class TestInvalidationStats:
    """Test invalidation statistics tracking."""

    def test_stats_update_with_optimization(self):
        """Test statistics update with optimization data."""
        stats = InvalidationStats()

        stats.update(
            keys_count=10, duration=1.5, reason=InvalidationReason.PARTIAL_CONTENT_CHANGE, preserved_keys=5, optimization_ratio=0.5
        )

        assert stats.total_invalidations == 1
        assert stats.partial_invalidations == 1
        assert stats.keys_invalidated == 10
        assert stats.keys_preserved == 5
        assert stats.avg_optimization_ratio == 0.5
        assert stats.avg_invalidation_time == 1.5

    def test_stats_multiple_updates(self):
        """Test statistics with multiple updates."""
        stats = InvalidationStats()

        # First update
        stats.update(10, 1.0, InvalidationReason.PARTIAL_CONTENT_CHANGE, 5, 0.5)

        # Second update
        stats.update(20, 2.0, InvalidationReason.CHUNK_MODIFIED, 10, 0.33)

        assert stats.total_invalidations == 2
        assert stats.partial_invalidations == 2
        assert stats.keys_invalidated == 30
        assert stats.keys_preserved == 15
        assert stats.avg_invalidation_time == 1.5
        # Average of 0.5 and 0.33
        assert abs(stats.avg_optimization_ratio - 0.415) < 0.01


class TestInvalidationReasons:
    """Test invalidation reason mapping."""

    def test_invalidation_reason_mapping(self, cache_service):
        """Test mapping from invalidation type to reason."""
        mappings = [
            (IncrementalInvalidationType.CONTENT_BASED, InvalidationReason.PARTIAL_CONTENT_CHANGE),
            (IncrementalInvalidationType.CHUNK_BASED, InvalidationReason.CHUNK_MODIFIED),
            (IncrementalInvalidationType.METADATA_ONLY, InvalidationReason.METADATA_ONLY_CHANGE),
            (IncrementalInvalidationType.DEPENDENCY_BASED, InvalidationReason.DEPENDENCY_CHANGED),
            (IncrementalInvalidationType.HYBRID, InvalidationReason.PARTIAL_CONTENT_CHANGE),
        ]

        for invalidation_type, expected_reason in mappings:
            reason = cache_service._get_invalidation_reason(invalidation_type)
            assert reason == expected_reason


class TestPartialInvalidationResult:
    """Test PartialInvalidationResult data class."""

    def test_partial_invalidation_result_creation(self):
        """Test creation and serialization of partial invalidation result."""
        result = PartialInvalidationResult(
            file_path="/test/file.py",
            invalidation_type=IncrementalInvalidationType.CHUNK_BASED,
            affected_chunks=["chunk_1", "chunk_2"],
            affected_cache_keys=["key1", "key2"],
            preservation_keys=["key3", "key4"],
            optimization_ratio=0.5,
            content_changes={"changed_chunks": 2, "total_chunks": 4},
        )

        data = result.to_dict()

        assert data["file_path"] == "/test/file.py"
        assert data["invalidation_type"] == "chunk_based"
        assert data["affected_chunks"] == ["chunk_1", "chunk_2"]
        assert data["optimization_ratio"] == 0.5
        assert data["content_changes"]["changed_chunks"] == 2


class TestCacheKeyGeneration:
    """Test cache key generation for chunks."""

    @pytest.mark.asyncio
    async def test_chunk_cache_key_generation(self, cache_service, sample_chunks):
        """Test generation of cache keys for chunks."""
        chunk = sample_chunks[0]

        keys = await cache_service._generate_chunk_cache_keys(chunk)

        expected_keys = [
            f"chunk:{chunk.chunk_id}",
            f"embedding:chunk:{chunk.chunk_id}",
            f"file:chunk:{chunk.file_path}:{chunk.chunk_id}",
            f"function:{chunk.name}:{chunk.file_path}",
            f"embedding:function:{chunk.name}:{chunk.file_path}",
        ]

        for expected_key in expected_keys:
            assert expected_key in keys


class TestTargetedInvalidation:
    """Test targeted invalidation based on partial results."""

    @pytest.mark.asyncio
    async def test_perform_targeted_invalidation(self, cache_service):
        """Test targeted invalidation execution."""
        partial_result = PartialInvalidationResult(
            file_path="/test/file.py",
            invalidation_type=IncrementalInvalidationType.CONTENT_BASED,
            affected_cache_keys=["file:test", "embedding:test"],
            preservation_keys=["search:test"],
        )

        # Mock service invalidation
        cache_service._invalidate_keys_in_service = AsyncMock()
        cache_service._get_dependent_keys = AsyncMock(return_value=[])

        affected_keys = await cache_service._perform_targeted_invalidation(partial_result, cascade=False)

        assert affected_keys == ["file:test", "embedding:test"]
        # Should have called invalidation for file and embedding services
        assert cache_service._invalidate_keys_in_service.call_count >= 1

    @pytest.mark.asyncio
    async def test_perform_targeted_invalidation_with_cascade(self, cache_service):
        """Test targeted invalidation with cascade."""
        partial_result = PartialInvalidationResult(
            file_path="/test/file.py",
            invalidation_type=IncrementalInvalidationType.CONTENT_BASED,
            affected_cache_keys=["file:test"],
            dependency_changes=["dep1"],
        )

        # Mock methods
        cache_service._invalidate_keys_in_service = AsyncMock()
        cache_service._get_dependent_keys = AsyncMock(return_value=["dependent:key"])
        cache_service._invalidate_dependent_keys = AsyncMock()

        affected_keys = await cache_service._perform_targeted_invalidation(partial_result, cascade=True)

        # Should include original keys plus dependent keys
        assert "file:test" in affected_keys
        assert "dependent:key" in affected_keys
        cache_service._invalidate_dependent_keys.assert_called_once()


class TestIntegration:
    """Integration tests for partial invalidation functionality."""

    @pytest.mark.asyncio
    async def test_full_partial_invalidation_workflow(self, config):
        """Test complete partial invalidation workflow."""
        service = CacheInvalidationService(config)

        # Mock initialization
        service._initialize_cache_services = AsyncMock()
        await service.initialize()

        file_path = "/test/file.py"
        old_content = "def old_function(): pass"
        new_content = "def new_function(): pass"

        # Mock all required methods
        service._analyze_partial_invalidation = AsyncMock(
            return_value=PartialInvalidationResult(
                file_path=file_path,
                invalidation_type=IncrementalInvalidationType.CONTENT_BASED,
                affected_cache_keys=["file:test"],
                preservation_keys=["search:test"],
                optimization_ratio=0.5,
            )
        )

        service._perform_targeted_invalidation = AsyncMock(return_value=["file:test"])

        # Perform partial invalidation
        event = await service.partial_invalidate_file_caches(file_path, old_content, new_content, "test_project")

        # Verify results
        assert isinstance(event, InvalidationEvent)
        assert event.reason == InvalidationReason.PARTIAL_CONTENT_CHANGE
        assert event.partial_result is not None
        assert event.partial_result.optimization_ratio == 0.5

        # Verify statistics were updated
        stats = service.get_invalidation_stats()
        assert stats.partial_invalidations == 1
        assert stats.avg_optimization_ratio == 0.5

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_fallback_to_full_invalidation(self, config):
        """Test fallback to full invalidation on errors."""
        service = CacheInvalidationService(config)

        # Mock initialization
        service._initialize_cache_services = AsyncMock()
        await service.initialize()

        file_path = "/test/file.py"

        # Mock analysis to raise exception
        service._analyze_partial_invalidation = AsyncMock(side_effect=Exception("Analysis failed"))

        # Mock full invalidation
        service.invalidate_file_caches = AsyncMock(
            return_value=InvalidationEvent(
                event_id="fallback",
                reason=InvalidationReason.FILE_MODIFIED,
                timestamp=datetime.now(),
                affected_keys=["file:test"],
            )
        )

        # Should fall back to full invalidation
        event = await service.partial_invalidate_file_caches(file_path, "old", "new")

        assert event.reason == InvalidationReason.FILE_MODIFIED
        service.invalidate_file_caches.assert_called_once()

        await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])
