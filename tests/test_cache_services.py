"""
Unit tests for cache services - Wave 15.1.1
Tests core cache functionality including RedisCacheService, MultiTierCacheService,
and specialized cache services.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis
from redis.exceptions import ConnectionError, RedisError

from src.services.cache_service import CacheService
from src.services.embedding_cache_service import EmbeddingCacheService
from src.services.file_cache_service import FileCacheService
from src.services.project_cache_service import ProjectCacheService
from src.services.resilient_cache_service import ResilientCacheService
from src.services.search_cache_service import SearchCacheService
from src.utils.cache_key_utils import generate_cache_key
from src.utils.resilient_redis_manager import ResilientRedisManager


class TestCacheService:
    """Test base CacheService functionality."""

    @pytest.fixture
    async def cache_service(self):
        """Create a cache service instance for testing."""
        service = CacheService()
        yield service
        # Cleanup
        await service.close()

    @pytest.fixture
    async def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=0)
        mock.expire = AsyncMock(return_value=True)
        mock.ttl = AsyncMock(return_value=-1)
        mock.ping = AsyncMock(return_value=True)
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_cache_service_initialization(self, cache_service):
        """Test cache service initialization."""
        assert cache_service is not None
        assert hasattr(cache_service, "get")
        assert hasattr(cache_service, "set")
        assert hasattr(cache_service, "delete")
        assert hasattr(cache_service, "clear")

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache_service, mock_redis):
        """Test cache get with cache miss."""
        cache_service._redis_client = mock_redis

        result = await cache_service.get("non_existent_key")
        assert result is None
        mock_redis.get.assert_called_once_with("non_existent_key")

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, cache_service, mock_redis):
        """Test cache get with cache hit."""
        test_data = {"test": "data", "number": 42}
        mock_redis.get.return_value = json.dumps(test_data)
        cache_service._redis_client = mock_redis

        result = await cache_service.get("existing_key")
        assert result == test_data
        mock_redis.get.assert_called_once_with("existing_key")

    @pytest.mark.asyncio
    async def test_cache_set(self, cache_service, mock_redis):
        """Test cache set operation."""
        cache_service._redis_client = mock_redis
        test_data = {"test": "data", "number": 42}

        result = await cache_service.set("test_key", test_data, ttl=3600)
        assert result is True
        mock_redis.set.assert_called_once()
        args = mock_redis.set.call_args[0]
        assert args[0] == "test_key"
        assert json.loads(args[1]) == test_data

    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_service, mock_redis):
        """Test cache delete operation."""
        cache_service._redis_client = mock_redis

        result = await cache_service.delete("test_key")
        assert result is True
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_cache_clear_pattern(self, cache_service, mock_redis):
        """Test cache clear with pattern."""
        mock_redis.scan = AsyncMock(return_value=(0, [b"prefix:key1", b"prefix:key2"]))
        cache_service._redis_client = mock_redis

        result = await cache_service.clear(pattern="prefix:*")
        assert result == 2
        mock_redis.delete.assert_called_once_with(b"prefix:key1", b"prefix:key2")

    @pytest.mark.asyncio
    async def test_cache_connection_error(self, cache_service):
        """Test cache behavior on connection error."""
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Connection refused")
        cache_service._redis_client = mock_redis

        result = await cache_service.get("test_key")
        assert result is None  # Should handle error gracefully

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, cache_service, mock_redis):
        """Test cache TTL and expiration."""
        cache_service._redis_client = mock_redis

        # Set with TTL
        await cache_service.set("expiring_key", {"data": "test"}, ttl=60)
        mock_redis.set.assert_called_once()

        # Check TTL
        mock_redis.ttl.return_value = 45
        ttl = await cache_service.get_ttl("expiring_key")
        assert ttl == 45


class TestResilientCacheService:
    """Test ResilientCacheService with failover and recovery."""

    @pytest.fixture
    async def resilient_cache(self):
        """Create a resilient cache service instance."""
        service = ResilientCacheService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_automatic_failover(self, resilient_cache):
        """Test automatic failover to backup cache."""
        # Mock primary failure
        primary_mock = AsyncMock()
        primary_mock.get.side_effect = ConnectionError("Primary failed")

        # Mock backup success
        backup_mock = AsyncMock()
        backup_mock.get.return_value = json.dumps({"source": "backup"})

        resilient_cache._primary_cache = primary_mock
        resilient_cache._backup_cache = backup_mock

        result = await resilient_cache.get("test_key")
        assert result == {"source": "backup"}
        backup_mock.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, resilient_cache):
        """Test circuit breaker pattern for failed connections."""
        mock_cache = AsyncMock()
        mock_cache.get.side_effect = ConnectionError("Connection failed")
        resilient_cache._primary_cache = mock_cache

        # Multiple failures should open circuit
        for _ in range(5):
            await resilient_cache.get("test_key")

        # Circuit should be open, not attempting connection
        assert resilient_cache._circuit_open is True

        # After cooldown, circuit should attempt reset
        await asyncio.sleep(resilient_cache._circuit_reset_timeout)
        resilient_cache._circuit_open = False

        mock_cache.get.side_effect = None
        mock_cache.get.return_value = json.dumps({"data": "recovered"})

        result = await resilient_cache.get("test_key")
        assert result == {"data": "recovered"}


class TestSpecializedCacheServices:
    """Test specialized cache services for different data types."""

    @pytest.mark.asyncio
    async def test_embedding_cache_service(self):
        """Test EmbeddingCacheService for vector embeddings."""
        service = EmbeddingCacheService()
        mock_redis = AsyncMock()
        service._redis_client = mock_redis

        # Test embedding storage
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"model": "nomic-embed-text", "dimension": 5}

        await service.set_embedding("doc_123", embedding, metadata)
        mock_redis.set.assert_called_once()

        # Test embedding retrieval
        mock_redis.get.return_value = json.dumps({"embedding": embedding, "metadata": metadata})

        result = await service.get_embedding("doc_123")
        assert result["embedding"] == embedding
        assert result["metadata"] == metadata

        await service.close()

    @pytest.mark.asyncio
    async def test_file_cache_service(self):
        """Test FileCacheService for file metadata caching."""
        service = FileCacheService()
        mock_redis = AsyncMock()
        service._redis_client = mock_redis

        # Test file metadata storage
        file_metadata = {"path": "/path/to/file.py", "size": 1024, "mtime": time.time(), "content_hash": "abc123", "chunks": 10}

        await service.set_file_metadata("file_123", file_metadata)
        mock_redis.set.assert_called_once()

        # Test batch retrieval
        mock_redis.mget.return_value = [json.dumps(file_metadata), None, json.dumps({**file_metadata, "path": "/path/to/file2.py"})]

        results = await service.get_files_batch(["file_1", "file_2", "file_3"])
        assert len(results) == 2  # Only non-null results

        await service.close()

    @pytest.mark.asyncio
    async def test_project_cache_service(self):
        """Test ProjectCacheService for project-level caching."""
        service = ProjectCacheService()
        mock_redis = AsyncMock()
        service._redis_client = mock_redis

        # Test project metadata
        project_data = {"name": "test_project", "files": 100, "total_chunks": 1000, "last_indexed": datetime.now().isoformat()}

        await service.set_project_data("proj_123", project_data)

        # Test namespace isolation
        await service.set_namespaced("proj_123", "config", {"key": "value"})
        expected_key = "project:proj_123:config"
        mock_redis.set.assert_called_with(expected_key, json.dumps({"key": "value"}), ex=None)

        await service.close()

    @pytest.mark.asyncio
    async def test_search_cache_service(self):
        """Test SearchCacheService for search result caching."""
        service = SearchCacheService()
        mock_redis = AsyncMock()
        service._redis_client = mock_redis

        # Test search result caching
        search_results = {
            "query": "test query",
            "results": [{"file": "file1.py", "score": 0.95}, {"file": "file2.py", "score": 0.87}],
            "total": 2,
            "search_time": 0.123,
        }

        service._hash_query("test query", {"n_results": 10})
        await service.cache_search_results("test query", search_results, {"n_results": 10})

        # Verify caching with appropriate TTL for search results
        mock_redis.set.assert_called_once()
        args = mock_redis.set.call_args
        assert args[1]["ex"] == 300  # 5 minute TTL for search results

        await service.close()


class TestCacheKeyGeneration:
    """Test cache key generation utilities."""

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        key = generate_cache_key("prefix", "identifier")
        assert key == "prefix:identifier"

    def test_generate_cache_key_with_params(self):
        """Test cache key generation with parameters."""
        params = {"user": "test", "project": "demo"}
        key = generate_cache_key("search", "query_123", params)
        assert "search:query_123" in key
        assert "user=test" in key or "project=demo" in key

    def test_generate_cache_key_deterministic(self):
        """Test cache key generation is deterministic."""
        params = {"b": 2, "a": 1, "c": 3}
        key1 = generate_cache_key("test", "id", params)
        key2 = generate_cache_key("test", "id", params)
        assert key1 == key2

    def test_generate_cache_key_special_chars(self):
        """Test cache key generation with special characters."""
        key = generate_cache_key("test", "id with spaces & special!")
        assert " " not in key
        assert "&" not in key
        assert "!" not in key


class TestCachePerformance:
    """Test cache performance characteristics."""

    @pytest.mark.asyncio
    async def test_batch_operations_performance(self):
        """Test performance of batch cache operations."""
        service = CacheService()
        mock_redis = AsyncMock()
        service._redis_client = mock_redis

        # Test batch set
        items = {f"key_{i}": {"data": i} for i in range(100)}
        start_time = time.time()

        # Simulate pipeline for batch operations
        mock_pipeline = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipeline
        mock_pipeline.__aenter__.return_value = mock_pipeline
        mock_pipeline.__aexit__.return_value = None
        mock_pipeline.execute.return_value = [True] * 100

        await service.set_many(items)

        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete within 1 second

        await service.close()

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test cache behavior under concurrent access."""
        service = CacheService()
        mock_redis = AsyncMock()
        service._redis_client = mock_redis

        # Simulate concurrent reads and writes
        async def concurrent_operation(i):
            if i % 2 == 0:
                await service.set(f"key_{i}", {"data": i})
            else:
                await service.get(f"key_{i}")

        tasks = [concurrent_operation(i) for i in range(50)]
        await asyncio.gather(*tasks)

        # Verify no race conditions or errors
        assert mock_redis.get.call_count == 25
        assert mock_redis.set.call_count == 25

        await service.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
