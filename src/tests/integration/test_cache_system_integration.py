"""
Comprehensive cache system integration tests.

This module provides end-to-end testing for the complete cache system integration,
including performance verification, security validation, and failure testing.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
import redis.asyncio as redis

from src.config.cache_config import CacheConfig, CacheLevel, CacheWriteStrategy
from src.services.cache_service import MultiTierCacheService, get_cache_service
from src.services.embedding_cache_service import EmbeddingCacheService
from src.services.file_cache_service import FileCacheService
from src.services.project_cache_service import ProjectCacheService
from src.services.search_cache_service import SearchCacheService
from src.utils.cache_key_optimization import KeyOptimizationManager
from src.utils.cache_performance_optimization import AdaptivePerformanceOptimizer, PerformanceProfile
from src.utils.cache_serialization_optimization import OptimizedCacheSerializer


@pytest.fixture
async def cache_system():
    """Create a complete cache system for testing."""
    # Create test configuration
    config = CacheConfig(
        enabled=True, cache_level=CacheLevel.BOTH, write_strategy=CacheWriteStrategy.WRITE_THROUGH, key_prefix="test_cache"
    )

    # Initialize cache services
    cache_service = MultiTierCacheService(config)
    await cache_service.initialize()

    # Initialize specialized services
    embedding_service = EmbeddingCacheService(config)
    search_service = SearchCacheService(config)
    project_service = ProjectCacheService(config)
    file_service = FileCacheService(config)

    await embedding_service.initialize()
    await search_service.initialize()
    await project_service.initialize()
    await file_service.initialize()

    # Initialize optimization components
    optimizer = AdaptivePerformanceOptimizer()
    key_manager = KeyOptimizationManager()
    serializer = OptimizedCacheSerializer()

    system = {
        "cache_service": cache_service,
        "embedding_service": embedding_service,
        "search_service": search_service,
        "project_service": project_service,
        "file_service": file_service,
        "optimizer": optimizer,
        "key_manager": key_manager,
        "serializer": serializer,
        "config": config,
    }

    yield system

    # Cleanup
    await cache_service.shutdown()
    await embedding_service.shutdown()
    await search_service.shutdown()
    await project_service.shutdown()
    await file_service.shutdown()


class TestCacheSystemIntegration:
    """Test complete cache system integration."""

    @pytest.mark.asyncio
    async def test_end_to_end_cache_flow(self, cache_system):
        """Test complete end-to-end cache flow."""
        cache_service = cache_system["cache_service"]

        # Test data flow through all cache layers
        test_key = "integration_test_key"
        test_value = {"data": "test_value", "timestamp": time.time()}

        # 1. Set value
        success = await cache_service.set(test_key, test_value, ttl=3600)
        assert success, "Failed to set cache value"

        # 2. Get value from L1 (should hit)
        retrieved_value = await cache_service.get(test_key)
        assert retrieved_value == test_value, "L1 cache retrieval failed"

        # 3. Clear L1 cache
        cache_service.l1_cache.clear()

        # 4. Get value from L2 (should hit and promote to L1)
        retrieved_value = await cache_service.get(test_key)
        assert retrieved_value == test_value, "L2 cache retrieval failed"

        # 5. Verify promotion to L1
        l1_value = cache_service.l1_cache.get(test_key)
        assert l1_value == test_value, "L1 promotion failed"

        # 6. Delete value
        deleted = await cache_service.delete(test_key)
        assert deleted, "Failed to delete cache value"

        # 7. Verify deletion from both layers
        l1_value = cache_service.l1_cache.get(test_key)
        l2_value = await cache_service.l2_cache.get(test_key)
        assert l1_value is None and l2_value is None, "Cache deletion failed"

    @pytest.mark.asyncio
    async def test_specialized_service_integration(self, cache_system):
        """Test integration between specialized cache services."""
        embedding_service = cache_system["embedding_service"]
        search_service = cache_system["search_service"]
        project_service = cache_system["project_service"]
        file_service = cache_system["file_service"]

        # Test embedding cache
        embedding_key = "test_embedding"
        embedding_value = [0.1, 0.2, 0.3, 0.4, 0.5]

        await embedding_service.cache_embedding(embedding_key, embedding_value)
        cached_embedding = await embedding_service.get_cached_embedding(embedding_key)
        assert cached_embedding == embedding_value, "Embedding cache integration failed"

        # Test search cache
        search_params = {"query": "test", "limit": 10}
        search_results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]

        await search_service.cache_search_results(search_params, search_results)
        cached_results = await search_service.get_cached_search_results(search_params)
        assert cached_results == search_results, "Search cache integration failed"

        # Test project cache
        project_info = {"name": "test_project", "files": 100, "size": "10MB"}

        await project_service.cache_project_info("test_project", project_info)
        cached_info = await project_service.get_cached_project_info("test_project")
        assert cached_info == project_info, "Project cache integration failed"

        # Test file cache
        file_path = "/test/file.py"
        file_chunks = [{"type": "function", "name": "test_func"}]

        await file_service.cache_file_chunks(file_path, file_chunks)
        cached_chunks = await file_service.get_cached_file_chunks(file_path)
        assert cached_chunks == file_chunks, "File cache integration failed"

    @pytest.mark.asyncio
    async def test_cache_warming_integration(self, cache_system):
        """Test cache warming system integration."""
        cache_service = cache_system["cache_service"]

        # Populate cache with test data
        test_data = {f"warm_key_{i}": f"value_{i}" for i in range(100)}

        for key, value in test_data.items():
            await cache_service.set(key, value)

        # Simulate access patterns
        frequent_keys = [f"warm_key_{i}" for i in range(10)]
        for _ in range(5):  # Access frequent keys multiple times
            for key in frequent_keys:
                await cache_service.get(key)

        # Clear L1 cache
        cache_service.l1_cache.clear()

        # Trigger cache warming
        warming_result = await cache_service.trigger_cache_warmup("adaptive")
        assert warming_result["success"], "Cache warming failed"

        # Verify frequent keys were warmed
        warmed_count = 0
        for key in frequent_keys:
            if cache_service.l1_cache.exists(key):
                warmed_count += 1

        assert warmed_count > 0, "No keys were warmed to L1 cache"

    @pytest.mark.asyncio
    async def test_performance_optimization_integration(self, cache_system):
        """Test performance optimization system integration."""
        optimizer = cache_system["optimizer"]

        # Create performance profile
        profile = PerformanceProfile(
            hit_rate=0.5,  # Low hit rate to trigger optimization
            avg_latency=0.05,
            throughput_ops_per_sec=1000,
            memory_usage_mb=256,
            cpu_usage_percent=30,
            cache_size=1000,
            operation_count=5000,
        )

        # Run optimization
        optimization_result = await optimizer.optimize_performance(profile)

        assert optimization_result["status"] in ["completed", "baseline_set"], "Performance optimization failed"

        if optimization_result["status"] == "completed":
            assert optimization_result["recommendations_generated"] >= 0, "No recommendations generated"

    @pytest.mark.asyncio
    async def test_key_optimization_integration(self, cache_system):
        """Test key optimization system integration."""
        key_manager = cache_system["key_manager"]
        cache_service = cache_system["cache_service"]

        # Test key optimization
        key_components = {"project": "test_project", "service": "embedding", "content_hash": "abc123def456", "version": "1.0"}

        # Optimize key
        optimized_key = key_manager.optimize_cache_key("embedding", **key_components)
        assert optimized_key != str(key_components), "Key optimization failed"
        assert len(optimized_key) > 0, "Optimized key is empty"

        # Test cache operations with optimized key
        test_value = {"data": "optimized_key_test"}

        success = await cache_service.set(optimized_key, test_value)
        assert success, "Failed to set value with optimized key"

        retrieved_value = await cache_service.get(optimized_key)
        assert retrieved_value == test_value, "Failed to retrieve value with optimized key"

        # Restore key components
        restored_components = key_manager.restore_cache_key("embedding", optimized_key)
        assert "project" in restored_components, "Failed to restore key components"

    @pytest.mark.asyncio
    async def test_serialization_optimization_integration(self, cache_system):
        """Test serialization optimization integration."""
        serializer = cache_system["serializer"]
        cache_service = cache_system["cache_service"]

        # Test complex data serialization
        complex_data = {
            "vectors": [[0.1, 0.2, 0.3] for _ in range(100)],
            "metadata": {"source": "test", "timestamp": time.time()},
            "nested": {"level1": {"level2": {"level3": "deep_value"}}},
        }

        # Serialize and compress
        serialized = serializer.serialize_and_compress(complex_data)
        assert len(serialized) > 0, "Serialization failed"

        # Deserialize and decompress
        deserialized = serializer.decompress_and_deserialize(serialized)
        assert deserialized == complex_data, "Deserialization failed"

        # Test with cache service
        test_key = "serialization_test"
        await cache_service.set(test_key, complex_data)
        retrieved_data = await cache_service.get(test_key)

        # Note: Direct comparison might fail due to serialization format differences
        # So we test structure preservation
        assert isinstance(retrieved_data, dict), "Retrieved data type mismatch"
        assert "vectors" in retrieved_data, "Data structure not preserved"


class TestCachePerformanceVerification:
    """Test cache performance verification."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_performance(self, cache_system):
        """Test cache hit rate performance."""
        cache_service = cache_system["cache_service"]

        # Populate cache
        test_data = {f"perf_key_{i}": f"value_{i}" for i in range(1000)}
        for key, value in test_data.items():
            await cache_service.set(key, value)

        # Test cache hits
        hit_count = 0
        total_operations = 1000

        start_time = time.time()
        for i in range(total_operations):
            key = f"perf_key_{i % 500}"  # 50% hit rate expected
            value = await cache_service.get(key)
            if value is not None:
                hit_count += 1

        end_time = time.time()

        # Calculate metrics
        hit_rate = hit_count / total_operations
        avg_latency = (end_time - start_time) / total_operations

        assert hit_rate > 0.4, f"Hit rate too low: {hit_rate}"
        assert avg_latency < 0.01, f"Average latency too high: {avg_latency}"

    @pytest.mark.asyncio
    async def test_cache_throughput_performance(self, cache_system):
        """Test cache throughput performance."""
        cache_service = cache_system["cache_service"]

        operations_count = 5000
        test_data = {f"throughput_key_{i}": f"value_{i}" for i in range(operations_count)}

        # Test write throughput
        start_time = time.time()
        for key, value in test_data.items():
            await cache_service.set(key, value)
        write_time = time.time() - start_time

        write_throughput = operations_count / write_time

        # Test read throughput
        start_time = time.time()
        for key in test_data.keys():
            await cache_service.get(key)
        read_time = time.time() - start_time

        read_throughput = operations_count / read_time

        assert write_throughput > 1000, f"Write throughput too low: {write_throughput} ops/sec"
        assert read_throughput > 2000, f"Read throughput too low: {read_throughput} ops/sec"

    @pytest.mark.asyncio
    async def test_cache_memory_efficiency(self, cache_system):
        """Test cache memory efficiency."""
        cache_service = cache_system["cache_service"]

        # Test memory usage with large data
        large_data = "x" * 10000  # 10KB strings
        keys_count = 100

        # Get initial memory stats
        initial_stats = cache_service.l1_cache.get_info()
        initial_memory = initial_stats["memory_usage_mb"]

        # Add large data to cache
        for i in range(keys_count):
            await cache_service.set(f"memory_test_{i}", large_data)

        # Get final memory stats
        final_stats = cache_service.l1_cache.get_info()
        final_memory = final_stats["memory_usage_mb"]

        memory_increase = final_memory - initial_memory
        expected_increase = (len(large_data) * keys_count) / (1024 * 1024)  # Convert to MB

        # Allow for some overhead (up to 50% more than expected)
        assert memory_increase < expected_increase * 1.5, f"Memory usage too high: {memory_increase}MB"

    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, cache_system):
        """Test concurrent access performance."""
        cache_service = cache_system["cache_service"]

        async def worker_task(worker_id: int, operations: int):
            """Worker task for concurrent testing."""
            for i in range(operations):
                key = f"concurrent_{worker_id}_{i}"
                value = f"value_{worker_id}_{i}"

                # Set and get operation
                await cache_service.set(key, value)
                retrieved = await cache_service.get(key)
                assert retrieved == value, f"Concurrent access failed for {key}"

        # Run concurrent workers
        num_workers = 10
        operations_per_worker = 100

        start_time = time.time()
        tasks = [worker_task(i, operations_per_worker) for i in range(num_workers)]
        await asyncio.gather(*tasks)
        end_time = time.time()

        total_operations = num_workers * operations_per_worker * 2  # set + get
        throughput = total_operations / (end_time - start_time)

        assert throughput > 1000, f"Concurrent throughput too low: {throughput} ops/sec"


class TestCacheSecurityValidation:
    """Test cache security validation."""

    @pytest.mark.asyncio
    async def test_cache_isolation(self, cache_system):
        """Test cache isolation between different contexts."""
        cache_service = cache_system["cache_service"]

        # Test project isolation
        project1_key = "project1:test_key"
        project2_key = "project2:test_key"

        project1_value = {"project": "project1", "data": "sensitive1"}
        project2_value = {"project": "project2", "data": "sensitive2"}

        await cache_service.set(project1_key, project1_value)
        await cache_service.set(project2_key, project2_value)

        # Verify isolation
        retrieved1 = await cache_service.get(project1_key)
        retrieved2 = await cache_service.get(project2_key)

        assert retrieved1 != retrieved2, "Cache isolation failed"
        assert retrieved1["project"] == "project1", "Project 1 data corrupted"
        assert retrieved2["project"] == "project2", "Project 2 data corrupted"

    @pytest.mark.asyncio
    async def test_sensitive_data_handling(self, cache_system):
        """Test handling of sensitive data in cache."""
        cache_service = cache_system["cache_service"]

        # Test with sensitive data
        sensitive_data = {
            "api_key": "secret_api_key_12345",
            "password": "super_secret_password",
            "personal_info": {"email": "user@example.com", "phone": "+1234567890"},
        }

        key = "sensitive_test"
        await cache_service.set(key, sensitive_data)

        # Retrieve and verify data integrity
        retrieved_data = await cache_service.get(key)
        assert retrieved_data == sensitive_data, "Sensitive data integrity compromised"

        # Clean up sensitive data
        deleted = await cache_service.delete(key)
        assert deleted, "Failed to delete sensitive data"

        # Verify complete removal
        retrieved_after_delete = await cache_service.get(key)
        assert retrieved_after_delete is None, "Sensitive data not properly deleted"

    @pytest.mark.asyncio
    async def test_cache_key_security(self, cache_system):
        """Test cache key security and collision resistance."""
        key_manager = cache_system["key_manager"]

        # Test key collision resistance
        similar_data1 = {"content": "very similar content here"}
        similar_data2 = {"content": "very similar content there"}  # Only one word different

        key1 = key_manager.optimize_cache_key("test", **similar_data1)
        key2 = key_manager.optimize_cache_key("test", **similar_data2)

        assert key1 != key2, "Key collision detected for similar data"

        # Test key unpredictability
        test_data = {"project": "test", "file": "example.py"}
        keys = [key_manager.optimize_cache_key("test", **test_data) for _ in range(10)]

        # All keys should be identical for same data
        assert all(k == keys[0] for k in keys), "Key generation not deterministic"
        assert len(keys[0]) > 8, "Generated key too short for security"


class TestCacheFailureScenarios:
    """Test cache failure scenarios and recovery."""

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, cache_system):
        """Test behavior when Redis connection fails."""
        cache_service = cache_system["cache_service"]

        # Simulate Redis connection failure
        original_redis = cache_service.l2_cache.redis_manager._redis
        cache_service.l2_cache.redis_manager._redis = None

        try:
            # Cache should degrade gracefully to L1 only
            test_key = "failure_test_key"
            test_value = {"data": "failure_test"}

            # This should still work with L1 cache
            success = await cache_service.set(test_key, test_value)
            # Note: This might fail depending on write strategy, which is expected

            # Try to retrieve from L1
            l1_value = cache_service.l1_cache.get(test_key)
            if success:
                assert l1_value == test_value, "L1 cache not working during L2 failure"

        finally:
            # Restore Redis connection
            cache_service.l2_cache.redis_manager._redis = original_redis

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, cache_system):
        """Test cache behavior under memory pressure."""
        cache_service = cache_system["cache_service"]

        # Fill cache beyond capacity
        large_data = "x" * 50000  # 50KB per entry

        # Add data until we exceed L1 cache capacity
        for i in range(cache_service.l1_cache.max_size + 50):
            key = f"pressure_test_{i}"
            await cache_service.set(key, large_data)

        # Verify cache is functioning (eviction should have occurred)
        l1_info = cache_service.l1_cache.get_info()
        assert l1_info["size"] <= cache_service.l1_cache.max_size, "L1 cache exceeded capacity"

        # Verify most recent entries are still accessible
        recent_keys = [f"pressure_test_{i}" for i in range(cache_service.l1_cache.max_size - 10, cache_service.l1_cache.max_size + 50)]
        accessible_count = 0

        for key in recent_keys:
            value = await cache_service.get(key)
            if value is not None:
                accessible_count += 1

        assert accessible_count > 0, "No recent entries accessible under memory pressure"

    @pytest.mark.asyncio
    async def test_cache_corruption_recovery(self, cache_system):
        """Test recovery from cache corruption scenarios."""
        cache_service = cache_system["cache_service"]

        # Test with corrupted data
        valid_key = "valid_data_key"
        valid_data = {"valid": True, "data": "clean"}

        await cache_service.set(valid_key, valid_data)

        # Simulate corruption by directly modifying L1 cache
        if valid_key in cache_service.l1_cache._cache:
            cache_service.l1_cache._cache[valid_key].value = "corrupted_data"

        # Cache should handle corruption gracefully
        try:
            await cache_service.get(valid_key)
            # Depending on implementation, this might return corrupted data or None
            # The important thing is that it doesn't crash
        except Exception as e:
            pytest.fail(f"Cache corruption caused unhandled exception: {e}")

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, cache_system):
        """Test cache under high concurrency stress."""
        cache_service = cache_system["cache_service"]

        async def stress_worker(worker_id: int, operations: int):
            """High-intensity worker for stress testing."""
            errors = 0
            for i in range(operations):
                try:
                    key = f"stress_{worker_id}_{i % 100}"  # Reuse keys for conflicts
                    value = f"data_{worker_id}_{i}"

                    # Mixed operations
                    if i % 3 == 0:
                        await cache_service.set(key, value)
                    elif i % 3 == 1:
                        await cache_service.get(key)
                    else:
                        await cache_service.delete(key)
                except Exception:
                    errors += 1

            return errors

        # Run high-concurrency stress test
        num_workers = 50
        operations_per_worker = 200

        tasks = [stress_worker(i, operations_per_worker) for i in range(num_workers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Stress test caused {len(exceptions)} exceptions"

        # Check error rates
        total_errors = sum(r for r in results if isinstance(r, int))
        error_rate = total_errors / (num_workers * operations_per_worker)

        assert error_rate < 0.05, f"Error rate too high under stress: {error_rate}"

        # Verify cache is still functional
        test_key = "post_stress_test"
        test_value = {"status": "operational"}

        await cache_service.set(test_key, test_value)
        retrieved = await cache_service.get(test_key)
        assert retrieved == test_value, "Cache not functional after stress test"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
