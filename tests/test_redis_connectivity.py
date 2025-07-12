"""
Integration tests for Redis connectivity - Wave 15.2.3
Tests Redis connection management, failover, and cluster operations.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis
from redis.exceptions import (
    ConnectionError,
    TimeoutError,
    RedisError,
    ClusterDownError,
    ReadOnlyError
)

from src.services.cache_service import CacheService
from src.services.resilient_cache_service import ResilientCacheService
from src.utils.resilient_redis_manager import (
    ResilientRedisManager,
    RedisConnectionPool,
    RedisClusterManager,
    RedisHealthChecker
)
from src.config.cache_config import RedisConfig


class TestRedisConnectivity:
    """Test Redis connectivity and connection management."""

    @pytest.fixture
    async def redis_config(self):
        """Create Redis configuration for testing."""
        return RedisConfig(
            host="localhost",
            port=6379,
            password=None,
            db=15,  # Use test database
            ssl=False,
            connection_timeout=5,
            max_connections=10,
            retry_attempts=3,
            retry_delay=0.1
        )

    @pytest.fixture
    async def redis_client(self, redis_config):
        """Create Redis client for testing."""
        client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            decode_responses=True,
            socket_connect_timeout=redis_config.connection_timeout,
            retry_on_timeout=True
        )
        
        # Clean test database
        try:
            await client.flushdb()
        except:
            pytest.skip("Redis not available for testing")
        
        yield client
        
        # Cleanup
        try:
            await client.flushdb()
            await client.close()
        except:
            pass

    @pytest.mark.asyncio
    async def test_basic_redis_operations(self, redis_client):
        """Test basic Redis operations."""
        # Test string operations
        await redis_client.set("test:string", "hello world")
        value = await redis_client.get("test:string")
        assert value == "hello world"
        
        # Test JSON operations
        data = {"name": "Test User", "id": 123, "active": True}
        await redis_client.set("test:json", json.dumps(data))
        retrieved = json.loads(await redis_client.get("test:json"))
        assert retrieved == data
        
        # Test TTL operations
        await redis_client.set("test:ttl", "expires", ex=2)
        ttl = await redis_client.ttl("test:ttl")
        assert 0 < ttl <= 2
        
        # Test delete operations
        await redis_client.delete("test:string", "test:json")
        assert await redis_client.get("test:string") is None

    @pytest.mark.asyncio
    async def test_redis_connection_pool(self, redis_config):
        """Test Redis connection pool management."""
        pool_manager = RedisConnectionPool(redis_config)
        
        # Initialize pool
        await pool_manager.initialize()
        
        # Test connection acquisition
        conn1 = await pool_manager.get_connection()
        conn2 = await pool_manager.get_connection()
        
        assert conn1 is not None
        assert conn2 is not None
        assert conn1 != conn2  # Different connections
        
        # Test pool statistics
        stats = await pool_manager.get_stats()
        assert stats["active_connections"] >= 2
        assert stats["total_connections"] <= redis_config.max_connections
        
        # Return connections
        await pool_manager.return_connection(conn1)
        await pool_manager.return_connection(conn2)
        
        # Test connection reuse
        conn3 = await pool_manager.get_connection()
        assert conn3 in [conn1, conn2]  # Reused connection
        
        await pool_manager.close()

    @pytest.mark.asyncio
    async def test_redis_health_checker(self, redis_client):
        """Test Redis health monitoring."""
        health_checker = RedisHealthChecker(redis_client)
        
        # Test healthy connection
        health = await health_checker.check_health()
        assert health["status"] == "healthy"
        assert health["response_time"] > 0
        assert health["memory_usage"] > 0
        
        # Test ping operation
        ping_result = await health_checker.ping()
        assert ping_result is True
        
        # Test info gathering
        info = await health_checker.get_info()
        assert "redis_version" in info
        assert "used_memory" in info
        assert "connected_clients" in info

    @pytest.mark.asyncio
    async def test_redis_connection_retry(self, redis_config):
        """Test Redis connection retry logic."""
        manager = ResilientRedisManager(redis_config)
        
        # Mock connection failure then success
        original_connect = redis.Redis.ping
        attempt_count = 0
        
        async def mock_ping(self):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Connection failed")
            return await original_connect(self)
        
        with patch.object(redis.Redis, 'ping', mock_ping):
            # Should retry and eventually succeed
            client = await manager.get_client()
            assert client is not None
            assert attempt_count == 3
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_redis_connection_timeout(self, redis_config):
        """Test Redis connection timeout handling."""
        # Set very short timeout
        short_timeout_config = redis_config
        short_timeout_config.connection_timeout = 0.001
        
        manager = ResilientRedisManager(short_timeout_config)
        
        # Mock slow connection
        with patch.object(redis.Redis, 'ping', side_effect=asyncio.sleep(1)):
            with pytest.raises((ConnectionError, TimeoutError)):
                await manager.get_client()
        
        await manager.close()


class TestResilientRedisManager:
    """Test resilient Redis manager functionality."""

    @pytest.fixture
    async def resilient_manager(self):
        """Create resilient Redis manager."""
        config = RedisConfig(
            host="localhost",
            port=6379,
            db=15,
            retry_attempts=3,
            retry_delay=0.1,
            circuit_breaker_enabled=True
        )
        manager = ResilientRedisManager(config)
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_automatic_failover(self, resilient_manager):
        """Test automatic failover to backup Redis."""
        # Configure backup Redis
        backup_config = RedisConfig(host="localhost", port=6380, db=15)
        resilient_manager.add_backup(backup_config)
        
        # Mock primary failure
        with patch.object(resilient_manager._primary_client, 'ping', side_effect=ConnectionError):
            # Should failover to backup
            client = await resilient_manager.get_client()
            assert client == resilient_manager._backup_clients[0]

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, resilient_manager):
        """Test circuit breaker for Redis connections."""
        # Simulate multiple failures
        with patch.object(resilient_manager._primary_client, 'ping', side_effect=ConnectionError):
            for _ in range(5):
                try:
                    await resilient_manager.get_client()
                except ConnectionError:
                    pass
        
        # Circuit should be open
        assert resilient_manager._circuit_breaker.is_open()
        
        # Wait for circuit reset
        await asyncio.sleep(resilient_manager._circuit_breaker.reset_timeout)
        
        # Should attempt to close circuit on success
        with patch.object(resilient_manager._primary_client, 'ping', return_value=True):
            client = await resilient_manager.get_client()
            assert client is not None

    @pytest.mark.asyncio
    async def test_connection_recovery(self, resilient_manager):
        """Test connection recovery after failure."""
        # Initial healthy state
        client = await resilient_manager.get_client()
        assert client is not None
        
        # Simulate connection loss
        resilient_manager._mark_unhealthy("Connection lost")
        
        # Mock recovery
        with patch.object(resilient_manager._primary_client, 'ping', return_value=True):
            # Should recover connection
            recovered_client = await resilient_manager.get_client()
            assert recovered_client is not None
            assert resilient_manager.is_healthy()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, resilient_manager):
        """Test handling of connection pool exhaustion."""
        # Get all available connections
        connections = []
        max_connections = resilient_manager._config.max_connections
        
        for _ in range(max_connections):
            conn = await resilient_manager.get_connection()
            connections.append(conn)
        
        # Next request should either wait or fail gracefully
        start_time = time.time()
        try:
            extra_conn = await asyncio.wait_for(
                resilient_manager.get_connection(),
                timeout=1.0
            )
            # If successful, return it
            await resilient_manager.return_connection(extra_conn)
        except asyncio.TimeoutError:
            # Expected behavior when pool is exhausted
            pass
        
        elapsed = time.time() - start_time
        assert elapsed >= 1.0  # Should have waited
        
        # Return connections
        for conn in connections:
            await resilient_manager.return_connection(conn)


class TestRedisClusterSupport:
    """Test Redis cluster support and operations."""

    @pytest.fixture
    async def cluster_manager(self):
        """Create Redis cluster manager for testing."""
        # Mock cluster configuration
        cluster_nodes = [
            {"host": "localhost", "port": 7000},
            {"host": "localhost", "port": 7001},
            {"host": "localhost", "port": 7002}
        ]
        
        manager = RedisClusterManager(cluster_nodes)
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_cluster_connection(self, cluster_manager):
        """Test Redis cluster connection."""
        # Mock cluster client
        mock_cluster = AsyncMock()
        mock_cluster.ping.return_value = True
        mock_cluster.cluster_info.return_value = {"cluster_state": "ok"}
        
        with patch('redis.asyncio.RedisCluster', return_value=mock_cluster):
            await cluster_manager.initialize()
            
            # Test cluster operations
            await cluster_manager.set("test:key", "value")
            value = await cluster_manager.get("test:key")
            
            mock_cluster.set.assert_called_once_with("test:key", "value")
            mock_cluster.get.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_cluster_node_failure(self, cluster_manager):
        """Test handling of cluster node failures."""
        mock_cluster = AsyncMock()
        
        # Simulate node failure
        mock_cluster.get.side_effect = ClusterDownError("Node down")
        
        with patch('redis.asyncio.RedisCluster', return_value=mock_cluster):
            await cluster_manager.initialize()
            
            # Should handle node failure gracefully
            try:
                await cluster_manager.get("test:key")
            except ClusterDownError:
                # Expected when cluster is down
                pass
            
            # Should attempt cluster recovery
            cluster_health = await cluster_manager.check_cluster_health()
            assert cluster_health["status"] in ["degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_cluster_slot_migration(self, cluster_manager):
        """Test handling of Redis cluster slot migration."""
        mock_cluster = AsyncMock()
        
        # Simulate slot migration
        mock_cluster.get.side_effect = [
            redis.exceptions.MovedError("MOVED 1234 localhost:7001"),
            "migrated_value"
        ]
        
        with patch('redis.asyncio.RedisCluster', return_value=mock_cluster):
            await cluster_manager.initialize()
            
            # Should handle MOVED error and retry
            value = await cluster_manager.get("test:key")
            assert mock_cluster.get.call_count == 2  # Initial + retry

    @pytest.mark.asyncio
    async def test_cluster_readonly_mode(self, cluster_manager):
        """Test handling of cluster readonly mode."""
        mock_cluster = AsyncMock()
        
        # Simulate readonly mode
        mock_cluster.set.side_effect = ReadOnlyError("Readonly mode")
        
        with patch('redis.asyncio.RedisCluster', return_value=mock_cluster):
            await cluster_manager.initialize()
            
            # Should handle readonly error
            with pytest.raises(ReadOnlyError):
                await cluster_manager.set("test:key", "value")
            
            # Should route to master node
            await cluster_manager.ensure_master_connection()


class TestRedisPerformanceOptimization:
    """Test Redis performance optimization features."""

    @pytest.mark.asyncio
    async def test_pipeline_operations(self, redis_client):
        """Test Redis pipeline for batch operations."""
        # Test manual pipeline
        pipeline = redis_client.pipeline()
        
        # Add multiple operations
        for i in range(100):
            pipeline.set(f"pipeline:key:{i}", f"value_{i}")
        
        # Execute pipeline
        start_time = time.time()
        results = await pipeline.execute()
        elapsed = time.time() - start_time
        
        assert len(results) == 100
        assert all(result is True for result in results)
        assert elapsed < 1.0  # Should be much faster than individual ops

    @pytest.mark.asyncio
    async def test_lua_script_execution(self, redis_client):
        """Test Lua script execution for atomic operations."""
        # Lua script for atomic increment with limit
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local current = redis.call('GET', key) or 0
        current = tonumber(current)
        
        if current >= limit then
            return -1
        else
            return redis.call('INCR', key)
        end
        """
        
        script = await redis_client.register_script(lua_script)
        
        # Test script execution
        await redis_client.set("counter:test", "5")
        
        # Should increment (under limit)
        result = await script(keys=["counter:test"], args=[10])
        assert result == 6
        
        # Set to limit
        await redis_client.set("counter:test", "10")
        
        # Should not increment (at limit)
        result = await script(keys=["counter:test"], args=[10])
        assert result == -1

    @pytest.mark.asyncio
    async def test_redis_compression(self, redis_client):
        """Test Redis data compression for large values."""
        import gzip
        
        # Large data that benefits from compression
        large_data = {"data": "x" * 10000, "metadata": {"size": "large"}}
        json_data = json.dumps(large_data)
        
        # Compress data
        compressed_data = gzip.compress(json_data.encode())
        
        # Store compressed
        await redis_client.set("compressed:test", compressed_data)
        
        # Retrieve and decompress
        retrieved = await redis_client.get("compressed:test")
        if isinstance(retrieved, str):
            retrieved = retrieved.encode()
        
        decompressed = gzip.decompress(retrieved).decode()
        restored_data = json.loads(decompressed)
        
        assert restored_data == large_data
        
        # Verify compression ratio
        compression_ratio = len(compressed_data) / len(json_data)
        assert compression_ratio < 0.5  # Should compress significantly

    @pytest.mark.asyncio
    async def test_redis_memory_optimization(self, redis_client):
        """Test Redis memory optimization techniques."""
        # Test memory-efficient data structures
        
        # Use hash for structured data instead of JSON strings
        user_data = {"name": "Test User", "email": "test@example.com", "age": 30}
        
        # Store as hash
        await redis_client.hset("user:hash:123", mapping=user_data)
        
        # Store as JSON string for comparison
        await redis_client.set("user:json:123", json.dumps(user_data))
        
        # Retrieve both
        hash_data = await redis_client.hgetall("user:hash:123")
        json_data = json.loads(await redis_client.get("user:json:123"))
        
        # Convert hash values back to correct types
        hash_data["age"] = int(hash_data["age"])
        
        assert hash_data == json_data
        
        # Check memory usage (hash should be more efficient for structured data)
        info = await redis_client.info("memory")
        memory_usage = info["used_memory"]
        assert memory_usage > 0


class TestRedisErrorRecovery:
    """Test Redis error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_network_partition_recovery(self):
        """Test recovery from network partitions."""
        config = RedisConfig(
            host="localhost",
            port=6379,
            db=15,
            retry_attempts=5,
            retry_delay=0.1
        )
        
        manager = ResilientRedisManager(config)
        
        # Simulate network partition
        with patch.object(manager._primary_client, 'ping', side_effect=ConnectionError("Network unreachable")):
            # Should mark as unhealthy
            try:
                await manager.get_client()
            except ConnectionError:
                pass
            
            assert not manager.is_healthy()
        
        # Simulate network recovery
        with patch.object(manager._primary_client, 'ping', return_value=True):
            # Should recover
            client = await manager.get_client()
            assert client is not None
            assert manager.is_healthy()
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, redis_client):
        """Test handling of Redis memory pressure."""
        # Fill Redis with data to approach memory limit
        large_value = "x" * 1000000  # 1MB per key
        
        try:
            for i in range(10):
                await redis_client.set(f"memory:pressure:{i}", large_value)
        except redis.exceptions.OutOfMemoryError:
            # Expected when Redis runs out of memory
            pass
        
        # Test eviction policy
        info = await redis_client.info("memory")
        if "maxmemory_policy" in info:
            assert info["maxmemory_policy"] in ["allkeys-lru", "volatile-lru", "allkeys-lfu"]
        
        # Clean up
        for i in range(10):
            try:
                await redis_client.delete(f"memory:pressure:{i}")
            except:
                pass

    @pytest.mark.asyncio
    async def test_redis_persistence_recovery(self, redis_client):
        """Test Redis persistence and recovery."""
        # Set data with persistence
        test_data = {"persistent": True, "timestamp": time.time()}
        await redis_client.set("persistence:test", json.dumps(test_data))
        
        # Force save to disk
        await redis_client.bgsave()
        
        # Wait for background save
        await asyncio.sleep(0.1)
        
        # Verify data exists
        retrieved = json.loads(await redis_client.get("persistence:test"))
        assert retrieved["persistent"] is True
        
        # Check last save time
        info = await redis_client.info("persistence")
        if "rdb_last_save_time" in info:
            last_save = info["rdb_last_save_time"]
            assert last_save > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])