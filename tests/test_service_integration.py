"""
Integration tests for service integration - Wave 15.2.1
Tests integration between cache services, invalidation, and monitoring.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis

from src.services.cache_invalidation_service import CacheInvalidationService
from src.services.cache_memory_pressure_service import CacheMemoryPressureService
from src.services.cache_performance_service import CachePerformanceService
from src.services.cache_service import CacheService
from src.services.cascade_invalidation_service import CascadeInvalidationService
from src.services.file_monitoring_service import FileMonitoringService
from src.services.security_audit_service import SecurityAuditService
from src.utils.secure_cache_utils import SecureCacheWrapper, SecurityContext


class TestServiceIntegrationBase:
    """Base class for service integration tests."""

    @pytest.fixture
    async def redis_client(self):
        """Create Redis client for testing."""
        client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        # Clean test database
        await client.flushdb()
        yield client
        await client.flushdb()
        await client.close()

    @pytest.fixture
    async def cache_service(self, redis_client):
        """Create cache service with real Redis."""
        service = CacheService()
        service._redis_client = redis_client
        yield service
        await service.close()

    @pytest.fixture
    async def invalidation_service(self, redis_client):
        """Create invalidation service."""
        service = CacheInvalidationService()
        service._cache_client = redis_client
        yield service
        await service.close()

    @pytest.fixture
    async def monitoring_service(self):
        """Create file monitoring service."""
        service = FileMonitoringService()
        yield service
        await service.stop()


class TestCacheServiceIntegration(TestServiceIntegrationBase):
    """Test integration between cache services."""

    @pytest.mark.asyncio
    async def test_cache_with_invalidation_integration(self, cache_service, invalidation_service):
        """Test cache service with invalidation service integration."""
        # Set cache data
        await cache_service.set("user:123", {"name": "Test User", "email": "test@example.com"})
        await cache_service.set("user:456", {"name": "Another User"})
        await cache_service.set("profile:123", {"bio": "Test bio"})

        # Verify data exists
        user_data = await cache_service.get("user:123")
        assert user_data["name"] == "Test User"

        # Invalidate user pattern
        result = await invalidation_service.invalidate_pattern("user:*")

        # Verify invalidation
        assert result["success"] is True
        assert result["invalidated_count"] == 2

        # Check data is gone
        user_data = await cache_service.get("user:123")
        assert user_data is None

        # Profile should still exist
        profile_data = await cache_service.get("profile:123")
        assert profile_data is not None

    @pytest.mark.asyncio
    async def test_cache_with_cascade_invalidation(self, cache_service, invalidation_service):
        """Test cache with cascade invalidation."""
        cascade_service = CascadeInvalidationService()
        cascade_service._cache_client = cache_service._redis_client

        # Set up data hierarchy
        await cache_service.set("user:123", {"name": "Test User"})
        await cache_service.set("profile:123", {"user_id": 123, "bio": "Bio"})
        await cache_service.set("settings:123", {"user_id": 123, "theme": "dark"})
        await cache_service.set("preferences:123", {"user_id": 123, "lang": "en"})

        # Register dependencies
        await cascade_service.register_dependency(parent="user:123", dependents=["profile:123", "settings:123", "preferences:123"])

        # Cascade invalidate
        result = await cascade_service.invalidate_cascade("user:123")

        # Verify all related data is invalidated
        assert result["total_invalidated"] == 4  # Parent + 3 dependents

        # Check all data is gone
        for key in ["user:123", "profile:123", "settings:123", "preferences:123"]:
            data = await cache_service.get(key)
            assert data is None

        await cascade_service.close()

    @pytest.mark.asyncio
    async def test_cache_with_security_integration(self, cache_service):
        """Test cache with security service integration."""
        security_context = SecurityContext(user_id="user123", tenant_id="tenant456", encryption_required=True, audit_enabled=True)

        audit_service = SecurityAuditService()
        secure_cache = SecureCacheWrapper(cache_service, security_context)

        # Store sensitive data
        sensitive_data = {"ssn": "123-45-6789", "credit_card": "4111111111111111", "api_key": "sk_live_abcdef123456"}

        await secure_cache.set("sensitive:user123", sensitive_data)

        # Verify data is encrypted in storage
        raw_data = await cache_service.get("sensitive:user123")
        assert "123-45-6789" not in str(raw_data)
        assert "4111111111111111" not in str(raw_data)

        # Verify data can be decrypted
        decrypted_data = await secure_cache.get("sensitive:user123")
        assert decrypted_data["ssn"] == "123-45-6789"

        # Check audit log
        audit_logs = await audit_service.get_audit_logs(user_id="user123")
        assert len(audit_logs) > 0

        await audit_service.close()


class TestPerformanceIntegration(TestServiceIntegrationBase):
    """Test performance monitoring integration."""

    @pytest.mark.asyncio
    async def test_cache_performance_monitoring(self, cache_service):
        """Test cache performance monitoring integration."""
        perf_service = CachePerformanceService()

        # Enable performance monitoring
        cache_service.enable_performance_monitoring(perf_service)

        # Perform various operations
        operations = [
            ("set", "perf:key:1", {"data": "test1"}),
            ("set", "perf:key:2", {"data": "test2"}),
            ("get", "perf:key:1", None),
            ("get", "perf:key:3", None),  # Cache miss
            ("delete", "perf:key:1", None),
        ]

        for op, key, value in operations:
            if op == "set":
                await cache_service.set(key, value)
            elif op == "get":
                await cache_service.get(key)
            elif op == "delete":
                await cache_service.delete(key)

        # Get performance metrics
        metrics = await perf_service.get_metrics()

        assert metrics["total_operations"] == 5
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1
        assert metrics["hit_rate"] == 0.5
        assert metrics["average_response_time"] > 0

        await perf_service.close()

    @pytest.mark.asyncio
    async def test_memory_pressure_integration(self, cache_service):
        """Test memory pressure service integration."""
        memory_service = CacheMemoryPressureService()

        # Connect services
        cache_service.set_memory_pressure_service(memory_service)
        memory_service.set_cache_service(cache_service)

        # Fill cache with data
        for i in range(100):
            await cache_service.set(f"memory:test:{i}", {"data": "x" * 1000, "priority": i % 3})  # Varying priorities

        # Simulate memory pressure
        await memory_service.simulate_memory_pressure(threshold=0.8)

        # Check that some data was evicted
        eviction_count = 0
        for i in range(100):
            data = await cache_service.get(f"memory:test:{i}")
            if data is None:
                eviction_count += 1

        assert eviction_count > 0  # Some data should be evicted

        await memory_service.close()


class TestFileMonitoringIntegration(TestServiceIntegrationBase):
    """Test file monitoring integration with cache services."""

    @pytest.mark.asyncio
    async def test_file_change_cache_invalidation(self, cache_service, invalidation_service, monitoring_service):
        """Test file change triggering cache invalidation."""
        # Connect services
        monitoring_service.set_invalidation_service(invalidation_service)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "model1.py"
            file2 = Path(temp_dir) / "model2.py"

            file1.write_text("class Model1: pass")
            file2.write_text("class Model2: pass")

            # Set up cache data
            await cache_service.set("cache:model1:data", {"version": 1})
            await cache_service.set("cache:model1:schema", {"fields": ["id", "name"]})
            await cache_service.set("cache:model2:data", {"version": 1})

            # Register file-to-cache mappings
            await monitoring_service.register_file_cache_mapping(str(file1), ["cache:model1:*"])
            await monitoring_service.register_file_cache_mapping(str(file2), ["cache:model2:*"])

            # Start monitoring
            await monitoring_service.start_monitoring(temp_dir)

            # Modify file1
            await asyncio.sleep(0.1)  # Ensure file watcher is ready
            file1.write_text("class Model1:\n    def __init__(self): pass")

            # Wait for file change detection and invalidation
            await asyncio.sleep(1.0)

            # Check that model1 cache was invalidated
            model1_data = await cache_service.get("cache:model1:data")
            model1_schema = await cache_service.get("cache:model1:schema")
            assert model1_data is None
            assert model1_schema is None

            # Check that model2 cache is still there
            model2_data = await cache_service.get("cache:model2:data")
            assert model2_data is not None

    @pytest.mark.asyncio
    async def test_batch_file_changes(self, cache_service, invalidation_service, monitoring_service):
        """Test handling of batch file changes."""
        monitoring_service.set_invalidation_service(invalidation_service)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            files = []
            for i in range(5):
                file_path = Path(temp_dir) / f"file{i}.py"
                file_path.write_text(f"# File {i}")
                files.append(file_path)

                # Set up cache for each file
                await cache_service.set(f"cache:file{i}:data", {"content": f"data{i}"})

                # Register mapping
                await monitoring_service.register_file_cache_mapping(str(file_path), [f"cache:file{i}:*"])

            # Start monitoring
            await monitoring_service.start_monitoring(temp_dir)
            await asyncio.sleep(0.1)

            # Modify all files simultaneously
            for i, file_path in enumerate(files):
                file_path.write_text(f"# Modified file {i}")

            # Wait for batch processing
            await asyncio.sleep(2.0)

            # Check all caches were invalidated
            for i in range(5):
                data = await cache_service.get(f"cache:file{i}:data")
                assert data is None


class TestCrossServiceCommunication(TestServiceIntegrationBase):
    """Test communication between different services."""

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, cache_service):
        """Test service health monitoring across services."""
        # Create health monitoring for multiple services
        perf_service = CachePerformanceService()
        memory_service = CacheMemoryPressureService()

        services = {"cache": cache_service, "performance": perf_service, "memory": memory_service}

        # Check health of all services
        health_status = {}
        for name, service in services.items():
            try:
                if hasattr(service, "health_check"):
                    health = await service.health_check()
                    health_status[name] = health
                else:
                    health_status[name] = {"status": "healthy", "message": "No health check available"}
            except Exception as e:
                health_status[name] = {"status": "unhealthy", "error": str(e)}

        # Verify health checks
        assert len(health_status) == 3
        for name, status in health_status.items():
            assert status["status"] in ["healthy", "unhealthy"]

        # Cleanup
        await perf_service.close()
        await memory_service.close()

    @pytest.mark.asyncio
    async def test_service_configuration_propagation(self, cache_service):
        """Test configuration changes propagating across services."""
        # Create services that depend on configuration
        perf_service = CachePerformanceService()

        # Initial configuration
        config = {"performance": {"metrics_enabled": True, "sampling_rate": 1.0, "batch_size": 100}}

        # Apply configuration to services
        await cache_service.update_config(config)
        await perf_service.update_config(config)

        # Verify configuration is applied
        cache_config = cache_service.get_config()
        perf_config = perf_service.get_config()

        assert cache_config["performance"]["metrics_enabled"] is True
        assert perf_config["performance"]["sampling_rate"] == 1.0

        # Update configuration
        new_config = {"performance": {"metrics_enabled": False, "sampling_rate": 0.5}}

        # Propagate changes
        await cache_service.update_config(new_config)
        await perf_service.update_config(new_config)

        # Verify updates
        updated_cache_config = cache_service.get_config()
        updated_perf_config = perf_service.get_config()

        assert updated_cache_config["performance"]["metrics_enabled"] is False
        assert updated_perf_config["performance"]["sampling_rate"] == 0.5

        await perf_service.close()

    @pytest.mark.asyncio
    async def test_service_dependency_management(self, cache_service):
        """Test service dependency management and startup order."""
        # Create services with dependencies
        services = {}
        startup_order = []

        class MockServiceWithDeps:
            def __init__(self, name, dependencies=None):
                self.name = name
                self.dependencies = dependencies or []
                self.started = False

            async def start(self):
                # Check dependencies are started
                for dep_name in self.dependencies:
                    if not services[dep_name].started:
                        raise Exception(f"Dependency {dep_name} not started")

                self.started = True
                startup_order.append(self.name)

            async def stop(self):
                self.started = False

        # Define service dependencies
        services["cache"] = MockServiceWithDeps("cache")
        services["invalidation"] = MockServiceWithDeps("invalidation", ["cache"])
        services["monitoring"] = MockServiceWithDeps("monitoring", ["cache", "invalidation"])
        services["performance"] = MockServiceWithDeps("performance", ["cache"])

        # Start services in dependency order
        def resolve_dependencies(service_name, visited=None):
            if visited is None:
                visited = set()

            if service_name in visited:
                return  # Already processed or circular dependency

            visited.add(service_name)
            service = services[service_name]

            # Start dependencies first
            for dep in service.dependencies:
                resolve_dependencies(dep, visited)

            if not service.started:
                asyncio.create_task(service.start())

        # Start all services
        for name in services.keys():
            resolve_dependencies(name)

        # Wait for all to start
        await asyncio.sleep(0.1)

        # Verify startup order respects dependencies
        assert startup_order.index("cache") < startup_order.index("invalidation")
        assert startup_order.index("cache") < startup_order.index("monitoring")
        assert startup_order.index("invalidation") < startup_order.index("monitoring")


class TestErrorPropagationIntegration(TestServiceIntegrationBase):
    """Test error handling and propagation across services."""

    @pytest.mark.asyncio
    async def test_cascade_error_handling(self, cache_service, invalidation_service):
        """Test error handling in cascade operations."""
        cascade_service = CascadeInvalidationService()
        cascade_service._cache_client = cache_service._redis_client

        # Set up data
        await cache_service.set("parent:1", {"data": "parent"})
        await cache_service.set("child:1", {"data": "child1"})
        await cache_service.set("child:2", {"data": "child2"})

        # Register dependencies
        await cascade_service.register_dependency("parent:1", ["child:1", "invalid:key", "child:2"])

        # Mock Redis to fail on specific key
        original_delete = cache_service._redis_client.delete

        async def mock_delete(*args):
            for key in args:
                if "invalid" in key:
                    raise redis.RedisError("Simulated Redis error")
            return await original_delete(*args)

        cache_service._redis_client.delete = mock_delete

        # Attempt cascade invalidation
        result = await cascade_service.invalidate_cascade("parent:1", continue_on_error=True)

        # Should handle partial failure gracefully
        assert result["total_invalidated"] == 3  # Parent + 2 successful children
        assert result["failed_count"] == 1
        assert "errors" in result

        await cascade_service.close()

    @pytest.mark.asyncio
    async def test_service_circuit_breaker(self, cache_service):
        """Test circuit breaker pattern for service failures."""
        from src.utils.circuit_breaker import CircuitBreaker

        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0, expected_exception=redis.RedisError)

        # Wrap cache operations with circuit breaker
        original_get = cache_service.get

        @circuit_breaker
        async def protected_get(key):
            return await original_get(key)

        # Simulate failures
        with patch.object(cache_service._redis_client, "get", side_effect=redis.RedisError("Connection failed")):
            # Should trip circuit breaker after threshold
            for i in range(5):
                try:
                    await protected_get(f"key:{i}")
                except (redis.RedisError, Exception):
                    pass

        # Circuit should be open
        assert circuit_breaker.state == "OPEN"

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should attempt to close circuit
        with patch.object(cache_service._redis_client, "get", return_value="test"):
            result = await protected_get("recovery:key")
            assert result == "test"
            assert circuit_breaker.state == "CLOSED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
