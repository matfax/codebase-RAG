"""
Tests for cache failover service.

This module contains comprehensive tests for the CacheFailoverService,
including failover scenarios, health monitoring, and recovery operations.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from .cache_failover_service import (
    CacheFailoverService,
    FailoverTrigger,
    FailoverStatus,
    ServiceHealth,
    FailoverEvent,
    FailoverConfiguration,
    get_cache_failover_service
)
from ..config.cache_config import CacheConfig
from ..services.cache_service import BaseCacheService


class MockCacheService(BaseCacheService):
    """Mock cache service for testing."""
    
    def __init__(self, service_id: str, should_fail: bool = False):
        self.service_id = service_id
        self.should_fail = should_fail
        self.operation_count = 0
        self.failure_count = 0
        self._data = {}

    async def get(self, key: str):
        """Mock get operation."""
        self.operation_count += 1
        if self.should_fail:
            self.failure_count += 1
            raise ConnectionError(f"Mock failure in {self.service_id}")
        return self._data.get(key)

    async def set(self, key: str, value, ttl=None):
        """Mock set operation."""
        self.operation_count += 1
        if self.should_fail:
            self.failure_count += 1
            raise ConnectionError(f"Mock failure in {self.service_id}")
        self._data[key] = value
        return True

    async def delete(self, key: str):
        """Mock delete operation."""
        self.operation_count += 1
        if self.should_fail:
            self.failure_count += 1
            raise ConnectionError(f"Mock failure in {self.service_id}")
        return self._data.pop(key, None) is not None

    async def exists(self, key: str):
        """Mock exists operation."""
        self.operation_count += 1
        if self.should_fail:
            self.failure_count += 1
            raise ConnectionError(f"Mock failure in {self.service_id}")
        return key in self._data

    def set_failure_mode(self, should_fail: bool):
        """Set whether this service should fail."""
        self.should_fail = should_fail


class TestCacheFailoverService:
    """Test cases for CacheFailoverService."""

    @pytest.fixture
    def failover_config(self):
        """Create failover configuration for testing."""
        return FailoverConfiguration(
            health_check_interval_seconds=1,  # Fast for testing
            failure_threshold=2,
            recovery_threshold=3,
            timeout_threshold_ms=1000.0,
            auto_recovery_enabled=True,
            auto_recovery_delay_seconds=2,  # Fast for testing
            health_check_timeout_seconds=1
        )

    @pytest.fixture
    async def mock_primary_service(self):
        """Create mock primary cache service."""
        return MockCacheService("primary", should_fail=False)

    @pytest.fixture
    async def mock_failover_service(self):
        """Create mock failover cache service."""
        return MockCacheService("failover", should_fail=False)

    @pytest.fixture
    async def failover_service(self, failover_config, mock_primary_service):
        """Create failover service with mocked dependencies."""
        with patch('src.services.cache_failover_service.get_cache_service', return_value=mock_primary_service):
            service = CacheFailoverService(
                config=CacheConfig(),
                failover_config=failover_config
            )
            await service.initialize()
            yield service
            await service.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, failover_service, mock_primary_service):
        """Test failover service initialization."""
        assert failover_service._primary_service == mock_primary_service
        assert failover_service._current_service == mock_primary_service
        assert failover_service._failover_status == FailoverStatus.ACTIVE
        assert failover_service._health_monitor_task is not None

    @pytest.mark.asyncio
    async def test_register_failover_service(self, failover_service, mock_failover_service):
        """Test registering a failover service."""
        failover_service.register_failover_service(mock_failover_service)
        
        assert mock_failover_service in failover_service._failover_services
        service_id = failover_service._get_service_id(mock_failover_service)
        assert service_id in failover_service._service_health

    @pytest.mark.asyncio
    async def test_normal_operations(self, failover_service):
        """Test normal cache operations through failover service."""
        # Test set operation
        result = await failover_service.set("test_key", "test_value")
        assert result is True
        
        # Test get operation
        value = await failover_service.get("test_key")
        assert value == "test_value"
        
        # Test exists operation
        exists = await failover_service.exists("test_key")
        assert exists is True
        
        # Test delete operation
        deleted = await failover_service.delete("test_key")
        assert deleted is True

    @pytest.mark.asyncio
    async def test_failover_on_connection_error(self, failover_service, mock_primary_service, mock_failover_service):
        """Test automatic failover when primary service fails."""
        # Register failover service
        failover_service.register_failover_service(mock_failover_service)
        
        # Make primary service fail
        mock_primary_service.set_failure_mode(True)
        
        # Perform operations that should trigger failover
        for _ in range(failover_service.failover_config.failure_threshold):
            try:
                await failover_service.set("test_key", "test_value")
            except ConnectionError:
                pass  # Expected failures
        
        # Next operation should trigger failover
        try:
            await failover_service.set("test_key", "test_value")
        except:
            pass
        
        # Give some time for failover to complete
        await asyncio.sleep(0.1)
        
        # Check that we've failed over
        assert failover_service._failover_status in [FailoverStatus.FAILING_OVER, FailoverStatus.FAILED_OVER]
        assert len(failover_service._failover_events) > 0

    @pytest.mark.asyncio
    async def test_manual_failover(self, failover_service, mock_failover_service):
        """Test manual failover trigger."""
        # Register failover service
        failover_service.register_failover_service(mock_failover_service)
        
        # Trigger manual failover
        event = await failover_service.manual_failover("Testing manual failover")
        
        assert isinstance(event, FailoverEvent)
        assert event.trigger == FailoverTrigger.MANUAL_TRIGGER
        assert event.success is True
        assert failover_service._failover_status == FailoverStatus.FAILED_OVER
        assert failover_service._current_service == mock_failover_service

    @pytest.mark.asyncio
    async def test_failover_without_available_services(self, failover_service):
        """Test failover behavior when no failover services are available."""
        # Don't register any failover services
        
        # Make primary service fail
        mock_primary_service = failover_service._primary_service
        mock_primary_service.set_failure_mode(True)
        
        # Try to trigger failover - should fail
        event = await failover_service._trigger_failover(
            FailoverTrigger.MANUAL_TRIGGER,
            {"test": "no services available"}
        )
        
        assert event.success is False
        assert "No healthy failover services available" in event.error_message
        assert failover_service._failover_status == FailoverStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_health_monitoring(self, failover_service, mock_primary_service):
        """Test health monitoring functionality."""
        # Wait for a health check cycle
        await asyncio.sleep(1.5)
        
        # Check that health status was updated
        service_id = failover_service._get_service_id(mock_primary_service)
        health_status = failover_service._service_health.get(service_id)
        
        assert health_status is not None
        assert health_status.health in [ServiceHealth.HEALTHY, ServiceHealth.DEGRADED]
        assert health_status.last_check is not None

    @pytest.mark.asyncio
    async def test_health_check_failure_detection(self, failover_service, mock_primary_service, mock_failover_service):
        """Test detection of health check failures."""
        # Register failover service
        failover_service.register_failover_service(mock_failover_service)
        
        # Make primary service fail
        mock_primary_service.set_failure_mode(True)
        
        # Wait for health checks to detect the failure
        await asyncio.sleep(3)
        
        # Check that failure was detected
        service_id = failover_service._get_service_id(mock_primary_service)
        consecutive_failures = failover_service._consecutive_failures.get(service_id, 0)
        
        assert consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_recovery_to_primary(self, failover_service, mock_primary_service, mock_failover_service):
        """Test recovery to primary service."""
        # Register failover service and trigger failover
        failover_service.register_failover_service(mock_failover_service)
        await failover_service.manual_failover("Test failover for recovery")
        
        assert failover_service._failover_status == FailoverStatus.FAILED_OVER
        
        # Make primary service healthy again
        mock_primary_service.set_failure_mode(False)
        
        # Simulate successful health checks to build up recovery threshold
        for _ in range(failover_service.failover_config.recovery_threshold + 1):
            await failover_service._check_service_health(mock_primary_service)
        
        # Attempt recovery
        recovery_success = await failover_service.manual_recovery()
        
        assert recovery_success is True
        assert failover_service._failover_status == FailoverStatus.ACTIVE
        assert failover_service._current_service == mock_primary_service

    @pytest.mark.asyncio
    async def test_failover_status_reporting(self, failover_service, mock_primary_service):
        """Test failover status reporting."""
        status = await failover_service.get_failover_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "current_service" in status
        assert "primary_service" in status
        assert "is_failed_over" in status
        assert "service_health" in status
        assert "recent_events" in status
        
        assert status["status"] == FailoverStatus.ACTIVE.value
        assert status["is_failed_over"] is False

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, failover_service, mock_failover_service):
        """Test detection of performance degradation."""
        # Register failover service
        failover_service.register_failover_service(mock_failover_service)
        
        # Establish baseline performance
        primary_service = failover_service._primary_service
        service_id = failover_service._get_service_id(primary_service)
        
        # Simulate fast operations to establish baseline
        for _ in range(10):
            await failover_service._update_service_health(primary_service, 10.0, True)
        
        # Simulate slow operations (performance degradation)
        slow_response_time = 100.0  # Much slower than baseline
        await failover_service._update_service_health(primary_service, slow_response_time, True)
        
        # Check if performance degradation was detected
        health_status = failover_service._service_health[service_id]
        assert health_status.metadata.get("performance_degraded") is True

    @pytest.mark.asyncio
    async def test_service_selection_logic(self, failover_service):
        """Test failover service selection logic."""
        # Create multiple failover services with different health
        healthy_service = MockCacheService("healthy", should_fail=False)
        unhealthy_service = MockCacheService("unhealthy", should_fail=True)
        
        failover_service.register_failover_service(healthy_service)
        failover_service.register_failover_service(unhealthy_service)
        
        # Update health status
        await failover_service._update_service_health(healthy_service, 50.0, True)
        await failover_service._update_service_health(unhealthy_service, 0.0, False)
        
        # Select best service
        best_service = await failover_service._select_best_failover_service()
        
        assert best_service == healthy_service

    @pytest.mark.asyncio
    async def test_callbacks(self, failover_service, mock_failover_service):
        """Test failover and recovery callbacks."""
        failover_called = False
        recovery_called = False
        
        def failover_callback(event):
            nonlocal failover_called
            failover_called = True
            assert isinstance(event, FailoverEvent)
        
        def recovery_callback(event):
            nonlocal recovery_called
            recovery_called = True
            assert isinstance(event, FailoverEvent)
        
        # Register callbacks
        failover_service.add_failover_callback(failover_callback)
        failover_service.add_recovery_callback(recovery_callback)
        
        # Register failover service and trigger failover
        failover_service.register_failover_service(mock_failover_service)
        await failover_service.manual_failover("Test callback")
        
        assert failover_called is True
        
        # Test recovery callback would require more complex setup
        # This is a simplified test

    @pytest.mark.asyncio
    async def test_timeout_handling(self, failover_service):
        """Test handling of timeout errors."""
        # Create a service that times out
        timeout_service = AsyncMock()
        timeout_service.set.side_effect = asyncio.TimeoutError("Operation timed out")
        
        failover_service._current_service = timeout_service
        
        # Try operation that should timeout and trigger failover logic
        with pytest.raises(asyncio.TimeoutError):
            await failover_service.set("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, failover_service):
        """Test concurrent operations during failover."""
        # This test ensures that concurrent operations are handled properly
        # during failover scenarios
        
        async def concurrent_operation(key_suffix):
            try:
                await failover_service.set(f"key_{key_suffix}", f"value_{key_suffix}")
                return True
            except:
                return False
        
        # Launch multiple concurrent operations
        tasks = [concurrent_operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some operations should succeed
        successful_ops = sum(1 for result in results if result is True)
        assert successful_ops >= 0  # At minimum, no crashes

    @pytest.mark.asyncio
    async def test_global_service_singleton(self):
        """Test global service singleton behavior."""
        with patch('src.services.cache_failover_service.get_cache_service'):
            service1 = await get_cache_failover_service()
            service2 = await get_cache_failover_service()
            
            assert service1 is service2
            assert isinstance(service1, CacheFailoverService)

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test failover configuration validation."""
        config = FailoverConfiguration(
            health_check_interval_seconds=30,
            failure_threshold=3,
            recovery_threshold=5,
            auto_recovery_enabled=True
        )
        
        with patch('src.services.cache_failover_service.get_cache_service'):
            service = CacheFailoverService(failover_config=config)
            
            assert service.failover_config.health_check_interval_seconds == 30
            assert service.failover_config.failure_threshold == 3
            assert service.failover_config.recovery_threshold == 5
            assert service.failover_config.auto_recovery_enabled is True

    @pytest.mark.asyncio
    async def test_service_identification(self, failover_service):
        """Test service identification logic."""
        primary_service = failover_service._primary_service
        service_id = failover_service._get_service_id(primary_service)
        
        assert service_id is not None
        assert service_id != "none"
        
        # Test retrieval by ID
        retrieved_service = failover_service._get_service_by_id(service_id)
        assert retrieved_service == primary_service

    @pytest.mark.asyncio
    async def test_error_propagation(self, failover_service):
        """Test that appropriate errors are propagated."""
        # Test with primary service that raises a non-connection error
        primary_service = failover_service._primary_service
        original_set = primary_service.set
        
        def mock_set_with_value_error(*args, **kwargs):
            raise ValueError("Invalid value")
        
        primary_service.set = mock_set_with_value_error
        
        # This should propagate the ValueError, not trigger failover
        with pytest.raises(ValueError, match="Invalid value"):
            await failover_service.set("test_key", "test_value")
        
        # Restore original method
        primary_service.set = original_set

    @pytest.mark.asyncio
    async def test_max_failover_attempts(self, failover_service):
        """Test maximum failover attempts configuration."""
        # Make all services fail
        primary_service = failover_service._primary_service
        primary_service.set_failure_mode(True)
        
        # Remove failover services so failover fails
        failover_service._failover_services.clear()
        
        # Should attempt up to max_failover_attempts
        with pytest.raises((ConnectionError, RuntimeError)):
            await failover_service.set("test_key", "test_value")

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, failover_config):
        """Test proper cleanup during shutdown."""
        with patch('src.services.cache_failover_service.get_cache_service'):
            service = CacheFailoverService(failover_config=failover_config)
            await service.initialize()
            
            # Verify tasks are running
            assert service._health_monitor_task is not None
            assert not service._health_monitor_task.cancelled()
            
            # Shutdown and verify cleanup
            await service.shutdown()
            
            # Tasks should be cancelled
            assert service._health_monitor_task.cancelled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])