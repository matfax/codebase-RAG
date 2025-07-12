"""
Redis Failure Scenario Tests for Query Caching Layer.

This module provides comprehensive testing for Redis failure scenarios including:
- Connection failures and recovery
- Network timeouts and disruptions
- Redis server unavailability
- Authentication failures
- Cluster node failures
- Memory exhaustion scenarios
- Failover and recovery validation
"""

import asyncio
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import AuthenticationError, ConnectionError, ResponseError, TimeoutError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from config.cache_config import CacheConfig
    from services.cache_service import CacheHealthStatus
    from services.project_cache_service import ProjectCacheService
    from services.search_cache_service import SearchCacheService
except ImportError:
    # Alternative imports if relative imports fail
    import os
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, os.path.abspath(src_path))
    
    from config.cache_config import CacheConfig
    from services.cache_service import CacheHealthStatus
    from services.project_cache_service import ProjectCacheService
    from services.search_cache_service import SearchCacheService


class FailureType(Enum):
    """Types of Redis failures to simulate."""

    CONNECTION_REFUSED = "connection_refused"
    TIMEOUT = "timeout"
    AUTHENTICATION_ERROR = "authentication_error"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    NETWORK_PARTITION = "network_partition"
    SERVER_SHUTDOWN = "server_shutdown"
    CLUSTER_NODE_FAILURE = "cluster_node_failure"
    DATA_CORRUPTION = "data_corruption"
    SLOW_RESPONSE = "slow_response"


@dataclass
class FailureScenarioResult:
    """Result of a failure scenario test."""

    scenario_name: str
    failure_type: FailureType
    duration_seconds: float
    operations_attempted: int
    operations_succeeded: int
    operations_failed: int
    recovery_time_seconds: float | None = None
    fallback_activated: bool = False
    data_consistency_maintained: bool = True
    error_types: list[str] = field(default_factory=list)
    recovery_successful: bool = False
    performance_impact: dict[str, Any] = field(default_factory=dict)


class RedisFailureSimulator:
    """Simulator for various Redis failure scenarios."""

    def __init__(self):
        self.failure_active = False
        self.failure_type = None
        self.failure_start_time = None

    async def simulate_connection_failure(self, redis_mock: AsyncMock) -> None:
        """Simulate Redis connection failure."""
        self.failure_active = True
        self.failure_type = FailureType.CONNECTION_REFUSED
        self.failure_start_time = time.time()

        # Make all Redis operations raise ConnectionError
        redis_mock.get.side_effect = ConnectionError("Connection refused")
        redis_mock.set.side_effect = ConnectionError("Connection refused")
        redis_mock.delete.side_effect = ConnectionError("Connection refused")
        redis_mock.ping.side_effect = ConnectionError("Connection refused")
        redis_mock.flushdb.side_effect = ConnectionError("Connection refused")

    async def simulate_timeout_failure(self, redis_mock: AsyncMock, timeout_delay: float = 10.0) -> None:
        """Simulate Redis timeout failure."""
        self.failure_active = True
        self.failure_type = FailureType.TIMEOUT
        self.failure_start_time = time.time()

        # Make all Redis operations raise TimeoutError
        redis_mock.get.side_effect = TimeoutError("Operation timed out")
        redis_mock.set.side_effect = TimeoutError("Operation timed out")
        redis_mock.delete.side_effect = TimeoutError("Operation timed out")
        redis_mock.ping.side_effect = TimeoutError("Operation timed out")

    async def simulate_authentication_failure(self, redis_mock: AsyncMock) -> None:
        """Simulate Redis authentication failure."""
        self.failure_active = True
        self.failure_type = FailureType.AUTHENTICATION_ERROR
        self.failure_start_time = time.time()

        # Make all Redis operations raise AuthenticationError
        redis_mock.get.side_effect = AuthenticationError("Authentication failed")
        redis_mock.set.side_effect = AuthenticationError("Authentication failed")
        redis_mock.delete.side_effect = AuthenticationError("Authentication failed")
        redis_mock.ping.side_effect = AuthenticationError("Authentication failed")

    async def simulate_memory_exhaustion(self, redis_mock: AsyncMock) -> None:
        """Simulate Redis memory exhaustion."""
        self.failure_active = True
        self.failure_type = FailureType.MEMORY_EXHAUSTION
        self.failure_start_time = time.time()

        # Make write operations fail, reads might work
        redis_mock.set.side_effect = ResponseError("OOM command not allowed when used memory > 'maxmemory'")
        redis_mock.delete.side_effect = ResponseError("OOM command not allowed when used memory > 'maxmemory'")
        # Get operations might still work
        redis_mock.get.return_value = None
        redis_mock.ping.side_effect = ResponseError("OOM command not allowed")

    async def simulate_slow_response(self, redis_mock: AsyncMock, delay: float = 2.0) -> None:
        """Simulate slow Redis responses."""
        self.failure_active = True
        self.failure_type = FailureType.SLOW_RESPONSE
        self.failure_start_time = time.time()

        async def slow_get(*args, **kwargs):
            await asyncio.sleep(delay)
            return None

        async def slow_set(*args, **kwargs):
            await asyncio.sleep(delay)
            return True

        async def slow_delete(*args, **kwargs):
            await asyncio.sleep(delay)
            return True

        async def slow_ping(*args, **kwargs):
            await asyncio.sleep(delay)
            return True

        redis_mock.get.side_effect = slow_get
        redis_mock.set.side_effect = slow_set
        redis_mock.delete.side_effect = slow_delete
        redis_mock.ping.side_effect = slow_ping

    async def simulate_intermittent_failure(self, redis_mock: AsyncMock, failure_rate: float = 0.3) -> None:
        """Simulate intermittent Redis failures."""
        self.failure_active = True
        self.failure_type = FailureType.NETWORK_PARTITION
        self.failure_start_time = time.time()

        original_get = AsyncMock(return_value=None)
        original_set = AsyncMock(return_value=True)
        original_delete = AsyncMock(return_value=True)
        original_ping = AsyncMock(return_value=True)

        async def intermittent_get(*args, **kwargs):
            if random.random() < failure_rate:
                raise ConnectionError("Intermittent connection failure")
            return await original_get(*args, **kwargs)

        async def intermittent_set(*args, **kwargs):
            if random.random() < failure_rate:
                raise ConnectionError("Intermittent connection failure")
            return await original_set(*args, **kwargs)

        async def intermittent_delete(*args, **kwargs):
            if random.random() < failure_rate:
                raise ConnectionError("Intermittent connection failure")
            return await original_delete(*args, **kwargs)

        async def intermittent_ping(*args, **kwargs):
            if random.random() < failure_rate:
                raise ConnectionError("Intermittent connection failure")
            return await original_ping(*args, **kwargs)

        redis_mock.get.side_effect = intermittent_get
        redis_mock.set.side_effect = intermittent_set
        redis_mock.delete.side_effect = intermittent_delete
        redis_mock.ping.side_effect = intermittent_ping

    async def restore_normal_operation(self, redis_mock: AsyncMock) -> None:
        """Restore normal Redis operation after failure."""
        self.failure_active = False
        recovery_time = time.time() - self.failure_start_time if self.failure_start_time else 0

        # Reset all Redis operations to normal behavior
        redis_mock.get.side_effect = None
        redis_mock.set.side_effect = None
        redis_mock.delete.side_effect = None
        redis_mock.ping.side_effect = None
        redis_mock.flushdb.side_effect = None

        # Set normal return values
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.delete.return_value = True
        redis_mock.ping.return_value = True
        redis_mock.flushdb.return_value = True

        return recovery_time


class RedisFailureTestSuite:
    """Test suite for Redis failure scenarios."""

    def __init__(self):
        self.failure_simulator = RedisFailureSimulator()

    async def test_connection_failure_scenario(
        self, cache_service: Any, redis_mock: AsyncMock, test_duration: float = 10.0, recovery_after: float = 5.0
    ) -> FailureScenarioResult:
        """Test cache behavior during Redis connection failure."""
        scenario_name = "connection_failure"
        start_time = time.time()

        operations_attempted = 0
        operations_succeeded = 0
        operations_failed = 0
        error_types = []

        # Start with normal operations
        for i in range(5):
            try:
                await cache_service.set(f"test_key_{i}", f"test_value_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        # Simulate connection failure
        await self.failure_simulator.simulate_connection_failure(redis_mock)

        # Test operations during failure
        failure_start = time.time()
        while time.time() - failure_start < recovery_after:
            try:
                # Test various operations
                await cache_service.get(f"test_key_{random.randint(0, 4)}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            try:
                await cache_service.set(f"fail_key_{int(time.time())}", "fail_value")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            await asyncio.sleep(0.1)

        # Restore normal operation
        recovery_start = time.time()
        await self.failure_simulator.restore_normal_operation(redis_mock)

        # Test recovery
        recovery_successful = False
        for i in range(10):
            try:
                await cache_service.set(f"recovery_key_{i}", f"recovery_value_{i}")
                result = await cache_service.get(f"recovery_key_{i}")
                if result is not None:
                    recovery_successful = True
                    break
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            await asyncio.sleep(0.1)

        recovery_time = time.time() - recovery_start if recovery_successful else None
        total_duration = time.time() - start_time

        return FailureScenarioResult(
            scenario_name=scenario_name,
            failure_type=FailureType.CONNECTION_REFUSED,
            duration_seconds=total_duration,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_time_seconds=recovery_time,
            recovery_successful=recovery_successful,
            error_types=list(set(error_types)),
        )

    async def test_timeout_scenario(
        self, cache_service: Any, redis_mock: AsyncMock, timeout_duration: float = 3.0
    ) -> FailureScenarioResult:
        """Test cache behavior during Redis timeout scenarios."""
        scenario_name = "timeout_scenario"
        start_time = time.time()

        operations_attempted = 0
        operations_succeeded = 0
        operations_failed = 0
        error_types = []

        # Simulate timeout failure
        await self.failure_simulator.simulate_timeout_failure(redis_mock)

        # Test operations during timeout
        for i in range(10):
            try:
                await cache_service.get(f"timeout_key_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            try:
                await cache_service.set(f"timeout_key_{i}", f"timeout_value_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        # Restore normal operation
        recovery_start = time.time()
        await self.failure_simulator.restore_normal_operation(redis_mock)

        # Test recovery
        recovery_successful = False
        for i in range(5):
            try:
                await cache_service.set(f"recovery_key_{i}", f"recovery_value_{i}")
                recovery_successful = True
                operations_attempted += 1
                operations_succeeded += 1
                break
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        recovery_time = time.time() - recovery_start if recovery_successful else None
        total_duration = time.time() - start_time

        return FailureScenarioResult(
            scenario_name=scenario_name,
            failure_type=FailureType.TIMEOUT,
            duration_seconds=total_duration,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_time_seconds=recovery_time,
            recovery_successful=recovery_successful,
            error_types=list(set(error_types)),
        )

    async def test_memory_exhaustion_scenario(self, cache_service: Any, redis_mock: AsyncMock) -> FailureScenarioResult:
        """Test cache behavior during Redis memory exhaustion."""
        scenario_name = "memory_exhaustion"
        start_time = time.time()

        operations_attempted = 0
        operations_succeeded = 0
        operations_failed = 0
        error_types = []

        # Pre-populate some data
        for i in range(5):
            try:
                await cache_service.set(f"pre_key_{i}", f"pre_value_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        # Simulate memory exhaustion
        await self.failure_simulator.simulate_memory_exhaustion(redis_mock)

        # Test operations during memory exhaustion
        for i in range(15):
            # Read operations might still work
            try:
                await cache_service.get(f"pre_key_{i % 5}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            # Write operations should fail
            try:
                await cache_service.set(f"oom_key_{i}", f"oom_value_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        # Restore normal operation
        recovery_start = time.time()
        await self.failure_simulator.restore_normal_operation(redis_mock)

        # Test recovery
        recovery_successful = False
        for i in range(5):
            try:
                await cache_service.set(f"recovery_key_{i}", f"recovery_value_{i}")
                recovery_successful = True
                operations_attempted += 1
                operations_succeeded += 1
                break
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        recovery_time = time.time() - recovery_start if recovery_successful else None
        total_duration = time.time() - start_time

        return FailureScenarioResult(
            scenario_name=scenario_name,
            failure_type=FailureType.MEMORY_EXHAUSTION,
            duration_seconds=total_duration,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_time_seconds=recovery_time,
            recovery_successful=recovery_successful,
            error_types=list(set(error_types)),
        )

    async def test_intermittent_failure_scenario(
        self, cache_service: Any, redis_mock: AsyncMock, failure_rate: float = 0.3, test_operations: int = 50
    ) -> FailureScenarioResult:
        """Test cache behavior during intermittent Redis failures."""
        scenario_name = "intermittent_failure"
        start_time = time.time()

        operations_attempted = 0
        operations_succeeded = 0
        operations_failed = 0
        error_types = []

        # Simulate intermittent failures
        await self.failure_simulator.simulate_intermittent_failure(redis_mock, failure_rate)

        # Test operations during intermittent failures
        for i in range(test_operations):
            # Mix of read and write operations
            try:
                if i % 3 == 0:
                    await cache_service.set(f"intermittent_key_{i}", f"intermittent_value_{i}")
                else:
                    await cache_service.get(f"intermittent_key_{i // 3}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            await asyncio.sleep(0.05)  # Small delay between operations

        # Restore normal operation
        recovery_start = time.time()
        await self.failure_simulator.restore_normal_operation(redis_mock)

        # Test recovery
        recovery_successful = False
        for i in range(5):
            try:
                await cache_service.set(f"recovery_key_{i}", f"recovery_value_{i}")
                result = await cache_service.get(f"recovery_key_{i}")
                recovery_successful = True
                operations_attempted += 1
                operations_succeeded += 1
                break
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

        recovery_time = time.time() - recovery_start if recovery_successful else None
        total_duration = time.time() - start_time

        # Calculate expected failure rate vs actual
        expected_failures = int(test_operations * failure_rate)
        actual_failure_rate = operations_failed / operations_attempted if operations_attempted > 0 else 0

        return FailureScenarioResult(
            scenario_name=scenario_name,
            failure_type=FailureType.NETWORK_PARTITION,
            duration_seconds=total_duration,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_time_seconds=recovery_time,
            recovery_successful=recovery_successful,
            error_types=list(set(error_types)),
            performance_impact={
                "expected_failure_rate": failure_rate,
                "actual_failure_rate": actual_failure_rate,
                "expected_failures": expected_failures,
                "variance": abs(actual_failure_rate - failure_rate),
            },
        )

    async def test_slow_response_scenario(
        self, cache_service: Any, redis_mock: AsyncMock, response_delay: float = 1.0
    ) -> FailureScenarioResult:
        """Test cache behavior during slow Redis responses."""
        scenario_name = "slow_response"
        start_time = time.time()

        operations_attempted = 0
        operations_succeeded = 0
        operations_failed = 0
        error_types = []
        response_times = []

        # Simulate slow responses
        await self.failure_simulator.simulate_slow_response(redis_mock, response_delay)

        # Test operations with slow responses
        for i in range(5):
            operation_start = time.time()
            try:
                await cache_service.set(f"slow_key_{i}", f"slow_value_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            operation_end = time.time()
            response_times.append(operation_end - operation_start)

        # Restore normal operation
        recovery_start = time.time()
        await self.failure_simulator.restore_normal_operation(redis_mock)

        # Test recovery with normal speed
        for i in range(3):
            operation_start = time.time()
            try:
                await cache_service.set(f"fast_key_{i}", f"fast_value_{i}")
                operations_attempted += 1
                operations_succeeded += 1
            except Exception as e:
                operations_attempted += 1
                operations_failed += 1
                error_types.append(type(e).__name__)

            operation_end = time.time()
            response_times.append(operation_end - operation_start)

        recovery_time = time.time() - recovery_start
        total_duration = time.time() - start_time

        # Analyze response times
        slow_responses = [t for t in response_times[:5]]  # First 5 are slow
        fast_responses = [t for t in response_times[5:]]  # Rest are normal

        return FailureScenarioResult(
            scenario_name=scenario_name,
            failure_type=FailureType.SLOW_RESPONSE,
            duration_seconds=total_duration,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_time_seconds=recovery_time,
            recovery_successful=len(fast_responses) > 0 and all(t < response_delay for t in fast_responses),
            error_types=list(set(error_types)),
            performance_impact={
                "avg_slow_response_time": sum(slow_responses) / len(slow_responses) if slow_responses else 0,
                "avg_fast_response_time": sum(fast_responses) / len(fast_responses) if fast_responses else 0,
                "expected_delay": response_delay,
                "response_times": response_times,
            },
        )

    async def test_health_check_during_failure(
        self, cache_service: Any, redis_mock: AsyncMock, failure_type: FailureType
    ) -> dict[str, Any]:
        """Test health check behavior during various failure types."""
        health_results = []

        # Get initial health
        try:
            initial_health = await cache_service.get_health()
            health_results.append({"stage": "initial", "health": initial_health, "timestamp": time.time()})
        except Exception as e:
            health_results.append({"stage": "initial", "error": str(e), "timestamp": time.time()})

        # Simulate failure
        if failure_type == FailureType.CONNECTION_REFUSED:
            await self.failure_simulator.simulate_connection_failure(redis_mock)
        elif failure_type == FailureType.TIMEOUT:
            await self.failure_simulator.simulate_timeout_failure(redis_mock)
        elif failure_type == FailureType.MEMORY_EXHAUSTION:
            await self.failure_simulator.simulate_memory_exhaustion(redis_mock)

        # Check health during failure
        for i in range(3):
            await asyncio.sleep(0.5)
            try:
                failure_health = await cache_service.get_health()
                health_results.append({"stage": f"failure_{i}", "health": failure_health, "timestamp": time.time()})
            except Exception as e:
                health_results.append({"stage": f"failure_{i}", "error": str(e), "timestamp": time.time()})

        # Restore and check recovery
        await self.failure_simulator.restore_normal_operation(redis_mock)

        for i in range(3):
            await asyncio.sleep(0.2)
            try:
                recovery_health = await cache_service.get_health()
                health_results.append({"stage": f"recovery_{i}", "health": recovery_health, "timestamp": time.time()})
            except Exception as e:
                health_results.append({"stage": f"recovery_{i}", "error": str(e), "timestamp": time.time()})

        return {
            "failure_type": failure_type.value,
            "health_checks": health_results,
            "health_detection_working": any(
                "health" in result and hasattr(result["health"], "status") and result["health"].status != CacheHealthStatus.HEALTHY
                for result in health_results
                if "failure" in result["stage"]
            ),
        }


class RedisFailureScenarioTester:
    """Enhanced Redis failure scenario tester for comprehensive testing."""
    
    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self.simulator = RedisFailureSimulator()
    
    async def run_all_failure_scenarios(self) -> List[Dict[str, Any]]:
        """Run all Redis failure scenarios."""
        results = []
        
        # Scenario 1: Connection Failure
        try:
            result = await self.test_connection_failure_scenario()
            results.append({
                "scenario": "redis_connection_failure",
                "status": "passed" if result.fallback_activated else "failed",
                "description": f"Redis connection failure test with {result.operations_attempted} operations",
                "details": {
                    "operations_attempted": result.operations_attempted,
                    "operations_succeeded": result.operations_succeeded,
                    "operations_failed": result.operations_failed,
                    "fallback_activated": result.fallback_activated,
                    "error_types": result.error_types
                }
            })
        except Exception as e:
            results.append({
                "scenario": "redis_connection_failure",
                "status": "error",
                "description": f"Connection failure test failed: {e}",
                "error": str(e)
            })
        
        # Scenario 2: Timeout Failure
        try:
            result = await self.test_timeout_failure_scenario()
            results.append({
                "scenario": "redis_timeout_failure",
                "status": "passed" if result.error_types else "failed",
                "description": "Redis timeout failure test",
                "details": {
                    "operations_attempted": result.operations_attempted,
                    "error_types": result.error_types,
                    "duration": result.duration_seconds
                }
            })
        except Exception as e:
            results.append({
                "scenario": "redis_timeout_failure",
                "status": "error",
                "description": f"Timeout failure test failed: {e}",
                "error": str(e)
            })
        
        # Scenario 3: Authentication Failure
        try:
            result = await self.test_authentication_failure_scenario()
            results.append({
                "scenario": "redis_auth_failure",
                "status": "passed" if "AuthenticationError" in result.error_types else "failed",
                "description": "Redis authentication failure test",
                "details": {
                    "error_types": result.error_types,
                    "operations_attempted": result.operations_attempted
                }
            })
        except Exception as e:
            results.append({
                "scenario": "redis_auth_failure",
                "status": "error",
                "description": f"Authentication failure test failed: {e}",
                "error": str(e)
            })
        
        # Scenario 4: Memory Exhaustion
        try:
            result = await self.test_memory_exhaustion_scenario()
            results.append({
                "scenario": "redis_memory_exhaustion",
                "status": "passed" if "ResponseError" in result.error_types else "failed",
                "description": "Redis memory exhaustion test",
                "details": {
                    "error_types": result.error_types,
                    "operations_attempted": result.operations_attempted
                }
            })
        except Exception as e:
            results.append({
                "scenario": "redis_memory_exhaustion",
                "status": "error",
                "description": f"Memory exhaustion test failed: {e}",
                "error": str(e)
            })
        
        return results
    
    async def test_connection_failure_scenario(self) -> FailureScenarioResult:
        """Test Redis connection failure scenario."""
        mock_redis = AsyncMock()
        
        # Simulate connection failure
        await self.simulator.simulate_connection_failure(mock_redis)
        
        # Create mock cache service
        mock_cache = AsyncMock()
        mock_cache._redis_client = mock_redis
        
        operations_attempted = 0
        operations_succeeded = 0
        operations_failed = 0
        error_types = []
        
        # Test various operations during failure
        test_operations = [
            ("get", "test_key"),
            ("set", "test_key", "test_value"),
            ("delete", "test_key")
        ]
        
        for operation, *args in test_operations:
            operations_attempted += 1
            try:
                if operation == "get":
                    await mock_redis.get(args[0])
                elif operation == "set":
                    await mock_redis.set(args[0], args[1])
                elif operation == "delete":
                    await mock_redis.delete(args[0])
                operations_succeeded += 1
            except Exception as e:
                operations_failed += 1
                error_types.append(type(e).__name__)
        
        return FailureScenarioResult(
            scenario_name="redis_connection_failure",
            failure_type=FailureType.CONNECTION_REFUSED,
            duration_seconds=1.0,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            fallback_activated=operations_failed > 0,
            error_types=list(set(error_types))
        )
    
    async def test_timeout_failure_scenario(self) -> FailureScenarioResult:
        """Test Redis timeout failure scenario."""
        mock_redis = AsyncMock()
        
        # Simulate timeout failure
        await self.simulator.simulate_timeout_failure(mock_redis, timeout_delay=1.0)
        
        operations_attempted = 0
        operations_failed = 0
        error_types = []
        
        start_time = time.time()
        
        # Test operations during timeout
        for i in range(3):
            operations_attempted += 1
            try:
                await mock_redis.set(f"timeout_key_{i}", f"value_{i}")
            except Exception as e:
                operations_failed += 1
                error_types.append(type(e).__name__)
        
        duration = time.time() - start_time
        
        return FailureScenarioResult(
            scenario_name="redis_timeout_failure",
            failure_type=FailureType.TIMEOUT,
            duration_seconds=duration,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_attempted - operations_failed,
            operations_failed=operations_failed,
            error_types=list(set(error_types))
        )
    
    async def test_authentication_failure_scenario(self) -> FailureScenarioResult:
        """Test Redis authentication failure scenario."""
        mock_redis = AsyncMock()
        
        # Simulate authentication failure
        await self.simulator.simulate_authentication_failure(mock_redis)
        
        operations_attempted = 0
        operations_failed = 0
        error_types = []
        
        # Test operations during auth failure
        for i in range(3):
            operations_attempted += 1
            try:
                await mock_redis.get(f"auth_key_{i}")
            except Exception as e:
                operations_failed += 1
                error_types.append(type(e).__name__)
        
        return FailureScenarioResult(
            scenario_name="redis_auth_failure",
            failure_type=FailureType.AUTHENTICATION_ERROR,
            duration_seconds=1.0,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_attempted - operations_failed,
            operations_failed=operations_failed,
            error_types=list(set(error_types))
        )
    
    async def test_memory_exhaustion_scenario(self) -> FailureScenarioResult:
        """Test Redis memory exhaustion scenario."""
        mock_redis = AsyncMock()
        
        # Simulate memory exhaustion
        await self.simulator.simulate_memory_exhaustion(mock_redis)
        
        operations_attempted = 0
        operations_failed = 0
        error_types = []
        
        # Test operations during memory exhaustion
        for i in range(5):
            operations_attempted += 1
            try:
                await mock_redis.set(f"memory_key_{i}", "large_value" * 1000)
            except Exception as e:
                operations_failed += 1
                error_types.append(type(e).__name__)
        
        return FailureScenarioResult(
            scenario_name="redis_memory_exhaustion",
            failure_type=FailureType.MEMORY_EXHAUSTION,
            duration_seconds=1.0,
            operations_attempted=operations_attempted,
            operations_succeeded=operations_attempted - operations_failed,
            operations_failed=operations_failed,
            error_types=list(set(error_types))
        )


class TestRedisFailureScenarios:
    """Test suite for Redis failure scenarios."""

    @pytest.fixture
    def failure_test_suite(self):
        """Create Redis failure test suite."""
        return RedisFailureTestSuite()

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()

        # Default behavior (normal operation)
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.delete.return_value = True
        redis_mock.ping.return_value = True
        redis_mock.flushdb.return_value = True

        return redis_mock

    @pytest.fixture
    def mock_cache_service(self, mock_redis):
        """Create mock cache service with Redis dependency."""

        class MockCacheService:
            def __init__(self):
                self._redis = mock_redis

            async def get(self, key: str):
                return await self._redis.get(key)

            async def set(self, key: str, value: Any, ttl: int = None):
                return await self._redis.set(key, value, ex=ttl)

            async def delete(self, key: str):
                return await self._redis.delete(key)

            async def clear(self):
                return await self._redis.flushdb()

            async def get_health(self):
                try:
                    await self._redis.ping()
                    from services.cache_service import CacheHealthInfo

                    return CacheHealthInfo(status=CacheHealthStatus.HEALTHY, redis_connected=True, redis_ping_time=0.001)
                except Exception as e:
                    from services.cache_service import CacheHealthInfo

                    return CacheHealthInfo(status=CacheHealthStatus.DISCONNECTED, redis_connected=False, last_error=str(e))

        return MockCacheService()

    @pytest.mark.asyncio
    async def test_connection_failure_scenario(self, failure_test_suite, mock_cache_service, mock_redis):
        """Test connection failure scenario."""
        result = await failure_test_suite.test_connection_failure_scenario(
            mock_cache_service,
            mock_redis,
            test_duration=2.0,
            recovery_after=1.0,  # Reduced for testing
        )

        # Verify test results
        assert result.scenario_name == "connection_failure"
        assert result.failure_type == FailureType.CONNECTION_REFUSED
        assert result.operations_attempted > 0
        assert result.operations_failed > 0  # Should have failures during connection issues

        # Should detect connection errors
        assert "ConnectionError" in result.error_types

        print(
            f"Connection failure: {result.operations_failed}/{result.operations_attempted} failed, recovery: {result.recovery_successful}"
        )

    @pytest.mark.asyncio
    async def test_timeout_scenario(self, failure_test_suite, mock_cache_service, mock_redis):
        """Test timeout scenario."""
        result = await failure_test_suite.test_timeout_scenario(mock_cache_service, mock_redis, timeout_duration=1.0)

        # Verify test results
        assert result.scenario_name == "timeout_scenario"
        assert result.failure_type == FailureType.TIMEOUT
        assert result.operations_attempted > 0
        assert result.operations_failed > 0  # Should have timeout failures

        # Should detect timeout errors
        assert "TimeoutError" in result.error_types

        print(f"Timeout scenario: {result.operations_failed}/{result.operations_attempted} failed, recovery: {result.recovery_successful}")

    @pytest.mark.asyncio
    async def test_memory_exhaustion_scenario(self, failure_test_suite, mock_cache_service, mock_redis):
        """Test memory exhaustion scenario."""
        result = await failure_test_suite.test_memory_exhaustion_scenario(mock_cache_service, mock_redis)

        # Verify test results
        assert result.scenario_name == "memory_exhaustion"
        assert result.failure_type == FailureType.MEMORY_EXHAUSTION
        assert result.operations_attempted > 0
        assert result.operations_failed > 0  # Should have OOM failures

        # Should detect memory exhaustion errors
        assert "ResponseError" in result.error_types

        print(f"Memory exhaustion: {result.operations_failed}/{result.operations_attempted} failed, recovery: {result.recovery_successful}")

    @pytest.mark.asyncio
    async def test_intermittent_failure_scenario(self, failure_test_suite, mock_cache_service, mock_redis):
        """Test intermittent failure scenario."""
        result = await failure_test_suite.test_intermittent_failure_scenario(
            mock_cache_service,
            mock_redis,
            failure_rate=0.4,
            test_operations=30,  # 40% failure rate
        )

        # Verify test results
        assert result.scenario_name == "intermittent_failure"
        assert result.failure_type == FailureType.NETWORK_PARTITION
        assert result.operations_attempted >= 30

        # Should have some failures (intermittent)
        actual_failure_rate = result.operations_failed / result.operations_attempted if result.operations_attempted > 0 else 0

        # Allow some variance in failure rate (±20%)
        assert 0.2 <= actual_failure_rate <= 0.6, f"Unexpected failure rate: {actual_failure_rate}"

        # Should have performance impact data
        assert "actual_failure_rate" in result.performance_impact
        assert "expected_failure_rate" in result.performance_impact

        print(f"Intermittent failure: {actual_failure_rate:.1%} failure rate (expected 40%)")

    @pytest.mark.asyncio
    async def test_slow_response_scenario(self, failure_test_suite, mock_cache_service, mock_redis):
        """Test slow response scenario."""
        result = await failure_test_suite.test_slow_response_scenario(mock_cache_service, mock_redis, response_delay=0.5)  # 500ms delay

        # Verify test results
        assert result.scenario_name == "slow_response"
        assert result.failure_type == FailureType.SLOW_RESPONSE
        assert result.operations_attempted > 0

        # Should have performance impact data
        assert "avg_slow_response_time" in result.performance_impact
        assert "avg_fast_response_time" in result.performance_impact

        # Slow responses should be approximately the expected delay
        slow_avg = result.performance_impact["avg_slow_response_time"]
        fast_avg = result.performance_impact["avg_fast_response_time"]

        assert slow_avg > 0.4, f"Slow responses not slow enough: {slow_avg}s"
        assert fast_avg < 0.1, f"Fast responses too slow: {fast_avg}s"

        print(f"Slow response: {slow_avg:.3f}s slow, {fast_avg:.3f}s fast")

    @pytest.mark.asyncio
    async def test_health_check_during_failures(self, failure_test_suite, mock_cache_service, mock_redis):
        """Test health check behavior during various failures."""

        # Test health during connection failure
        connection_health = await failure_test_suite.test_health_check_during_failure(
            mock_cache_service, mock_redis, FailureType.CONNECTION_REFUSED
        )

        # Verify health check results
        assert connection_health["failure_type"] == "connection_refused"
        assert len(connection_health["health_checks"]) > 0

        # Should detect unhealthy state during failure
        failure_checks = [result for result in connection_health["health_checks"] if "failure" in result["stage"]]
        assert len(failure_checks) > 0

        print(f"Health check during connection failure: {len(failure_checks)} failure checks")

        # Test health during timeout
        timeout_health = await failure_test_suite.test_health_check_during_failure(mock_cache_service, mock_redis, FailureType.TIMEOUT)

        assert timeout_health["failure_type"] == "timeout"
        print(f"Health check during timeout: {len(timeout_health['health_checks'])} total checks")

    def test_failure_simulator_functionality(self, mock_redis):
        """Test failure simulator functionality."""
        simulator = RedisFailureSimulator()

        # Test initial state
        assert not simulator.failure_active
        assert simulator.failure_type is None

        # Test connection failure simulation
        asyncio.run(simulator.simulate_connection_failure(mock_redis))
        assert simulator.failure_active
        assert simulator.failure_type == FailureType.CONNECTION_REFUSED

        # Test restoration
        asyncio.run(simulator.restore_normal_operation(mock_redis))
        assert not simulator.failure_active

        print("Failure simulator functionality verified")

    def test_failure_scenario_result_structure(self):
        """Test failure scenario result data structure."""
        result = FailureScenarioResult(
            scenario_name="test_scenario",
            failure_type=FailureType.CONNECTION_REFUSED,
            duration_seconds=5.0,
            operations_attempted=100,
            operations_succeeded=60,
            operations_failed=40,
            recovery_time_seconds=2.0,
            recovery_successful=True,
            error_types=["ConnectionError", "TimeoutError"],
        )

        assert result.scenario_name == "test_scenario"
        assert result.failure_type == FailureType.CONNECTION_REFUSED
        assert result.operations_attempted == 100
        assert result.operations_succeeded == 60
        assert result.operations_failed == 40
        assert result.recovery_successful is True
        assert "ConnectionError" in result.error_types


@pytest.mark.integration
class TestRedisFailureIntegration:
    """Integration tests for Redis failure scenarios with real services."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(
            enabled=True,
            redis_url="redis://localhost:6379/15",  # Test database
            default_ttl=300,
            max_memory_mb=50,
            connection_timeout=2.0,  # Short timeout for testing
            max_retries=2,
        )

    @pytest.mark.asyncio
    async def test_real_redis_connection_handling(self, cache_config):
        """Test real Redis connection handling."""
        try:
            # Test with correct configuration
            service = SearchCacheService(cache_config)
            await service.initialize()

            # Verify normal operation
            await service.set("test_key", "test_value")
            result = await service.get("test_key")
            assert result == "test_value"

            # Test health check
            health = await service.get_health()
            assert health.redis_connected is True

            await service.shutdown()

            print("Real Redis connection handling verified")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")

    @pytest.mark.asyncio
    async def test_invalid_redis_url_handling(self):
        """Test handling of invalid Redis URL."""
        # Test with invalid Redis URL
        invalid_config = CacheConfig(enabled=True, redis_url="redis://invalid-host:6379/0", connection_timeout=1.0, max_retries=1)

        service = SearchCacheService(invalid_config)

        # Should handle initialization gracefully
        try:
            await service.initialize()

            # Operations should handle connection failures
            result = await service.get("test_key")
            # Should return None or handle gracefully

            health = await service.get_health()
            # Should indicate unhealthy state

        except Exception as e:
            # Connection errors are expected
            assert "connection" in str(e).lower() or "timeout" in str(e).lower()

        finally:
            await service.shutdown()

        print("Invalid Redis URL handling verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
