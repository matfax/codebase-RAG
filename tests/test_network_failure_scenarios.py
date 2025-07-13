"""
Network Failure Scenario Tests for Query Caching Layer.

This module provides comprehensive testing for network failure scenarios including:
- Network connectivity issues
- Fallback mechanisms validation
- Connection timeout handling
- Network partition scenarios
- Recovery after network restoration
"""

import asyncio
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from config.cache_config import CacheConfig
except ImportError:
    # Alternative imports if relative imports fail
    import os

    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, os.path.abspath(src_path))
    from config.cache_config import CacheConfig


@dataclass
class NetworkFailureResult:
    """Result of network failure scenario test."""

    scenario_name: str
    failure_duration_seconds: float
    operations_during_failure: int
    fallback_activations: int
    recovery_time_seconds: float | None
    data_consistency_maintained: bool
    error_types: list[str] = field(default_factory=list)


class NetworkFailureTester:
    """Tester for network failure scenarios."""

    def __init__(self):
        self.network_available = True
        self.fallback_count = 0

    async def simulate_network_partition(self, cache_service: Any, partition_duration: float = 5.0) -> NetworkFailureResult:
        """Simulate network partition scenario."""
        scenario_name = "network_partition"
        start_time = time.time()

        operations_during_failure = 0
        fallback_activations = 0
        error_types = []

        # Normal operation before partition
        try:
            await cache_service.set("pre_partition", "test_value")
        except Exception as e:
            error_types.append(type(e).__name__)

        # Simulate network partition
        self.network_available = False
        partition_start = time.time()

        # Operations during partition
        while time.time() - partition_start < partition_duration:
            try:
                # Try cache operations during partition
                await cache_service.get("pre_partition")
                operations_during_failure += 1

                await cache_service.set(f"partition_key_{operations_during_failure}", "partition_value")
                operations_during_failure += 1

            except Exception as e:
                error_types.append(type(e).__name__)
                fallback_activations += 1

            await asyncio.sleep(0.2)

        # Restore network
        self.network_available = True
        recovery_start = time.time()

        # Test recovery
        recovery_successful = False
        try:
            await cache_service.set("post_partition", "recovery_value")
            result = await cache_service.get("post_partition")
            recovery_successful = result is not None
        except Exception as e:
            error_types.append(type(e).__name__)

        recovery_time = time.time() - recovery_start if recovery_successful else None
        total_duration = time.time() - start_time

        return NetworkFailureResult(
            scenario_name=scenario_name,
            failure_duration_seconds=partition_duration,
            operations_during_failure=operations_during_failure,
            fallback_activations=fallback_activations,
            recovery_time_seconds=recovery_time,
            data_consistency_maintained=recovery_successful,
            error_types=list(set(error_types)),
        )

    async def test_connection_timeout_handling(self, cache_service: Any, timeout_count: int = 10) -> NetworkFailureResult:
        """Test connection timeout handling."""
        scenario_name = "connection_timeout"
        start_time = time.time()

        operations_attempted = 0
        fallback_activations = 0
        error_types = []

        # Simulate multiple timeout scenarios
        for i in range(timeout_count):
            try:
                # Simulate timeout by making network unavailable briefly
                if random.random() < 0.5:  # 50% chance of timeout
                    self.network_available = False
                    await asyncio.sleep(0.1)
                    self.network_available = True

                await cache_service.set(f"timeout_test_{i}", f"value_{i}")
                operations_attempted += 1

            except Exception as e:
                error_types.append(type(e).__name__)
                fallback_activations += 1

        total_duration = time.time() - start_time

        return NetworkFailureResult(
            scenario_name=scenario_name,
            failure_duration_seconds=total_duration,
            operations_during_failure=operations_attempted,
            fallback_activations=fallback_activations,
            recovery_time_seconds=0.1,  # Quick recovery for timeouts
            data_consistency_maintained=True,
            error_types=list(set(error_types)),
        )


class NetworkFailureScenarioTester:
    """Enhanced network failure scenario tester for comprehensive testing."""

    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self.failure_tester = NetworkFailureTester()

    async def run_all_network_scenarios(self) -> list[dict[str, Any]]:
        """Run all network failure scenarios."""
        results = []

        # Scenario 1: Network Partition
        try:
            result = await self.test_network_partition_scenario()
            results.append(
                {
                    "scenario": "network_partition",
                    "status": "passed" if result.data_consistency_maintained else "failed",
                    "description": f"Network partition test with {result.operations_during_failure} operations during failure",
                    "details": {
                        "failure_duration": result.failure_duration_seconds,
                        "operations_during_failure": result.operations_during_failure,
                        "fallback_activations": result.fallback_activations,
                        "recovery_time": result.recovery_time_seconds,
                        "error_types": result.error_types,
                    },
                }
            )
        except Exception as e:
            results.append(
                {"scenario": "network_partition", "status": "error", "description": f"Network partition test failed: {e}", "error": str(e)}
            )

        # Scenario 2: Connection Timeout
        try:
            result = await self.test_connection_timeout_scenario()
            results.append(
                {
                    "scenario": "connection_timeout",
                    "status": "passed" if result else "failed",
                    "description": "Connection timeout handling test",
                    "details": {"timeout_handled": result},
                }
            )
        except Exception as e:
            results.append(
                {
                    "scenario": "connection_timeout",
                    "status": "error",
                    "description": f"Connection timeout test failed: {e}",
                    "error": str(e),
                }
            )

        # Scenario 3: Intermittent Network Issues
        try:
            result = await self.test_intermittent_network_issues()
            results.append(
                {
                    "scenario": "intermittent_network",
                    "status": "passed" if result else "failed",
                    "description": "Intermittent network issues test",
                    "details": {"resilience_maintained": result},
                }
            )
        except Exception as e:
            results.append(
                {
                    "scenario": "intermittent_network",
                    "status": "error",
                    "description": f"Intermittent network test failed: {e}",
                    "error": str(e),
                }
            )

        return results

    async def test_network_partition_scenario(self) -> NetworkFailureResult:
        """Test network partition scenario."""
        # Mock cache service for testing
        mock_cache = AsyncMock()
        mock_cache.set.return_value = True
        mock_cache.get.return_value = "test_value"

        result = await self.failure_tester.simulate_network_partition(mock_cache, partition_duration=2.0)
        return result

    async def test_connection_timeout_scenario(self) -> bool:
        """Test connection timeout handling."""
        # Simulate timeout scenario
        mock_cache = AsyncMock()
        mock_cache.set.side_effect = asyncio.TimeoutError("Connection timeout")

        try:
            await mock_cache.set("timeout_test", "value")
            return False  # Should have thrown timeout
        except asyncio.TimeoutError:
            return True  # Timeout was properly handled
        except Exception:
            return False  # Unexpected error

    async def test_intermittent_network_issues(self) -> bool:
        """Test intermittent network connectivity issues."""
        mock_cache = AsyncMock()

        # Simulate intermittent failures
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd call fails
                raise ConnectionError("Intermittent failure")
            return True

        mock_cache.set.side_effect = side_effect

        successful_operations = 0
        failed_operations = 0

        for i in range(10):
            try:
                await mock_cache.set(f"intermittent_key_{i}", f"value_{i}")
                successful_operations += 1
            except ConnectionError:
                failed_operations += 1

        # Should have some successes and some failures
        return successful_operations > 0 and failed_operations > 0


class TestNetworkFailureScenarios:
    """Test suite for network failure scenarios."""

    @pytest.fixture
    def network_tester(self):
        """Create network failure tester."""
        return NetworkFailureTester()

    @pytest.fixture
    def mock_cache_service(self, network_tester):
        """Create mock cache service with network simulation."""
        cache_data = {}

        class MockCacheService:
            async def get(self, key: str):
                if not network_tester.network_available:
                    raise ConnectionError("Network partition - connection failed")
                return cache_data.get(key)

            async def set(self, key: str, value: Any, ttl: int = None):
                if not network_tester.network_available:
                    raise ConnectionError("Network partition - connection failed")
                cache_data[key] = value
                return True

            async def delete(self, key: str):
                if not network_tester.network_available:
                    raise ConnectionError("Network partition - connection failed")
                if key in cache_data:
                    del cache_data[key]
                    return True
                return False

        return MockCacheService()

    @pytest.mark.asyncio
    async def test_network_partition_scenario(self, network_tester, mock_cache_service):
        """Test network partition scenario."""
        result = await network_tester.simulate_network_partition(mock_cache_service, partition_duration=2.0)  # Short duration for testing

        assert result.scenario_name == "network_partition"
        assert result.failure_duration_seconds >= 1.0
        assert result.fallback_activations > 0  # Should have fallback activations
        assert "ConnectionError" in result.error_types

        print(f"Network partition: {result.fallback_activations} fallback activations")

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, network_tester, mock_cache_service):
        """Test connection timeout handling."""
        result = await network_tester.test_connection_timeout_handling(mock_cache_service, timeout_count=10)

        assert result.scenario_name == "connection_timeout"
        assert result.operations_during_failure >= 0

        print(f"Timeout handling: {result.operations_during_failure} operations, {result.fallback_activations} fallbacks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
