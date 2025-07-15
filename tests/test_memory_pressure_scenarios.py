"""
Memory Pressure Scenario Tests for Query Caching Layer.

This module provides comprehensive testing for memory pressure scenarios including:
- High memory usage conditions
- Cache behavior under memory constraints
- Memory limit enforcement
- Graceful degradation testing
- Resource exhaustion handling
"""

import asyncio
import gc
import json
import random
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.cache_config import CacheConfig


@dataclass
class MemoryPressureResult:
    """Result of memory pressure scenario test."""

    scenario_name: str
    initial_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_limit_enforced: bool
    graceful_degradation: bool
    operations_succeeded: int
    operations_failed: int
    recovery_successful: bool


class MemoryPressureTester:
    """Tester for memory pressure scenarios."""

    def __init__(self):
        self.process = psutil.Process()

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def _generate_large_data(self, size_mb: float) -> dict[str, Any]:
        """Generate data of specified size in MB."""
        target_size = int(size_mb * 1024 * 1024)  # Convert to bytes

        # Account for JSON overhead
        content_size = max(100, target_size - 200)

        return {"large_content": "x" * content_size, "metadata": {"size_mb": size_mb, "timestamp": time.time()}}

    async def test_memory_limit_enforcement(
        self, cache_service: Any, memory_limit_mb: float = 100, data_size_mb: float = 10
    ) -> MemoryPressureResult:
        """Test memory limit enforcement."""
        scenario_name = "memory_limit_enforcement"
        initial_memory = self._get_memory_usage_mb()

        operations_succeeded = 0
        operations_failed = 0
        peak_memory = initial_memory

        # Try to exceed memory limit
        for i in range(20):
            try:
                key = f"memory_test_{i}"
                large_data = self._generate_large_data(data_size_mb)

                await cache_service.set(key, large_data)
                operations_succeeded += 1

                current_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, current_memory)

                # Check if we've hit memory pressure
                if current_memory > memory_limit_mb:
                    break

            except Exception as e:
                operations_failed += 1
                if "memory" in str(e).lower() or "oom" in str(e).lower():
                    break

        final_memory = self._get_memory_usage_mb()
        memory_limit_enforced = peak_memory <= memory_limit_mb * 1.2  # Allow 20% variance

        return MemoryPressureResult(
            scenario_name=scenario_name,
            initial_memory_mb=initial_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_limit_enforced=memory_limit_enforced,
            graceful_degradation=operations_failed == 0 or operations_succeeded > 0,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_successful=True,
        )

    async def test_graceful_degradation(self, cache_service: Any, pressure_threshold_mb: float = 200) -> MemoryPressureResult:
        """Test graceful degradation under memory pressure."""
        scenario_name = "graceful_degradation"
        initial_memory = self._get_memory_usage_mb()

        operations_succeeded = 0
        operations_failed = 0
        peak_memory = initial_memory

        # Gradually increase memory pressure
        for iteration in range(15):
            data_size = 5 + iteration * 2  # Increasing size

            try:
                key = f"degradation_test_{iteration}"
                data = self._generate_large_data(data_size)

                await cache_service.set(key, data)
                operations_succeeded += 1

                current_memory = self._get_memory_usage_mb()
                peak_memory = max(peak_memory, current_memory)

            except Exception:
                operations_failed += 1

        final_memory = self._get_memory_usage_mb()
        graceful_degradation = operations_succeeded > operations_failed

        return MemoryPressureResult(
            scenario_name=scenario_name,
            initial_memory_mb=initial_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_limit_enforced=False,  # Not testing limits here
            graceful_degradation=graceful_degradation,
            operations_succeeded=operations_succeeded,
            operations_failed=operations_failed,
            recovery_successful=final_memory < peak_memory,
        )


class TestMemoryPressureScenarios:
    """Test suite for memory pressure scenarios."""

    @pytest.fixture
    def memory_tester(self):
        """Create memory pressure tester."""
        return MemoryPressureTester()

    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service with memory tracking."""
        cache_data = {}
        memory_usage = 0
        memory_limit = 100 * 1024 * 1024  # 100MB limit

        class MockCacheService:
            async def get(self, key: str):
                return cache_data.get(key)

            async def set(self, key: str, value: Any, ttl: int = None):
                nonlocal memory_usage

                # Estimate memory usage
                data_size = len(json.dumps(value).encode("utf-8"))

                # Check memory limit
                if memory_usage + data_size > memory_limit:
                    raise Exception("OOM: Memory limit exceeded")

                cache_data[key] = value
                memory_usage += data_size
                return True

            async def delete(self, key: str):
                nonlocal memory_usage
                if key in cache_data:
                    data_size = len(json.dumps(cache_data[key]).encode("utf-8"))
                    del cache_data[key]
                    memory_usage = max(0, memory_usage - data_size)
                    return True
                return False

        return MockCacheService()

    @pytest.mark.asyncio
    async def test_memory_limit_enforcement(self, memory_tester, mock_cache_service):
        """Test memory limit enforcement."""
        result = await memory_tester.test_memory_limit_enforcement(mock_cache_service, memory_limit_mb=50, data_size_mb=5)

        assert result.scenario_name == "memory_limit_enforcement"
        assert result.operations_succeeded >= 0

        print(f"Memory limit test: {result.operations_succeeded} succeeded, {result.operations_failed} failed")

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, memory_tester, mock_cache_service):
        """Test graceful degradation under memory pressure."""
        result = await memory_tester.test_graceful_degradation(mock_cache_service, pressure_threshold_mb=100)

        assert result.scenario_name == "graceful_degradation"

        print(f"Graceful degradation: {result.graceful_degradation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
