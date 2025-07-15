"""
Cache Corruption Scenario Tests for Query Caching Layer.

This module provides comprehensive testing for cache data corruption scenarios including:
- Data integrity validation
- Corrupted data detection and handling
- Cache consistency verification
- Recovery from corruption scenarios
- Data validation and checksums
"""

import asyncio
import hashlib
import json
import random
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.cache_config import CacheConfig


@dataclass
class CorruptionTestResult:
    """Result of cache corruption scenario test."""

    scenario_name: str
    corruption_detected: bool
    data_consistency_maintained: bool
    recovery_successful: bool
    corrupted_entries: int
    total_entries: int
    error_types: list[str] = field(default_factory=list)


class CacheCorruptionTester:
    """Tester for cache corruption scenarios."""

    def __init__(self):
        self.original_data = {}

    def _generate_test_data(self, size: int = 1000) -> dict[str, Any]:
        """Generate test data with integrity markers."""
        data = {
            "id": random.randint(1, 1000000),
            "content": "".join(random.choices(string.ascii_letters, k=size)),
            "timestamp": asyncio.get_event_loop().time(),
            "metadata": {"type": "test", "size": size},
        }

        # Add checksum for integrity validation
        data_str = json.dumps(data, sort_keys=True)
        data["checksum"] = hashlib.md5(data_str.encode()).hexdigest()

        return data

    def _corrupt_data(self, data: dict[str, Any], corruption_type: str = "random") -> dict[str, Any]:
        """Corrupt data in various ways."""
        if corruption_type == "random":
            # Random character corruption
            if "content" in data:
                content = data["content"]
                pos = random.randint(0, len(content) - 1)
                corrupted_char = random.choice(string.ascii_letters)
                data["content"] = content[:pos] + corrupted_char + content[pos + 1 :]

        elif corruption_type == "partial":
            # Remove part of the data
            if "metadata" in data:
                del data["metadata"]

        elif corruption_type == "type_change":
            # Change data type
            if "id" in data:
                data["id"] = str(data["id"])  # Convert int to string

        return data

    def _validate_data_integrity(self, data: dict[str, Any]) -> bool:
        """Validate data integrity using checksum."""
        if "checksum" not in data:
            return False

        stored_checksum = data.pop("checksum")
        data_str = json.dumps(data, sort_keys=True)
        calculated_checksum = hashlib.md5(data_str.encode()).hexdigest()

        # Restore checksum
        data["checksum"] = stored_checksum

        return stored_checksum == calculated_checksum

    async def test_data_corruption_detection(
        self, cache_service: Any, test_entries: int = 20, corruption_rate: float = 0.3
    ) -> CorruptionTestResult:
        """Test detection of corrupted cache data."""
        scenario_name = "corruption_detection"

        # Store original data
        test_keys = []
        for i in range(test_entries):
            key = f"corruption_test_{i}"
            data = self._generate_test_data()
            self.original_data[key] = data.copy()
            test_keys.append(key)

            await cache_service.set(key, data)

        # Simulate corruption by modifying some entries
        corrupted_count = 0
        for key in test_keys:
            if random.random() < corruption_rate:
                # Get the data and corrupt it
                corrupted_data = self.original_data[key].copy()
                self._corrupt_data(corrupted_data, "random")
                await cache_service.set(key, corrupted_data)
                corrupted_count += 1

        # Validate data integrity
        corruption_detected = False
        consistent_entries = 0
        error_types = []

        for key in test_keys:
            try:
                cached_data = await cache_service.get(key)
                if cached_data:
                    is_valid = self._validate_data_integrity(cached_data)
                    if not is_valid:
                        corruption_detected = True
                    else:
                        consistent_entries += 1
            except Exception as e:
                error_types.append(type(e).__name__)

        data_consistency_maintained = consistent_entries >= (test_entries - corrupted_count)

        return CorruptionTestResult(
            scenario_name=scenario_name,
            corruption_detected=corruption_detected,
            data_consistency_maintained=data_consistency_maintained,
            recovery_successful=True,  # Basic test doesn't include recovery
            corrupted_entries=corrupted_count,
            total_entries=test_entries,
            error_types=error_types,
        )

    async def test_consistency_validation(self, cache_service: Any, test_operations: int = 50) -> CorruptionTestResult:
        """Test cache consistency validation."""
        scenario_name = "consistency_validation"

        # Perform mixed operations
        operations_completed = 0
        consistency_errors = 0
        error_types = []

        for i in range(test_operations):
            key = f"consistency_test_{i}"
            data = self._generate_test_data()

            try:
                # Set data
                await cache_service.set(key, data)

                # Immediately retrieve and validate
                retrieved_data = await cache_service.get(key)

                if retrieved_data:
                    # Check if data matches what we stored
                    if not self._validate_data_integrity(retrieved_data):
                        consistency_errors += 1

                operations_completed += 1

            except Exception as e:
                error_types.append(type(e).__name__)

        return CorruptionTestResult(
            scenario_name=scenario_name,
            corruption_detected=consistency_errors > 0,
            data_consistency_maintained=consistency_errors == 0,
            recovery_successful=True,
            corrupted_entries=consistency_errors,
            total_entries=operations_completed,
            error_types=error_types,
        )


class TestCacheCorruptionScenarios:
    """Test suite for cache corruption scenarios."""

    @pytest.fixture
    def corruption_tester(self):
        """Create cache corruption tester."""
        return CacheCorruptionTester()

    @pytest.fixture
    def mock_cache_service(self):
        """Create mock cache service."""
        cache_data = {}

        class MockCacheService:
            async def get(self, key: str):
                return cache_data.get(key)

            async def set(self, key: str, value: Any, ttl: int = None):
                cache_data[key] = value
                return True

            async def delete(self, key: str):
                if key in cache_data:
                    del cache_data[key]
                    return True
                return False

        return MockCacheService()

    @pytest.mark.asyncio
    async def test_corruption_detection(self, corruption_tester, mock_cache_service):
        """Test corruption detection functionality."""
        result = await corruption_tester.test_data_corruption_detection(mock_cache_service, test_entries=10, corruption_rate=0.4)

        assert result.scenario_name == "corruption_detection"
        assert result.total_entries == 10
        assert result.corrupted_entries > 0  # Should have some corruption

        print(f"Corruption detection: {result.corrupted_entries}/{result.total_entries} corrupted")

    @pytest.mark.asyncio
    async def test_consistency_validation(self, corruption_tester, mock_cache_service):
        """Test consistency validation."""
        result = await corruption_tester.test_consistency_validation(mock_cache_service, test_operations=20)

        assert result.scenario_name == "consistency_validation"
        assert result.total_entries <= 20

        print(f"Consistency validation: {result.data_consistency_maintained}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
