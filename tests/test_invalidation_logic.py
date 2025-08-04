"""
Unit tests for cache invalidation logic - Wave 15.1.4
Tests invalidation services, cascade operations, and file monitoring integration.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.cache_invalidation_service import (
    CacheInvalidationService,
    InvalidationEvent,
    InvalidationMetrics,
    InvalidationStrategy,
    get_cache_invalidation_service,
)
from src.services.cascade_invalidation_service import (
    CascadeInvalidationService,
    CircularDependencyError,
    DependencyGraph,
    InvalidationChain,
)
from src.services.file_monitoring_service import FileChangeEvent, FileChangeType, FileMonitoringService
from src.services.file_system_event_handler import EventPriority, EventType, FileSystemEventHandler


class TestCacheInvalidationService:
    """Test CacheInvalidationService core functionality."""

    @pytest.fixture
    async def invalidation_service(self):
        """Create invalidation service instance."""
        service = CacheInvalidationService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_basic_invalidation(self, invalidation_service):
        """Test basic cache invalidation."""
        # Mock cache client
        mock_cache = AsyncMock()
        mock_cache.delete.return_value = True
        invalidation_service._cache_client = mock_cache

        # Invalidate single key
        result = await invalidation_service.invalidate("cache:key:123")

        assert result["success"] is True
        assert result["invalidated_count"] == 1
        mock_cache.delete.assert_called_once_with("cache:key:123")

    @pytest.mark.asyncio
    async def test_pattern_invalidation(self, invalidation_service):
        """Test pattern-based invalidation."""
        mock_cache = AsyncMock()
        mock_cache.scan.return_value = (0, [b"cache:user:123", b"cache:user:456", b"cache:user:789"])
        mock_cache.delete.return_value = 3
        invalidation_service._cache_client = mock_cache

        # Invalidate by pattern
        result = await invalidation_service.invalidate_pattern("cache:user:*")

        assert result["success"] is True
        assert result["invalidated_count"] == 3
        assert result["pattern"] == "cache:user:*"

    @pytest.mark.asyncio
    async def test_batch_invalidation(self, invalidation_service):
        """Test batch invalidation of multiple keys."""
        mock_cache = AsyncMock()
        mock_cache.delete.return_value = 5
        invalidation_service._cache_client = mock_cache

        keys = [f"cache:key:{i}" for i in range(5)]
        result = await invalidation_service.invalidate_batch(keys)

        assert result["success"] is True
        assert result["invalidated_count"] == 5
        assert result["batch_size"] == 5

    @pytest.mark.asyncio
    async def test_conditional_invalidation(self, invalidation_service):
        """Test conditional invalidation based on criteria."""
        mock_cache = AsyncMock()

        # Mock cache entries with metadata
        cache_entries = {
            "cache:old:1": {"data": "test", "_metadata": {"created": time.time() - 7200}},
            "cache:new:1": {"data": "test", "_metadata": {"created": time.time() - 300}},
            "cache:old:2": {"data": "test", "_metadata": {"created": time.time() - 3700}},
        }

        mock_cache.scan.return_value = (0, list(cache_entries.keys()))
        mock_cache.get.side_effect = lambda k: json.dumps(cache_entries.get(k))
        invalidation_service._cache_client = mock_cache

        # Invalidate entries older than 1 hour
        def condition(metadata):
            return time.time() - metadata.get("created", 0) > 3600

        result = await invalidation_service.invalidate_conditional(pattern="cache:*", condition=condition)

        assert result["evaluated_count"] == 3
        assert result["invalidated_count"] == 2

    @pytest.mark.asyncio
    async def test_invalidation_with_strategy(self, invalidation_service):
        """Test invalidation with different strategies."""
        mock_cache = AsyncMock()
        invalidation_service._cache_client = mock_cache

        # Test immediate invalidation
        await invalidation_service.invalidate("cache:key:123", strategy=InvalidationStrategy.IMMEDIATE)
        mock_cache.delete.assert_called_once()

        # Test lazy invalidation (mark for deletion)
        mock_cache.reset_mock()
        await invalidation_service.invalidate("cache:key:456", strategy=InvalidationStrategy.LAZY)
        mock_cache.set.assert_called_once()  # Should mark as invalid

        # Test scheduled invalidation
        mock_cache.reset_mock()
        await invalidation_service.invalidate("cache:key:789", strategy=InvalidationStrategy.SCHEDULED, delay_seconds=60)
        # Should be queued for later execution
        assert len(invalidation_service._scheduled_invalidations) > 0

    @pytest.mark.asyncio
    async def test_invalidation_metrics(self, invalidation_service):
        """Test invalidation metrics collection."""
        mock_cache = AsyncMock()
        mock_cache.delete.return_value = 1
        invalidation_service._cache_client = mock_cache

        # Perform multiple invalidations
        for i in range(10):
            await invalidation_service.invalidate(f"cache:key:{i}")

        metrics = invalidation_service.get_metrics()

        assert metrics["total_invalidations"] == 10
        assert metrics["success_count"] == 10
        assert metrics["failure_count"] == 0
        assert metrics["average_duration"] > 0

    @pytest.mark.asyncio
    async def test_invalidation_event_handling(self, invalidation_service):
        """Test invalidation event generation and handling."""
        events_received = []

        # Register event handler
        def event_handler(event: InvalidationEvent):
            events_received.append(event)

        invalidation_service.on_invalidation(event_handler)

        # Perform invalidation
        await invalidation_service.invalidate("cache:key:123")

        assert len(events_received) == 1
        assert events_received[0].key == "cache:key:123"
        assert events_received[0].success is True


class TestCascadeInvalidationService:
    """Test cascade invalidation functionality."""

    @pytest.fixture
    async def cascade_service(self):
        """Create cascade invalidation service instance."""
        service = CascadeInvalidationService()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_dependency_registration(self, cascade_service):
        """Test dependency registration and retrieval."""
        # Register dependencies
        await cascade_service.register_dependency(parent="cache:user:123", dependents=["cache:profile:123", "cache:settings:123"])

        deps = await cascade_service.get_dependents("cache:user:123")
        assert len(deps) == 2
        assert "cache:profile:123" in deps
        assert "cache:settings:123" in deps

    @pytest.mark.asyncio
    async def test_cascade_invalidation(self, cascade_service):
        """Test cascading invalidation through dependencies."""
        mock_cache = AsyncMock()
        mock_cache.delete.side_effect = lambda keys: len(keys) if isinstance(keys, list) else 1
        cascade_service._cache_client = mock_cache

        # Set up dependency chain
        await cascade_service.register_dependency(parent="cache:user:123", dependents=["cache:profile:123", "cache:preferences:123"])
        await cascade_service.register_dependency(parent="cache:profile:123", dependents=["cache:avatar:123", "cache:bio:123"])

        # Invalidate parent
        result = await cascade_service.invalidate_cascade("cache:user:123")

        assert result["total_invalidated"] == 5  # Parent + 4 dependents
        assert result["cascade_depth"] == 2

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, cascade_service):
        """Test detection of circular dependencies."""
        # Create circular dependency
        await cascade_service.register_dependency("cache:a", ["cache:b"])
        await cascade_service.register_dependency("cache:b", ["cache:c"])

        # This should raise an error
        with pytest.raises(CircularDependencyError):
            await cascade_service.register_dependency("cache:c", ["cache:a"])

    @pytest.mark.asyncio
    async def test_selective_cascade(self, cascade_service):
        """Test selective cascade with filters."""
        mock_cache = AsyncMock()
        cascade_service._cache_client = mock_cache

        # Set up dependencies
        await cascade_service.register_dependency(
            parent="cache:product:123", dependents=["cache:price:123", "cache:stock:123", "cache:reviews:123", "cache:images:123"]
        )

        # Cascade with filter - only price and stock
        result = await cascade_service.invalidate_cascade("cache:product:123", filter_fn=lambda key: "price" in key or "stock" in key)

        assert result["total_invalidated"] == 3  # Parent + 2 filtered dependents

    @pytest.mark.asyncio
    async def test_cascade_with_failure_handling(self, cascade_service):
        """Test cascade invalidation with failure handling."""
        mock_cache = AsyncMock()

        # Simulate some failures
        def delete_side_effect(key):
            if "fail" in key:
                raise Exception("Delete failed")
            return 1

        mock_cache.delete.side_effect = delete_side_effect
        cascade_service._cache_client = mock_cache

        # Set up dependencies with some that will fail
        await cascade_service.register_dependency(parent="cache:parent", dependents=["cache:child1", "cache:fail:child2", "cache:child3"])

        result = await cascade_service.invalidate_cascade("cache:parent", continue_on_error=True)

        assert result["total_invalidated"] == 3  # Parent + 2 successful
        assert result["failed_count"] == 1


class TestFileMonitoringIntegration:
    """Test file monitoring integration with cache invalidation."""

    @pytest.fixture
    async def file_monitor(self):
        """Create file monitoring service instance."""
        service = FileMonitoringService()
        yield service
        await service.stop()

    @pytest.mark.asyncio
    async def test_file_change_invalidation(self, file_monitor):
        """Test cache invalidation on file changes."""
        invalidation_service = AsyncMock()
        file_monitor.set_invalidation_service(invalidation_service)

        # Register file-to-cache mapping
        await file_monitor.register_file_cache_mapping(file_path="/src/models/user.py", cache_patterns=["cache:user:*", "cache:profile:*"])

        # Simulate file change
        event = FileChangeEvent(path="/src/models/user.py", change_type=FileChangeType.MODIFIED, timestamp=time.time())

        await file_monitor.handle_file_change(event)

        # Verify invalidation was triggered
        assert invalidation_service.invalidate_pattern.call_count == 2
        invalidation_service.invalidate_pattern.assert_any_call("cache:user:*")
        invalidation_service.invalidate_pattern.assert_any_call("cache:profile:*")

    @pytest.mark.asyncio
    async def test_batch_file_changes(self, file_monitor):
        """Test handling batch file changes efficiently."""
        invalidation_service = AsyncMock()
        file_monitor.set_invalidation_service(invalidation_service)

        # Register mappings
        mappings = {"/src/file1.py": ["cache:type1:*"], "/src/file2.py": ["cache:type2:*"], "/src/file3.py": ["cache:type3:*"]}

        for file_path, patterns in mappings.items():
            await file_monitor.register_file_cache_mapping(file_path, patterns)

        # Simulate rapid file changes
        events = [FileChangeEvent(path, FileChangeType.MODIFIED, time.time()) for path in mappings.keys()]

        # Process in batch
        await file_monitor.handle_batch_changes(events)

        # Should batch invalidations efficiently
        assert invalidation_service.invalidate_pattern.call_count <= len(events)

    @pytest.mark.asyncio
    async def test_file_monitoring_with_debouncing(self, file_monitor):
        """Test file change debouncing to prevent rapid invalidations."""
        invalidation_service = AsyncMock()
        file_monitor.set_invalidation_service(invalidation_service)
        file_monitor.set_debounce_interval(0.5)  # 500ms debounce

        # Register mapping
        await file_monitor.register_file_cache_mapping("/src/volatile.py", ["cache:volatile:*"])

        # Simulate rapid changes to same file
        for i in range(5):
            event = FileChangeEvent(path="/src/volatile.py", change_type=FileChangeType.MODIFIED, timestamp=time.time())
            await file_monitor.handle_file_change(event)
            await asyncio.sleep(0.1)  # 100ms between changes

        # Wait for debounce
        await asyncio.sleep(0.6)

        # Should only invalidate once due to debouncing
        assert invalidation_service.invalidate_pattern.call_count == 1


class TestInvalidationStrategies:
    """Test different invalidation strategies and patterns."""

    @pytest.mark.asyncio
    async def test_ttl_based_invalidation(self):
        """Test TTL-based automatic invalidation."""
        service = CacheInvalidationService()
        mock_cache = AsyncMock()
        service._cache_client = mock_cache

        # Set up TTL monitoring
        await service.enable_ttl_monitoring(scan_interval=1, batch_size=100)  # 1 second

        # Mock expired entries
        mock_cache.scan.return_value = (0, [b"cache:expired:1", b"cache:expired:2", b"cache:valid:1"])
        mock_cache.ttl.side_effect = [-1, -1, 3600]  # First two expired

        # Run TTL check
        expired = await service.check_and_invalidate_expired()

        assert expired["expired_count"] == 2
        assert mock_cache.delete.call_count == 1  # Batch delete

    @pytest.mark.asyncio
    async def test_memory_pressure_invalidation(self):
        """Test invalidation under memory pressure."""
        service = CacheInvalidationService()
        mock_cache = AsyncMock()
        service._cache_client = mock_cache

        # Mock memory info
        mock_cache.info.return_value = {"used_memory": 900_000_000, "maxmemory": 1_000_000_000}  # 900MB  # 1GB limit

        # Set up LRU data for eviction
        mock_cache.scan.return_value = (0, [b"cache:lru:1", b"cache:lru:2"])

        # Trigger memory pressure invalidation
        result = await service.handle_memory_pressure(threshold=0.8)

        assert result["memory_freed"] > 0
        assert result["keys_evicted"] > 0

    @pytest.mark.asyncio
    async def test_priority_based_invalidation(self):
        """Test priority-based selective invalidation."""
        service = CacheInvalidationService()

        # Define priority rules
        priority_rules = {
            "cache:critical:*": 10,  # Highest priority (keep)
            "cache:important:*": 5,
            "cache:normal:*": 3,
            "cache:low:*": 1,  # Lowest priority (evict first)
        }

        service.set_priority_rules(priority_rules)

        # Test eviction order
        candidates = ["cache:critical:1", "cache:low:1", "cache:normal:1", "cache:low:2"]

        eviction_order = service.get_eviction_order(candidates)

        assert eviction_order[0] in ["cache:low:1", "cache:low:2"]
        assert eviction_order[-1] == "cache:critical:1"


class TestInvalidationMetrics:
    """Test invalidation metrics and monitoring."""

    def test_invalidation_metrics_collection(self):
        """Test collection of invalidation metrics."""
        metrics = InvalidationMetrics()

        # Record various operations
        metrics.record_invalidation("cache:key:1", success=True, duration=0.001)
        metrics.record_invalidation("cache:key:2", success=True, duration=0.002)
        metrics.record_invalidation("cache:key:3", success=False, duration=0.005)

        stats = metrics.get_statistics()

        assert stats["total_invalidations"] == 3
        assert stats["success_rate"] == 2 / 3
        assert stats["average_duration"] == pytest.approx(0.00267, rel=0.01)

    def test_invalidation_patterns_analysis(self):
        """Test analysis of invalidation patterns."""
        metrics = InvalidationMetrics()

        # Simulate invalidation pattern
        for i in range(100):
            key = f"cache:user:{i % 10}"
            metrics.record_invalidation(key, success=True, duration=0.001)

        patterns = metrics.analyze_patterns()

        # Should identify hot keys
        assert len(patterns["hot_keys"]) > 0
        assert patterns["hot_keys"][0]["count"] == 10

        # Should identify key patterns
        assert "cache:user:*" in patterns["common_patterns"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
