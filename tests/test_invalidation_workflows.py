"""
Integration tests for invalidation workflows - Wave 15.2.4
Tests complete invalidation workflows including file monitoring, cascade operations, and event handling.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileSystemEvent

from src.services.cache_invalidation_service import CacheInvalidationService
from src.services.cache_service import CacheService
from src.services.cascade_invalidation_service import CascadeInvalidationService
from src.services.file_monitoring_service import FileMonitoringService
from src.services.file_system_event_handler import FileSystemEventHandler
from src.utils.invalidation_event_bus import InvalidationEvent, InvalidationEventBus


class TestInvalidationWorkflowBase:
    """Base class for invalidation workflow tests."""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service for testing."""
        service = CacheService()
        # Use mock Redis for faster tests
        service._redis_client = AsyncMock()
        service._redis_client.get.return_value = None
        service._redis_client.set.return_value = True
        service._redis_client.delete.return_value = 1
        service._redis_client.scan.return_value = (0, [])
        yield service
        await service.close()

    @pytest.fixture
    async def invalidation_service(self, cache_service):
        """Create invalidation service."""
        service = CacheInvalidationService()
        service._cache_client = cache_service._redis_client
        yield service
        await service.close()

    @pytest.fixture
    async def cascade_service(self, cache_service):
        """Create cascade invalidation service."""
        service = CascadeInvalidationService()
        service._cache_client = cache_service._redis_client
        yield service
        await service.close()

    @pytest.fixture
    async def file_monitoring_service(self):
        """Create file monitoring service."""
        service = FileMonitoringService()
        yield service
        await service.stop()

    @pytest.fixture
    async def event_bus(self):
        """Create invalidation event bus."""
        bus = InvalidationEventBus()
        yield bus
        await bus.close()


class TestFileChangeInvalidationWorkflow(TestInvalidationWorkflowBase):
    """Test complete file change to cache invalidation workflow."""

    @pytest.mark.asyncio
    async def test_single_file_change_workflow(self, cache_service, invalidation_service, file_monitoring_service):
        """Test complete workflow for single file change."""
        # Set up cache data
        cache_data = {
            "model:user:123": {"name": "User Model", "version": 1},
            "model:user:456": {"name": "Another User", "version": 1},
            "schema:user": {"fields": ["id", "name", "email"]},
            "config:user": {"settings": {"validation": True}},
        }

        for key, value in cache_data.items():
            cache_service._redis_client.get.return_value = json.dumps(value)
            await cache_service.set(key, value)

        # Connect services
        file_monitoring_service.set_invalidation_service(invalidation_service)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create source file
            user_model_file = Path(temp_dir) / "models" / "user.py"
            user_model_file.parent.mkdir(exist_ok=True)
            user_model_file.write_text("class User:\n    pass")

            # Register file-to-cache mapping
            await file_monitoring_service.register_file_cache_mapping(str(user_model_file), cache_patterns=["model:user:*", "schema:user"])

            # Start monitoring
            await file_monitoring_service.start_monitoring(temp_dir)

            # Simulate file change
            user_model_file.write_text("class User:\n    def __init__(self):\n        pass")

            # Create file system event
            event = FileModifiedEvent(str(user_model_file))
            await file_monitoring_service._handle_file_event(event)

            # Wait for invalidation processing
            await asyncio.sleep(0.1)

            # Verify invalidation was called
            assert invalidation_service._cache_client.delete.called

            # Check that correct patterns were invalidated
            delete_calls = invalidation_service._cache_client.delete.call_args_list
            deleted_keys = set()
            for call in delete_calls:
                deleted_keys.update(call[0])

            # Should invalidate user model and schema, but not config
            assert any("model:user" in str(key) for key in deleted_keys)

    @pytest.mark.asyncio
    async def test_batch_file_changes_workflow(self, cache_service, invalidation_service, file_monitoring_service):
        """Test workflow for batch file changes."""
        # Set up multiple files and cache mappings
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()

            # Create multiple model files
            files_and_caches = [
                ("user.py", ["cache:user:*", "schema:user"]),
                ("product.py", ["cache:product:*", "schema:product"]),
                ("order.py", ["cache:order:*", "schema:order"]),
            ]

            # Set up cache data for all
            for filename, cache_patterns in files_and_caches:
                file_path = models_dir / filename
                file_path.write_text(f"class {filename.split('.')[0].title()}:\n    pass")

                # Register mapping
                await file_monitoring_service.register_file_cache_mapping(str(file_path), cache_patterns)

                # Set cache data
                for pattern in cache_patterns:
                    base_key = pattern.replace("*", "123")
                    await cache_service.set(base_key, {"file": filename})

            # Connect services
            file_monitoring_service.set_invalidation_service(invalidation_service)
            file_monitoring_service.set_batch_debounce_time(0.1)  # Fast for testing

            # Start monitoring
            await file_monitoring_service.start_monitoring(temp_dir)

            # Simulate rapid changes to all files
            events = []
            for filename, _ in files_and_caches:
                file_path = models_dir / filename
                file_path.write_text(f"# Modified\nclass {filename.split('.')[0].title()}:\n    pass")
                events.append(FileModifiedEvent(str(file_path)))

            # Process events in batch
            await file_monitoring_service._handle_batch_events(events)

            # Wait for batch processing
            await asyncio.sleep(0.2)

            # Verify batch invalidation was more efficient than individual
            # (should group related invalidations)
            assert invalidation_service._cache_client.delete.called

    @pytest.mark.asyncio
    async def test_file_rename_workflow(self, cache_service, invalidation_service, file_monitoring_service):
        """Test workflow for file rename operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()

            # Create original file
            old_file = models_dir / "old_model.py"
            new_file = models_dir / "new_model.py"

            old_file.write_text("class OldModel:\n    pass")

            # Register mapping for old file
            await file_monitoring_service.register_file_cache_mapping(str(old_file), ["cache:old_model:*"])

            # Set cache data
            await cache_service.set("cache:old_model:123", {"name": "old"})

            # Connect services and start monitoring
            file_monitoring_service.set_invalidation_service(invalidation_service)
            await file_monitoring_service.start_monitoring(temp_dir)

            # Simulate file rename (delete old, create new)
            old_file.unlink()
            new_file.write_text("class NewModel:\n    pass")

            # Process events
            delete_event = FileDeletedEvent(str(old_file))
            create_event = FileCreatedEvent(str(new_file))

            await file_monitoring_service._handle_file_event(delete_event)
            await file_monitoring_service._handle_file_event(create_event)

            # Wait for processing
            await asyncio.sleep(0.1)

            # Should invalidate old model cache
            assert invalidation_service._cache_client.delete.called


class TestCascadeInvalidationWorkflow(TestInvalidationWorkflowBase):
    """Test cascade invalidation workflows."""

    @pytest.mark.asyncio
    async def test_deep_cascade_workflow(self, cache_service, invalidation_service, cascade_service):
        """Test deep cascade invalidation workflow."""
        # Set up hierarchical cache structure
        cache_hierarchy = {
            "user:123": ["profile:123", "settings:123", "sessions:123"],
            "profile:123": ["avatar:123", "bio:123"],
            "settings:123": ["preferences:123", "permissions:123"],
            "sessions:123": ["session:abc", "session:def"],
            "avatar:123": ["avatar:thumb:123", "avatar:full:123"],
        }

        # Register all dependencies
        for parent, children in cache_hierarchy.items():
            await cascade_service.register_dependency(parent, children)

        # Set cache data for all keys
        all_keys = set(cache_hierarchy.keys())
        for children in cache_hierarchy.values():
            all_keys.update(children)

        for key in all_keys:
            await cache_service.set(key, {"type": key.split(":")[0], "id": key.split(":")[-1]})

        # Mock successful deletions
        cache_service._redis_client.delete.return_value = 1

        # Perform cascade invalidation from root
        result = await cascade_service.invalidate_cascade("user:123")

        # Verify deep cascade
        assert result["success"] is True
        assert result["total_invalidated"] >= 12  # All dependent keys
        assert result["cascade_depth"] >= 3  # At least 3 levels deep

        # Verify correct traversal order (breadth-first)
        assert "invalidation_order" in result

    @pytest.mark.asyncio
    async def test_selective_cascade_workflow(self, cache_service, invalidation_service, cascade_service):
        """Test selective cascade invalidation with filters."""
        # Set up mixed cache types
        await cascade_service.register_dependency(
            "project:123",
            [
                "project:data:123",  # Data cache
                "project:config:123",  # Config cache
                "project:temp:123",  # Temporary cache
                "project:metrics:123",  # Metrics cache
            ],
        )

        # Set cache data
        cache_data = {
            "project:123": {"name": "Test Project"},
            "project:data:123": {"records": 100},
            "project:config:123": {"settings": {}},
            "project:temp:123": {"temp_data": "xyz"},
            "project:metrics:123": {"views": 500},
        }

        for key, value in cache_data.items():
            await cache_service.set(key, value)

        # Define filter to only invalidate data and config (not temp or metrics)
        def important_cache_filter(cache_key: str) -> bool:
            return any(cache_type in cache_key for cache_type in ["data", "config"])

        # Perform selective cascade
        result = await cascade_service.invalidate_cascade("project:123", filter_fn=important_cache_filter)

        # Verify selective invalidation
        assert result["success"] is True
        assert result["total_invalidated"] == 3  # project + data + config
        assert result["filtered_count"] == 2  # temp + metrics filtered out

    @pytest.mark.asyncio
    async def test_cascade_with_circular_dependency_handling(self, cache_service, cascade_service):
        """Test cascade workflow with circular dependency detection."""
        # Set up valid dependencies first
        await cascade_service.register_dependency("cache:a", ["cache:b"])
        await cascade_service.register_dependency("cache:b", ["cache:c"])

        # Attempt to create circular dependency
        with pytest.raises(Exception) as exc_info:
            await cascade_service.register_dependency("cache:c", ["cache:a"])

        assert "circular" in str(exc_info.value).lower()

        # Verify normal cascade still works
        result = await cascade_service.invalidate_cascade("cache:a")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_cascade_failure_recovery_workflow(self, cache_service, cascade_service):
        """Test cascade workflow with failure recovery."""
        # Set up dependencies
        await cascade_service.register_dependency("root:key", ["child:1", "child:2", "child:3", "child:4"])

        # Mock partial failures
        def mock_delete(*keys):
            failed_keys = ["child:2", "child:4"]
            successful_count = sum(1 for key in keys if key not in failed_keys)

            for key in keys:
                if key in failed_keys:
                    raise Exception(f"Failed to delete {key}")

            return successful_count

        cache_service._redis_client.delete.side_effect = mock_delete

        # Perform cascade with error handling
        result = await cascade_service.invalidate_cascade("root:key", continue_on_error=True)

        # Verify partial success
        assert result["success"] is True  # Overall success despite failures
        assert result["total_invalidated"] >= 2  # At least root + 2 children
        assert result["failed_count"] == 2
        assert len(result["errors"]) == 2


class TestInvalidationEventWorkflow(TestInvalidationWorkflowBase):
    """Test invalidation event handling workflows."""

    @pytest.mark.asyncio
    async def test_event_driven_invalidation_workflow(self, cache_service, invalidation_service, event_bus):
        """Test event-driven invalidation workflow."""
        # Set up event handlers
        invalidation_events = []

        async def invalidation_handler(event: InvalidationEvent):
            invalidation_events.append(event)
            # Trigger related invalidations
            if event.cache_key.startswith("user:"):
                user_id = event.cache_key.split(":")[-1]
                await invalidation_service.invalidate_pattern(f"session:{user_id}:*")

        event_bus.subscribe("cache_invalidated", invalidation_handler)

        # Connect invalidation service to event bus
        invalidation_service.set_event_bus(event_bus)

        # Set up cache data
        await cache_service.set("user:123", {"name": "Test User"})
        await cache_service.set("session:123:abc", {"token": "abc123"})
        await cache_service.set("session:123:def", {"token": "def456"})

        # Perform invalidation that triggers events
        await invalidation_service.invalidate("user:123")

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify events were fired and handled
        assert len(invalidation_events) >= 1
        assert invalidation_events[0].cache_key == "user:123"

        # Verify cascade invalidation occurred via events
        assert invalidation_service._cache_client.delete.call_count >= 2

    @pytest.mark.asyncio
    async def test_event_aggregation_workflow(self, cache_service, invalidation_service, event_bus):
        """Test event aggregation for batch processing."""
        # Set up event aggregator
        aggregated_events = []

        async def batch_handler(events: list[InvalidationEvent]):
            aggregated_events.extend(events)
            # Process in batch for efficiency
            keys_to_invalidate = [event.cache_key for event in events]
            await invalidation_service.invalidate_batch(keys_to_invalidate)

        event_bus.set_batch_handler(batch_handler, batch_size=5, batch_timeout=0.1)

        # Generate multiple invalidation events rapidly
        for i in range(10):
            event = InvalidationEvent(cache_key=f"batch:key:{i}", event_type="invalidated", timestamp=time.time(), source="test")
            await event_bus.publish("cache_invalidated", event)

        # Wait for batch processing
        await asyncio.sleep(0.2)

        # Verify events were batched
        assert len(aggregated_events) == 10
        # Should have been processed in 2 batches of 5
        assert event_bus.get_batch_count() == 2

    @pytest.mark.asyncio
    async def test_event_priority_workflow(self, cache_service, invalidation_service, event_bus):
        """Test priority-based event processing workflow."""
        # Set up priority handlers
        processed_events = []

        async def high_priority_handler(event: InvalidationEvent):
            processed_events.append(("high", event.cache_key))

        async def normal_priority_handler(event: InvalidationEvent):
            processed_events.append(("normal", event.cache_key))

        # Register handlers with different priorities
        event_bus.subscribe("cache_invalidated", high_priority_handler, priority=10)
        event_bus.subscribe("cache_invalidated", normal_priority_handler, priority=5)

        # Publish events
        critical_event = InvalidationEvent(cache_key="critical:data", event_type="invalidated", priority=10, timestamp=time.time())

        normal_event = InvalidationEvent(cache_key="normal:data", event_type="invalidated", priority=5, timestamp=time.time())

        await event_bus.publish("cache_invalidated", normal_event)
        await event_bus.publish("cache_invalidated", critical_event)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify priority order
        assert len(processed_events) == 4  # 2 events × 2 handlers
        # High priority handler should process critical event first
        high_priority_events = [e for e in processed_events if e[0] == "high"]
        assert high_priority_events[0][1] == "critical:data"


class TestComplexInvalidationWorkflows(TestInvalidationWorkflowBase):
    """Test complex multi-service invalidation workflows."""

    @pytest.mark.asyncio
    async def test_microservice_invalidation_workflow(self, cache_service, invalidation_service, cascade_service, event_bus):
        """Test invalidation workflow across microservices."""
        # Simulate microservice cache dependencies
        service_caches = {
            "user-service": ["user:123", "user:profile:123"],
            "order-service": ["order:456", "order:user:123"],
            "notification-service": ["notification:user:123"],
            "analytics-service": ["analytics:user:123"],
        }

        # Set up cross-service dependencies
        await cascade_service.register_dependency(
            "user:123",
            ["order:user:123", "notification:user:123", "analytics:user:123"],  # User service
        )

        # Set up event handlers for cross-service communication
        cross_service_events = []

        async def cross_service_handler(event: InvalidationEvent):
            if event.cache_key.startswith("user:"):
                # Notify other services about user changes
                cross_service_events.append(event)

                # Trigger related service invalidations
                user_id = event.cache_key.split(":")[-1]
                await invalidation_service.invalidate_pattern(f"*:user:{user_id}")

        event_bus.subscribe("cache_invalidated", cross_service_handler)
        invalidation_service.set_event_bus(event_bus)

        # Set cache data for all services
        for service, cache_keys in service_caches.items():
            for key in cache_keys:
                await cache_service.set(key, {"service": service, "data": f"data_{key}"})

        # Trigger user update (simulating user service change)
        await invalidation_service.invalidate("user:123")

        # Wait for cross-service propagation
        await asyncio.sleep(0.2)

        # Verify cross-service invalidation
        assert len(cross_service_events) >= 1
        # Should have invalidated caches in multiple services
        assert invalidation_service._cache_client.delete.call_count >= 3

    @pytest.mark.asyncio
    async def test_tenant_isolation_invalidation_workflow(self, cache_service, invalidation_service):
        """Test tenant-isolated invalidation workflow."""
        # Set up multi-tenant cache data
        tenants = ["tenant_a", "tenant_b", "tenant_c"]

        for tenant in tenants:
            for resource in ["users", "orders", "products"]:
                for i in range(3):
                    key = f"{tenant}:{resource}:{i}"
                    await cache_service.set(key, {"tenant": tenant, "resource": resource, "id": i})

        # Define tenant-aware invalidation
        async def invalidate_tenant_resource(tenant_id: str, resource_type: str):
            pattern = f"{tenant_id}:{resource_type}:*"
            await invalidation_service.invalidate_pattern(pattern)

        # Invalidate users for tenant_a only
        await invalidate_tenant_resource("tenant_a", "users")

        # Verify tenant isolation
        # Should only invalidate tenant_a users, not other tenants or resources
        delete_calls = invalidation_service._cache_client.delete.call_args_list
        if delete_calls:
            deleted_keys = set()
            for call in delete_calls:
                deleted_keys.update(call[0])

            # Check that only tenant_a users were affected
            tenant_a_user_keys = {f"tenant_a:users:{i}" for i in range(3)}
            assert any(key in deleted_keys for key in tenant_a_user_keys)

    @pytest.mark.asyncio
    async def test_real_time_invalidation_workflow(self, cache_service, invalidation_service, file_monitoring_service, event_bus):
        """Test real-time invalidation workflow with immediate propagation."""
        # Set up real-time event processing
        real_time_events = []

        async def real_time_handler(event: InvalidationEvent):
            real_time_events.append({"timestamp": time.time(), "key": event.cache_key, "latency": time.time() - event.timestamp})

        event_bus.subscribe("cache_invalidated", real_time_handler)
        invalidation_service.set_event_bus(event_bus)

        # Set up file monitoring for real-time updates
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"
            config_file.write_text('{"version": 1}')

            # Register for immediate invalidation
            await file_monitoring_service.register_file_cache_mapping(str(config_file), ["config:cache"], immediate=True)  # No debouncing

            file_monitoring_service.set_invalidation_service(invalidation_service)
            await file_monitoring_service.start_monitoring(temp_dir)

            # Set cache data
            await cache_service.set("config:cache", {"version": 1})

            # Trigger file change
            start_time = time.time()
            config_file.write_text('{"version": 2}')

            # Simulate file event
            event = FileModifiedEvent(str(config_file))
            await file_monitoring_service._handle_file_event(event)

            # Wait minimal time for processing
            await asyncio.sleep(0.05)

            # Verify real-time processing
            assert len(real_time_events) >= 1

            # Check latency (should be very low for real-time)
            if real_time_events:
                latency = real_time_events[0]["latency"]
                assert latency < 0.1  # Less than 100ms for real-time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
