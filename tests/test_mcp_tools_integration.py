"""
Integration tests for MCP tools - Wave 15.2.2
Tests MCP tool integration with cache services and real workflows.
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

from src.services.cache_invalidation_service import CacheInvalidationService
from src.services.cache_performance_service import CachePerformanceService
from src.services.cache_service import CacheService
from src.tools.cache.cache_management import clear_cache_tool, get_cache_stats_tool, set_cache_tool, warm_cache_tool
from src.tools.cache.cascade_invalidation_tools import cascade_invalidate_tool, get_dependency_graph_tool, register_dependency_tool
from src.tools.core.cache_alert_management import acknowledge_alert_tool, get_cache_alerts_tool, set_alert_threshold_tool


class TestMCPToolsBase:
    """Base class for MCP tools integration tests."""

    @pytest.fixture
    async def mock_cache_service(self):
        """Create mock cache service."""
        service = AsyncMock(spec=CacheService)
        service.get.return_value = None
        service.set.return_value = True
        service.delete.return_value = True
        service.clear.return_value = 10
        service.stats.return_value = {"keys": 100, "memory_usage": "50MB", "hit_rate": 0.85}
        return service

    @pytest.fixture
    async def mock_invalidation_service(self):
        """Create mock invalidation service."""
        service = AsyncMock(spec=CacheInvalidationService)
        service.invalidate.return_value = {"success": True, "invalidated_count": 1}
        service.invalidate_pattern.return_value = {"success": True, "invalidated_count": 5}
        return service


class TestCacheManagementTools(TestMCPToolsBase):
    """Test cache management MCP tools."""

    @pytest.mark.asyncio
    async def test_set_cache_tool(self, mock_cache_service):
        """Test set_cache_tool MCP integration."""
        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            # Test basic set operation
            result = await set_cache_tool({"key": "test:key:123", "value": {"name": "Test User", "id": 123}, "ttl": 3600})

            assert result["success"] is True
            assert result["key"] == "test:key:123"
            mock_cache_service.set.assert_called_once_with("test:key:123", {"name": "Test User", "id": 123}, ttl=3600)

    @pytest.mark.asyncio
    async def test_set_cache_tool_with_encryption(self, mock_cache_service):
        """Test set_cache_tool with encryption enabled."""
        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            result = await set_cache_tool(
                {"key": "secure:user:456", "value": {"ssn": "123-45-6789", "api_key": "secret"}, "ttl": 1800, "encrypted": True}
            )

            assert result["success"] is True
            assert result["encrypted"] is True
            # Verify encryption was applied
            mock_cache_service.set_encrypted.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_stats_tool(self, mock_cache_service):
        """Test get_cache_stats_tool MCP integration."""
        mock_cache_service.get_stats.return_value = {
            "total_keys": 1500,
            "memory_usage_mb": 128,
            "hit_rate": 0.92,
            "miss_rate": 0.08,
            "operations_per_second": 2500,
            "average_response_time_ms": 0.8,
            "connection_count": 50,
            "evicted_keys": 25,
        }

        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            result = await get_cache_stats_tool({})

            assert result["total_keys"] == 1500
            assert result["memory_usage_mb"] == 128
            assert result["hit_rate"] == 0.92
            assert result["performance"]["operations_per_second"] == 2500
            assert "health_status" in result

    @pytest.mark.asyncio
    async def test_clear_cache_tool_pattern(self, mock_cache_service):
        """Test clear_cache_tool with pattern matching."""
        mock_cache_service.clear.return_value = 45

        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            result = await clear_cache_tool({"pattern": "user:session:*", "confirm": True})

            assert result["success"] is True
            assert result["cleared_count"] == 45
            assert result["pattern"] == "user:session:*"
            mock_cache_service.clear.assert_called_once_with(pattern="user:session:*")

    @pytest.mark.asyncio
    async def test_warm_cache_tool(self, mock_cache_service):
        """Test warm_cache_tool MCP integration."""
        # Mock data loader
        mock_data_loader = AsyncMock()
        mock_data_loader.load_user_data.return_value = {"id": 123, "name": "Test User"}
        mock_data_loader.load_product_data.return_value = {"id": 456, "name": "Test Product"}

        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            with patch("src.tools.cache.cache_management.get_data_loader", return_value=mock_data_loader):
                result = await warm_cache_tool(
                    {
                        "entries": [
                            {"key": "user:123", "loader": "load_user_data", "params": {"user_id": 123}},
                            {"key": "product:456", "loader": "load_product_data", "params": {"product_id": 456}},
                        ],
                        "batch_size": 10,
                    }
                )

                assert result["success"] is True
                assert result["warmed_count"] == 2
                assert result["failed_count"] == 0
                assert mock_cache_service.set.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_tool_error_handling(self, mock_cache_service):
        """Test MCP tool error handling."""
        # Simulate cache service error
        mock_cache_service.set.side_effect = Exception("Redis connection failed")

        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            result = await set_cache_tool({"key": "test:key", "value": {"data": "test"}})

            assert result["success"] is False
            assert "error" in result
            assert "Redis connection failed" in result["error"]


class TestCascadeInvalidationTools(TestMCPToolsBase):
    """Test cascade invalidation MCP tools."""

    @pytest.mark.asyncio
    async def test_register_dependency_tool(self, mock_invalidation_service):
        """Test register_dependency_tool MCP integration."""
        mock_cascade_service = AsyncMock()
        mock_cascade_service.register_dependency.return_value = True

        with patch("src.tools.cache.cascade_invalidation_tools.get_cascade_service", return_value=mock_cascade_service):
            result = await register_dependency_tool(
                {"parent": "user:123", "dependents": ["profile:123", "settings:123", "preferences:123"]}
            )

            assert result["success"] is True
            assert result["parent"] == "user:123"
            assert len(result["dependents"]) == 3
            mock_cascade_service.register_dependency.assert_called_once_with("user:123", ["profile:123", "settings:123", "preferences:123"])

    @pytest.mark.asyncio
    async def test_cascade_invalidate_tool(self, mock_invalidation_service):
        """Test cascade_invalidate_tool MCP integration."""
        mock_cascade_service = AsyncMock()
        mock_cascade_service.invalidate_cascade.return_value = {
            "success": True,
            "total_invalidated": 8,
            "cascade_depth": 3,
            "invalidation_tree": {"user:123": ["profile:123", "settings:123"], "profile:123": ["avatar:123", "bio:123"]},
        }

        with patch("src.tools.cache.cascade_invalidation_tools.get_cascade_service", return_value=mock_cascade_service):
            result = await cascade_invalidate_tool({"key": "user:123", "max_depth": 5, "continue_on_error": True})

            assert result["success"] is True
            assert result["total_invalidated"] == 8
            assert result["cascade_depth"] == 3
            assert "invalidation_tree" in result

    @pytest.mark.asyncio
    async def test_get_dependency_graph_tool(self, mock_invalidation_service):
        """Test get_dependency_graph_tool MCP integration."""
        mock_cascade_service = AsyncMock()
        mock_cascade_service.get_dependency_graph.return_value = {
            "nodes": ["user:123", "profile:123", "settings:123"],
            "edges": [{"from": "user:123", "to": "profile:123"}, {"from": "user:123", "to": "settings:123"}],
            "depth_map": {"user:123": 0, "profile:123": 1, "settings:123": 1},
        }

        with patch("src.tools.cache.cascade_invalidation_tools.get_cascade_service", return_value=mock_cascade_service):
            result = await get_dependency_graph_tool({"root_key": "user:123", "max_depth": 3})

            assert len(result["nodes"]) == 3
            assert len(result["edges"]) == 2
            assert "depth_map" in result

    @pytest.mark.asyncio
    async def test_cascade_tool_circular_dependency_detection(self):
        """Test cascade tool circular dependency detection."""
        mock_cascade_service = AsyncMock()
        mock_cascade_service.register_dependency.side_effect = Exception("Circular dependency detected")

        with patch("src.tools.cache.cascade_invalidation_tools.get_cascade_service", return_value=mock_cascade_service):
            result = await register_dependency_tool({"parent": "user:123", "dependents": ["profile:123"]})

            assert result["success"] is False
            assert "circular dependency" in result["error"].lower()


class TestCacheAlertTools(TestMCPToolsBase):
    """Test cache alert management MCP tools."""

    @pytest.mark.asyncio
    async def test_get_cache_alerts_tool(self):
        """Test get_cache_alerts_tool MCP integration."""
        mock_alert_service = AsyncMock()
        mock_alert_service.get_active_alerts.return_value = [
            {
                "id": "alert_001",
                "type": "memory_pressure",
                "severity": "warning",
                "message": "Cache memory usage at 85%",
                "created_at": datetime.now().isoformat(),
                "threshold": 0.8,
                "current_value": 0.85,
            },
            {
                "id": "alert_002",
                "type": "hit_rate_low",
                "severity": "critical",
                "message": "Cache hit rate below threshold",
                "created_at": datetime.now().isoformat(),
                "threshold": 0.7,
                "current_value": 0.65,
            },
        ]

        with patch("src.tools.core.cache_alert_management.get_alert_service", return_value=mock_alert_service):
            result = await get_cache_alerts_tool({"severity": "warning", "include_resolved": False})

            assert len(result["alerts"]) == 2
            assert result["alerts"][0]["type"] == "memory_pressure"
            assert result["alerts"][1]["severity"] == "critical"
            assert result["summary"]["total_alerts"] == 2

    @pytest.mark.asyncio
    async def test_set_alert_threshold_tool(self):
        """Test set_alert_threshold_tool MCP integration."""
        mock_alert_service = AsyncMock()
        mock_alert_service.set_threshold.return_value = True

        with patch("src.tools.core.cache_alert_management.get_alert_service", return_value=mock_alert_service):
            result = await set_alert_threshold_tool({"metric": "memory_usage", "threshold": 0.9, "severity": "critical", "enabled": True})

            assert result["success"] is True
            assert result["metric"] == "memory_usage"
            assert result["threshold"] == 0.9
            mock_alert_service.set_threshold.assert_called_once()

    @pytest.mark.asyncio
    async def test_acknowledge_alert_tool(self):
        """Test acknowledge_alert_tool MCP integration."""
        mock_alert_service = AsyncMock()
        mock_alert_service.acknowledge_alert.return_value = {
            "acknowledged": True,
            "acknowledged_by": "user123",
            "acknowledged_at": datetime.now().isoformat(),
        }

        with patch("src.tools.core.cache_alert_management.get_alert_service", return_value=mock_alert_service):
            result = await acknowledge_alert_tool(
                {"alert_id": "alert_001", "user_id": "user123", "note": "Investigating memory pressure issue"}
            )

            assert result["success"] is True
            assert result["acknowledged"] is True
            assert result["note"] == "Investigating memory pressure issue"


class TestMCPToolWorkflows(TestMCPToolsBase):
    """Test complete MCP tool workflows."""

    @pytest.mark.asyncio
    async def test_cache_management_workflow(self, mock_cache_service):
        """Test complete cache management workflow using MCP tools."""
        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            # Step 1: Check cache stats
            stats_result = await get_cache_stats_tool({})
            assert "total_keys" in stats_result

            # Step 2: Set some cache data
            set_result = await set_cache_tool({"key": "workflow:test:1", "value": {"step": 1, "data": "test"}, "ttl": 3600})
            assert set_result["success"] is True

            # Step 3: Warm cache with additional data
            warm_result = await warm_cache_tool(
                {
                    "entries": [
                        {"key": "workflow:test:2", "loader": "load_test_data", "params": {"id": 2}},
                        {"key": "workflow:test:3", "loader": "load_test_data", "params": {"id": 3}},
                    ]
                }
            )
            assert warm_result["success"] is True

            # Step 4: Clear specific pattern
            clear_result = await clear_cache_tool({"pattern": "workflow:test:*", "confirm": True})
            assert clear_result["success"] is True

    @pytest.mark.asyncio
    async def test_invalidation_workflow(self):
        """Test complete invalidation workflow using MCP tools."""
        mock_cascade_service = AsyncMock()
        mock_cascade_service.register_dependency.return_value = True
        mock_cascade_service.invalidate_cascade.return_value = {"success": True, "total_invalidated": 5}

        with patch("src.tools.cache.cascade_invalidation_tools.get_cascade_service", return_value=mock_cascade_service):
            # Step 1: Register dependencies
            reg_result = await register_dependency_tool({"parent": "workflow:parent", "dependents": ["workflow:child1", "workflow:child2"]})
            assert reg_result["success"] is True

            # Step 2: Get dependency graph
            graph_result = await get_dependency_graph_tool({"root_key": "workflow:parent"})
            assert "nodes" in graph_result

            # Step 3: Cascade invalidate
            invalidate_result = await cascade_invalidate_tool({"key": "workflow:parent", "continue_on_error": True})
            assert invalidate_result["success"] is True

    @pytest.mark.asyncio
    async def test_monitoring_workflow(self):
        """Test complete monitoring workflow using MCP tools."""
        mock_alert_service = AsyncMock()
        mock_alert_service.set_threshold.return_value = True
        mock_alert_service.get_active_alerts.return_value = []

        with patch("src.tools.core.cache_alert_management.get_alert_service", return_value=mock_alert_service):
            # Step 1: Set monitoring thresholds
            threshold_result = await set_alert_threshold_tool({"metric": "memory_usage", "threshold": 0.85, "severity": "warning"})
            assert threshold_result["success"] is True

            # Step 2: Check alerts
            alerts_result = await get_cache_alerts_tool({})
            assert "alerts" in alerts_result

            # Step 3: If alerts exist, acknowledge them
            if alerts_result["summary"]["total_alerts"] > 0:
                ack_result = await acknowledge_alert_tool(
                    {"alert_id": alerts_result["alerts"][0]["id"], "user_id": "system", "note": "Automated acknowledgment"}
                )
                assert ack_result["success"] is True


class TestMCPToolsPerformance(TestMCPToolsBase):
    """Test MCP tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_batch_operations_performance(self, mock_cache_service):
        """Test performance of batch MCP operations."""
        # Test warming large number of cache entries
        large_entries = [{"key": f"perf:test:{i}", "loader": "load_data", "params": {"id": i}} for i in range(100)]

        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            start_time = time.time()

            result = await warm_cache_tool({"entries": large_entries, "batch_size": 20, "max_concurrent": 5})

            elapsed = time.time() - start_time

            assert result["success"] is True
            assert result["warmed_count"] == 100
            assert elapsed < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self, mock_cache_service):
        """Test MCP tool timeout handling."""

        # Simulate slow cache operation
        async def slow_set(*args, **kwargs):
            await asyncio.sleep(2.0)
            return True

        mock_cache_service.set.side_effect = slow_set

        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            # Set short timeout
            result = await asyncio.wait_for(set_cache_tool({"key": "slow:operation", "value": {"data": "test"}}), timeout=1.0)

            # Should handle timeout gracefully
            assert "timeout" in result or result["success"] is False

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, mock_cache_service):
        """Test concurrent execution of MCP tools."""
        with patch("src.tools.cache.cache_management.get_cache_service", return_value=mock_cache_service):
            # Execute multiple tools concurrently
            tasks = []

            for i in range(10):
                task = set_cache_tool({"key": f"concurrent:test:{i}", "value": {"index": i}})
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All operations should succeed
            successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            assert successful == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
