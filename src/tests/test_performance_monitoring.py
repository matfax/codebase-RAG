"""
Tests for performance monitoring and metrics collection services.

This module tests the comprehensive performance monitoring infrastructure
including monitoring service, dashboard service, and integration service.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.performance_dashboard_service import DashboardConfig, DashboardMetrics, PerformanceChartData, PerformanceDashboardService
from src.services.performance_integration_service import PerformanceIntegrationConfig, PerformanceIntegrationService, PipelineHealthStatus
from src.services.performance_monitoring_service import (
    ComponentMetrics,
    PerformanceAlert,
    PerformanceMonitoringConfig,
    PerformanceMonitoringService,
    PipelinePerformanceSnapshot,
)


class TestPerformanceMonitoringConfig:
    """Test performance monitoring configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = PerformanceMonitoringConfig()

        assert config.enable_monitoring is True
        assert config.enable_alerts is True
        assert config.enable_system_monitoring is True
        assert config.snapshot_interval_seconds == 30.0
        assert config.max_snapshots_in_memory == 1000
        assert "max_processing_time_ms" in config.alert_thresholds
        assert config.alert_thresholds["max_processing_time_ms"] == 30000.0

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "PERF_MONITOR_ENABLED": "false",
                "PERF_MONITOR_ALERTS": "false",
                "PERF_MONITOR_INTERVAL": "60",
                "PERF_MONITOR_THRESHOLD_MAX_PROCESSING_TIME_MS": "45000",
            },
        ):
            config = PerformanceMonitoringConfig.from_env()

            assert config.enable_monitoring is False
            assert config.enable_alerts is False
            assert config.snapshot_interval_seconds == 60.0
            assert config.alert_thresholds.get("max_processing_time_ms") == 45000.0


class TestComponentMetrics:
    """Test component metrics functionality."""

    def test_metrics_creation(self):
        """Test creation of component metrics."""
        metrics = ComponentMetrics(component_name="test_component")

        assert metrics.component_name == "test_component"
        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.min_processing_time_ms == float("inf")
        assert metrics.max_processing_time_ms == 0.0

    def test_timing_updates(self):
        """Test timing statistics updates."""
        metrics = ComponentMetrics(component_name="test")

        # Update with timing data
        metrics.total_operations = 3
        metrics.update_timing(100.0)
        metrics.update_timing(200.0)
        metrics.update_timing(150.0)

        assert metrics.total_processing_time_ms == 450.0
        assert metrics.average_processing_time_ms == 150.0
        assert metrics.min_processing_time_ms == 100.0
        assert metrics.max_processing_time_ms == 200.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ComponentMetrics(component_name="test")

        metrics.total_operations = 100
        metrics.successful_operations = 85
        metrics.failed_operations = 15
        metrics.update_success_rate()

        assert metrics.success_rate_percent == 85.0
        assert metrics.error_rate_percent == 15.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        metrics = ComponentMetrics(component_name="test")

        metrics.cache_hits = 80
        metrics.cache_misses = 20

        assert metrics.cache_hit_rate_percent == 80.0

        # Test with no cache operations
        metrics.cache_hits = 0
        metrics.cache_misses = 0
        assert metrics.cache_hit_rate_percent == 0.0


class TestPerformanceAlert:
    """Test performance alert functionality."""

    def test_alert_creation(self):
        """Test creation of performance alert."""
        alert = PerformanceAlert(
            alert_id="test_alert_123",
            alert_type="warning",
            message="Test alert message",
            metric_name="error_rate_percent",
            current_value=10.5,
            threshold_value=5.0,
            timestamp=time.time(),
            component="test_component",
        )

        assert alert.alert_id == "test_alert_123"
        assert alert.alert_type == "warning"
        assert alert.message == "Test alert message"
        assert alert.current_value == 10.5
        assert alert.threshold_value == 5.0
        assert alert.resolved is False
        assert alert.resolution_timestamp is None


@pytest.mark.asyncio
class TestPerformanceMonitoringService:
    """Test the performance monitoring service."""

    async def test_service_initialization(self):
        """Test service initialization."""
        config = PerformanceMonitoringConfig(enable_system_monitoring=False)
        service = PerformanceMonitoringService(config=config)

        assert service.config == config
        assert len(service._component_metrics) == 0
        assert len(service._active_operations) == 0
        assert len(service._snapshots) == 0

    async def test_disabled_monitoring(self):
        """Test behavior when monitoring is disabled."""
        config = PerformanceMonitoringConfig(enable_monitoring=False)
        service = PerformanceMonitoringService(config=config)

        # Operations should be no-ops
        service.start_operation("test_component", "test_op")
        service.complete_operation("test_component", "test_op", success=True)

        assert len(service._component_metrics) == 0
        assert len(service._active_operations) == 0

    async def test_operation_tracking(self):
        """Test operation tracking functionality."""
        service = PerformanceMonitoringService()

        # Start operation
        service.start_operation("test_component", "test_op", {"key": "value"})

        assert "test_component" in service._component_metrics
        assert "test_component" in service._active_operations
        assert "test_op" in service._active_operations["test_component"]

        # Complete operation
        service.complete_operation(
            component="test_component", operation_id="test_op", success=True, items_processed=10, cache_hits=5, cache_misses=2
        )

        metrics = service._component_metrics["test_component"]
        assert metrics.total_operations == 1
        assert metrics.successful_operations == 1
        assert metrics.failed_operations == 0
        assert metrics.cache_hits == 5
        assert metrics.cache_misses == 2
        assert "test_op" not in service._active_operations["test_component"]

    def test_cache_operation_recording(self):
        """Test cache operation recording."""
        service = PerformanceMonitoringService()

        # Record cache operations
        service.record_cache_operation("cache_component", "get", hit=True, processing_time_ms=5.0)
        service.record_cache_operation("cache_component", "get", hit=False, processing_time_ms=10.0)
        service.record_cache_operation("cache_component", "put", hit=True, processing_time_ms=2.0)

        metrics = service._component_metrics["cache_component"]
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1
        assert metrics.cache_hit_rate_percent == 66.66666666666666

    def test_calls_detected_recording(self):
        """Test function calls detected recording."""
        service = PerformanceMonitoringService()

        service.record_calls_detected("extractor", 25)
        service.record_calls_detected("extractor", 30)

        assert service._global_stats["total_calls_detected"] == 55

    def test_performance_snapshot_creation(self):
        """Test performance snapshot creation."""
        service = PerformanceMonitoringService()

        # Add some metrics
        service.start_operation("test_component", "op1")
        service.complete_operation("test_component", "op1", success=True, items_processed=5)
        service.record_calls_detected("test_component", 20)

        # Create snapshot
        snapshot = service.create_performance_snapshot()

        assert isinstance(snapshot, PipelinePerformanceSnapshot)
        assert snapshot.total_calls_detected == 20
        assert "test_component" in snapshot.components
        assert snapshot.efficiency_score >= 0
        assert snapshot.efficiency_score <= 100

    def test_performance_report_generation(self):
        """Test performance report generation."""
        service = PerformanceMonitoringService()

        # Add some data
        service.start_operation("test_component", "op1")
        service.complete_operation("test_component", "op1", success=True, items_processed=5)

        report = service.get_performance_report()

        assert "current_snapshot" in report
        assert "component_performance" in report
        assert "global_statistics" in report
        assert "recent_alerts" in report
        assert "optimization_recommendations" in report

    def test_alert_handling(self):
        """Test alert handling and deduplication."""
        config = PerformanceMonitoringConfig(enable_alerts=True, alert_thresholds={"max_error_rate_percent": 5.0})
        service = PerformanceMonitoringService(config=config)

        # Create component with high error rate
        service.start_operation("failing_component", "op1")
        service.complete_operation("failing_component", "op1", success=False)

        service.start_operation("failing_component", "op2")
        service.complete_operation("failing_component", "op2", success=False)

        service.start_operation("failing_component", "op3")
        service.complete_operation("failing_component", "op3", success=True)

        # Should trigger error rate alert (2/3 = 66.7% error rate)
        assert len(service._alerts) > 0

        # Check alert details
        error_alert = next((a for a in service._alerts if "error rate" in a.message.lower()), None)
        assert error_alert is not None
        assert error_alert.component == "failing_component"
        assert error_alert.alert_type in ["warning", "error"]

    async def test_monitoring_loop(self):
        """Test the monitoring loop functionality."""
        config = PerformanceMonitoringConfig(enable_monitoring=True, snapshot_interval_seconds=0.1)  # Very short interval for testing
        service = PerformanceMonitoringService(config=config)

        # Start monitoring
        await service.start_monitoring()

        # Add some operations
        service.start_operation("test_component", "op1")
        service.complete_operation("test_component", "op1", success=True)

        # Wait for at least one snapshot
        await asyncio.sleep(0.2)

        # Stop monitoring
        await service.stop_monitoring()

        # Should have created snapshots
        assert len(service._snapshots) > 0

    def test_statistics_and_cleanup(self):
        """Test statistics collection and data cleanup."""
        service = PerformanceMonitoringService()

        # Add operations and data
        for i in range(10):
            service.start_operation("test_component", f"op_{i}")
            service.complete_operation("test_component", f"op_{i}", success=True)

        # Get statistics
        stats = service.get_component_statistics("test_component")
        assert stats is not None
        assert stats["operational_metrics"]["total_operations"] == 10
        assert stats["operational_metrics"]["success_rate_percent"] == 100.0

        # Clear performance data
        service.clear_performance_data()
        assert len(service._component_metrics) == 0
        assert len(service._snapshots) == 0


class TestDashboardConfig:
    """Test dashboard configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = DashboardConfig()

        assert config.enable_dashboard is True
        assert config.update_interval_seconds == 5.0
        assert config.max_chart_data_points == 100
        assert config.enable_real_time_alerts is True

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ", {"DASHBOARD_ENABLED": "false", "DASHBOARD_UPDATE_INTERVAL": "10", "DASHBOARD_MAX_DATA_POINTS": "200"}
        ):
            config = DashboardConfig.from_env()

            assert config.enable_dashboard is False
            assert config.update_interval_seconds == 10.0
            assert config.max_chart_data_points == 200


class TestPerformanceChartData:
    """Test performance chart data functionality."""

    def test_chart_creation(self):
        """Test chart data creation."""
        chart = PerformanceChartData(timestamps=[], values=[], labels=[], chart_type="line", title="Test Chart", y_axis_label="Test Values")

        assert chart.chart_type == "line"
        assert chart.title == "Test Chart"
        assert len(chart.timestamps) == 0

    def test_data_point_addition(self):
        """Test adding data points to chart."""
        chart = PerformanceChartData(timestamps=[], values=[], labels=[], chart_type="line", title="Test Chart", y_axis_label="Test Values")

        # Add data points
        chart.add_data_point(1000.0, 50.0, "50%")
        chart.add_data_point(2000.0, 75.0, "75%")

        assert len(chart.timestamps) == 2
        assert len(chart.values) == 2
        assert len(chart.labels) == 2
        assert chart.values[0] == 50.0
        assert chart.values[1] == 75.0

    def test_data_point_limiting(self):
        """Test limiting data points to prevent memory issues."""
        chart = PerformanceChartData(timestamps=[], values=[], labels=[], chart_type="line", title="Test Chart", y_axis_label="Test Values")

        # Add more data points than limit
        for i in range(150):
            chart.add_data_point(float(i), float(i * 2), f"label_{i}")

        # Limit to 100 points
        chart.limit_data_points(100)

        assert len(chart.timestamps) == 100
        assert len(chart.values) == 100
        assert len(chart.labels) == 100

        # Should keep the most recent data
        assert chart.timestamps[0] == 50.0  # (150 - 100)
        assert chart.timestamps[-1] == 149.0


class TestDashboardMetrics:
    """Test dashboard metrics functionality."""

    def test_metrics_from_snapshot(self):
        """Test creating dashboard metrics from performance snapshot."""
        # Create mock snapshot
        mock_components = {
            "component1": ComponentMetrics(
                component_name="component1",
                total_operations=100,
                successful_operations=95,
                failed_operations=5,
                cache_hits=80,
                cache_misses=20,
            ),
            "component2": ComponentMetrics(
                component_name="component2",
                total_operations=50,
                successful_operations=50,
                failed_operations=0,
                cache_hits=30,
                cache_misses=10,
            ),
        }

        mock_snapshot = PipelinePerformanceSnapshot(
            timestamp=time.time(),
            total_pipeline_time_ms=5000.0,
            total_files_processed=200,
            total_calls_detected=1500,
            components=mock_components,
            system_metrics={"memory_usage_mb": 512.0, "cpu_percent": 25.0},
            alerts=[],
            efficiency_score=85.0,
        )

        start_time = time.time() - 3600  # 1 hour ago
        dashboard_metrics = DashboardMetrics.from_snapshot(mock_snapshot, start_time)

        assert dashboard_metrics.pipeline_efficiency == 85.0
        assert dashboard_metrics.total_operations == 150  # 100 + 50
        assert dashboard_metrics.active_operations == 2
        assert dashboard_metrics.memory_usage_mb == 512.0
        assert dashboard_metrics.cpu_percent == 25.0

        # Check calculated cache hit rate: (80+30)/(80+20+30+10) * 100 = 110/140 * 100 ≈ 78.57%
        expected_cache_hit_rate = (80 + 30) / (80 + 20 + 30 + 10) * 100
        assert abs(dashboard_metrics.cache_hit_rate - expected_cache_hit_rate) < 0.01

        # Check calculated error rate: (5+0)/(95+5+50+0) * 100 = 5/150 * 100 ≈ 3.33%
        expected_error_rate = (5 + 0) / (95 + 5 + 50 + 0) * 100
        assert abs(dashboard_metrics.error_rate - expected_error_rate) < 0.01


@pytest.mark.asyncio
class TestPerformanceDashboardService:
    """Test the performance dashboard service."""

    async def test_dashboard_initialization(self):
        """Test dashboard service initialization."""
        mock_monitoring_service = Mock()
        config = DashboardConfig(enable_dashboard=False)  # Disable for testing

        dashboard = PerformanceDashboardService(monitoring_service=mock_monitoring_service, config=config)

        assert dashboard.monitoring_service == mock_monitoring_service
        assert dashboard.config == config
        assert dashboard._is_running is False
        assert len(dashboard._charts) > 0  # Should initialize charts

    async def test_chart_initialization(self):
        """Test chart initialization."""
        mock_monitoring_service = Mock()
        dashboard = PerformanceDashboardService(monitoring_service=mock_monitoring_service, config=DashboardConfig(enable_dashboard=False))

        # Check that charts are initialized
        expected_charts = [
            "efficiency_score",
            "calls_per_second",
            "files_per_second",
            "memory_usage",
            "cpu_usage",
            "cache_hit_rate",
            "error_rate",
            "component_performance",
        ]

        for chart_name in expected_charts:
            assert chart_name in dashboard._charts
            chart = dashboard._charts[chart_name]
            assert isinstance(chart, PerformanceChartData)
            assert len(chart.timestamps) == 0

    def test_dashboard_data_retrieval(self):
        """Test dashboard data retrieval."""
        mock_monitoring_service = Mock()
        dashboard = PerformanceDashboardService(monitoring_service=mock_monitoring_service, config=DashboardConfig(enable_dashboard=False))

        # Set some current metrics
        dashboard._current_metrics = DashboardMetrics(
            pipeline_efficiency=80.0,
            total_operations=100,
            active_operations=5,
            calls_per_second=50.0,
            files_per_second=10.0,
            memory_usage_mb=512.0,
            cpu_percent=25.0,
            cache_hit_rate=85.0,
            error_rate=2.0,
            active_alerts=1,
            uptime_seconds=3600.0,
        )

        dashboard_data = dashboard.get_dashboard_data()

        assert "current_metrics" in dashboard_data
        assert "charts" in dashboard_data
        assert "recent_alerts" in dashboard_data
        assert dashboard_data["current_metrics"]["pipeline_efficiency"] == 80.0
        assert dashboard_data["current_metrics"]["total_operations"] == 100

    def test_performance_report_generation(self):
        """Test performance report generation."""
        mock_monitoring_service = Mock()
        mock_monitoring_service.get_performance_report.return_value = {
            "current_snapshot": {"efficiency_score": 85.0},
            "component_performance": {},
            "system_metrics": {},
            "optimization_recommendations": [],
        }

        dashboard = PerformanceDashboardService(monitoring_service=mock_monitoring_service, config=DashboardConfig(enable_dashboard=False))

        report = dashboard.generate_performance_report()

        assert "report_metadata" in report
        assert "performance_summary" in report
        assert "current_state" in report
        assert "alert_summary" in report

    def test_performance_forecast(self):
        """Test performance forecasting."""
        mock_monitoring_service = Mock()
        dashboard = PerformanceDashboardService(monitoring_service=mock_monitoring_service, config=DashboardConfig(enable_dashboard=False))

        # Add some historical metrics
        base_time = time.time() - 3600
        for i in range(20):
            metrics = DashboardMetrics(
                pipeline_efficiency=80.0 - (i * 0.5),  # Declining trend
                total_operations=100 + i,
                active_operations=5,
                calls_per_second=50.0 + (i * 0.1),
                files_per_second=10.0,
                memory_usage_mb=512.0 + (i * 2.0),  # Increasing memory
                cpu_percent=25.0,
                cache_hit_rate=85.0,
                error_rate=2.0,
                active_alerts=1,
                uptime_seconds=base_time + (i * 60),
            )
            dashboard._metrics_history.append(metrics)

        forecast = dashboard.get_performance_forecast(hours_ahead=12)

        assert "forecast_period_hours" in forecast
        assert "current_metrics" in forecast
        assert "forecasted_metrics" in forecast
        assert "trends" in forecast
        assert "warnings" in forecast

        # Should forecast declining efficiency
        assert forecast["forecasted_metrics"]["pipeline_efficiency"] < forecast["current_metrics"]["pipeline_efficiency"]


class TestPerformanceIntegrationConfig:
    """Test performance integration configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = PerformanceIntegrationConfig()

        assert config.enable_monitoring is True
        assert config.enable_dashboard is True
        assert config.enable_alerts is True
        assert config.enable_auto_optimization is True
        assert config.performance_reporting_interval_minutes == 30
        assert config.auto_optimization_interval_minutes == 60

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "PERF_INTEGRATION_MONITORING": "false",
                "PERF_INTEGRATION_DASHBOARD": "false",
                "PERF_INTEGRATION_REPORT_INTERVAL": "60",
                "PERF_INTEGRATION_CRITICAL_ERROR_RATE": "15.0",
            },
        ):
            config = PerformanceIntegrationConfig.from_env()

            assert config.enable_monitoring is False
            assert config.enable_dashboard is False
            assert config.performance_reporting_interval_minutes == 60
            assert config.critical_error_rate_threshold == 15.0


@pytest.mark.asyncio
class TestPerformanceIntegrationService:
    """Test the performance integration service."""

    async def test_service_initialization(self):
        """Test service initialization."""
        config = PerformanceIntegrationConfig(enable_monitoring=False, enable_dashboard=False)

        service = PerformanceIntegrationService(config=config)
        assert service.config == config
        assert service._is_running is False
        assert len(service._registered_components) == 0

    async def test_component_registration(self):
        """Test component registration."""
        # Create mock components
        mock_cache = Mock()
        mock_cache.get_statistics.return_value = {"cache_size": 100}

        mock_extractor = Mock()
        mock_extractor.get_statistics.return_value = {"total_extractions": 50}

        service = PerformanceIntegrationService(breadcrumb_cache_service=mock_cache, concurrent_extractor=mock_extractor)

        await service._register_components()

        assert "breadcrumb_cache" in service._registered_components
        assert "concurrent_extractor" in service._registered_components
        assert service._component_health["breadcrumb_cache"] is True
        assert service._component_health["concurrent_extractor"] is True

    def test_operation_tracking_integration(self):
        """Test operation tracking integration."""
        # Create mock monitoring service
        mock_monitoring = Mock()
        service = PerformanceIntegrationService()
        service.monitoring_service = mock_monitoring

        # Track an operation
        service.track_operation(
            component="test_component",
            operation_name="test_operation",
            duration_ms=100.0,
            success=True,
            items_processed=5,
            cache_hits=3,
            cache_misses=1,
        )

        # Should call monitoring service methods
        assert mock_monitoring.start_operation.called
        assert mock_monitoring.complete_operation.called

    def test_health_status_creation(self):
        """Test health status creation."""
        health_status = PipelineHealthStatus(
            overall_health="good",
            health_score=75.0,
            active_components=8,
            total_components=10,
            active_alerts=2,
            critical_alerts=0,
            performance_issues=["Minor cache hit rate degradation"],
            recommendations=["Consider increasing cache size"],
            uptime_hours=24.5,
        )

        assert health_status.overall_health == "good"
        assert health_status.health_score == 75.0
        assert health_status.active_components == 8
        assert len(health_status.performance_issues) == 1
        assert len(health_status.recommendations) == 1

    async def test_health_check_performance(self):
        """Test health check performance."""
        service = PerformanceIntegrationService()

        # Mock monitoring service with performance data
        mock_monitoring = Mock()
        mock_monitoring.get_performance_report.return_value = {
            "current_snapshot": {"efficiency_score": 85.0, "active_alerts": 1},
            "component_performance": {"component1": {"error_rate_percent": 2.0, "cache_hit_rate_percent": 80.0}},
            "system_metrics": {"memory_usage_mb": 512.0},
        }
        service.monitoring_service = mock_monitoring

        # Perform health check
        health_status = await service._perform_health_check()

        assert isinstance(health_status, PipelineHealthStatus)
        assert health_status.health_score > 0
        assert health_status.overall_health in ["excellent", "good", "warning", "critical"]

    def test_performance_summary(self):
        """Test performance summary generation."""
        service = PerformanceIntegrationService()
        service._is_running = True

        # Create mock health status
        service._last_health_status = PipelineHealthStatus(
            overall_health="good",
            health_score=75.0,
            active_components=8,
            total_components=10,
            active_alerts=2,
            critical_alerts=0,
            performance_issues=[],
            recommendations=[],
            uptime_hours=12.0,
        )

        # Add some component health
        service._component_health = {"component1": True, "component2": True, "component3": False}

        summary = service.get_performance_summary()

        assert summary["service_status"] == "running"
        assert summary["overall_health"] == "good"
        assert summary["health_score"] == 75.0
        assert summary["healthy_components"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
