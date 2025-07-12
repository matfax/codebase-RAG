"""
Tests for cache memory usage reporting service.

This module provides comprehensive tests for the CacheMemoryReporter including:
- Report generation testing
- Dashboard functionality testing
- Alert system testing
- Export functionality testing
- Integration testing with leak detector and profiler
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .cache_memory_reporter import (
    AlertSeverity,
    CacheMemoryReporter,
    DashboardMetrics,
    MemoryAlert,
    MemoryReport,
    ReportFormat,
    ReportingConfig,
    ReportType,
)


class TestCacheMemoryReporter:
    """Test suite for CacheMemoryReporter."""

    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        return ReportingConfig(
            enable_automatic_reports=False,  # Disable for testing
            enable_alerts=True,
            dashboard_update_interval_seconds=1,  # Fast updates for testing
            alert_check_interval_seconds=1,  # Fast checks for testing
            export_directory=temp_dir,
            memory_usage_warning_threshold_mb=100.0,
            memory_usage_critical_threshold_mb=200.0,
            memory_growth_rate_warning_mb_per_hour=50.0,
            memory_growth_rate_critical_mb_per_hour=100.0,
        )

    @pytest.fixture
    async def reporter(self, test_config):
        """Create and initialize memory reporter for testing."""
        with (
            patch("src.services.cache_memory_reporter.get_leak_detector") as mock_leak_detector,
            patch("src.services.cache_memory_reporter.get_memory_profiler") as mock_profiler,
        ):
            # Mock the detector and profiler
            mock_leak_detector.return_value = AsyncMock()
            mock_profiler.return_value = AsyncMock()

            reporter = CacheMemoryReporter(test_config)
            await reporter.initialize()

            yield reporter

            await reporter.shutdown()

    @pytest.fixture
    def mock_memory_stats(self):
        """Mock memory statistics."""
        return {
            "rss_mb": 512.0,
            "system_memory": {"total_mb": 8192.0, "available_mb": 4096.0},
        }

    @pytest.fixture
    def mock_cache_profile(self):
        """Mock cache profile data."""
        return {
            "cache_name": "test_cache",
            "memory_usage": {
                "current_memory_mb": 128.0,
                "peak_memory_mb": 256.0,
                "total_allocated_mb": 512.0,
                "total_deallocated_mb": 384.0,
            },
            "allocation_stats": {
                "total_allocations": 1000,
                "total_deallocations": 800,
                "memory_turnover_ratio": 0.75,
            },
            "efficiency_metrics": {"memory_efficiency_ratio": 0.8, "fragmentation_ratio": 0.1},
            "is_active": True,
            "recent_events": 50,
        }

    async def test_initialization(self, test_config, temp_dir):
        """Test reporter initialization."""
        with (
            patch("src.services.cache_memory_reporter.get_leak_detector") as mock_leak_detector,
            patch("src.services.cache_memory_reporter.get_memory_profiler") as mock_profiler,
        ):
            mock_leak_detector.return_value = AsyncMock()
            mock_profiler.return_value = AsyncMock()

            reporter = CacheMemoryReporter(test_config)

            assert not reporter.is_running
            assert len(reporter.reports) == 0
            assert len(reporter.alerts) == 0

            await reporter.initialize()

            assert reporter.is_running
            assert reporter.leak_detector is not None
            assert reporter.memory_profiler is not None

            # Check export directory was created
            assert Path(temp_dir).exists()

            await reporter.shutdown()
            assert not reporter.is_running

    async def test_summary_report_generation(self, reporter):
        """Test summary report generation."""
        with (
            patch("src.services.cache_memory_reporter.get_total_cache_memory_usage", return_value=256.0),
            patch("src.services.cache_memory_reporter.get_memory_stats") as mock_stats,
            patch("src.services.cache_memory_reporter.get_system_memory_pressure", return_value="normal"),
        ):
            mock_stats.return_value = {
                "rss_mb": 512.0,
                "system_memory": {"available_mb": 4096.0},
            }

            report = await reporter.generate_report(ReportType.SUMMARY)

            assert report.report_type == ReportType.SUMMARY
            assert report.cache_name is None
            assert "current_memory" in report.summary
            assert report.summary["current_memory"]["total_cache_memory_mb"] == 256.0
            assert len(reporter.reports) == 1
            assert report.report_id in reporter.reports_by_id

    async def test_cache_specific_report(self, reporter, mock_cache_profile):
        """Test cache-specific report generation."""
        cache_name = "test_cache"

        # Mock memory profiler methods
        reporter.memory_profiler.get_cache_profile.return_value = mock_cache_profile
        reporter.memory_profiler.get_memory_trend.return_value = {
            "trend": "increasing",
            "avg_memory_mb": 150.0,
            "min_memory_mb": 100.0,
            "max_memory_mb": 200.0,
        }

        with (
            patch("src.services.cache_memory_reporter.get_total_cache_memory_usage", return_value=256.0),
            patch("src.services.cache_memory_reporter.get_memory_stats") as mock_stats,
            patch("src.services.cache_memory_reporter.get_system_memory_pressure", return_value="normal"),
        ):
            mock_stats.return_value = {
                "rss_mb": 512.0,
                "system_memory": {"available_mb": 4096.0},
            }

            report = await reporter.generate_report(ReportType.DETAILED, cache_name=cache_name)

            assert report.cache_name == cache_name
            assert "cache_specific" in report.summary
            assert report.summary["cache_specific"]["cache_name"] == cache_name
            assert "trends" in report.summary

    async def test_trend_analysis_report(self, reporter):
        """Test trend analysis report generation."""
        cache_name = "test_cache"

        # Mock trend data
        reporter.memory_profiler.get_memory_trend.side_effect = [
            {"trend": "increasing", "avg_memory_mb": 100.0},  # Short term
            {"trend": "stable", "avg_memory_mb": 95.0},  # Medium term
            {"trend": "decreasing", "avg_memory_mb": 90.0},  # Long term
        ]

        report = await reporter.generate_report(ReportType.TREND_ANALYSIS, cache_name=cache_name)

        assert "short_term" in report.trends
        assert "medium_term" in report.trends
        assert "long_term" in report.trends
        assert report.trends["short_term"]["trend"] == "increasing"

    async def test_leak_analysis_report(self, reporter):
        """Test leak analysis report generation."""
        # Mock leak detector data
        reporter.leak_detector.get_leak_summary.return_value = {
            "total_leaks": 3,
            "leaks_by_type": {"gradual_growth": 2, "sudden_spike": 1},
            "leaks_by_severity": {"high": 2, "medium": 1},
        }
        reporter.leak_detector.get_detection_stats.return_value = {
            "total_leaks_detected": 10,
            "detection_accuracy": 0.9,
        }

        report = await reporter.generate_report(ReportType.LEAK_ANALYSIS)

        assert "summary" in report.leaks
        assert "detection_statistics" in report.leaks
        assert report.leaks["summary"]["total_leaks"] == 3

    async def test_dashboard_metrics_generation(self, reporter, mock_cache_profile):
        """Test real-time dashboard metrics generation."""
        # Mock dependencies
        reporter.memory_profiler.active_profiles = {"test_cache": MagicMock()}
        reporter.memory_profiler.get_cache_profile.return_value = mock_cache_profile
        reporter.leak_detector.get_leak_summary.return_value = {"total_leaks": 0}

        with (
            patch("src.services.cache_memory_reporter.get_total_cache_memory_usage", return_value=256.0),
            patch("src.services.cache_memory_reporter.get_memory_stats") as mock_stats,
            patch("src.services.cache_memory_reporter.get_system_memory_pressure", return_value="normal"),
        ):
            mock_stats.return_value = {
                "rss_mb": 512.0,
                "system_memory": {"available_mb": 4096.0},
            }

            dashboard = await reporter.get_real_time_dashboard()

            assert isinstance(dashboard, DashboardMetrics)
            assert "total_cache_memory_mb" in dashboard.system_metrics
            assert "test_cache" in dashboard.cache_metrics
            assert dashboard.cache_metrics["test_cache"]["current_memory_mb"] == 128.0
            assert "memory_utilization_percent" in dashboard.performance_indicators

    async def test_alert_generation_memory_threshold(self, reporter):
        """Test alert generation for memory thresholds."""
        with patch("src.services.cache_memory_reporter.get_total_cache_memory_usage", return_value=250.0):  # Above critical threshold
            alerts = await reporter.check_and_generate_alerts()

            assert len(alerts) > 0
            critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
            assert len(critical_alerts) > 0
            assert "critical threshold" in critical_alerts[0].message.lower()

    async def test_alert_generation_cache_specific(self, reporter, mock_cache_profile):
        """Test cache-specific alert generation."""
        cache_name = "test_cache"

        # Mock high memory usage cache
        high_memory_profile = mock_cache_profile.copy()
        high_memory_profile["memory_usage"]["current_memory_mb"] = 600.0  # High usage
        high_memory_profile["efficiency_metrics"]["memory_efficiency_ratio"] = 0.3  # Low efficiency

        reporter.memory_profiler.active_profiles = {cache_name: MagicMock()}
        reporter.memory_profiler.get_cache_profile.return_value = high_memory_profile

        alerts = await reporter.check_and_generate_alerts()

        cache_alerts = [a for a in alerts if a.cache_name == cache_name]
        assert len(cache_alerts) >= 1  # Should have alerts for high memory and low efficiency

    async def test_alert_growth_rate_detection(self, reporter):
        """Test memory growth rate alert detection."""
        # Simulate increasing memory usage in dashboard history
        for i in range(15):
            dashboard = DashboardMetrics(
                timestamp=time.time() - (15 - i) * 30,  # 30 second intervals
                system_metrics={"total_cache_memory_mb": 100.0 + i * 20},  # Rapid growth
            )
            reporter.dashboard_history.append(dashboard)

        alerts = await reporter.check_and_generate_alerts()

        growth_alerts = [a for a in alerts if a.alert_type == "memory_growth_rate"]
        assert len(growth_alerts) > 0
        assert growth_alerts[0].severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]

    async def test_leak_based_alerts(self, reporter):
        """Test leak-based alert generation."""
        # Mock leak detector with critical leaks
        reporter.leak_detector.get_leak_summary.return_value = {
            "total_leaks": 5,
            "leaks_by_severity": {"critical": 2, "high": 3},
        }

        alerts = await reporter.check_and_generate_alerts()

        leak_alerts = [a for a in alerts if a.alert_type == "memory_leaks"]
        assert len(leak_alerts) > 0
        critical_leak_alerts = [a for a in leak_alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_leak_alerts) > 0

    async def test_alert_acknowledgment_and_resolution(self, reporter):
        """Test alert acknowledgment and resolution."""
        # Create a test alert
        alert = MemoryAlert(
            alert_id="test_alert_1",
            cache_name="test_cache",
            alert_type="test_type",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            timestamp=time.time(),
            current_value=150.0,
            threshold_value=100.0,
            metric_name="test_metric",
        )

        reporter.alerts.append(alert)
        reporter.alerts_by_id[alert.alert_id] = alert
        reporter.active_alerts["test_cache:test_type:test_metric"] = alert

        # Test acknowledgment
        result = reporter.acknowledge_alert(alert.alert_id)
        assert result is True
        assert alert.acknowledged is True

        # Test resolution
        result = reporter.resolve_alert(alert.alert_id)
        assert result is True
        assert alert.resolved is True
        assert alert.resolution_time is not None
        assert "test_cache:test_type:test_metric" not in reporter.active_alerts

    async def test_report_export_json(self, reporter, temp_dir):
        """Test JSON report export."""
        # Generate a test report
        report = await reporter.generate_report(ReportType.SUMMARY)

        # Export to JSON
        output_path = await reporter.export_report(report.report_id, ReportFormat.JSON)

        assert Path(output_path).exists()
        assert output_path.endswith(".json")

        # Verify content
        with open(output_path) as f:
            exported_data = json.load(f)

        assert exported_data["report_id"] == report.report_id
        assert exported_data["report_type"] == report.report_type.value

    async def test_report_export_csv(self, reporter, temp_dir):
        """Test CSV report export."""
        # Generate a test report
        report = await reporter.generate_report(ReportType.SUMMARY)

        # Export to CSV
        output_path = await reporter.export_report(report.report_id, ReportFormat.CSV)

        assert Path(output_path).exists()
        assert output_path.endswith(".csv")

        # Verify it's valid CSV
        with open(output_path) as f:
            content = f.read()

        assert "Report Type" in content
        assert report.report_type.value in content

    async def test_report_export_html(self, reporter, temp_dir):
        """Test HTML report export."""
        # Generate a test report with recommendations
        report = await reporter.generate_report(ReportType.SUMMARY, include_recommendations=True)

        # Export to HTML
        output_path = await reporter.export_report(report.report_id, ReportFormat.HTML)

        assert Path(output_path).exists()
        assert output_path.endswith(".html")

        # Verify HTML content
        with open(output_path) as f:
            content = f.read()

        assert "<html>" in content
        assert "Memory Usage Report" in content
        assert report.report_type.value in content

    async def test_report_export_text(self, reporter, temp_dir):
        """Test plain text report export."""
        # Generate a test report
        report = await reporter.generate_report(ReportType.SUMMARY)

        # Export to text
        output_path = await reporter.export_report(report.report_id, ReportFormat.TEXT)

        assert Path(output_path).exists()
        assert output_path.endswith(".text")

        # Verify text content
        with open(output_path) as f:
            content = f.read()

        assert "Memory Usage Report" in content
        assert report.report_type.value in content

    async def test_report_statistics(self, reporter):
        """Test report statistics tracking."""
        # Generate some reports
        await reporter.generate_report(ReportType.SUMMARY)
        await reporter.generate_report(ReportType.DETAILED)
        await reporter.generate_report(ReportType.TREND_ANALYSIS)

        stats = reporter.get_report_statistics()

        assert stats["total_reports"] == 3
        assert stats["generation_stats"]["total_reports_generated"] == 3
        assert stats["generation_stats"]["reports_by_type"]["summary"] == 1
        assert stats["generation_stats"]["reports_by_type"]["detailed"] == 1

    async def test_alert_summary(self, reporter):
        """Test alert summary generation."""
        # Create test alerts
        alerts = [
            MemoryAlert(
                alert_id=f"alert_{i}",
                cache_name=f"cache_{i % 2}",
                alert_type="test_type",
                severity=AlertSeverity.WARNING if i % 2 == 0 else AlertSeverity.CRITICAL,
                message=f"Test alert {i}",
                timestamp=time.time() - i * 60,  # Different timestamps
                current_value=100.0 + i,
                threshold_value=100.0,
                metric_name="test_metric",
            )
            for i in range(5)
        ]

        for alert in alerts:
            reporter.alerts.append(alert)
            reporter.alerts_by_id[alert.alert_id] = alert
            if not alert.resolved:
                reporter.active_alerts[f"{alert.cache_name}:{alert.alert_type}:{alert.metric_name}_{alert.alert_id}"] = alert

        summary = reporter.get_alert_summary()

        assert summary["total_alerts"] == 5
        assert summary["by_severity"]["warning"] == 3  # alerts 0, 2, 4
        assert summary["by_severity"]["critical"] == 2  # alerts 1, 3
        assert len(summary["recent_alerts"]) <= 10

    async def test_dashboard_trend_analysis(self, reporter):
        """Test dashboard trend analysis."""
        # Add dashboard history with memory trends
        base_time = time.time()
        for i in range(20):
            dashboard = DashboardMetrics(
                timestamp=base_time - (20 - i) * 30,
                cache_metrics={
                    "test_cache": {"current_memory_mb": 100.0 + i * 5},  # Increasing trend
                },
            )
            reporter.dashboard_history.append(dashboard)

        trends = reporter._analyze_dashboard_trends("test_cache")

        assert "data_points" in trends
        assert trends["data_points"] == 20
        assert trends["trend"] == "increasing"
        assert trends["growth_rate_mb_per_hour"] > 0

    async def test_recommendation_generation(self, reporter):
        """Test recommendation generation based on report data."""
        # Create a report with high memory usage
        report = MemoryReport(
            report_id="test_report",
            report_type=ReportType.SUMMARY,
            cache_name=None,
            timestamp=time.time(),
            start_time=time.time() - 3600,
            end_time=time.time(),
            summary={
                "current_memory": {"total_cache_memory_mb": 2500.0},  # Above critical threshold
                "trends": {"trend_direction": "increasing"},
            },
            leaks={"summary": {"total_leaks": 2}},
            detailed_metrics={"memory_hotspots": [{"cache_name": "test", "total_size_mb": 100}]},
        )

        recommendations = await reporter._generate_recommendations(report)

        assert len(recommendations) > 0
        critical_found = any("CRITICAL" in rec for rec in recommendations)
        trend_found = any("trending upward" in rec for rec in recommendations)
        leak_found = any("leaks detected" in rec for rec in recommendations)
        hotspot_found = any("hotspots detected" in rec for rec in recommendations)

        assert critical_found
        assert trend_found
        assert leak_found
        assert hotspot_found

    async def test_error_handling_invalid_report_export(self, reporter):
        """Test error handling for invalid report export."""
        with pytest.raises(ValueError, match="Report .* not found"):
            await reporter.export_report("nonexistent_report", ReportFormat.JSON)

    async def test_error_handling_unsupported_format(self, reporter):
        """Test error handling for unsupported export format."""
        # Generate a test report
        report = await reporter.generate_report(ReportType.SUMMARY)

        # Try to export with invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            await reporter.export_report(report.report_id, "invalid_format")

    async def test_integration_with_memory_profiler(self, reporter, mock_cache_profile):
        """Test integration with memory profiler."""
        cache_name = "integration_test_cache"

        # Mock profiler methods
        reporter.memory_profiler.get_cache_profile.return_value = mock_cache_profile
        reporter.memory_profiler.get_memory_trend.return_value = {"trend": "stable"}
        reporter.memory_profiler.get_allocation_patterns.return_value = {"allocations": {"count": 100}}
        reporter.memory_profiler.get_memory_hotspots.return_value = [{"cache_name": cache_name, "total_size_mb": 50}]
        reporter.memory_profiler.get_performance_metrics.return_value = {"allocation_times": {"avg": 0.05}}

        # Generate detailed report
        report = await reporter.generate_report(ReportType.DETAILED, cache_name=cache_name)

        # Verify integration
        assert "allocation_patterns" in report.detailed_metrics
        assert "memory_hotspots" in report.detailed_metrics
        assert "performance_metrics" in report.detailed_metrics
        assert "cache_profile" in report.detailed_metrics

    async def test_integration_with_leak_detector(self, reporter):
        """Test integration with leak detector."""
        # Mock leak detector methods
        reporter.leak_detector.get_leak_summary.return_value = {
            "total_leaks": 3,
            "leaks_by_type": {"gradual_growth": 2, "sudden_spike": 1},
            "leaks_by_severity": {"high": 2, "medium": 1},
            "recent_leaks": [{"timestamp": time.time(), "type": "gradual_growth"}],
        }
        reporter.leak_detector.get_detection_stats.return_value = {
            "total_leaks_detected": 10,
            "detection_accuracy": 0.85,
        }
        reporter.leak_detector.get_retention_analysis.return_value = {
            "retention_ratio": 0.9,
            "is_leak_suspected": True,
        }

        # Generate leak analysis report
        report = await reporter.generate_report(ReportType.LEAK_ANALYSIS, cache_name="test_cache")

        # Verify integration
        assert report.leaks["summary"]["total_leaks"] == 3
        assert "detection_statistics" in report.leaks
        assert "retention_analysis" in report.leaks

    async def test_concurrent_operations(self, reporter):
        """Test concurrent report generation and alert checking."""
        # Create multiple concurrent operations
        tasks = [
            reporter.generate_report(ReportType.SUMMARY),
            reporter.generate_report(ReportType.DETAILED),
            reporter.check_and_generate_alerts(),
            reporter.get_real_time_dashboard(),
        ]

        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)

        # Verify reports were generated
        assert len(reporter.reports) >= 2

    async def test_memory_cleanup(self, reporter):
        """Test memory cleanup functionality."""
        # Generate old reports and alerts
        old_timestamp = time.time() - (reporter.config.report_retention_hours + 1) * 3600

        old_report = MemoryReport(
            report_id="old_report",
            report_type=ReportType.SUMMARY,
            cache_name=None,
            timestamp=old_timestamp,
            start_time=old_timestamp - 3600,
            end_time=old_timestamp,
        )

        old_alert = MemoryAlert(
            alert_id="old_alert",
            cache_name="test_cache",
            alert_type="test_type",
            severity=AlertSeverity.WARNING,
            message="Old alert",
            timestamp=old_timestamp,
            current_value=100.0,
            threshold_value=50.0,
            metric_name="test_metric",
        )

        reporter.reports.append(old_report)
        reporter.reports_by_id[old_report.report_id] = old_report
        reporter.alerts.append(old_alert)
        reporter.alerts_by_id[old_alert.alert_id] = old_alert

        # Generate new report and alert
        new_report = await reporter.generate_report(ReportType.SUMMARY)

        # Trigger cleanup manually
        await reporter._cleanup_loop()  # This would normally be called by the background task

        # Verify old data was cleaned up
        assert old_report.report_id not in reporter.reports_by_id
        assert old_alert.alert_id not in reporter.alerts_by_id
        assert new_report.report_id in reporter.reports_by_id


class TestReportingConfig:
    """Test suite for ReportingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportingConfig()

        assert config.enable_automatic_reports is True
        assert config.enable_alerts is True
        assert config.memory_usage_warning_threshold_mb == 1000.0
        assert config.memory_usage_critical_threshold_mb == 2000.0
        assert config.export_directory == "memory_reports"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ReportingConfig(
            enable_automatic_reports=False,
            memory_usage_warning_threshold_mb=500.0,
            export_directory="/custom/path",
        )

        assert config.enable_automatic_reports is False
        assert config.memory_usage_warning_threshold_mb == 500.0
        assert config.export_directory == "/custom/path"


class TestDataModels:
    """Test suite for data models."""

    def test_memory_report_properties(self):
        """Test MemoryReport properties."""
        start_time = time.time() - 7200  # 2 hours ago
        end_time = time.time() - 3600  # 1 hour ago
        timestamp = time.time() - 1800  # 30 minutes ago

        report = MemoryReport(
            report_id="test_report",
            report_type=ReportType.SUMMARY,
            cache_name="test_cache",
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
        )

        assert report.duration_hours == pytest.approx(1.0, rel=0.1)
        assert report.age_hours == pytest.approx(0.5, rel=0.1)

    def test_memory_alert_properties(self):
        """Test MemoryAlert properties."""
        timestamp = time.time() - 300  # 5 minutes ago
        resolution_time = time.time() - 60  # 1 minute ago

        alert = MemoryAlert(
            alert_id="test_alert",
            cache_name="test_cache",
            alert_type="test_type",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            timestamp=timestamp,
            current_value=150.0,
            threshold_value=100.0,
            metric_name="test_metric",
            resolved=True,
            resolution_time=resolution_time,
        )

        assert alert.age_minutes == pytest.approx(5.0, rel=0.1)
        assert alert.duration_minutes == pytest.approx(4.0, rel=0.1)

    def test_dashboard_metrics_creation(self):
        """Test DashboardMetrics creation."""
        timestamp = time.time()

        dashboard = DashboardMetrics(
            timestamp=timestamp,
            system_metrics={"total_memory": 1000.0},
            cache_metrics={"cache1": {"memory": 256.0}},
        )

        assert dashboard.timestamp == timestamp
        assert dashboard.system_metrics["total_memory"] == 1000.0
        assert dashboard.cache_metrics["cache1"]["memory"] == 256.0


# Integration test helpers
class MockCacheMemoryLeakDetector:
    """Mock leak detector for testing."""

    def get_leak_summary(self, cache_name=None):
        return {"total_leaks": 0, "leaks_by_type": {}, "leaks_by_severity": {}}

    def get_detection_stats(self):
        return {"total_leaks_detected": 0, "detection_accuracy": 1.0}

    def get_retention_analysis(self, cache_name):
        return None


class MockCacheMemoryProfiler:
    """Mock memory profiler for testing."""

    def __init__(self):
        self.active_profiles = {}

    def get_cache_profile(self, cache_name):
        return None

    def get_memory_trend(self, cache_name=None, window_minutes=60):
        return {"error": "No data available"}

    def get_allocation_patterns(self, cache_name=None, window_minutes=60):
        return {"error": "No data available"}

    def get_memory_hotspots(self, cache_name=None, min_allocations=5):
        return []

    def get_performance_metrics(self):
        return {"allocation_times": {"count": 0}}


if __name__ == "__main__":
    pytest.main([__file__])
