"""
Performance Dashboard Service for Call Detection Pipeline Visualization.

This service provides real-time performance visualization, reporting, and
dashboard capabilities for the enhanced function call detection pipeline.
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.services.performance_monitoring_service import (
    ComponentMetrics,
    PerformanceAlert,
    PerformanceMonitoringService,
    PipelinePerformanceSnapshot,
)


@dataclass
class DashboardConfig:
    """Configuration for the performance dashboard."""

    enable_dashboard: bool = True
    update_interval_seconds: float = 5.0
    max_chart_data_points: int = 100
    enable_real_time_alerts: bool = True
    export_reports: bool = True
    report_export_interval_minutes: int = 60

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            enable_dashboard=os.getenv("DASHBOARD_ENABLED", "true").lower() == "true",
            update_interval_seconds=float(os.getenv("DASHBOARD_UPDATE_INTERVAL", "5")),
            max_chart_data_points=int(os.getenv("DASHBOARD_MAX_DATA_POINTS", "100")),
            enable_real_time_alerts=os.getenv("DASHBOARD_REAL_TIME_ALERTS", "true").lower() == "true",
            export_reports=os.getenv("DASHBOARD_EXPORT_REPORTS", "true").lower() == "true",
            report_export_interval_minutes=int(os.getenv("DASHBOARD_REPORT_INTERVAL", "60")),
        )


@dataclass
class PerformanceChartData:
    """Data structure for performance charts."""

    timestamps: list[float]
    values: list[float]
    labels: list[str]
    chart_type: str  # 'line', 'bar', 'area'
    title: str
    y_axis_label: str
    color: str = "#007bff"

    def add_data_point(self, timestamp: float, value: float, label: str = ""):
        """Add a new data point to the chart."""
        self.timestamps.append(timestamp)
        self.values.append(value)
        self.labels.append(label)

    def limit_data_points(self, max_points: int):
        """Limit the number of data points to prevent memory issues."""
        if len(self.timestamps) > max_points:
            excess = len(self.timestamps) - max_points
            self.timestamps = self.timestamps[excess:]
            self.values = self.values[excess:]
            self.labels = self.labels[excess:]


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""

    pipeline_efficiency: float
    total_operations: int
    active_operations: int
    calls_per_second: float
    files_per_second: float
    memory_usage_mb: float
    cpu_percent: float
    cache_hit_rate: float
    error_rate: float
    active_alerts: int
    uptime_seconds: float

    @classmethod
    def from_snapshot(cls, snapshot: PipelinePerformanceSnapshot, start_time: float) -> "DashboardMetrics":
        """Create dashboard metrics from performance snapshot."""
        total_ops = sum(m.total_operations for m in snapshot.components.values())
        active_ops = len([m for m in snapshot.components.values() if m.total_operations > 0])

        # Calculate aggregate cache hit rate
        total_hits = sum(m.cache_hits for m in snapshot.components.values())
        total_misses = sum(m.cache_misses for m in snapshot.components.values())
        cache_hit_rate = total_hits / (total_hits + total_misses) * 100 if (total_hits + total_misses) > 0 else 0

        # Calculate aggregate error rate
        total_successful = sum(m.successful_operations for m in snapshot.components.values())
        total_failed = sum(m.failed_operations for m in snapshot.components.values())
        error_rate = total_failed / (total_successful + total_failed) * 100 if (total_successful + total_failed) > 0 else 0

        return cls(
            pipeline_efficiency=snapshot.efficiency_score,
            total_operations=total_ops,
            active_operations=active_ops,
            calls_per_second=snapshot.calls_per_second,
            files_per_second=snapshot.files_per_second,
            memory_usage_mb=snapshot.system_metrics.get("memory_usage_mb", 0),
            cpu_percent=snapshot.system_metrics.get("cpu_percent", 0),
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate,
            active_alerts=len([a for a in snapshot.alerts if not a.resolved]),
            uptime_seconds=snapshot.timestamp - start_time,
        )


class PerformanceDashboardService:
    """
    Performance dashboard service for real-time visualization of call detection pipeline metrics.

    This service provides:
    - Real-time performance charts and graphs
    - Component-specific performance visualization
    - Alert dashboard with real-time notifications
    - Performance trend analysis and forecasting
    - Exportable performance reports
    - System health overview
    """

    def __init__(self, monitoring_service: PerformanceMonitoringService, config: DashboardConfig | None = None):
        """
        Initialize the performance dashboard service.

        Args:
            monitoring_service: Performance monitoring service
            config: Dashboard configuration
        """
        self.monitoring_service = monitoring_service
        self.config = config or DashboardConfig.from_env()
        self.logger = logging.getLogger(__name__)

        # Dashboard state
        self._is_running = False
        self._dashboard_task: asyncio.Task | None = None
        self._report_export_task: asyncio.Task | None = None
        self._start_time = time.time()

        # Chart data storage
        self._charts: dict[str, PerformanceChartData] = {}
        self._initialize_charts()

        # Dashboard metrics
        self._current_metrics: DashboardMetrics | None = None
        self._metrics_history: list[DashboardMetrics] = []

        # Alert tracking
        self._alert_history: list[PerformanceAlert] = []

        self.logger.info(f"PerformanceDashboardService initialized with config: {self.config}")

    def _initialize_charts(self):
        """Initialize performance charts."""
        chart_configs = [
            ("efficiency_score", "Pipeline Efficiency", "Efficiency (%)", "line", "#28a745"),
            ("calls_per_second", "Function Calls Detection Rate", "Calls/Second", "line", "#007bff"),
            ("files_per_second", "File Processing Rate", "Files/Second", "line", "#17a2b8"),
            ("memory_usage", "Memory Usage", "Memory (MB)", "area", "#ffc107"),
            ("cpu_usage", "CPU Usage", "CPU (%)", "area", "#fd7e14"),
            ("cache_hit_rate", "Cache Hit Rate", "Hit Rate (%)", "line", "#6610f2"),
            ("error_rate", "Error Rate", "Error Rate (%)", "line", "#dc3545"),
            ("component_performance", "Component Performance", "Operations/Second", "bar", "#20c997"),
        ]

        for chart_id, title, y_label, chart_type, color in chart_configs:
            self._charts[chart_id] = PerformanceChartData(
                timestamps=[], values=[], labels=[], chart_type=chart_type, title=title, y_axis_label=y_label, color=color
            )

    async def start_dashboard(self):
        """Start the performance dashboard service."""
        if not self.config.enable_dashboard:
            self.logger.info("Performance dashboard disabled")
            return

        if self._is_running:
            self.logger.warning("Performance dashboard already running")
            return

        try:
            self._is_running = True
            self._start_time = time.time()

            # Start dashboard update task
            self._dashboard_task = asyncio.create_task(self._dashboard_update_loop())

            # Start report export task if enabled
            if self.config.export_reports:
                self._report_export_task = asyncio.create_task(self._report_export_loop())

            self.logger.info("Performance dashboard started")

        except Exception as e:
            self.logger.error(f"Error starting performance dashboard: {e}")
            self._is_running = False

    async def stop_dashboard(self):
        """Stop the performance dashboard service."""
        self._is_running = False

        # Stop dashboard task
        if self._dashboard_task:
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass

        # Stop report export task
        if self._report_export_task:
            self._report_export_task.cancel()
            try:
                await self._report_export_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance dashboard stopped")

    async def _dashboard_update_loop(self):
        """Main dashboard update loop."""
        self.logger.info("Dashboard update loop started")

        while self._is_running:
            try:
                # Get current performance snapshot
                snapshot = self.monitoring_service.create_performance_snapshot()

                # Update dashboard metrics
                self._current_metrics = DashboardMetrics.from_snapshot(snapshot, self._start_time)
                self._metrics_history.append(self._current_metrics)

                # Limit history size
                if len(self._metrics_history) > self.config.max_chart_data_points:
                    self._metrics_history.pop(0)

                # Update charts
                self._update_charts(snapshot)

                # Update alert history
                self._update_alert_history(snapshot.alerts)

                # Wait for next update
                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(5.0)

        self.logger.info("Dashboard update loop stopped")

    def _update_charts(self, snapshot: PipelinePerformanceSnapshot):
        """Update chart data with new snapshot."""
        timestamp = snapshot.timestamp

        # Update efficiency chart
        self._charts["efficiency_score"].add_data_point(timestamp, snapshot.efficiency_score, f"{snapshot.efficiency_score:.1f}%")

        # Update rate charts
        self._charts["calls_per_second"].add_data_point(timestamp, snapshot.calls_per_second, f"{snapshot.calls_per_second:.1f}")

        self._charts["files_per_second"].add_data_point(timestamp, snapshot.files_per_second, f"{snapshot.files_per_second:.1f}")

        # Update system charts
        memory_mb = snapshot.system_metrics.get("memory_usage_mb", 0)
        cpu_percent = snapshot.system_metrics.get("cpu_percent", 0)

        self._charts["memory_usage"].add_data_point(timestamp, memory_mb, f"{memory_mb:.1f} MB")

        self._charts["cpu_usage"].add_data_point(timestamp, cpu_percent, f"{cpu_percent:.1f}%")

        # Update cache hit rate
        if self._current_metrics:
            self._charts["cache_hit_rate"].add_data_point(
                timestamp, self._current_metrics.cache_hit_rate, f"{self._current_metrics.cache_hit_rate:.1f}%"
            )

            self._charts["error_rate"].add_data_point(
                timestamp, self._current_metrics.error_rate, f"{self._current_metrics.error_rate:.1f}%"
            )

        # Update component performance chart
        component_chart = self._charts["component_performance"]
        component_chart.timestamps = [timestamp]
        component_chart.values = []
        component_chart.labels = []

        for name, metrics in snapshot.components.items():
            if metrics.total_operations > 0:
                ops_per_second = metrics.total_operations / ((timestamp - self._start_time) or 1)
                component_chart.values.append(ops_per_second)
                component_chart.labels.append(name)

        # Limit chart data points
        for chart in self._charts.values():
            chart.limit_data_points(self.config.max_chart_data_points)

    def _update_alert_history(self, current_alerts: list[PerformanceAlert]):
        """Update alert history with current alerts."""
        # Add new alerts to history
        for alert in current_alerts:
            if alert not in self._alert_history:
                self._alert_history.append(alert)

        # Limit alert history size
        if len(self._alert_history) > 500:
            self._alert_history = self._alert_history[-500:]

    async def _report_export_loop(self):
        """Loop for exporting performance reports."""
        self.logger.info("Report export loop started")

        while self._is_running:
            try:
                # Wait for export interval
                await asyncio.sleep(self.config.report_export_interval_minutes * 60)

                # Export performance report
                await self._export_performance_report()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in report export loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

        self.logger.info("Report export loop stopped")

    async def _export_performance_report(self):
        """Export a performance report."""
        try:
            report = self.generate_performance_report()

            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Performance report exported to {filename}")

        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get current dashboard data for visualization."""
        return {
            "current_metrics": {
                "pipeline_efficiency": self._current_metrics.pipeline_efficiency if self._current_metrics else 0,
                "total_operations": self._current_metrics.total_operations if self._current_metrics else 0,
                "active_operations": self._current_metrics.active_operations if self._current_metrics else 0,
                "calls_per_second": self._current_metrics.calls_per_second if self._current_metrics else 0,
                "files_per_second": self._current_metrics.files_per_second if self._current_metrics else 0,
                "memory_usage_mb": self._current_metrics.memory_usage_mb if self._current_metrics else 0,
                "cpu_percent": self._current_metrics.cpu_percent if self._current_metrics else 0,
                "cache_hit_rate": self._current_metrics.cache_hit_rate if self._current_metrics else 0,
                "error_rate": self._current_metrics.error_rate if self._current_metrics else 0,
                "active_alerts": self._current_metrics.active_alerts if self._current_metrics else 0,
                "uptime_seconds": self._current_metrics.uptime_seconds if self._current_metrics else 0,
            },
            "charts": {
                name: {
                    "timestamps": chart.timestamps,
                    "values": chart.values,
                    "labels": chart.labels,
                    "chart_type": chart.chart_type,
                    "title": chart.title,
                    "y_axis_label": chart.y_axis_label,
                    "color": chart.color,
                }
                for name, chart in self._charts.items()
            },
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                }
                for alert in self._alert_history[-10:]  # Last 10 alerts
            ],
            "is_running": self._is_running,
            "last_updated": time.time(),
        }

    def get_component_dashboard(self, component: str) -> dict[str, Any] | None:
        """Get dashboard data for a specific component."""
        component_stats = self.monitoring_service.get_component_statistics(component)
        if not component_stats:
            return None

        # Get component-specific chart data
        component_chart_data = {}
        for chart_name, chart in self._charts.items():
            if component in chart_name:
                component_chart_data[chart_name] = {
                    "timestamps": chart.timestamps,
                    "values": chart.values,
                    "labels": chart.labels,
                    "chart_type": chart.chart_type,
                    "title": chart.title,
                    "y_axis_label": chart.y_axis_label,
                    "color": chart.color,
                }

        return {
            "component_name": component,
            "statistics": component_stats,
            "charts": component_chart_data,
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                }
                for alert in self._alert_history
                if alert.component == component and not alert.resolved
            ],
        }

    def get_alert_dashboard(self) -> dict[str, Any]:
        """Get alert dashboard data."""
        recent_alerts = self._alert_history[-50:]  # Last 50 alerts

        # Group alerts by type and component
        alerts_by_type = {}
        alerts_by_component = {}

        for alert in recent_alerts:
            # By type
            if alert.alert_type not in alerts_by_type:
                alerts_by_type[alert.alert_type] = []
            alerts_by_type[alert.alert_type].append(alert)

            # By component
            if alert.component not in alerts_by_component:
                alerts_by_component[alert.component] = []
            alerts_by_component[alert.component].append(alert)

        # Calculate alert statistics
        total_alerts = len(recent_alerts)
        resolved_alerts = len([a for a in recent_alerts if a.resolved])
        active_alerts = total_alerts - resolved_alerts

        return {
            "alert_summary": {
                "total_alerts": total_alerts,
                "active_alerts": active_alerts,
                "resolved_alerts": resolved_alerts,
                "resolution_rate": (resolved_alerts / total_alerts * 100) if total_alerts > 0 else 100,
            },
            "alerts_by_type": {alert_type: len(alerts) for alert_type, alerts in alerts_by_type.items()},
            "alerts_by_component": {component: len(alerts) for component, alerts in alerts_by_component.items()},
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type,
                    "message": alert.message,
                    "component": alert.component,
                    "timestamp": alert.timestamp,
                    "resolved": alert.resolved,
                    "age_minutes": (time.time() - alert.timestamp) / 60,
                }
                for alert in recent_alerts
            ],
        }

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        current_time = time.time()

        # Calculate report period
        if self._metrics_history:
            start_time = self._metrics_history[0].uptime_seconds + self._start_time
            report_period_hours = (current_time - start_time) / 3600
        else:
            report_period_hours = 0

        # Calculate average metrics
        if self._metrics_history:
            avg_efficiency = statistics.mean(m.pipeline_efficiency for m in self._metrics_history)
            avg_calls_per_second = statistics.mean(m.calls_per_second for m in self._metrics_history)
            avg_files_per_second = statistics.mean(m.files_per_second for m in self._metrics_history)
            avg_memory = statistics.mean(m.memory_usage_mb for m in self._metrics_history)
            avg_cpu = statistics.mean(m.cpu_percent for m in self._metrics_history)
            avg_cache_hit_rate = statistics.mean(m.cache_hit_rate for m in self._metrics_history)
            avg_error_rate = statistics.mean(m.error_rate for m in self._metrics_history)
        else:
            avg_efficiency = avg_calls_per_second = avg_files_per_second = 0
            avg_memory = avg_cpu = avg_cache_hit_rate = avg_error_rate = 0

        # Get monitoring service report
        monitoring_report = self.monitoring_service.get_performance_report()

        report = {
            "report_metadata": {
                "generated_at": current_time,
                "generated_at_iso": datetime.fromtimestamp(current_time).isoformat(),
                "report_period_hours": report_period_hours,
                "dashboard_uptime_hours": (current_time - self._start_time) / 3600,
            },
            "performance_summary": {
                "average_pipeline_efficiency": avg_efficiency,
                "average_calls_per_second": avg_calls_per_second,
                "average_files_per_second": avg_files_per_second,
                "average_memory_usage_mb": avg_memory,
                "average_cpu_percent": avg_cpu,
                "average_cache_hit_rate": avg_cache_hit_rate,
                "average_error_rate": avg_error_rate,
            },
            "current_state": monitoring_report["current_snapshot"],
            "component_performance": monitoring_report["component_performance"],
            "system_metrics": monitoring_report["system_metrics"],
            "alert_summary": self.get_alert_dashboard()["alert_summary"],
            "performance_trends": monitoring_report.get("performance_trends", {}),
            "optimization_recommendations": monitoring_report["optimization_recommendations"],
        }

        return report

    def get_performance_forecast(self, hours_ahead: int = 24) -> dict[str, Any]:
        """Generate performance forecast based on trends."""
        if len(self._metrics_history) < 10:
            return {"error": "Insufficient data for forecasting"}

        # Simple linear trend analysis
        recent_metrics = self._metrics_history[-20:]  # Last 20 data points

        # Calculate trends
        efficiency_trend = self._calculate_linear_trend([m.pipeline_efficiency for m in recent_metrics])
        calls_trend = self._calculate_linear_trend([m.calls_per_second for m in recent_metrics])
        memory_trend = self._calculate_linear_trend([m.memory_usage_mb for m in recent_metrics])

        # Project trends forward
        current_efficiency = recent_metrics[-1].pipeline_efficiency
        current_calls = recent_metrics[-1].calls_per_second
        current_memory = recent_metrics[-1].memory_usage_mb

        forecast_efficiency = current_efficiency + (efficiency_trend * hours_ahead)
        forecast_calls = current_calls + (calls_trend * hours_ahead)
        forecast_memory = current_memory + (memory_trend * hours_ahead)

        return {
            "forecast_period_hours": hours_ahead,
            "current_metrics": {
                "pipeline_efficiency": current_efficiency,
                "calls_per_second": current_calls,
                "memory_usage_mb": current_memory,
            },
            "forecasted_metrics": {
                "pipeline_efficiency": max(0, min(100, forecast_efficiency)),
                "calls_per_second": max(0, forecast_calls),
                "memory_usage_mb": max(0, forecast_memory),
            },
            "trends": {
                "efficiency_trend_per_hour": efficiency_trend,
                "calls_trend_per_hour": calls_trend,
                "memory_trend_per_hour": memory_trend,
            },
            "warnings": self._generate_forecast_warnings(forecast_efficiency, forecast_calls, forecast_memory),
        }

    def _calculate_linear_trend(self, values: list[float]) -> float:
        """Calculate linear trend (slope) for a series of values."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_values = list(range(n))

        # Calculate linear regression slope
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values, strict=False))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _generate_forecast_warnings(self, forecast_efficiency: float, forecast_calls: float, forecast_memory: float) -> list[str]:
        """Generate warnings based on forecasted metrics."""
        warnings = []

        if forecast_efficiency < 60:
            warnings.append("Pipeline efficiency may drop below 60% in the forecast period")

        if forecast_calls < 1:
            warnings.append("Call detection rate may become very low in the forecast period")

        if forecast_memory > 2048:
            warnings.append("Memory usage may exceed 2GB in the forecast period")

        return warnings

    async def shutdown(self):
        """Shutdown the performance dashboard service."""
        self.logger.info("Shutting down PerformanceDashboardService")
        await self.stop_dashboard()
        self.logger.info("PerformanceDashboardService shutdown complete")
