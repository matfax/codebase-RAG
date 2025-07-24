"""
Cache memory usage reporting service.

This module provides comprehensive cache memory usage reporting capabilities including
real-time dashboards, historical analytics, trend analysis, export functionality,
alerting, and integration with leak detection and optimization systems.
"""

import asyncio
import csv
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Union

from ..utils.memory_utils import (
    CacheMemoryEvent,
    get_memory_stats,
    get_system_memory_pressure,
    get_total_cache_memory_usage,
)
from .cache_memory_leak_detector import CacheMemoryLeakDetector, get_leak_detector
from .cache_memory_profiler import CacheMemoryProfiler, get_memory_profiler


class ReportFormat(Enum):
    """Report export formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"


class ReportType(Enum):
    """Types of memory reports."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    TREND_ANALYSIS = "trend_analysis"
    LEAK_ANALYSIS = "leak_analysis"
    OPTIMIZATION = "optimization"
    DASHBOARD = "dashboard"
    ALERT = "alert"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MemoryReport:
    """Comprehensive memory usage report."""

    report_id: str
    report_type: ReportType
    cache_name: str | None
    timestamp: float
    start_time: float
    end_time: float
    summary: dict[str, Any] = field(default_factory=dict)
    detailed_metrics: dict[str, Any] = field(default_factory=dict)
    trends: dict[str, Any] = field(default_factory=dict)
    leaks: dict[str, Any] = field(default_factory=dict)
    optimizations: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        """Report duration in hours."""
        return (self.end_time - self.start_time) / 3600.0

    @property
    def age_hours(self) -> float:
        """Report age in hours."""
        return (time.time() - self.timestamp) / 3600.0


@dataclass
class MemoryAlert:
    """Memory usage alert."""

    alert_id: str
    cache_name: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: float
    current_value: float
    threshold_value: float
    metric_name: str
    recommendations: list[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_minutes(self) -> float:
        """Alert age in minutes."""
        return (time.time() - self.timestamp) / 60.0

    @property
    def duration_minutes(self) -> float | None:
        """Alert duration in minutes if resolved."""
        if self.resolution_time:
            return (self.resolution_time - self.timestamp) / 60.0
        return None


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""

    timestamp: float
    system_metrics: dict[str, Any] = field(default_factory=dict)
    cache_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    leak_summary: dict[str, Any] = field(default_factory=dict)
    optimization_summary: dict[str, Any] = field(default_factory=dict)
    active_alerts: list[MemoryAlert] = field(default_factory=list)
    trends: dict[str, Any] = field(default_factory=dict)
    performance_indicators: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportingConfig:
    """Configuration for memory reporting."""

    # Report generation settings
    enable_automatic_reports: bool = True
    report_generation_interval_hours: int = 1
    detailed_report_interval_hours: int = 24

    # Dashboard settings
    dashboard_update_interval_seconds: int = 30
    dashboard_history_hours: int = 24

    # Alert settings
    enable_alerts: bool = True
    alert_check_interval_seconds: int = 60

    # Memory thresholds for alerts
    memory_usage_warning_threshold_mb: float = 1000.0
    memory_usage_critical_threshold_mb: float = 2000.0
    memory_growth_rate_warning_mb_per_hour: float = 100.0
    memory_growth_rate_critical_mb_per_hour: float = 500.0

    # Export settings
    enable_export: bool = True
    export_directory: str = "memory_reports"
    max_export_files: int = 100

    # Data retention
    max_reports: int = 1000
    max_alerts: int = 5000
    report_retention_hours: int = 168  # 1 week
    alert_retention_hours: int = 720  # 30 days


class CacheMemoryReporter:
    """
    Comprehensive cache memory usage reporting service.

    This service provides:
    - Real-time memory usage dashboards
    - Historical trend analysis and reporting
    - Memory leak detection integration
    - Optimization recommendation integration
    - Alert generation and management
    - Export capabilities for reports
    - Automated reporting schedules
    """

    def __init__(self, config: ReportingConfig | None = None):
        self.config = config or ReportingConfig()
        self.logger = logging.getLogger(__name__)

        # Service dependencies
        self.leak_detector: CacheMemoryLeakDetector | None = None
        self.memory_profiler: CacheMemoryProfiler | None = None

        # Reporting state
        self.is_running = False

        # Generated reports
        self.reports: deque[MemoryReport] = deque(maxlen=self.config.max_reports)
        self.reports_by_id: dict[str, MemoryReport] = {}

        # Active alerts
        self.alerts: deque[MemoryAlert] = deque(maxlen=self.config.max_alerts)
        self.alerts_by_id: dict[str, MemoryAlert] = {}
        self.active_alerts: dict[str, MemoryAlert] = {}

        # Dashboard data
        self.dashboard_history: deque[DashboardMetrics] = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.current_dashboard: DashboardMetrics | None = None

        # Background tasks
        self.dashboard_task: asyncio.Task | None = None
        self.report_task: asyncio.Task | None = None
        self.alert_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None

        # Report statistics
        self.report_stats = {
            "total_reports_generated": 0,
            "reports_by_type": defaultdict(int),
            "total_alerts_generated": 0,
            "alerts_by_severity": defaultdict(int),
            "export_count": 0,
        }

    async def initialize(self) -> None:
        """Initialize the memory reporter."""
        self.is_running = True

        # Initialize service dependencies
        self.leak_detector = await get_leak_detector()
        self.memory_profiler = await get_memory_profiler()

        # Create export directory
        if self.config.enable_export:
            export_path = Path(self.config.export_directory)
            export_path.mkdir(exist_ok=True)

        # Start background tasks
        if self.config.dashboard_update_interval_seconds > 0:
            self.dashboard_task = asyncio.create_task(self._dashboard_update_loop())

        if self.config.enable_automatic_reports:
            self.report_task = asyncio.create_task(self._report_generation_loop())

        if self.config.enable_alerts:
            self.alert_task = asyncio.create_task(self._alert_monitoring_loop())

        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info("Cache memory reporter initialized")

    async def shutdown(self) -> None:
        """Shutdown the memory reporter."""
        self.is_running = False

        # Cancel background tasks
        tasks = [self.dashboard_task, self.report_task, self.alert_task, self.cleanup_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.logger.info("Cache memory reporter shutdown")

    async def generate_report(
        self,
        report_type: ReportType,
        cache_name: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        include_recommendations: bool = True,
    ) -> MemoryReport:
        """Generate a comprehensive memory usage report."""
        current_time = time.time()
        start_time = start_time or (current_time - 3600)  # Default to last hour
        end_time = end_time or current_time

        report_id = f"{report_type.value}_{cache_name or 'all'}_{int(current_time)}"

        report = MemoryReport(
            report_id=report_id,
            report_type=report_type,
            cache_name=cache_name,
            timestamp=current_time,
            start_time=start_time,
            end_time=end_time,
        )

        # Generate report sections based on type
        if report_type in [ReportType.SUMMARY, ReportType.DETAILED, ReportType.DASHBOARD]:
            report.summary = await self._generate_summary_metrics(cache_name, start_time, end_time)

        if report_type in [ReportType.DETAILED, ReportType.DASHBOARD]:
            report.detailed_metrics = await self._generate_detailed_metrics(cache_name, start_time, end_time)

        if report_type in [ReportType.TREND_ANALYSIS, ReportType.DETAILED, ReportType.DASHBOARD]:
            report.trends = await self._generate_trend_analysis(cache_name, start_time, end_time)

        if report_type in [ReportType.LEAK_ANALYSIS, ReportType.DETAILED]:
            report.leaks = await self._generate_leak_analysis(cache_name)

        if report_type in [ReportType.OPTIMIZATION, ReportType.DETAILED]:
            report.optimizations = await self._generate_optimization_analysis(cache_name)

        if include_recommendations:
            report.recommendations = await self._generate_recommendations(report)

        # Add alerts if relevant
        if cache_name:
            report.alerts = [asdict(alert) for alert in self.active_alerts.values() if alert.cache_name == cache_name]
        else:
            report.alerts = [asdict(alert) for alert in self.active_alerts.values()]

        # Store report
        self.reports.append(report)
        self.reports_by_id[report_id] = report

        # Update statistics
        self.report_stats["total_reports_generated"] += 1
        self.report_stats["reports_by_type"][report_type.value] += 1

        self.logger.info(f"Generated {report_type.value} report for cache '{cache_name or 'all'}' (ID: {report_id})")

        return report

    async def _generate_summary_metrics(self, cache_name: str | None, start_time: float, end_time: float) -> dict[str, Any]:
        """Generate summary metrics."""
        current_memory = get_total_cache_memory_usage()
        memory_stats = get_memory_stats()

        # Get cache-specific metrics if requested
        if cache_name and self.memory_profiler:
            cache_profile = self.memory_profiler.get_cache_profile(cache_name)
            cache_trends = self.memory_profiler.get_memory_trend(cache_name, window_minutes=60)
        else:
            cache_profile = None
            cache_trends = self.memory_profiler.get_memory_trend(window_minutes=60) if self.memory_profiler else {}

        summary = {
            "timestamp": time.time(),
            "cache_name": cache_name,
            "duration_hours": (end_time - start_time) / 3600.0,
            "current_memory": {
                "total_cache_memory_mb": current_memory,
                "system_memory_usage_mb": memory_stats["rss_mb"],
                "system_memory_available_mb": memory_stats["system_memory"]["available_mb"],
                "memory_pressure": get_system_memory_pressure(),
            },
        }

        if cache_profile:
            summary["cache_specific"] = {
                "cache_name": cache_name,
                "current_memory_mb": cache_profile["memory_usage"]["current_memory_mb"],
                "peak_memory_mb": cache_profile["memory_usage"]["peak_memory_mb"],
                "total_allocations": cache_profile["allocation_stats"]["total_allocations"],
                "memory_efficiency": cache_profile["efficiency_metrics"]["memory_efficiency_ratio"],
                "is_active": cache_profile["is_active"],
            }

        if cache_trends and not cache_trends.get("error"):
            summary["trends"] = {
                "trend_direction": cache_trends.get("trend", "stable"),
                "avg_memory_mb": cache_trends.get("avg_memory_mb", 0),
                "min_memory_mb": cache_trends.get("min_memory_mb", 0),
                "max_memory_mb": cache_trends.get("max_memory_mb", 0),
            }

        return summary

    async def _generate_detailed_metrics(self, cache_name: str | None, start_time: float, end_time: float) -> dict[str, Any]:
        """Generate detailed metrics."""
        detailed = {}

        if self.memory_profiler:
            # Get allocation patterns
            allocation_patterns = self.memory_profiler.get_allocation_patterns(cache_name, window_minutes=60)
            if not allocation_patterns.get("error"):
                detailed["allocation_patterns"] = allocation_patterns

            # Get memory hotspots
            hotspots = self.memory_profiler.get_memory_hotspots(cache_name, min_allocations=3)
            detailed["memory_hotspots"] = hotspots

            # Get performance metrics
            performance = self.memory_profiler.get_performance_metrics()
            detailed["performance_metrics"] = performance

            # Get cache profile if specific cache requested
            if cache_name:
                cache_profile = self.memory_profiler.get_cache_profile(cache_name)
                if cache_profile:
                    detailed["cache_profile"] = cache_profile

        return detailed

    async def _generate_trend_analysis(self, cache_name: str | None, start_time: float, end_time: float) -> dict[str, Any]:
        """Generate trend analysis."""
        trends = {}

        if self.memory_profiler:
            # Short-term trends (1 hour)
            short_trend = self.memory_profiler.get_memory_trend(cache_name, window_minutes=60)
            if not short_trend.get("error"):
                trends["short_term"] = short_trend

            # Medium-term trends (6 hours)
            medium_trend = self.memory_profiler.get_memory_trend(cache_name, window_minutes=360)
            if not medium_trend.get("error"):
                trends["medium_term"] = medium_trend

            # Long-term trends (24 hours)
            long_trend = self.memory_profiler.get_memory_trend(cache_name, window_minutes=1440)
            if not long_trend.get("error"):
                trends["long_term"] = long_trend

        # Add dashboard history trends if available
        if self.dashboard_history:
            dashboard_trends = self._analyze_dashboard_trends(cache_name)
            trends["dashboard_trends"] = dashboard_trends

        return trends

    async def _generate_leak_analysis(self, cache_name: str | None) -> dict[str, Any]:
        """Generate leak analysis."""
        leaks = {}

        if self.leak_detector:
            # Get leak summary
            leak_summary = self.leak_detector.get_leak_summary(cache_name)
            leaks["summary"] = leak_summary

            # Get detection statistics
            detection_stats = self.leak_detector.get_detection_stats()
            leaks["detection_statistics"] = detection_stats

            # Get retention analysis if cache-specific
            if cache_name:
                retention_analysis = self.leak_detector.get_retention_analysis(cache_name)
                if retention_analysis:
                    leaks["retention_analysis"] = retention_analysis

        return leaks

    async def _generate_optimization_analysis(self, cache_name: str | None) -> dict[str, Any]:
        """Generate optimization analysis."""
        optimizations = {}

        # TODO: Integrate with cache memory optimizer when available
        # For now, provide basic optimization insights based on available data

        if self.memory_profiler and cache_name:
            cache_profile = self.memory_profiler.get_cache_profile(cache_name)
            if cache_profile:
                memory_efficiency = cache_profile["efficiency_metrics"]["memory_efficiency_ratio"]
                turnover_ratio = cache_profile["allocation_stats"]["memory_turnover_ratio"]

                optimization_recommendations = []
                if memory_efficiency < 0.7:
                    optimization_recommendations.append("Consider implementing more aggressive eviction policies")
                if turnover_ratio < 0.5:
                    optimization_recommendations.append("Low memory turnover detected - review cache TTL settings")

                optimizations["basic_analysis"] = {
                    "memory_efficiency": memory_efficiency,
                    "turnover_ratio": turnover_ratio,
                    "recommendations": optimization_recommendations,
                }

        return optimizations

    async def _generate_recommendations(self, report: MemoryReport) -> list[str]:
        """Generate actionable recommendations based on report data."""
        recommendations = []

        # Memory usage recommendations
        if "current_memory" in report.summary:
            current_mb = report.summary["current_memory"]["total_cache_memory_mb"]
            if current_mb > self.config.memory_usage_critical_threshold_mb:
                recommendations.append("CRITICAL: Cache memory usage exceeds critical threshold - immediate action required")
            elif current_mb > self.config.memory_usage_warning_threshold_mb:
                recommendations.append("WARNING: Cache memory usage is high - consider optimization")

        # Trend-based recommendations
        if "trends" in report.summary and report.summary["trends"].get("trend_direction") == "increasing":
            recommendations.append("Memory usage is trending upward - monitor for potential leaks")

        # Leak-based recommendations
        if report.leaks and report.leaks.get("summary", {}).get("total_leaks", 0) > 0:
            recommendations.append("Memory leaks detected - review leak analysis section for details")

        # Performance recommendations
        if "performance_metrics" in report.detailed_metrics:
            perf = report.detailed_metrics["performance_metrics"]
            if perf.get("allocation_times", {}).get("avg", 0) > 0.1:  # 100ms average
                recommendations.append("High allocation times detected - consider optimizing allocation patterns")

        # Hotspot recommendations
        if "memory_hotspots" in report.detailed_metrics and report.detailed_metrics["memory_hotspots"]:
            recommendations.append("Memory hotspots detected - review frequently allocated keys")

        return recommendations

    def _analyze_dashboard_trends(self, cache_name: str | None) -> dict[str, Any]:
        """Analyze trends from dashboard history."""
        if not self.dashboard_history:
            return {"error": "No dashboard history available"}

        # Extract memory values over time
        if cache_name:
            memory_values = [metrics.cache_metrics.get(cache_name, {}).get("current_memory_mb", 0) for metrics in self.dashboard_history]
        else:
            memory_values = [
                sum(cache_data.get("current_memory_mb", 0) for cache_data in metrics.cache_metrics.values())
                for metrics in self.dashboard_history
            ]

        if not memory_values or all(v == 0 for v in memory_values):
            return {"error": "No memory data available"}

        # Calculate trend metrics
        start_value = memory_values[0]
        end_value = memory_values[-1]
        max_value = max(memory_values)
        min_value = min(memory_values)
        avg_value = sum(memory_values) / len(memory_values)

        # Calculate growth rate
        time_span_hours = len(memory_values) * (self.config.dashboard_update_interval_seconds / 3600.0)
        growth_rate_mb_per_hour = (end_value - start_value) / time_span_hours if time_span_hours > 0 else 0

        return {
            "data_points": len(memory_values),
            "time_span_hours": time_span_hours,
            "start_memory_mb": start_value,
            "end_memory_mb": end_value,
            "min_memory_mb": min_value,
            "max_memory_mb": max_value,
            "avg_memory_mb": avg_value,
            "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
            "trend": "increasing" if growth_rate_mb_per_hour > 1 else "decreasing" if growth_rate_mb_per_hour < -1 else "stable",
        }

    async def get_real_time_dashboard(self) -> DashboardMetrics:
        """Get real-time dashboard metrics."""
        current_time = time.time()

        # Get system metrics
        memory_stats = get_memory_stats()
        system_metrics = {
            "timestamp": current_time,
            "total_cache_memory_mb": get_total_cache_memory_usage(),
            "system_memory_mb": memory_stats["rss_mb"],
            "system_memory_available_mb": memory_stats["system_memory"]["available_mb"],
            "memory_pressure": get_system_memory_pressure(),
        }

        # Get cache-specific metrics
        cache_metrics = {}
        if self.memory_profiler:
            # Get all active cache profiles
            for cache_name in list(self.memory_profiler.active_profiles.keys()):
                profile = self.memory_profiler.get_cache_profile(cache_name)
                if profile:
                    cache_metrics[cache_name] = {
                        "current_memory_mb": profile["memory_usage"]["current_memory_mb"],
                        "peak_memory_mb": profile["memory_usage"]["peak_memory_mb"],
                        "total_allocations": profile["allocation_stats"]["total_allocations"],
                        "memory_efficiency": profile["efficiency_metrics"]["memory_efficiency_ratio"],
                        "recent_events": profile["recent_events"],
                    }

        # Get leak summary
        leak_summary = {}
        if self.leak_detector:
            leak_summary = self.leak_detector.get_leak_summary()

        # Get optimization summary (placeholder for now)
        optimization_summary = {"available": False}

        # Get active alerts
        active_alerts = list(self.active_alerts.values())

        # Generate trends from recent dashboard history
        trends = {}
        if len(self.dashboard_history) >= 2:
            recent_metrics = list(self.dashboard_history)[-10:]  # Last 10 data points
            if recent_metrics:
                cache_trends = {}
                for cache_name in cache_metrics:
                    cache_values = [m.cache_metrics.get(cache_name, {}).get("current_memory_mb", 0) for m in recent_metrics]
                    if cache_values:
                        cache_trends[cache_name] = {
                            "trend": "increasing" if cache_values[-1] > cache_values[0] else "decreasing",
                            "avg_mb": sum(cache_values) / len(cache_values),
                        }
                trends["cache_trends"] = cache_trends

        # Calculate performance indicators
        performance_indicators = {
            "memory_utilization_percent": min(100.0, (system_metrics["total_cache_memory_mb"] / 1000.0) * 100),
            "active_caches": len(cache_metrics),
            "active_alerts": len(active_alerts),
            "memory_growth_rate": trends.get("system_trend", {}).get("growth_rate_mb_per_hour", 0),
        }

        dashboard = DashboardMetrics(
            timestamp=current_time,
            system_metrics=system_metrics,
            cache_metrics=cache_metrics,
            leak_summary=leak_summary,
            optimization_summary=optimization_summary,
            active_alerts=active_alerts,
            trends=trends,
            performance_indicators=performance_indicators,
        )

        self.current_dashboard = dashboard
        return dashboard

    async def check_and_generate_alerts(self) -> list[MemoryAlert]:
        """Check for alert conditions and generate alerts."""
        new_alerts = []

        # Get current memory metrics
        total_memory = get_total_cache_memory_usage()

        # Check system-wide memory thresholds
        if total_memory > self.config.memory_usage_critical_threshold_mb:
            alert = await self._create_alert(
                cache_name="system",
                alert_type="memory_usage",
                severity=AlertSeverity.CRITICAL,
                message=f"System cache memory usage ({total_memory:.1f}MB) exceeds critical threshold",
                current_value=total_memory,
                threshold_value=self.config.memory_usage_critical_threshold_mb,
                metric_name="total_cache_memory_mb",
            )
            if alert:
                new_alerts.append(alert)
        elif total_memory > self.config.memory_usage_warning_threshold_mb:
            alert = await self._create_alert(
                cache_name="system",
                alert_type="memory_usage",
                severity=AlertSeverity.WARNING,
                message=f"System cache memory usage ({total_memory:.1f}MB) exceeds warning threshold",
                current_value=total_memory,
                threshold_value=self.config.memory_usage_warning_threshold_mb,
                metric_name="total_cache_memory_mb",
            )
            if alert:
                new_alerts.append(alert)

        # Check cache-specific thresholds
        if self.memory_profiler:
            for cache_name in list(self.memory_profiler.active_profiles.keys()):
                cache_alerts = await self._check_cache_alerts(cache_name)
                new_alerts.extend(cache_alerts)

        # Check memory growth rate
        if len(self.dashboard_history) >= 12:  # Need at least 6 minutes of data
            growth_alerts = await self._check_growth_rate_alerts()
            new_alerts.extend(growth_alerts)

        # Check for leak alerts
        if self.leak_detector:
            leak_alerts = await self._check_leak_alerts()
            new_alerts.extend(leak_alerts)

        return new_alerts

    async def _create_alert(
        self,
        cache_name: str,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        current_value: float,
        threshold_value: float,
        metric_name: str,
        recommendations: list[str] | None = None,
    ) -> MemoryAlert | None:
        """Create a new alert if it doesn't already exist."""
        alert_key = f"{cache_name}:{alert_type}:{metric_name}"

        # Check if similar alert already exists
        if alert_key in self.active_alerts:
            existing = self.active_alerts[alert_key]
            # Update existing alert
            existing.current_value = current_value
            existing.timestamp = time.time()
            return None

        alert_id = f"alert_{int(time.time())}_{cache_name}_{alert_type}"

        alert = MemoryAlert(
            alert_id=alert_id,
            cache_name=cache_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time(),
            current_value=current_value,
            threshold_value=threshold_value,
            metric_name=metric_name,
            recommendations=recommendations or [],
        )

        # Store alert
        self.alerts.append(alert)
        self.alerts_by_id[alert_id] = alert
        self.active_alerts[alert_key] = alert

        # Update statistics
        self.report_stats["total_alerts_generated"] += 1
        self.report_stats["alerts_by_severity"][severity.value] += 1

        self.logger.warning(f"Generated {severity.value} alert for {cache_name}: {message}")

        return alert

    async def _check_cache_alerts(self, cache_name: str) -> list[MemoryAlert]:
        """Check for cache-specific alerts."""
        alerts = []

        if not self.memory_profiler:
            return alerts

        cache_profile = self.memory_profiler.get_cache_profile(cache_name)
        if not cache_profile:
            return alerts

        current_memory = cache_profile["memory_usage"]["current_memory_mb"]
        memory_efficiency = cache_profile["efficiency_metrics"]["memory_efficiency_ratio"]

        # Check cache memory usage
        if current_memory > 500:  # Cache-specific threshold
            alert = await self._create_alert(
                cache_name=cache_name,
                alert_type="cache_memory_usage",
                severity=AlertSeverity.WARNING,
                message=f"Cache '{cache_name}' memory usage ({current_memory:.1f}MB) is high",
                current_value=current_memory,
                threshold_value=500.0,
                metric_name="current_memory_mb",
                recommendations=["Review cache eviction policies", "Consider reducing cache size limits"],
            )
            if alert:
                alerts.append(alert)

        # Check memory efficiency
        if memory_efficiency < 0.5:
            alert = await self._create_alert(
                cache_name=cache_name,
                alert_type="memory_efficiency",
                severity=AlertSeverity.WARNING,
                message=f"Cache '{cache_name}' has low memory efficiency ({memory_efficiency:.2f})",
                current_value=memory_efficiency,
                threshold_value=0.5,
                metric_name="memory_efficiency_ratio",
                recommendations=["Review memory allocation patterns", "Optimize data structures"],
            )
            if alert:
                alerts.append(alert)

        return alerts

    async def _check_growth_rate_alerts(self) -> list[MemoryAlert]:
        """Check for memory growth rate alerts."""
        alerts = []

        # Calculate growth rate from recent history
        recent_metrics = list(self.dashboard_history)[-12:]  # Last 6 minutes
        if len(recent_metrics) < 2:
            return alerts

        start_memory = recent_metrics[0].system_metrics["total_cache_memory_mb"]
        end_memory = recent_metrics[-1].system_metrics["total_cache_memory_mb"]
        time_span_hours = len(recent_metrics) * (self.config.dashboard_update_interval_seconds / 3600.0)

        if time_span_hours > 0:
            growth_rate = (end_memory - start_memory) / time_span_hours

            if growth_rate > self.config.memory_growth_rate_critical_mb_per_hour:
                alert = await self._create_alert(
                    cache_name="system",
                    alert_type="memory_growth_rate",
                    severity=AlertSeverity.CRITICAL,
                    message=f"Critical memory growth rate detected: {growth_rate:.1f}MB/hour",
                    current_value=growth_rate,
                    threshold_value=self.config.memory_growth_rate_critical_mb_per_hour,
                    metric_name="growth_rate_mb_per_hour",
                    recommendations=["Investigate memory leaks", "Review allocation patterns", "Consider emergency cache clearing"],
                )
                if alert:
                    alerts.append(alert)
            elif growth_rate > self.config.memory_growth_rate_warning_mb_per_hour:
                alert = await self._create_alert(
                    cache_name="system",
                    alert_type="memory_growth_rate",
                    severity=AlertSeverity.WARNING,
                    message=f"High memory growth rate detected: {growth_rate:.1f}MB/hour",
                    current_value=growth_rate,
                    threshold_value=self.config.memory_growth_rate_warning_mb_per_hour,
                    metric_name="growth_rate_mb_per_hour",
                    recommendations=["Monitor memory usage closely", "Review cache policies"],
                )
                if alert:
                    alerts.append(alert)

        return alerts

    async def _check_leak_alerts(self) -> list[MemoryAlert]:
        """Check for leak-based alerts."""
        alerts = []

        if not self.leak_detector:
            return alerts

        leak_summary = self.leak_detector.get_leak_summary()
        total_leaks = leak_summary.get("total_leaks", 0)

        if total_leaks > 0:
            high_severity_leaks = leak_summary.get("leaks_by_severity", {}).get("high", 0)
            critical_leaks = leak_summary.get("leaks_by_severity", {}).get("critical", 0)

            if critical_leaks > 0:
                alert = await self._create_alert(
                    cache_name="system",
                    alert_type="memory_leaks",
                    severity=AlertSeverity.CRITICAL,
                    message=f"{critical_leaks} critical memory leaks detected",
                    current_value=critical_leaks,
                    threshold_value=0,
                    metric_name="critical_leaks",
                    recommendations=["Review leak analysis report", "Investigate memory allocation patterns"],
                )
                if alert:
                    alerts.append(alert)
            elif high_severity_leaks > 0:
                alert = await self._create_alert(
                    cache_name="system",
                    alert_type="memory_leaks",
                    severity=AlertSeverity.WARNING,
                    message=f"{high_severity_leaks} high-severity memory leaks detected",
                    current_value=high_severity_leaks,
                    threshold_value=0,
                    metric_name="high_severity_leaks",
                    recommendations=["Monitor memory leaks", "Consider leak remediation"],
                )
                if alert:
                    alerts.append(alert)

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts_by_id:
            alert = self.alerts_by_id[alert_id]
            alert.acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts_by_id:
            alert = self.alerts_by_id[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()

            # Remove from active alerts
            alert_key = f"{alert.cache_name}:{alert.alert_type}:{alert.metric_name}"
            self.active_alerts.pop(alert_key, None)

            self.logger.info(f"Alert {alert_id} resolved")
            return True
        return False

    async def export_report(self, report_id: str, format_type: ReportFormat, output_path: str | None = None) -> str:
        """Export a report in the specified format."""
        if report_id not in self.reports_by_id:
            raise ValueError(f"Report {report_id} not found")

        report = self.reports_by_id[report_id]

        if not output_path:
            timestamp = datetime.fromtimestamp(report.timestamp).strftime("%Y%m%d_%H%M%S")
            filename = f"{report.report_type.value}_{report.cache_name or 'all'}_{timestamp}.{format_type.value}"
            output_path = str(Path(self.config.export_directory) / filename)

        if format_type == ReportFormat.JSON:
            content = json.dumps(asdict(report), indent=2, default=str)
        elif format_type == ReportFormat.CSV:
            content = await self._export_csv(report)
        elif format_type == ReportFormat.HTML:
            content = await self._export_html(report)
        elif format_type == ReportFormat.TEXT:
            content = await self._export_text(report)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

        # Update statistics
        self.report_stats["export_count"] += 1

        self.logger.info(f"Exported report {report_id} to {output_path}")
        return output_path

    async def _export_csv(self, report: MemoryReport) -> str:
        """Export report as CSV."""
        output = StringIO()
        writer = csv.writer(output)

        # Write header information
        writer.writerow(["Report Type", "Cache Name", "Timestamp", "Duration Hours"])
        writer.writerow([report.report_type.value, report.cache_name or "All", report.timestamp, report.duration_hours])
        writer.writerow([])

        # Write summary metrics
        if report.summary:
            writer.writerow(["Summary Metrics"])
            for key, value in report.summary.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        writer.writerow([f"{key}.{subkey}", subvalue])
                else:
                    writer.writerow([key, value])
            writer.writerow([])

        # Write recommendations
        if report.recommendations:
            writer.writerow(["Recommendations"])
            for rec in report.recommendations:
                writer.writerow([rec])

        return output.getvalue()

    async def _export_html(self, report: MemoryReport) -> str:
        """Export report as HTML."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Memory Report - {report.report_type.value}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin: 10px 0; }}
                .alert {{ background-color: #ffebee; padding: 10px; margin: 5px 0; border-left: 4px solid #f44336; }}
                .recommendation {{ background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-left: 4px solid #2196f3; }}
            </style>
        </head>
        <body>
            <h1>Memory Usage Report</h1>
            <div class="metric">
                <strong>Report Type:</strong> {report.report_type.value}<br>
                <strong>Cache:</strong> {report.cache_name or "All Caches"}<br>
                <strong>Generated:</strong> {datetime.fromtimestamp(report.timestamp)}<br>
                <strong>Duration:</strong> {report.duration_hours:.2f} hours
            </div>
        """

        # Add summary section
        if report.summary:
            html += "<h2>Summary</h2>"
            html += self._dict_to_html_table(report.summary)

        # Add alerts section
        if report.alerts:
            html += "<h2>Active Alerts</h2>"
            for alert in report.alerts:
                html += f'<div class="alert"><strong>{alert["severity"].upper()}:</strong> {alert["message"]}</div>'

        # Add recommendations section
        if report.recommendations:
            html += "<h2>Recommendations</h2>"
            for rec in report.recommendations:
                html += f'<div class="recommendation">{rec}</div>'

        html += "</body></html>"
        return html

    def _dict_to_html_table(self, data: dict, max_depth: int = 3, current_depth: int = 0) -> str:
        """Convert dictionary to HTML table."""
        if current_depth >= max_depth:
            return str(data)

        html = "<table>"
        for key, value in data.items():
            html += f"<tr><th>{key}</th><td>"
            if isinstance(value, dict):
                html += self._dict_to_html_table(value, max_depth, current_depth + 1)
            else:
                html += str(value)
            html += "</td></tr>"
        html += "</table>"
        return html

    async def _export_text(self, report: MemoryReport) -> str:
        """Export report as plain text."""
        lines = [
            f"Memory Usage Report - {report.report_type.value}",
            "=" * 50,
            f"Cache: {report.cache_name or 'All Caches'}",
            f"Generated: {datetime.fromtimestamp(report.timestamp)}",
            f"Duration: {report.duration_hours:.2f} hours",
            "",
        ]

        if report.summary:
            lines.append("SUMMARY")
            lines.append("-" * 20)
            lines.extend(self._dict_to_text_lines(report.summary))
            lines.append("")

        if report.alerts:
            lines.append("ACTIVE ALERTS")
            lines.append("-" * 20)
            for alert in report.alerts:
                lines.append(f"[{alert['severity'].upper()}] {alert['message']}")
            lines.append("")

        if report.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 20)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _dict_to_text_lines(self, data: dict, indent: int = 0) -> list[str]:
        """Convert dictionary to text lines."""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{'  ' * indent}{key}:")
                lines.extend(self._dict_to_text_lines(value, indent + 1))
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return lines

    async def _dashboard_update_loop(self) -> None:
        """Background task for updating dashboard metrics."""
        while self.is_running:
            try:
                dashboard = await self.get_real_time_dashboard()
                self.dashboard_history.append(dashboard)

                await asyncio.sleep(self.config.dashboard_update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")

    async def _report_generation_loop(self) -> None:
        """Background task for automatic report generation."""
        while self.is_running:
            try:
                # Generate summary report
                await self.generate_report(ReportType.SUMMARY)

                # Generate detailed report less frequently
                if time.time() % (self.config.detailed_report_interval_hours * 3600) < 60:
                    await self.generate_report(ReportType.DETAILED)

                await asyncio.sleep(self.config.report_generation_interval_hours * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in report generation loop: {e}")

    async def _alert_monitoring_loop(self) -> None:
        """Background task for monitoring and generating alerts."""
        while self.is_running:
            try:
                new_alerts = await self.check_and_generate_alerts()
                if new_alerts:
                    self.logger.info(f"Generated {len(new_alerts)} new alerts")

                await asyncio.sleep(self.config.alert_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old data."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                current_time = time.time()

                # Clean up old reports
                report_cutoff = current_time - (self.config.report_retention_hours * 3600)
                old_reports = [r for r in self.reports if r.timestamp < report_cutoff]
                for report in old_reports:
                    self.reports.remove(report)
                    self.reports_by_id.pop(report.report_id, None)

                # Clean up old alerts
                alert_cutoff = current_time - (self.config.alert_retention_hours * 3600)
                old_alerts = [a for a in self.alerts if a.timestamp < alert_cutoff]
                for alert in old_alerts:
                    self.alerts.remove(alert)
                    self.alerts_by_id.pop(alert.alert_id, None)

                if old_reports or old_alerts:
                    self.logger.info(f"Cleaned up {len(old_reports)} old reports and {len(old_alerts)} old alerts")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    def get_report_statistics(self) -> dict[str, Any]:
        """Get reporting statistics."""
        return {
            "total_reports": len(self.reports),
            "total_alerts": len(self.alerts),
            "active_alerts": len(self.active_alerts),
            "dashboard_history_points": len(self.dashboard_history),
            "generation_stats": dict(self.report_stats),
            "current_dashboard_available": self.current_dashboard is not None,
        }

    def get_alert_summary(self, include_resolved: bool = False) -> dict[str, Any]:
        """Get alert summary."""
        alerts_to_include = list(self.alerts) if include_resolved else list(self.active_alerts.values())

        summary = {
            "total_alerts": len(alerts_to_include),
            "by_severity": defaultdict(int),
            "by_cache": defaultdict(int),
            "by_type": defaultdict(int),
            "recent_alerts": [],
        }

        for alert in alerts_to_include:
            summary["by_severity"][alert.severity.value] += 1
            summary["by_cache"][alert.cache_name] += 1
            summary["by_type"][alert.alert_type] += 1

        # Get recent alerts (last 10)
        recent_alerts = sorted(alerts_to_include, key=lambda a: a.timestamp, reverse=True)[:10]
        summary["recent_alerts"] = [
            {
                "alert_id": alert.alert_id,
                "cache_name": alert.cache_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "age_minutes": alert.age_minutes,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
            }
            for alert in recent_alerts
        ]

        return summary


# Global reporter instance
_memory_reporter: CacheMemoryReporter | None = None


async def get_memory_reporter(config: ReportingConfig | None = None) -> CacheMemoryReporter:
    """Get the global memory reporter instance."""
    global _memory_reporter
    if _memory_reporter is None:
        _memory_reporter = CacheMemoryReporter(config)
        await _memory_reporter.initialize()
    return _memory_reporter


async def shutdown_memory_reporter() -> None:
    """Shutdown the global memory reporter."""
    global _memory_reporter
    if _memory_reporter:
        await _memory_reporter.shutdown()
        _memory_reporter = None
