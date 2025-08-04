"""
Enhanced Real-Time Performance Monitoring Service - Wave 5.0 Implementation.

This service provides comprehensive real-time performance monitoring capabilities including:
- Real-time metrics collection and aggregation
- Performance alerting and notification system
- Automated performance optimization and tuning
- Performance data visualization and reporting
- Bottleneck identification and analysis
- Optimization recommendations engine

Integration with existing cache performance monitoring for unified performance tracking.
"""

import asyncio
import json
import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

from src.utils.performance_monitor import (
    CachePerformanceMonitor,
    MemoryMonitor,
    RealTimeCacheStatsReporter,
    get_cache_performance_monitor,
    get_real_time_cache_reporter,
)


class PerformanceMetricType(Enum):
    """Types of performance metrics tracked by the system."""

    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""

    metric_id: str
    metric_type: PerformanceMetricType
    component: str
    value: float
    unit: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def age_seconds(self) -> float:
        """Get the age of this metric in seconds."""
        return time.time() - self.timestamp


@dataclass
class PerformanceAlert:
    """Represents a performance alert."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: float
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledgment_time: float | None = None
    resolution_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get the age of this alert in seconds."""
        return time.time() - self.timestamp

    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.ACTIVE


@dataclass
class PerformanceThreshold:
    """Defines a performance threshold for alerting."""

    metric_type: PerformanceMetricType
    component: str
    threshold_value: float
    comparison_operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    severity: AlertSeverity
    enabled: bool = True
    cooldown_seconds: float = 300.0  # 5 minutes default cooldown
    consecutive_violations: int = 1  # Number of consecutive violations required

    def evaluate(self, value: float) -> bool:
        """Evaluate if the value violates this threshold."""
        if not self.enabled:
            return False

        if self.comparison_operator == "gt":
            return value > self.threshold_value
        elif self.comparison_operator == "lt":
            return value < self.threshold_value
        elif self.comparison_operator == "eq":
            return value == self.threshold_value
        elif self.comparison_operator == "gte":
            return value >= self.threshold_value
        elif self.comparison_operator == "lte":
            return value <= self.threshold_value
        else:
            return False


@dataclass
class ComponentPerformanceStats:
    """Performance statistics for a system component."""

    component_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.failed_operations / self.total_operations) * 100.0

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_operations == 0:
            return 0.0
        return self.total_response_time / self.total_operations

    def update_operation(self, response_time: float, success: bool = True, memory_mb: float = 0.0):
        """Update stats with a new operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.total_response_time += response_time
        self.min_response_time = min(self.min_response_time, response_time)
        self.max_response_time = max(self.max_response_time, response_time)

        if memory_mb > 0:
            self.memory_usage_mb = memory_mb

        self.last_updated = time.time()


class RealTimePerformanceMonitor:
    """
    Enhanced real-time performance monitoring system.

    Provides comprehensive performance monitoring with:
    - Real-time metric collection and aggregation
    - Configurable alerting and thresholds
    - Performance trend analysis
    - Automated optimization recommendations
    - Integration with existing cache monitoring
    """

    def __init__(
        self,
        collection_interval: float = 5.0,
        retention_hours: int = 24,
        enable_alerting: bool = True,
        alert_callback: Callable | None = None,
    ):
        """
        Initialize the real-time performance monitor.

        Args:
            collection_interval: Seconds between metric collections
            retention_hours: Hours to retain performance data
            enable_alerting: Whether to enable performance alerting
            alert_callback: Optional callback for alert notifications
        """
        self.collection_interval = collection_interval
        self.retention_hours = retention_hours
        self.enable_alerting = enable_alerting
        self.alert_callback = alert_callback

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Performance data storage
        self._metrics: deque = deque(maxlen=int((retention_hours * 3600) / collection_interval))
        self._component_stats: dict[str, ComponentPerformanceStats] = {}
        self._alerts: deque = deque(maxlen=10000)  # Keep last 10k alerts
        self._thresholds: list[PerformanceThreshold] = []

        # Trend analysis
        self._trend_data: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._threshold_violations: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

        # Monitoring control
        self._monitoring_task: asyncio.Task | None = None
        self._is_running = False
        self._lock = threading.RLock()

        # System monitoring
        self._system_monitor = psutil.Process()
        self._memory_monitor = MemoryMonitor()

        # Integration with existing cache monitoring
        self._cache_monitor = get_cache_performance_monitor()
        self._cache_reporter = get_real_time_cache_reporter()

        # Performance optimization tracking
        self._optimization_recommendations: list[dict[str, Any]] = []
        self._bottlenecks: dict[str, dict[str, Any]] = {}

        # Initialize default thresholds
        self._initialize_default_thresholds()

        self.logger.info(f"RealTimePerformanceMonitor initialized with {collection_interval}s interval")

    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                component="*",  # Apply to all components
                threshold_value=15000,  # 15 seconds
                comparison_operator="gt",
                severity=AlertSeverity.WARNING,
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                component="*",
                threshold_value=30000,  # 30 seconds
                comparison_operator="gt",
                severity=AlertSeverity.ERROR,
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                component="*",
                threshold_value=2048,  # 2GB
                comparison_operator="gt",
                severity=AlertSeverity.WARNING,
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.CPU_USAGE,
                component="*",
                threshold_value=80.0,  # 80%
                comparison_operator="gt",
                severity=AlertSeverity.WARNING,
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.ERROR_RATE,
                component="*",
                threshold_value=5.0,  # 5%
                comparison_operator="gt",
                severity=AlertSeverity.ERROR,
            ),
            PerformanceThreshold(
                metric_type=PerformanceMetricType.CACHE_HIT_RATE,
                component="*",
                threshold_value=70.0,  # 70%
                comparison_operator="lt",
                severity=AlertSeverity.WARNING,
            ),
        ]

        self._thresholds.extend(default_thresholds)

    async def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self._is_running:
            self.logger.warning("Performance monitoring already running")
            return

        try:
            self._is_running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Start cache reporter integration
            self._cache_reporter.start_reporting()
            self._cache_reporter.subscribe(self._handle_cache_metrics)

            self.logger.info("Real-time performance monitoring started")

        except Exception as e:
            self.logger.error(f"Error starting performance monitoring: {e}")
            self._is_running = False
            raise

    async def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop cache reporter
        self._cache_reporter.stop_reporting()
        self._cache_reporter.unsubscribe(self._handle_cache_metrics)

        self.logger.info("Real-time performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for collecting performance metrics."""
        self.logger.info("Performance monitoring loop started")

        while self._is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Collect component metrics
                await self._collect_component_metrics()

                # Analyze trends and detect bottlenecks
                await self._analyze_performance_trends()

                # Generate optimization recommendations
                await self._generate_optimization_recommendations()

                # Clean up old data
                await self._cleanup_old_data()

                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(min(self.collection_interval, 30.0))

        self.logger.info("Performance monitoring loop stopped")

    async def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            current_time = time.time()

            # Memory metrics
            memory_info = self._system_monitor.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            # CPU metrics
            cpu_percent = self._system_monitor.cpu_percent()

            # I/O metrics
            io_counters = self._system_monitor.io_counters()
            read_bytes = io_counters.read_bytes if io_counters else 0
            write_bytes = io_counters.write_bytes if io_counters else 0

            # Create system metrics
            metrics = [
                PerformanceMetric(
                    metric_id=f"system_memory_{current_time}",
                    metric_type=PerformanceMetricType.MEMORY_USAGE,
                    component="system",
                    value=memory_mb,
                    unit="MB",
                    timestamp=current_time,
                ),
                PerformanceMetric(
                    metric_id=f"system_cpu_{current_time}",
                    metric_type=PerformanceMetricType.CPU_USAGE,
                    component="system",
                    value=cpu_percent,
                    unit="%",
                    timestamp=current_time,
                ),
                PerformanceMetric(
                    metric_id=f"system_disk_read_{current_time}",
                    metric_type=PerformanceMetricType.DISK_IO,
                    component="system",
                    value=read_bytes,
                    unit="bytes",
                    timestamp=current_time,
                    tags=["read"],
                ),
                PerformanceMetric(
                    metric_id=f"system_disk_write_{current_time}",
                    metric_type=PerformanceMetricType.DISK_IO,
                    component="system",
                    value=write_bytes,
                    unit="bytes",
                    timestamp=current_time,
                    tags=["write"],
                ),
            ]

            # Store metrics
            with self._lock:
                for metric in metrics:
                    self._metrics.append(metric)
                    self._trend_data[f"{metric.component}_{metric.metric_type.value}"].append(metric.value)

            # Check thresholds
            if self.enable_alerting:
                await self._check_thresholds(metrics)

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def _collect_component_metrics(self):
        """Collect component-specific performance metrics."""
        try:
            current_time = time.time()

            # Get cache metrics from integrated cache monitor
            cache_metrics = self._cache_monitor.get_aggregated_metrics()

            if cache_metrics and "summary" in cache_metrics:
                summary = cache_metrics["summary"]

                # Cache hit rate metric
                hit_rate = summary.get("overall_hit_rate", 0.0) * 100
                cache_hit_metric = PerformanceMetric(
                    metric_id=f"cache_hit_rate_{current_time}",
                    metric_type=PerformanceMetricType.CACHE_HIT_RATE,
                    component="cache_system",
                    value=hit_rate,
                    unit="%",
                    timestamp=current_time,
                )

                # Cache response time metric
                avg_response_time = summary.get("average_response_time_ms", 0.0)
                cache_response_metric = PerformanceMetric(
                    metric_id=f"cache_response_time_{current_time}",
                    metric_type=PerformanceMetricType.RESPONSE_TIME,
                    component="cache_system",
                    value=avg_response_time,
                    unit="ms",
                    timestamp=current_time,
                )

                # Cache error rate metric
                error_rate = summary.get("overall_error_rate", 0.0) * 100
                cache_error_metric = PerformanceMetric(
                    metric_id=f"cache_error_rate_{current_time}",
                    metric_type=PerformanceMetricType.ERROR_RATE,
                    component="cache_system",
                    value=error_rate,
                    unit="%",
                    timestamp=current_time,
                )

                metrics = [cache_hit_metric, cache_response_metric, cache_error_metric]

                # Store metrics
                with self._lock:
                    for metric in metrics:
                        self._metrics.append(metric)
                        self._trend_data[f"{metric.component}_{metric.metric_type.value}"].append(metric.value)

                # Check thresholds
                if self.enable_alerting:
                    await self._check_thresholds(metrics)

        except Exception as e:
            self.logger.error(f"Error collecting component metrics: {e}")

    def _handle_cache_metrics(self, cache_report: dict[str, Any]):
        """Handle cache metrics from the integrated cache reporter."""
        try:
            current_time = time.time()

            # Extract relevant metrics from cache report
            summary = cache_report.get("summary", {})

            # Convert cache metrics to performance metrics
            if summary:
                metrics = []

                # Overall cache performance
                if "overall_hit_rate" in summary:
                    metrics.append(
                        PerformanceMetric(
                            metric_id=f"cache_realtime_hit_rate_{current_time}",
                            metric_type=PerformanceMetricType.CACHE_HIT_RATE,
                            component="cache_realtime",
                            value=summary["overall_hit_rate"] * 100,
                            unit="%",
                            timestamp=current_time,
                            metadata={"source": "cache_reporter"},
                        )
                    )

                # Store metrics without triggering alerts (already handled by cache system)
                with self._lock:
                    for metric in metrics:
                        self._metrics.append(metric)
                        self._trend_data[f"{metric.component}_{metric.metric_type.value}"].append(metric.value)

        except Exception as e:
            self.logger.error(f"Error handling cache metrics: {e}")

    async def _check_thresholds(self, metrics: list[PerformanceMetric]):
        """Check metrics against configured thresholds and generate alerts."""
        try:
            current_time = time.time()

            for metric in metrics:
                for threshold in self._thresholds:
                    # Check if threshold applies to this metric
                    if threshold.metric_type == metric.metric_type and (
                        threshold.component == "*" or threshold.component == metric.component
                    ):

                        # Check if threshold is violated
                        if threshold.evaluate(metric.value):
                            violation_key = f"{metric.component}_{threshold.metric_type.value}"

                            # Track consecutive violations
                            self._threshold_violations[violation_key].append(current_time)

                            # Check if we have enough consecutive violations
                            recent_violations = [
                                t for t in self._threshold_violations[violation_key] if current_time - t <= threshold.cooldown_seconds
                            ]

                            if len(recent_violations) >= threshold.consecutive_violations:
                                # Check cooldown period
                                last_alert_time = self._get_last_alert_time(metric.component, threshold.metric_type.value)

                                if not last_alert_time or current_time - last_alert_time >= threshold.cooldown_seconds:

                                    # Generate alert
                                    alert = PerformanceAlert(
                                        alert_id=str(uuid.uuid4()),
                                        alert_type=f"{threshold.metric_type.value}_threshold",
                                        severity=threshold.severity,
                                        component=metric.component,
                                        metric_name=threshold.metric_type.value,
                                        current_value=metric.value,
                                        threshold_value=threshold.threshold_value,
                                        message=self._generate_alert_message(metric, threshold),
                                        timestamp=current_time,
                                        metadata={
                                            "metric_id": metric.metric_id,
                                            "threshold_operator": threshold.comparison_operator,
                                            "consecutive_violations": len(recent_violations),
                                        },
                                    )

                                    await self._handle_alert(alert)
                        else:
                            # Clear violations for this metric if threshold is not violated
                            violation_key = f"{metric.component}_{threshold.metric_type.value}"
                            if violation_key in self._threshold_violations:
                                # Keep only recent violations
                                self._threshold_violations[violation_key] = deque(
                                    [
                                        t for t in self._threshold_violations[violation_key] if current_time - t <= 60.0
                                    ],  # Keep violations from last minute
                                    maxlen=10,
                                )

        except Exception as e:
            self.logger.error(f"Error checking thresholds: {e}")

    def _generate_alert_message(self, metric: PerformanceMetric, threshold: PerformanceThreshold) -> str:
        """Generate a human-readable alert message."""
        operator_text = {
            "gt": "exceeded",
            "lt": "fell below",
            "gte": "exceeded or equals",
            "lte": "fell below or equals",
            "eq": "equals",
        }.get(threshold.comparison_operator, "violated")

        return (
            f"{metric.component} {metric.metric_type.value} {operator_text} threshold: "
            f"{metric.value:.2f} {metric.unit} "
            f"(threshold: {threshold.threshold_value:.2f} {metric.unit})"
        )

    def _get_last_alert_time(self, component: str, metric_name: str) -> float | None:
        """Get the timestamp of the last alert for a component/metric combination."""
        for alert in reversed(self._alerts):
            if alert.component == component and alert.metric_name == metric_name and alert.status == AlertStatus.ACTIVE:
                return alert.timestamp
        return None

    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle a new performance alert."""
        try:
            # Store alert
            with self._lock:
                self._alerts.append(alert)

            # Log alert
            self.logger.warning(f"Performance alert: {alert.message}")

            # Call alert callback if provided
            if self.alert_callback:
                try:
                    if asyncio.iscoroutinefunction(self.alert_callback):
                        await self.alert_callback(alert)
                    else:
                        self.alert_callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")

        except Exception as e:
            self.logger.error(f"Error handling alert: {e}")

    async def _analyze_performance_trends(self):
        """Analyze performance trends and detect patterns."""
        try:
            current_time = time.time()

            # Analyze trends for key metrics
            for trend_key, values in self._trend_data.items():
                if len(values) >= 10:  # Need at least 10 data points
                    recent_values = list(values)[-10:]  # Last 10 values
                    older_values = list(values)[-20:-10] if len(values) >= 20 else []

                    if older_values:
                        recent_avg = statistics.mean(recent_values)
                        older_avg = statistics.mean(older_values)

                        # Calculate trend percentage
                        trend_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0

                        # Detect significant trends (>20% change)
                        if abs(trend_percent) > 20:
                            component, metric_type = trend_key.rsplit("_", 1)

                            # Determine if this is a concerning trend
                            is_concerning = self._is_concerning_trend(metric_type, trend_percent)

                            if is_concerning:
                                # Generate trend alert
                                alert = PerformanceAlert(
                                    alert_id=str(uuid.uuid4()),
                                    alert_type="performance_trend",
                                    severity=AlertSeverity.WARNING,
                                    component=component,
                                    metric_name=f"{metric_type}_trend",
                                    current_value=recent_avg,
                                    threshold_value=older_avg,
                                    message=f"{component} {metric_type} trending {'up' if trend_percent > 0 else 'down'} by {abs(trend_percent):.1f}%",
                                    timestamp=current_time,
                                    metadata={
                                        "trend_percent": trend_percent,
                                        "recent_avg": recent_avg,
                                        "older_avg": older_avg,
                                        "trend_type": "degradation" if is_concerning else "improvement",
                                    },
                                )

                                await self._handle_alert(alert)

        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")

    def _is_concerning_trend(self, metric_type: str, trend_percent: float) -> bool:
        """Determine if a performance trend is concerning."""
        # Metrics where increase is bad
        bad_increase_metrics = ["response_time", "memory_usage", "cpu_usage", "error_rate", "disk_io"]
        # Metrics where decrease is bad
        bad_decrease_metrics = ["cache_hit_rate", "throughput"]

        if metric_type in bad_increase_metrics and trend_percent > 0:
            return True
        elif metric_type in bad_decrease_metrics and trend_percent < 0:
            return True

        return False

    async def _generate_optimization_recommendations(self):
        """Generate performance optimization recommendations."""
        try:
            current_time = time.time()
            recommendations = []

            # Analyze recent metrics for optimization opportunities
            recent_metrics = [m for m in self._metrics if current_time - m.timestamp <= 300]  # Last 5 minutes

            if not recent_metrics:
                return

            # Group metrics by component and type
            metrics_by_component = defaultdict(lambda: defaultdict(list))
            for metric in recent_metrics:
                metrics_by_component[metric.component][metric.metric_type.value].append(metric.value)

            # Generate recommendations based on metric patterns
            for component, metrics in metrics_by_component.items():
                # High response time recommendation
                if "response_time" in metrics:
                    avg_response_time = statistics.mean(metrics["response_time"])
                    if avg_response_time > 10000:  # > 10 seconds
                        recommendations.append(
                            {
                                "type": "response_time_optimization",
                                "component": component,
                                "priority": "high" if avg_response_time > 20000 else "medium",
                                "message": f"High response time detected in {component}",
                                "current_value": avg_response_time,
                                "recommendations": [
                                    "Consider increasing cache sizes",
                                    "Optimize database queries",
                                    "Add connection pooling",
                                    "Implement request batching",
                                ],
                                "timestamp": current_time,
                            }
                        )

                # Low cache hit rate recommendation
                if "cache_hit_rate" in metrics:
                    avg_hit_rate = statistics.mean(metrics["cache_hit_rate"])
                    if avg_hit_rate < 70:  # < 70%
                        recommendations.append(
                            {
                                "type": "cache_optimization",
                                "component": component,
                                "priority": "medium",
                                "message": f"Low cache hit rate in {component}",
                                "current_value": avg_hit_rate,
                                "recommendations": [
                                    "Increase cache TTL settings",
                                    "Optimize cache key generation",
                                    "Implement cache warming",
                                    "Review cache eviction policies",
                                ],
                                "timestamp": current_time,
                            }
                        )

                # High memory usage recommendation
                if "memory_usage" in metrics:
                    avg_memory = statistics.mean(metrics["memory_usage"])
                    if avg_memory > 1024:  # > 1GB
                        recommendations.append(
                            {
                                "type": "memory_optimization",
                                "component": component,
                                "priority": "high" if avg_memory > 2048 else "medium",
                                "message": f"High memory usage in {component}",
                                "current_value": avg_memory,
                                "recommendations": [
                                    "Implement memory pooling",
                                    "Optimize data structures",
                                    "Add memory-based cache eviction",
                                    "Review object lifecycle management",
                                ],
                                "timestamp": current_time,
                            }
                        )

            # Store recommendations
            with self._lock:
                self._optimization_recommendations.extend(recommendations)
                # Keep only recent recommendations
                cutoff_time = current_time - (self.retention_hours * 3600)
                self._optimization_recommendations = [r for r in self._optimization_recommendations if r["timestamp"] > cutoff_time]

        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")

    async def _cleanup_old_data(self):
        """Clean up old performance data to manage memory usage."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (self.retention_hours * 3600)

            # Clean up old alerts
            with self._lock:
                # Remove resolved alerts older than retention period
                active_alerts = []
                for alert in self._alerts:
                    if alert.status == AlertStatus.ACTIVE or alert.timestamp > cutoff_time:
                        active_alerts.append(alert)

                self._alerts.clear()
                self._alerts.extend(active_alerts)

                # Clean up trend data (already limited by deque maxlen)
                # Clean up threshold violations
                for key in list(self._threshold_violations.keys()):
                    recent_violations = [
                        t for t in self._threshold_violations[key] if current_time - t <= 3600  # Keep violations from last hour
                    ]
                    if recent_violations:
                        self._threshold_violations[key] = deque(recent_violations, maxlen=10)
                    else:
                        del self._threshold_violations[key]

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")

    def record_custom_metric(
        self,
        component: str,
        metric_name: str,
        value: float,
        unit: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        """Record a custom performance metric."""
        try:
            current_time = time.time()

            metric = PerformanceMetric(
                metric_id=f"{component}_{metric_name}_{current_time}",
                metric_type=PerformanceMetricType.CUSTOM,
                component=component,
                value=value,
                unit=unit,
                timestamp=current_time,
                metadata=metadata or {},
                tags=tags or [],
            )

            with self._lock:
                self._metrics.append(metric)
                self._trend_data[f"{component}_{metric_name}"].append(value)

            # Check custom thresholds
            if self.enable_alerting:
                asyncio.create_task(self._check_thresholds([metric]))

        except Exception as e:
            self.logger.error(f"Error recording custom metric: {e}")

    def add_threshold(self, threshold: PerformanceThreshold):
        """Add a custom performance threshold."""
        with self._lock:
            self._thresholds.append(threshold)
        self.logger.info(f"Added performance threshold for {threshold.component} {threshold.metric_type.value}")

    def remove_threshold(self, metric_type: PerformanceMetricType, component: str):
        """Remove performance thresholds for a specific metric/component."""
        with self._lock:
            self._thresholds = [t for t in self._thresholds if not (t.metric_type == metric_type and t.component == component)]
        self.logger.info(f"Removed performance thresholds for {component} {metric_type.value}")

    def get_current_metrics(self, component: str | None = None) -> list[PerformanceMetric]:
        """Get current performance metrics, optionally filtered by component."""
        current_time = time.time()
        recent_cutoff = current_time - (self.collection_interval * 2)  # Last 2 intervals

        with self._lock:
            recent_metrics = [m for m in self._metrics if m.timestamp > recent_cutoff]

            if component:
                recent_metrics = [m for m in recent_metrics if m.component == component]

            return recent_metrics

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[PerformanceAlert]:
        """Get active performance alerts, optionally filtered by severity."""
        with self._lock:
            active_alerts = [alert for alert in self._alerts if alert.is_active]

            if severity:
                active_alerts = [alert for alert in active_alerts if alert.severity == severity]

            return active_alerts

    def get_optimization_recommendations(self, component: str | None = None, priority: str | None = None) -> list[dict[str, Any]]:
        """Get optimization recommendations, optionally filtered."""
        with self._lock:
            recommendations = list(self._optimization_recommendations)

            if component:
                recommendations = [r for r in recommendations if r["component"] == component]

            if priority:
                recommendations = [r for r in recommendations if r["priority"] == priority]

            return recommendations

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a comprehensive performance summary."""
        current_time = time.time()
        recent_cutoff = current_time - 300  # Last 5 minutes

        with self._lock:
            recent_metrics = [m for m in self._metrics if m.timestamp > recent_cutoff]
            active_alerts = [a for a in self._alerts if a.is_active]

            # Group metrics by component
            components = defaultdict(lambda: defaultdict(list))
            for metric in recent_metrics:
                components[metric.component][metric.metric_type.value].append(metric.value)

            # Calculate summary statistics
            component_summaries = {}
            for component, metrics in components.items():
                component_summaries[component] = {}
                for metric_type, values in metrics.items():
                    if values:
                        component_summaries[component][metric_type] = {
                            "current": values[-1],
                            "average": statistics.mean(values),
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                        }

            return {
                "timestamp": current_time,
                "monitoring_status": "active" if self._is_running else "stopped",
                "total_metrics_collected": len(self._metrics),
                "recent_metrics_count": len(recent_metrics),
                "active_alerts_count": len(active_alerts),
                "alerts_by_severity": {
                    severity.value: len([a for a in active_alerts if a.severity == severity]) for severity in AlertSeverity
                },
                "component_summaries": component_summaries,
                "optimization_recommendations_count": len(self._optimization_recommendations),
                "high_priority_recommendations": len([r for r in self._optimization_recommendations if r.get("priority") == "high"]),
                "cache_integration_status": "active" if self._cache_reporter._reporting_enabled else "inactive",
            }

    async def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """Acknowledge a performance alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledgment_time = time.time()
                    alert.metadata["acknowledged_by"] = user
                    self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                    return True
        return False

    async def resolve_alert(self, alert_id: str, user: str = "system", resolution_note: str = ""):
        """Resolve a performance alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolution_time = time.time()
                    alert.metadata["resolved_by"] = user
                    if resolution_note:
                        alert.metadata["resolution_note"] = resolution_note
                    self.logger.info(f"Alert {alert_id} resolved by {user}")
                    return True
        return False

    async def shutdown(self):
        """Shutdown the performance monitor."""
        self.logger.info("Shutting down RealTimePerformanceMonitor")
        await self.stop_monitoring()

        # Clear data
        with self._lock:
            self._metrics.clear()
            self._component_stats.clear()
            self._alerts.clear()
            self._trend_data.clear()
            self._threshold_violations.clear()
            self._optimization_recommendations.clear()

        self.logger.info("RealTimePerformanceMonitor shutdown complete")


# Global instance
_performance_monitor: RealTimePerformanceMonitor | None = None


def get_performance_monitor() -> RealTimePerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = RealTimePerformanceMonitor()
    return _performance_monitor


def initialize_performance_monitoring(
    collection_interval: float = 5.0, retention_hours: int = 24, enable_alerting: bool = True, alert_callback: Callable | None = None
) -> RealTimePerformanceMonitor:
    """Initialize the global performance monitoring system."""
    global _performance_monitor
    _performance_monitor = RealTimePerformanceMonitor(
        collection_interval=collection_interval,
        retention_hours=retention_hours,
        enable_alerting=enable_alerting,
        alert_callback=alert_callback,
    )
    return _performance_monitor
