"""
Performance Monitoring Service for Call Detection Pipeline.

This service provides comprehensive performance monitoring and metrics collection
for the entire enhanced function call detection pipeline, including caching,
concurrent processing, Tree-sitter optimizations, and incremental detection.
"""

import asyncio
import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

from src.utils.performance_monitor import PerformanceMonitor


@dataclass
class PerformanceAlert:
    """Represents a performance alert condition."""

    alert_id: str
    alert_type: str  # 'warning', 'error', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: float
    component: str
    resolved: bool = False
    resolution_timestamp: float | None = None


@dataclass
class ComponentMetrics:
    """Metrics for a specific component in the call detection pipeline."""

    component_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float("inf")
    max_processing_time_ms: float = 0.0
    operations_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_rate_percent: float = 0.0
    throughput_items_per_second: float = 0.0

    def update_timing(self, processing_time_ms: float):
        """Update timing statistics with new measurement."""
        self.total_processing_time_ms += processing_time_ms
        self.min_processing_time_ms = min(self.min_processing_time_ms, processing_time_ms)
        self.max_processing_time_ms = max(self.max_processing_time_ms, processing_time_ms)

        if self.total_operations > 0:
            self.average_processing_time_ms = self.total_processing_time_ms / self.total_operations

    def update_success_rate(self):
        """Update error rate calculation."""
        if self.total_operations > 0:
            self.error_rate_percent = (self.failed_operations / self.total_operations) * 100

    @property
    def success_rate_percent(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100

    @property
    def cache_hit_rate_percent(self) -> float:
        """Calculate cache hit rate as percentage."""
        total_cache_operations = self.cache_hits + self.cache_misses
        if total_cache_operations == 0:
            return 0.0
        return (self.cache_hits / total_cache_operations) * 100


@dataclass
class PipelinePerformanceSnapshot:
    """Complete performance snapshot of the call detection pipeline."""

    timestamp: float
    total_pipeline_time_ms: float
    total_files_processed: int
    total_calls_detected: int
    components: dict[str, ComponentMetrics]
    system_metrics: dict[str, float]
    alerts: list[PerformanceAlert]
    efficiency_score: float  # 0-100 score based on multiple factors

    @property
    def calls_per_second(self) -> float:
        """Calculate calls detected per second."""
        if self.total_pipeline_time_ms == 0:
            return 0.0
        return self.total_calls_detected / (self.total_pipeline_time_ms / 1000)

    @property
    def files_per_second(self) -> float:
        """Calculate files processed per second."""
        if self.total_pipeline_time_ms == 0:
            return 0.0
        return self.total_files_processed / (self.total_pipeline_time_ms / 1000)


@dataclass
class PerformanceMonitoringConfig:
    """Configuration for performance monitoring."""

    enable_monitoring: bool = True
    enable_alerts: bool = True
    enable_system_monitoring: bool = True
    snapshot_interval_seconds: float = 30.0
    max_snapshots_in_memory: int = 1000
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "max_processing_time_ms": 30000.0,  # 30 seconds
            "max_error_rate_percent": 5.0,  # 5% error rate
            "min_cache_hit_rate_percent": 70.0,  # 70% cache hit rate
            "max_memory_usage_mb": 2048.0,  # 2GB memory
            "min_efficiency_score": 60.0,  # 60% efficiency
        }
    )
    persistent_storage_path: str | None = None
    enable_performance_tuning: bool = True
    enable_trend_analysis: bool = True

    @classmethod
    def from_env(cls) -> "PerformanceMonitoringConfig":
        """Create configuration from environment variables."""
        import os

        thresholds = {}
        threshold_prefix = "PERF_MONITOR_THRESHOLD_"
        for key, value in os.environ.items():
            if key.startswith(threshold_prefix):
                metric_name = key[len(threshold_prefix) :].lower()
                thresholds[metric_name] = float(value)

        return cls(
            enable_monitoring=os.getenv("PERF_MONITOR_ENABLED", "true").lower() == "true",
            enable_alerts=os.getenv("PERF_MONITOR_ALERTS", "true").lower() == "true",
            enable_system_monitoring=os.getenv("PERF_MONITOR_SYSTEM", "true").lower() == "true",
            snapshot_interval_seconds=float(os.getenv("PERF_MONITOR_INTERVAL", "30")),
            max_snapshots_in_memory=int(os.getenv("PERF_MONITOR_MAX_SNAPSHOTS", "1000")),
            alert_thresholds=thresholds if thresholds else cls().alert_thresholds,
            persistent_storage_path=os.getenv("PERF_MONITOR_STORAGE_PATH"),
            enable_performance_tuning=os.getenv("PERF_MONITOR_TUNING", "true").lower() == "true",
            enable_trend_analysis=os.getenv("PERF_MONITOR_TRENDS", "true").lower() == "true",
        )


class PerformanceMonitoringService:
    """
    Comprehensive performance monitoring service for the enhanced function call detection pipeline.

    This service provides:
    - Real-time performance metrics collection from all pipeline components
    - System resource monitoring (CPU, memory, I/O)
    - Performance alerting with configurable thresholds
    - Historical trend analysis and performance optimization recommendations
    - Integration with caching, concurrent processing, Tree-sitter, and incremental detection
    """

    def __init__(self, config: PerformanceMonitoringConfig | None = None, alert_callback: callable | None = None):
        """
        Initialize the performance monitoring service.

        Args:
            config: Monitoring configuration
            alert_callback: Optional callback for performance alerts
        """
        self.config = config or PerformanceMonitoringConfig.from_env()
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(__name__)

        # Component tracking
        self._component_metrics: dict[str, ComponentMetrics] = {}
        self._active_operations: dict[str, dict[str, float]] = defaultdict(dict)  # component -> operation_id -> start_time

        # Performance snapshots
        self._snapshots: deque = deque(maxlen=self.config.max_snapshots_in_memory)
        self._alerts: list[PerformanceAlert] = []

        # Monitoring control
        self._monitoring_task: asyncio.Task | None = None
        self._is_running = False
        self._lock = threading.RLock()

        # System monitoring
        if self.config.enable_system_monitoring:
            self._system_monitor = psutil.Process()
        else:
            self._system_monitor = None

        # Performance analysis
        self._performance_trends: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._optimization_recommendations: list[dict[str, Any]] = []

        # Global statistics
        self._global_stats = {
            "total_pipeline_operations": 0,
            "total_processing_time_ms": 0.0,
            "total_calls_detected": 0,
            "total_files_processed": 0,
            "service_start_time": time.time(),
        }

        self.logger.info(f"PerformanceMonitoringService initialized with config: {self.config}")

    async def start_monitoring(self):
        """Start the performance monitoring service."""
        if not self.config.enable_monitoring:
            self.logger.info("Performance monitoring disabled")
            return

        if self._is_running:
            self.logger.warning("Performance monitoring already running")
            return

        try:
            self._is_running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Performance monitoring started")

        except Exception as e:
            self.logger.error(f"Error starting performance monitoring: {e}")
            self._is_running = False

    async def stop_monitoring(self):
        """Stop the performance monitoring service."""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Save final snapshot if persistent storage enabled
        if self.config.persistent_storage_path:
            await self._save_performance_data()

        self.logger.info("Performance monitoring stopped")

    def start_operation(self, component: str, operation_id: str, metadata: dict[str, Any] | None = None):
        """
        Start tracking a performance operation.

        Args:
            component: Component name (e.g., 'breadcrumb_cache', 'concurrent_extractor')
            operation_id: Unique operation identifier
            metadata: Optional metadata about the operation
        """
        if not self.config.enable_monitoring:
            return

        with self._lock:
            self._active_operations[component][operation_id] = time.time()

            # Initialize component metrics if not exists
            if component not in self._component_metrics:
                self._component_metrics[component] = ComponentMetrics(component_name=component)

            self._component_metrics[component].total_operations += 1

    def complete_operation(
        self,
        component: str,
        operation_id: str,
        success: bool = True,
        items_processed: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Complete tracking a performance operation.

        Args:
            component: Component name
            operation_id: Operation identifier
            success: Whether operation was successful
            items_processed: Number of items processed
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            metadata: Optional metadata about completion
        """
        if not self.config.enable_monitoring:
            return

        with self._lock:
            start_time = self._active_operations[component].pop(operation_id, None)
            if start_time is None:
                self.logger.warning(f"Operation {operation_id} not found for component {component}")
                return

            processing_time_ms = (time.time() - start_time) * 1000
            metrics = self._component_metrics[component]

            # Update timing metrics
            metrics.update_timing(processing_time_ms)

            # Update success/failure counts
            if success:
                metrics.successful_operations += 1
            else:
                metrics.failed_operations += 1

            metrics.update_success_rate()

            # Update cache metrics
            metrics.cache_hits += cache_hits
            metrics.cache_misses += cache_misses

            # Update throughput metrics
            if processing_time_ms > 0:
                metrics.throughput_items_per_second = items_processed / (processing_time_ms / 1000)

            # Check for alerts
            if self.config.enable_alerts:
                self._check_component_alerts(component, metrics)

            # Update global stats
            self._global_stats["total_processing_time_ms"] += processing_time_ms
            if items_processed > 0:
                self._global_stats["total_files_processed"] += items_processed

    def record_calls_detected(self, component: str, calls_count: int):
        """Record number of function calls detected."""
        if not self.config.enable_monitoring:
            return

        with self._lock:
            self._global_stats["total_calls_detected"] += calls_count

    def record_cache_operation(self, component: str, operation_type: str, hit: bool, processing_time_ms: float = 0):
        """Record a cache operation."""
        if not self.config.enable_monitoring:
            return

        with self._lock:
            if component not in self._component_metrics:
                self._component_metrics[component] = ComponentMetrics(component_name=component)

            metrics = self._component_metrics[component]
            if hit:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1

            if processing_time_ms > 0:
                metrics.update_timing(processing_time_ms)

    def record_system_metrics(self) -> dict[str, float]:
        """Record current system metrics."""
        if not self.config.enable_system_monitoring or not self._system_monitor:
            return {}

        try:
            # Memory usage
            memory_info = self._system_monitor.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)

            # CPU usage
            cpu_percent = self._system_monitor.cpu_percent()

            # I/O statistics
            io_counters = self._system_monitor.io_counters()
            read_bytes = io_counters.read_bytes if io_counters else 0
            write_bytes = io_counters.write_bytes if io_counters else 0

            # Number of threads
            num_threads = self._system_monitor.num_threads()

            system_metrics = {
                "memory_usage_mb": memory_usage_mb,
                "cpu_percent": cpu_percent,
                "read_bytes": read_bytes,
                "write_bytes": write_bytes,
                "num_threads": num_threads,
                "timestamp": time.time(),
            }

            # Update component memory usage
            for component_metrics in self._component_metrics.values():
                component_metrics.memory_usage_mb = memory_usage_mb / len(self._component_metrics)

            return system_metrics

        except Exception as e:
            self.logger.error(f"Error recording system metrics: {e}")
            return {}

    def create_performance_snapshot(self) -> PipelinePerformanceSnapshot:
        """Create a complete performance snapshot."""
        with self._lock:
            # Record system metrics
            system_metrics = self.record_system_metrics()

            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score()

            # Create snapshot
            snapshot = PipelinePerformanceSnapshot(
                timestamp=time.time(),
                total_pipeline_time_ms=self._global_stats["total_processing_time_ms"],
                total_files_processed=self._global_stats["total_files_processed"],
                total_calls_detected=self._global_stats["total_calls_detected"],
                components=self._component_metrics.copy(),
                system_metrics=system_metrics,
                alerts=[alert for alert in self._alerts if not alert.resolved],
                efficiency_score=efficiency_score,
            )

            # Store snapshot
            self._snapshots.append(snapshot)

            # Update performance trends
            if self.config.enable_trend_analysis:
                self._update_performance_trends(snapshot)

            return snapshot

    def _calculate_efficiency_score(self) -> float:
        """Calculate overall pipeline efficiency score (0-100)."""
        scores = []

        # Cache hit rate score (0-30 points)
        total_cache_hits = sum(m.cache_hits for m in self._component_metrics.values())
        total_cache_misses = sum(m.cache_misses for m in self._component_metrics.values())
        total_cache_ops = total_cache_hits + total_cache_misses

        if total_cache_ops > 0:
            cache_hit_rate = total_cache_hits / total_cache_ops
            cache_score = min(30, cache_hit_rate * 30)
            scores.append(cache_score)

        # Success rate score (0-25 points)
        total_successful = sum(m.successful_operations for m in self._component_metrics.values())
        total_operations = sum(m.total_operations for m in self._component_metrics.values())

        if total_operations > 0:
            success_rate = total_successful / total_operations
            success_score = success_rate * 25
            scores.append(success_score)

        # Performance score based on processing time (0-25 points)
        avg_processing_times = [m.average_processing_time_ms for m in self._component_metrics.values() if m.average_processing_time_ms > 0]
        if avg_processing_times:
            # Lower processing time is better
            avg_time = statistics.mean(avg_processing_times)
            # Score based on inverse of processing time (capped at reasonable limits)
            time_score = max(0, 25 - (avg_time / 1000))  # 1 second = 1 point deduction
            scores.append(min(25, time_score))

        # Memory efficiency score (0-20 points)
        if self._component_metrics:
            avg_memory = statistics.mean(m.memory_usage_mb for m in self._component_metrics.values())
            # Score based on memory usage (less is better)
            memory_score = max(0, 20 - (avg_memory / 100))  # 100MB = 1 point deduction
            scores.append(min(20, memory_score))

        return sum(scores) if scores else 50.0  # Default middle score

    def _update_performance_trends(self, snapshot: PipelinePerformanceSnapshot):
        """Update performance trend analysis."""
        # Update trends for key metrics
        self._performance_trends["efficiency_score"].append(snapshot.efficiency_score)
        self._performance_trends["calls_per_second"].append(snapshot.calls_per_second)
        self._performance_trends["files_per_second"].append(snapshot.files_per_second)

        for component_name, metrics in snapshot.components.items():
            trend_key = f"{component_name}_cache_hit_rate"
            self._performance_trends[trend_key].append(metrics.cache_hit_rate_percent)

            trend_key = f"{component_name}_avg_time"
            self._performance_trends[trend_key].append(metrics.average_processing_time_ms)

        # Generate optimization recommendations based on trends
        if self.config.enable_performance_tuning:
            self._generate_optimization_recommendations()

    def _generate_optimization_recommendations(self):
        """Generate performance optimization recommendations based on trends."""
        recommendations = []

        # Analyze efficiency score trend
        efficiency_scores = list(self._performance_trends["efficiency_score"])
        if len(efficiency_scores) >= 10:
            recent_scores = efficiency_scores[-5:]
            earlier_scores = efficiency_scores[-10:-5]

            if statistics.mean(recent_scores) < statistics.mean(earlier_scores) - 5:
                recommendations.append(
                    {
                        "type": "performance_degradation",
                        "component": "pipeline",
                        "message": "Pipeline efficiency has decreased recently",
                        "suggestion": "Consider increasing cache sizes or optimizing slow components",
                        "priority": "high",
                    }
                )

        # Analyze cache hit rates
        for trend_key, values in self._performance_trends.items():
            if "cache_hit_rate" in trend_key and len(values) >= 5:
                recent_hit_rate = statistics.mean(values[-5:])
                if recent_hit_rate < 70:
                    component = trend_key.split("_cache_hit_rate")[0]
                    recommendations.append(
                        {
                            "type": "low_cache_hit_rate",
                            "component": component,
                            "message": f"Low cache hit rate ({recent_hit_rate:.1f}%) for {component}",
                            "suggestion": "Consider increasing cache size or improving cache key strategy",
                            "priority": "medium",
                        }
                    )

        # Update recommendations (keep only recent ones)
        self._optimization_recommendations = recommendations[-50:]  # Keep last 50 recommendations

    def _check_component_alerts(self, component: str, metrics: ComponentMetrics):
        """Check for performance alerts on component metrics."""
        current_time = time.time()

        # Check error rate
        if metrics.error_rate_percent > self.config.alert_thresholds.get("max_error_rate_percent", 5.0):
            alert = PerformanceAlert(
                alert_id=f"{component}_error_rate_{current_time}",
                alert_type="warning",
                message=f"High error rate in {component}: {metrics.error_rate_percent:.1f}%",
                metric_name="error_rate_percent",
                current_value=metrics.error_rate_percent,
                threshold_value=self.config.alert_thresholds["max_error_rate_percent"],
                timestamp=current_time,
                component=component,
            )
            self._add_alert(alert)

        # Check cache hit rate
        if metrics.cache_hit_rate_percent < self.config.alert_thresholds.get("min_cache_hit_rate_percent", 70.0):
            alert = PerformanceAlert(
                alert_id=f"{component}_cache_hit_rate_{current_time}",
                alert_type="warning",
                message=f"Low cache hit rate in {component}: {metrics.cache_hit_rate_percent:.1f}%",
                metric_name="cache_hit_rate_percent",
                current_value=metrics.cache_hit_rate_percent,
                threshold_value=self.config.alert_thresholds["min_cache_hit_rate_percent"],
                timestamp=current_time,
                component=component,
            )
            self._add_alert(alert)

        # Check processing time
        if metrics.average_processing_time_ms > self.config.alert_thresholds.get("max_processing_time_ms", 30000.0):
            alert = PerformanceAlert(
                alert_id=f"{component}_processing_time_{current_time}",
                alert_type="error",
                message=f"High processing time in {component}: {metrics.average_processing_time_ms:.1f}ms",
                metric_name="average_processing_time_ms",
                current_value=metrics.average_processing_time_ms,
                threshold_value=self.config.alert_thresholds["max_processing_time_ms"],
                timestamp=current_time,
                component=component,
            )
            self._add_alert(alert)

    def _add_alert(self, alert: PerformanceAlert):
        """Add a new performance alert."""
        # Check if similar alert already exists
        existing_alert = next(
            (
                a
                for a in self._alerts
                if a.component == alert.component
                and a.metric_name == alert.metric_name
                and not a.resolved
                and abs(a.timestamp - alert.timestamp) < 300
            ),  # 5 minutes
            None,
        )

        if existing_alert:
            # Update existing alert
            existing_alert.current_value = alert.current_value
            existing_alert.timestamp = alert.timestamp
        else:
            # Add new alert
            self._alerts.append(alert)

            # Call alert callback if provided
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")

        # Limit number of alerts in memory
        if len(self._alerts) > 1000:
            # Remove oldest resolved alerts
            resolved_alerts = [a for a in self._alerts if a.resolved]
            if resolved_alerts:
                resolved_alerts.sort(key=lambda x: x.timestamp)
                self._alerts = [a for a in self._alerts if not a.resolved or a not in resolved_alerts[:500]]

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        self.logger.info("Performance monitoring loop started")

        while self._is_running:
            try:
                # Create performance snapshot
                snapshot = self.create_performance_snapshot()

                # Log summary information
                self.logger.debug(
                    f"Performance snapshot: {snapshot.total_files_processed} files, "
                    f"{snapshot.total_calls_detected} calls, "
                    f"efficiency: {snapshot.efficiency_score:.1f}%"
                )

                # Wait for next interval
                await asyncio.sleep(self.config.snapshot_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying

        self.logger.info("Performance monitoring loop stopped")

    async def _save_performance_data(self):
        """Save performance data to persistent storage."""
        if not self.config.persistent_storage_path:
            return

        try:
            storage_path = Path(self.config.persistent_storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            data = {
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "total_pipeline_time_ms": s.total_pipeline_time_ms,
                        "total_files_processed": s.total_files_processed,
                        "total_calls_detected": s.total_calls_detected,
                        "efficiency_score": s.efficiency_score,
                        "components": {
                            name: {
                                "total_operations": m.total_operations,
                                "successful_operations": m.successful_operations,
                                "failed_operations": m.failed_operations,
                                "average_processing_time_ms": m.average_processing_time_ms,
                                "cache_hit_rate_percent": m.cache_hit_rate_percent,
                                "error_rate_percent": m.error_rate_percent,
                            }
                            for name, m in s.components.items()
                        },
                    }
                    for s in list(self._snapshots)[-100:]  # Save last 100 snapshots
                ],
                "global_stats": self._global_stats,
                "optimization_recommendations": self._optimization_recommendations[-20:],  # Last 20 recommendations
            }

            with open(storage_path, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Performance data saved to {storage_path}")

        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")

    def get_performance_report(self, include_trends: bool = True) -> dict[str, Any]:
        """Get comprehensive performance report."""
        current_snapshot = self.create_performance_snapshot()

        report = {
            "current_snapshot": {
                "timestamp": current_snapshot.timestamp,
                "efficiency_score": current_snapshot.efficiency_score,
                "total_files_processed": current_snapshot.total_files_processed,
                "total_calls_detected": current_snapshot.total_calls_detected,
                "calls_per_second": current_snapshot.calls_per_second,
                "files_per_second": current_snapshot.files_per_second,
                "active_alerts": len([a for a in current_snapshot.alerts if not a.resolved]),
            },
            "component_performance": {
                name: {
                    "total_operations": m.total_operations,
                    "success_rate_percent": m.success_rate_percent,
                    "average_processing_time_ms": m.average_processing_time_ms,
                    "cache_hit_rate_percent": m.cache_hit_rate_percent,
                    "throughput_items_per_second": m.throughput_items_per_second,
                    "error_rate_percent": m.error_rate_percent,
                }
                for name, m in current_snapshot.components.items()
            },
            "system_metrics": current_snapshot.system_metrics,
            "global_statistics": self._global_stats,
            "recent_alerts": [
                {
                    "alert_type": a.alert_type,
                    "message": a.message,
                    "component": a.component,
                    "timestamp": a.timestamp,
                    "resolved": a.resolved,
                }
                for a in self._alerts[-10:]  # Last 10 alerts
            ],
            "optimization_recommendations": self._optimization_recommendations[-10:],  # Last 10 recommendations
        }

        if include_trends and self.config.enable_trend_analysis:
            report["performance_trends"] = {
                name: list(values)[-20:] for name, values in self._performance_trends.items()  # Last 20 data points
            }

        return report

    def get_component_statistics(self, component: str) -> dict[str, Any] | None:
        """Get detailed statistics for a specific component."""
        if component not in self._component_metrics:
            return None

        metrics = self._component_metrics[component]

        return {
            "component_name": metrics.component_name,
            "operational_metrics": {
                "total_operations": metrics.total_operations,
                "successful_operations": metrics.successful_operations,
                "failed_operations": metrics.failed_operations,
                "success_rate_percent": metrics.success_rate_percent,
                "error_rate_percent": metrics.error_rate_percent,
            },
            "performance_metrics": {
                "average_processing_time_ms": metrics.average_processing_time_ms,
                "min_processing_time_ms": metrics.min_processing_time_ms,
                "max_processing_time_ms": metrics.max_processing_time_ms,
                "total_processing_time_ms": metrics.total_processing_time_ms,
                "throughput_items_per_second": metrics.throughput_items_per_second,
            },
            "cache_metrics": {
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "cache_hit_rate_percent": metrics.cache_hit_rate_percent,
            },
            "resource_metrics": {"memory_usage_mb": metrics.memory_usage_mb},
        }

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolution_timestamp = time.time()
                    self.logger.info(f"Alert resolved: {alert_id}")
                    break

    def clear_performance_data(self):
        """Clear all performance data (for testing or reset)."""
        with self._lock:
            self._component_metrics.clear()
            self._active_operations.clear()
            self._snapshots.clear()
            self._alerts.clear()
            self._performance_trends.clear()
            self._optimization_recommendations.clear()

            self._global_stats = {
                "total_pipeline_operations": 0,
                "total_processing_time_ms": 0.0,
                "total_calls_detected": 0,
                "total_files_processed": 0,
                "service_start_time": time.time(),
            }

        self.logger.info("Performance data cleared")

    def export_performance_data(self, format_type: str = "json") -> str | dict[str, Any]:
        """Export performance data in specified format."""
        report = self.get_performance_report()

        if format_type.lower() == "json":
            return json.dumps(report, indent=2)
        elif format_type.lower() == "dict":
            return report
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    async def shutdown(self):
        """Shutdown the performance monitoring service."""
        self.logger.info("Shutting down PerformanceMonitoringService")
        await self.stop_monitoring()
        self.clear_performance_data()
        self.logger.info("PerformanceMonitoringService shutdown complete")
