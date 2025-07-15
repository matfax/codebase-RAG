"""
Cache performance degradation detection and handling service.

This service monitors cache performance, detects degradation patterns,
and implements automatic remediation strategies to maintain optimal performance.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..services.cache_service import get_cache_service
from ..utils.telemetry import get_telemetry_manager, trace_cache_operation


class PerformanceDegradationType(Enum):
    """Types of performance degradation."""

    SLOW_RESPONSE_TIME = "slow_response_time"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_HIT_RATE = "low_hit_rate"
    MEMORY_PRESSURE = "memory_pressure"
    CONNECTION_SATURATION = "connection_saturation"
    CPU_INTENSIVE_OPERATIONS = "cpu_intensive_operations"
    NETWORK_LATENCY = "network_latency"
    DISK_IO_BOTTLENECK = "disk_io_bottleneck"


class PerformanceMetricType(Enum):
    """Types of performance metrics."""

    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    HIT_RATE = "hit_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    CONNECTION_COUNT = "connection_count"


class RemediationAction(Enum):
    """Types of remediation actions."""

    CACHE_EVICTION = "cache_eviction"
    CONNECTION_POOL_RESTART = "connection_pool_restart"
    GARBAGE_COLLECTION = "garbage_collection"
    CACHE_WARMUP = "cache_warmup"
    LOAD_BALANCING = "load_balancing"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"
    ALERT_NOTIFICATION = "alert_notification"
    AUTO_SCALING = "auto_scaling"


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""

    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    operation_type: str | None = None
    service_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""

    metric_type: PerformanceMetricType
    baseline_value: float
    variance: float
    confidence_interval: float
    sample_count: int
    established_at: datetime
    last_updated: datetime


@dataclass
class DegradationEvent:
    """Represents a performance degradation event."""

    event_id: str
    degradation_type: PerformanceDegradationType
    metric_type: PerformanceMetricType
    detected_at: datetime
    current_value: float
    baseline_value: float
    degradation_ratio: float
    severity: str  # low, medium, high, critical
    affected_operations: list[str]
    remediation_actions: list[RemediationAction]
    resolved_at: datetime | None = None
    resolution_successful: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RemediationResult:
    """Result of a remediation action."""

    action: RemediationAction
    started_at: datetime
    completed_at: datetime | None
    success: bool
    error_message: str | None = None
    performance_improvement: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfiguration:
    """Configuration for performance monitoring and remediation."""

    monitoring_interval_seconds: int = 60
    baseline_window_size: int = 1000
    baseline_min_samples: int = 100
    degradation_threshold_ratio: float = 2.0  # 2x baseline
    critical_threshold_ratio: float = 5.0  # 5x baseline
    error_rate_threshold: float = 0.1  # 10% error rate
    hit_rate_threshold: float = 0.7  # 70% hit rate
    memory_usage_threshold: float = 0.8  # 80% memory usage
    auto_remediation_enabled: bool = True
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "response_time_p95": 1000.0,  # 1 second
            "error_rate": 0.05,  # 5%
            "hit_rate": 0.8,  # 80%
            "memory_usage": 0.85,  # 85%
        }
    )


class CachePerformanceService:
    """Service for monitoring and handling cache performance degradation."""

    def __init__(self, config: CacheConfig | None = None, perf_config: PerformanceConfiguration | None = None):
        """Initialize the cache performance service."""
        self.config = config or get_global_cache_config()
        self.perf_config = perf_config or PerformanceConfiguration()
        self.logger = logging.getLogger(__name__)
        self._telemetry = get_telemetry_manager()

        # Performance data storage
        self._metrics: dict[PerformanceMetricType, deque[PerformanceMetric]] = defaultdict(lambda: deque(maxlen=10000))
        self._baselines: dict[PerformanceMetricType, PerformanceBaseline] = {}
        self._degradation_events: list[DegradationEvent] = []
        self._active_degradations: dict[str, DegradationEvent] = {}

        # Remediation tracking
        self._remediation_history: list[RemediationResult] = []
        self._remediation_callbacks: dict[RemediationAction, list[Callable]] = defaultdict(list)

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._baseline_update_task: asyncio.Task | None = None

        # Cache service reference
        self._cache_service = None

        # Performance counters
        self._operation_counts: dict[str, int] = defaultdict(int)
        self._operation_times: dict[str, list[float]] = defaultdict(list)
        self._operation_errors: dict[str, int] = defaultdict(int)

    async def initialize(self):
        """Initialize the performance monitoring service."""
        try:
            self._cache_service = await get_cache_service()

            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._baseline_update_task = asyncio.create_task(self._baseline_update_loop())

            self.logger.info("Cache performance monitoring service initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize performance service: {e}")
            raise

    async def shutdown(self):
        """Shutdown the performance monitoring service."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._baseline_update_task:
            self._baseline_update_task.cancel()
            try:
                await self._baseline_update_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Cache performance monitoring service shutdown")

    @trace_cache_operation("performance_record_metric")
    async def record_metric(
        self,
        metric_type: PerformanceMetricType,
        value: float,
        operation_type: str | None = None,
        service_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            operation_type=operation_type,
            service_id=service_id,
            metadata=metadata or {},
        )

        self._metrics[metric_type].append(metric)

        # Check for degradation
        await self._check_degradation(metric)

    async def record_operation_performance(self, operation: str, duration_ms: float, success: bool, metadata: dict[str, Any] | None = None):
        """Record performance data for a cache operation."""
        self._operation_counts[operation] += 1
        self._operation_times[operation].append(duration_ms)

        if not success:
            self._operation_errors[operation] += 1

        # Record metrics
        await self.record_metric(PerformanceMetricType.RESPONSE_TIME, duration_ms, operation_type=operation, metadata=metadata)

        # Calculate and record error rate
        total_ops = self._operation_counts[operation]
        error_count = self._operation_errors[operation]
        error_rate = error_count / total_ops if total_ops > 0 else 0.0

        await self.record_metric(PerformanceMetricType.ERROR_RATE, error_rate, operation_type=operation)

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get current performance summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "baselines": {},
            "active_degradations": len(self._active_degradations),
            "total_degradation_events": len(self._degradation_events),
            "operation_stats": {},
        }

        # Current metrics summary
        for metric_type, metrics in self._metrics.items():
            if metrics:
                recent_metrics = list(metrics)[-100:]  # Last 100 metrics
                values = [m.value for m in recent_metrics]

                summary["metrics"][metric_type.value] = {
                    "current": values[-1] if values else 0,
                    "average": statistics.mean(values) if values else 0,
                    "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else (values[-1] if values else 0),
                    "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else (values[-1] if values else 0),
                    "sample_count": len(values),
                }

        # Baseline information
        for metric_type, baseline in self._baselines.items():
            summary["baselines"][metric_type.value] = {
                "baseline_value": baseline.baseline_value,
                "variance": baseline.variance,
                "confidence_interval": baseline.confidence_interval,
                "sample_count": baseline.sample_count,
                "last_updated": baseline.last_updated.isoformat(),
            }

        # Operation statistics
        for operation, count in self._operation_counts.items():
            times = self._operation_times[operation]
            errors = self._operation_errors[operation]

            summary["operation_stats"][operation] = {
                "total_operations": count,
                "error_count": errors,
                "error_rate": errors / count if count > 0 else 0,
                "average_time_ms": statistics.mean(times) if times else 0,
                "p95_time_ms": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else (times[-1] if times else 0),
            }

        return summary

    async def get_degradation_events(self, limit: int = 50, active_only: bool = False) -> list[dict[str, Any]]:
        """Get recent degradation events."""
        events = list(self._active_degradations.values()) if active_only else self._degradation_events

        # Sort by detected time (newest first)
        sorted_events = sorted(events, key=lambda e: e.detected_at, reverse=True)

        # Limit results
        limited_events = sorted_events[:limit]

        # Format for response
        formatted_events = []
        for event in limited_events:
            formatted_events.append(
                {
                    "event_id": event.event_id,
                    "degradation_type": event.degradation_type.value,
                    "metric_type": event.metric_type.value,
                    "detected_at": event.detected_at.isoformat(),
                    "current_value": event.current_value,
                    "baseline_value": event.baseline_value,
                    "degradation_ratio": event.degradation_ratio,
                    "severity": event.severity,
                    "affected_operations": event.affected_operations,
                    "remediation_actions": [action.value for action in event.remediation_actions],
                    "resolved_at": event.resolved_at.isoformat() if event.resolved_at else None,
                    "resolution_successful": event.resolution_successful,
                    "metadata": event.metadata,
                }
            )

        return formatted_events

    async def trigger_manual_remediation(
        self, action: RemediationAction, target_metric: PerformanceMetricType | None = None, metadata: dict[str, Any] | None = None
    ) -> RemediationResult:
        """Manually trigger a remediation action."""
        result = RemediationResult(action=action, started_at=datetime.now(), completed_at=None, success=False, metadata=metadata or {})

        try:
            self.logger.info(f"Starting manual remediation: {action.value}")

            # Execute remediation action
            if action == RemediationAction.CACHE_EVICTION:
                await self._perform_cache_eviction()
            elif action == RemediationAction.CONNECTION_POOL_RESTART:
                await self._restart_connection_pool()
            elif action == RemediationAction.GARBAGE_COLLECTION:
                await self._trigger_garbage_collection()
            elif action == RemediationAction.CACHE_WARMUP:
                await self._perform_cache_warmup()
            else:
                raise ValueError(f"Unsupported remediation action: {action}")

            result.completed_at = datetime.now()
            result.success = True

            # Measure improvement if target metric specified
            if target_metric and target_metric in self._metrics:
                before_metrics = list(self._metrics[target_metric])[-10:]
                await asyncio.sleep(5)  # Wait for effect
                after_metrics = list(self._metrics[target_metric])[-5:]

                if before_metrics and after_metrics:
                    before_avg = statistics.mean([m.value for m in before_metrics])
                    after_avg = statistics.mean([m.value for m in after_metrics])
                    result.performance_improvement = (before_avg - after_avg) / before_avg

            self.logger.info(f"Manual remediation completed: {action.value}")

        except Exception as e:
            result.completed_at = datetime.now()
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Manual remediation failed: {action.value}, error: {e}")

        self._remediation_history.append(result)
        return result

    async def _monitoring_loop(self):
        """Background loop for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(self.perf_config.monitoring_interval_seconds)
                await self._collect_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    async def _baseline_update_loop(self):
        """Background loop for updating performance baselines."""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                await self._update_baselines()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in baseline update loop: {e}")

    async def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        try:
            if not self._cache_service:
                return

            # Collect cache service metrics
            if hasattr(self._cache_service, "get_tier_stats"):
                stats = self._cache_service.get_tier_stats()

                # Extract metrics from stats
                if "l1_stats" in stats:
                    l1_stats = stats["l1_stats"]
                    if "hit_rate" in l1_stats:
                        await self.record_metric(PerformanceMetricType.HIT_RATE, l1_stats["hit_rate"], service_id="L1")

            # Memory usage (if available)
            try:
                import psutil

                memory_percent = psutil.virtual_memory().percent / 100.0
                await self.record_metric(PerformanceMetricType.MEMORY_USAGE, memory_percent)

                cpu_percent = psutil.cpu_percent(interval=1) / 100.0
                await self.record_metric(PerformanceMetricType.CPU_USAGE, cpu_percent)
            except ImportError:
                # psutil not available, skip system metrics
                pass

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    async def _update_baselines(self):
        """Update performance baselines based on recent data."""
        for metric_type, metrics in self._metrics.items():
            if len(metrics) < self.perf_config.baseline_min_samples:
                continue

            # Get recent stable metrics (exclude outliers)
            recent_values = [m.value for m in list(metrics)[-self.perf_config.baseline_window_size :]]

            if len(recent_values) < self.perf_config.baseline_min_samples:
                continue

            # Calculate baseline statistics
            mean_value = statistics.mean(recent_values)
            variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0
            std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

            # Update or create baseline
            if metric_type in self._baselines:
                baseline = self._baselines[metric_type]
                # Use exponential moving average for baseline updates
                alpha = 0.1  # Smoothing factor
                baseline.baseline_value = alpha * mean_value + (1 - alpha) * baseline.baseline_value
                baseline.variance = variance
                baseline.confidence_interval = 1.96 * std_dev  # 95% confidence interval
                baseline.sample_count = len(recent_values)
                baseline.last_updated = datetime.now()
            else:
                self._baselines[metric_type] = PerformanceBaseline(
                    metric_type=metric_type,
                    baseline_value=mean_value,
                    variance=variance,
                    confidence_interval=1.96 * std_dev,
                    sample_count=len(recent_values),
                    established_at=datetime.now(),
                    last_updated=datetime.now(),
                )

    async def _check_degradation(self, metric: PerformanceMetric):
        """Check if a metric indicates performance degradation."""
        metric_type = metric.metric_type

        if metric_type not in self._baselines:
            return  # No baseline to compare against

        baseline = self._baselines[metric_type]
        degradation_ratio = metric.value / baseline.baseline_value if baseline.baseline_value > 0 else 1.0

        # Check for different types of degradation
        degradation_detected = False
        degradation_type = None
        severity = "low"

        if metric_type == PerformanceMetricType.RESPONSE_TIME:
            if degradation_ratio >= self.perf_config.critical_threshold_ratio:
                degradation_detected = True
                degradation_type = PerformanceDegradationType.SLOW_RESPONSE_TIME
                severity = "critical"
            elif degradation_ratio >= self.perf_config.degradation_threshold_ratio:
                degradation_detected = True
                degradation_type = PerformanceDegradationType.SLOW_RESPONSE_TIME
                severity = "high" if degradation_ratio >= 3.0 else "medium"

        elif metric_type == PerformanceMetricType.ERROR_RATE:
            if metric.value >= self.perf_config.error_rate_threshold:
                degradation_detected = True
                degradation_type = PerformanceDegradationType.HIGH_ERROR_RATE
                severity = "critical" if metric.value >= 0.2 else "high"

        elif metric_type == PerformanceMetricType.HIT_RATE:
            if metric.value <= self.perf_config.hit_rate_threshold:
                degradation_detected = True
                degradation_type = PerformanceDegradationType.LOW_HIT_RATE
                severity = "medium"

        elif metric_type == PerformanceMetricType.MEMORY_USAGE:
            if metric.value >= self.perf_config.memory_usage_threshold:
                degradation_detected = True
                degradation_type = PerformanceDegradationType.MEMORY_PRESSURE
                severity = "high" if metric.value >= 0.95 else "medium"

        if degradation_detected:
            await self._handle_degradation(degradation_type, metric, baseline, degradation_ratio, severity)

    async def _handle_degradation(
        self,
        degradation_type: PerformanceDegradationType,
        metric: PerformanceMetric,
        baseline: PerformanceBaseline,
        degradation_ratio: float,
        severity: str,
    ):
        """Handle detected performance degradation."""
        event_id = f"degradation_{int(time.time())}_{degradation_type.value}"

        # Check if this is a duplicate of an active degradation
        for active_event in self._active_degradations.values():
            if active_event.degradation_type == degradation_type and active_event.metric_type == metric.metric_type:
                # Update existing event
                active_event.current_value = metric.value
                active_event.degradation_ratio = degradation_ratio
                return

        # Create new degradation event
        event = DegradationEvent(
            event_id=event_id,
            degradation_type=degradation_type,
            metric_type=metric.metric_type,
            detected_at=metric.timestamp,
            current_value=metric.value,
            baseline_value=baseline.baseline_value,
            degradation_ratio=degradation_ratio,
            severity=severity,
            affected_operations=[metric.operation_type] if metric.operation_type else [],
            remediation_actions=[],
            metadata=metric.metadata,
        )

        # Determine remediation actions
        remediation_actions = self._determine_remediation_actions(degradation_type, severity)
        event.remediation_actions = remediation_actions

        # Store event
        self._degradation_events.append(event)
        self._active_degradations[event_id] = event

        self.logger.warning(
            f"Performance degradation detected: {degradation_type.value}, " f"severity: {severity}, ratio: {degradation_ratio:.2f}"
        )

        # Execute automatic remediation if enabled
        if self.perf_config.auto_remediation_enabled and remediation_actions:
            await self._execute_auto_remediation(event)

    def _determine_remediation_actions(self, degradation_type: PerformanceDegradationType, severity: str) -> list[RemediationAction]:
        """Determine appropriate remediation actions for degradation."""
        actions = []

        if degradation_type == PerformanceDegradationType.SLOW_RESPONSE_TIME:
            if severity in ["high", "critical"]:
                actions.extend([RemediationAction.CACHE_EVICTION, RemediationAction.GARBAGE_COLLECTION])
            if severity == "critical":
                actions.append(RemediationAction.CONNECTION_POOL_RESTART)

        elif degradation_type == PerformanceDegradationType.HIGH_ERROR_RATE:
            actions.extend([RemediationAction.CONNECTION_POOL_RESTART, RemediationAction.CIRCUIT_BREAKER_TRIP])

        elif degradation_type == PerformanceDegradationType.LOW_HIT_RATE:
            actions.append(RemediationAction.CACHE_WARMUP)

        elif degradation_type == PerformanceDegradationType.MEMORY_PRESSURE:
            actions.extend([RemediationAction.CACHE_EVICTION, RemediationAction.GARBAGE_COLLECTION])

        # Always add alerting for significant issues
        if severity in ["high", "critical"]:
            actions.append(RemediationAction.ALERT_NOTIFICATION)

        return actions

    async def _execute_auto_remediation(self, event: DegradationEvent):
        """Execute automatic remediation for a degradation event."""
        for action in event.remediation_actions:
            try:
                result = await self.trigger_manual_remediation(
                    action, target_metric=event.metric_type, metadata={"event_id": event.event_id, "auto_remediation": True}
                )

                if result.success:
                    self.logger.info(f"Auto remediation successful: {action.value} for event {event.event_id}")
                else:
                    self.logger.warning(f"Auto remediation failed: {action.value} for event {event.event_id}")

            except Exception as e:
                self.logger.error(f"Error in auto remediation {action.value}: {e}")

    async def _perform_cache_eviction(self):
        """Perform cache eviction to free memory."""
        if self._cache_service and hasattr(self._cache_service, "l1_cache"):
            # Evict a portion of L1 cache
            l1_cache = self._cache_service.l1_cache
            if hasattr(l1_cache, "_cache") and l1_cache._cache:
                # Evict 25% of cache entries
                keys_to_evict = list(l1_cache._cache.keys())[: len(l1_cache._cache) // 4]
                for key in keys_to_evict:
                    l1_cache._cache.pop(key, None)

    async def _restart_connection_pool(self):
        """Restart connection pool to resolve connection issues."""
        # This would restart Redis connection pool
        # Implementation depends on cache service architecture
        self.logger.info("Connection pool restart triggered")

    async def _trigger_garbage_collection(self):
        """Trigger garbage collection to free memory."""
        import gc

        gc.collect()

    async def _perform_cache_warmup(self):
        """Perform cache warmup to improve hit rates."""
        # This would implement cache warmup logic
        # Implementation depends on specific cache warming strategy
        self.logger.info("Cache warmup triggered")


# Global service instance
_performance_service = None


async def get_cache_performance_service() -> CachePerformanceService:
    """Get the global cache performance service instance."""
    global _performance_service
    if _performance_service is None:
        _performance_service = CachePerformanceService()
        await _performance_service.initialize()
    return _performance_service


async def cleanup_performance_service():
    """Clean up the performance service instance."""
    global _performance_service
    if _performance_service is not None:
        await _performance_service.shutdown()
        _performance_service = None
