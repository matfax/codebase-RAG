"""
Performance monitoring and progress tracking utilities for codebase indexing operations.

Extended with comprehensive cache metrics collection for all cache services.
"""

import asyncio
import json
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import psutil


@dataclass
class ProcessingStats:
    """Statistics for a processing operation."""

    total_items: int
    processed_items: int = 0
    failed_items: int = 0
    start_time: float = field(default_factory=time.time)
    stage_times: dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0


class CacheOperation(Enum):
    """Types of cache operations for metrics tracking."""

    GET = "get"
    SET = "set"
    DELETE = "delete"
    EXISTS = "exists"
    CLEAR = "clear"
    BATCH_GET = "batch_get"
    BATCH_SET = "batch_set"
    INVALIDATE = "invalidate"


@dataclass
class CacheMetrics:
    """Comprehensive cache metrics for performance monitoring."""

    # Hit/Miss Statistics
    hit_count: int = 0
    miss_count: int = 0

    # Operation Statistics
    get_count: int = 0
    set_count: int = 0
    delete_count: int = 0
    exists_count: int = 0
    clear_count: int = 0
    batch_get_count: int = 0
    batch_set_count: int = 0
    invalidate_count: int = 0

    # Error Statistics
    error_count: int = 0
    timeout_count: int = 0
    connection_error_count: int = 0

    # Performance Statistics
    total_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    response_time_samples: int = 0

    # Memory and Size Statistics
    current_size_bytes: int = 0
    max_size_bytes: int = 0
    eviction_count: int = 0
    memory_pressure_events: int = 0

    # Cache Size and Cleanup Tracking
    size_history: list[tuple[float, int]] = field(default_factory=list)  # (timestamp, size_bytes)
    cleanup_events: list[dict[str, Any]] = field(default_factory=list)  # Cleanup event details
    last_cleanup_time: float = 0.0
    cleanup_frequency_seconds: float = 0.0  # Average time between cleanups
    size_growth_rate_bytes_per_second: float = 0.0  # Rate of size growth
    avg_size_bytes: float = 0.0  # Average size over time
    size_variance: float = 0.0  # Size variance for stability measurement

    # Cache-specific Statistics
    l1_hit_count: int = 0  # In-memory cache hits
    l2_hit_count: int = 0  # Redis cache hits
    promotion_count: int = 0  # L2 to L1 promotions
    demotion_count: int = 0  # L1 to L2 demotions

    # Timestamps
    first_operation_time: float = field(default_factory=time.time)
    last_operation_time: float = field(default_factory=time.time)
    last_reset_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_gets = self.hit_count + self.miss_count
        return self.hit_count / total_gets if total_gets > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate

    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / self.response_time_samples if self.response_time_samples > 0 else 0.0

    @property
    def total_operations(self) -> int:
        """Calculate total number of operations."""
        return (
            self.get_count
            + self.set_count
            + self.delete_count
            + self.exists_count
            + self.clear_count
            + self.batch_get_count
            + self.batch_set_count
            + self.invalidate_count
        )

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.total_operations
        return self.error_count / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 (in-memory) cache hit rate."""
        total_hits = self.l1_hit_count + self.l2_hit_count
        return self.l1_hit_count / total_hits if total_hits > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """Calculate L2 (Redis) cache hit rate."""
        total_hits = self.l1_hit_count + self.l2_hit_count
        return self.l2_hit_count / total_hits if total_hits > 0 else 0.0

    def record_operation(self, operation: CacheOperation, response_time: float = 0.0, hit: bool = False) -> None:
        """Record a cache operation."""
        self.last_operation_time = time.time()

        # Update operation counters
        if operation == CacheOperation.GET:
            self.get_count += 1
            if hit:
                self.hit_count += 1
            else:
                self.miss_count += 1
        elif operation == CacheOperation.SET:
            self.set_count += 1
        elif operation == CacheOperation.DELETE:
            self.delete_count += 1
        elif operation == CacheOperation.EXISTS:
            self.exists_count += 1
        elif operation == CacheOperation.CLEAR:
            self.clear_count += 1
        elif operation == CacheOperation.BATCH_GET:
            self.batch_get_count += 1
        elif operation == CacheOperation.BATCH_SET:
            self.batch_set_count += 1
        elif operation == CacheOperation.INVALIDATE:
            self.invalidate_count += 1

        # Update response time statistics
        if response_time > 0:
            self.total_response_time += response_time
            self.response_time_samples += 1
            self.min_response_time = min(self.min_response_time, response_time)
            self.max_response_time = max(self.max_response_time, response_time)

    def record_error(self, error_type: str = "general") -> None:
        """Record a cache error."""
        self.error_count += 1
        if error_type == "timeout":
            self.timeout_count += 1
        elif error_type == "connection":
            self.connection_error_count += 1

    def record_cache_level_hit(self, level: str) -> None:
        """Record a cache hit at specific level."""
        if level == "l1":
            self.l1_hit_count += 1
        elif level == "l2":
            self.l2_hit_count += 1

    def record_eviction(self) -> None:
        """Record a cache eviction event."""
        self.eviction_count += 1

    def record_memory_pressure(self) -> None:
        """Record a memory pressure event."""
        self.memory_pressure_events += 1

    def update_size(self, size_bytes: int) -> None:
        """Update current cache size and track size history."""
        current_time = time.time()
        self.current_size_bytes = size_bytes
        self.max_size_bytes = max(self.max_size_bytes, size_bytes)

        # Record size history
        self.size_history.append((current_time, size_bytes))

        # Keep only recent history (last 100 entries)
        if len(self.size_history) > 100:
            self.size_history = self.size_history[-100:]

        # Calculate size growth rate if we have enough data
        if len(self.size_history) >= 2:
            time_span = self.size_history[-1][0] - self.size_history[0][0]
            if time_span > 0:
                size_change = self.size_history[-1][1] - self.size_history[0][1]
                self.size_growth_rate_bytes_per_second = size_change / time_span

        # Calculate average size and variance
        if len(self.size_history) >= 2:
            sizes = [entry[1] for entry in self.size_history]
            self.avg_size_bytes = sum(sizes) / len(sizes)

            # Calculate variance
            if len(sizes) > 1:
                variance_sum = sum((size - self.avg_size_bytes) ** 2 for size in sizes)
                self.size_variance = variance_sum / len(sizes)

    def record_cleanup_event(self, cleanup_type: str, items_removed: int = 0, bytes_freed: int = 0, reason: str = "") -> None:
        """Record a cache cleanup/eviction event."""
        current_time = time.time()

        cleanup_event = {
            "timestamp": current_time,
            "type": cleanup_type,  # "eviction", "ttl_cleanup", "manual_clear", "size_limit", etc.
            "items_removed": items_removed,
            "bytes_freed": bytes_freed,
            "reason": reason,
        }

        self.cleanup_events.append(cleanup_event)

        # Keep only recent cleanup events (last 50 events)
        if len(self.cleanup_events) > 50:
            self.cleanup_events = self.cleanup_events[-50:]

        # Calculate cleanup frequency
        if len(self.cleanup_events) >= 2:
            time_span = self.cleanup_events[-1]["timestamp"] - self.cleanup_events[0]["timestamp"]
            if time_span > 0:
                self.cleanup_frequency_seconds = time_span / (len(self.cleanup_events) - 1)

        self.last_cleanup_time = current_time

    def get_size_statistics(self) -> dict[str, Any]:
        """Get detailed size statistics."""
        if not self.size_history:
            return {
                "current_size_bytes": self.current_size_bytes,
                "max_size_bytes": self.max_size_bytes,
                "avg_size_bytes": 0,
                "size_variance": 0,
                "size_growth_rate_bytes_per_second": 0,
                "size_stability": "unknown",
                "samples": 0,
            }

        # Calculate size stability
        stability = "stable"
        if self.size_variance > (self.avg_size_bytes * 0.1) ** 2:  # High variance
            stability = "unstable"
        elif abs(self.size_growth_rate_bytes_per_second) > 1024:  # Growing/shrinking > 1KB/s
            stability = "growing" if self.size_growth_rate_bytes_per_second > 0 else "shrinking"

        return {
            "current_size_bytes": self.current_size_bytes,
            "current_size_mb": round(self.current_size_bytes / 1024 / 1024, 2),
            "max_size_bytes": self.max_size_bytes,
            "max_size_mb": round(self.max_size_bytes / 1024 / 1024, 2),
            "avg_size_bytes": round(self.avg_size_bytes, 2),
            "avg_size_mb": round(self.avg_size_bytes / 1024 / 1024, 2),
            "size_variance": round(self.size_variance, 2),
            "size_growth_rate_bytes_per_second": round(self.size_growth_rate_bytes_per_second, 2),
            "size_growth_rate_mb_per_hour": round(self.size_growth_rate_bytes_per_second * 3600 / 1024 / 1024, 2),
            "size_stability": stability,
            "samples": len(self.size_history),
            "tracking_duration_seconds": self.size_history[-1][0] - self.size_history[0][0] if len(self.size_history) >= 2 else 0,
        }

    def get_cleanup_statistics(self) -> dict[str, Any]:
        """Get detailed cleanup statistics."""
        if not self.cleanup_events:
            return {
                "total_cleanup_events": 0,
                "cleanup_frequency_seconds": 0,
                "cleanup_frequency_minutes": 0,
                "last_cleanup_time": 0,
                "time_since_last_cleanup_seconds": 0,
                "cleanup_types": {},
                "total_items_removed": 0,
                "total_bytes_freed": 0,
                "avg_items_per_cleanup": 0,
                "avg_bytes_per_cleanup": 0,
            }

        # Calculate cleanup type distribution
        cleanup_types = {}
        total_items_removed = 0
        total_bytes_freed = 0

        for event in self.cleanup_events:
            cleanup_type = event["type"]
            cleanup_types[cleanup_type] = cleanup_types.get(cleanup_type, 0) + 1
            total_items_removed += event["items_removed"]
            total_bytes_freed += event["bytes_freed"]

        current_time = time.time()

        return {
            "total_cleanup_events": len(self.cleanup_events),
            "cleanup_frequency_seconds": round(self.cleanup_frequency_seconds, 2),
            "cleanup_frequency_minutes": round(self.cleanup_frequency_seconds / 60, 2),
            "last_cleanup_time": self.last_cleanup_time,
            "time_since_last_cleanup_seconds": round(current_time - self.last_cleanup_time, 2) if self.last_cleanup_time > 0 else 0,
            "cleanup_types": cleanup_types,
            "total_items_removed": total_items_removed,
            "total_bytes_freed": total_bytes_freed,
            "total_bytes_freed_mb": round(total_bytes_freed / 1024 / 1024, 2),
            "avg_items_per_cleanup": round(total_items_removed / len(self.cleanup_events), 2),
            "avg_bytes_per_cleanup": round(total_bytes_freed / len(self.cleanup_events), 2),
            "avg_bytes_per_cleanup_mb": round(total_bytes_freed / len(self.cleanup_events) / 1024 / 1024, 2),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.hit_count = 0
        self.miss_count = 0
        self.get_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.exists_count = 0
        self.clear_count = 0
        self.batch_get_count = 0
        self.batch_set_count = 0
        self.invalidate_count = 0
        self.error_count = 0
        self.timeout_count = 0
        self.connection_error_count = 0
        self.total_response_time = 0.0
        self.min_response_time = float("inf")
        self.max_response_time = 0.0
        self.response_time_samples = 0
        self.current_size_bytes = 0
        self.max_size_bytes = 0
        self.eviction_count = 0
        self.memory_pressure_events = 0
        self.l1_hit_count = 0
        self.l2_hit_count = 0
        self.promotion_count = 0
        self.demotion_count = 0
        self.last_reset_time = time.time()

        # Reset size and cleanup tracking
        self.size_history.clear()
        self.cleanup_events.clear()
        self.last_cleanup_time = 0.0
        self.cleanup_frequency_seconds = 0.0
        self.size_growth_rate_bytes_per_second = 0.0
        self.avg_size_bytes = 0.0
        self.size_variance = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "operations": {
                "get": self.get_count,
                "set": self.set_count,
                "delete": self.delete_count,
                "exists": self.exists_count,
                "clear": self.clear_count,
                "batch_get": self.batch_get_count,
                "batch_set": self.batch_set_count,
                "invalidate": self.invalidate_count,
                "total": self.total_operations,
            },
            "errors": {
                "total": self.error_count,
                "timeout": self.timeout_count,
                "connection": self.connection_error_count,
                "rate": self.error_rate,
            },
            "performance": {
                "average_response_time_ms": self.average_response_time * 1000,
                "min_response_time_ms": self.min_response_time * 1000 if self.min_response_time != float("inf") else 0,
                "max_response_time_ms": self.max_response_time * 1000,
                "total_response_time_ms": self.total_response_time * 1000,
                "samples": self.response_time_samples,
            },
            "memory": {
                "current_size_bytes": self.current_size_bytes,
                "max_size_bytes": self.max_size_bytes,
                "eviction_count": self.eviction_count,
                "memory_pressure_events": self.memory_pressure_events,
            },
            "cache_levels": {
                "l1_hit_count": self.l1_hit_count,
                "l2_hit_count": self.l2_hit_count,
                "l1_hit_rate": self.l1_hit_rate,
                "l2_hit_rate": self.l2_hit_rate,
                "promotion_count": self.promotion_count,
                "demotion_count": self.demotion_count,
            },
            "timestamps": {
                "first_operation_time": self.first_operation_time,
                "last_operation_time": self.last_operation_time,
                "last_reset_time": self.last_reset_time,
                "uptime_seconds": time.time() - self.first_operation_time,
            },
            "size_tracking": self.get_size_statistics(),
            "cleanup_tracking": self.get_cleanup_statistics(),
        }

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100.0

    @property
    def processing_rate(self) -> float:
        """Calculate items processed per second."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed

    @property
    def eta_seconds(self) -> float | None:
        """Calculate estimated time to completion in seconds."""
        remaining_items = self.total_items - self.processed_items
        if remaining_items <= 0 or self.processing_rate == 0:
            return None
        return remaining_items / self.processing_rate

    @property
    def eta_formatted(self) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds
        if eta is None:
            return "N/A"

        if eta < 60:
            return f"{eta:.0f}s"
        elif eta < 3600:
            return f"{eta / 60:.0f}m {eta % 60:.0f}s"
        else:
            hours = eta // 3600
            minutes = (eta % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


class ProgressTracker:
    """Thread-safe progress tracker with ETA estimation and memory monitoring."""

    def __init__(self, total_items: int, description: str = "Processing"):
        self.stats = ProcessingStats(total_items=total_items)
        self.description = description
        self._lock = threading.Lock()
        self._stage_start_times: dict[str, float] = {}

    def start_stage(self, stage_name: str) -> None:
        """Mark the start of a processing stage."""
        with self._lock:
            self._stage_start_times[stage_name] = time.time()

    def end_stage(self, stage_name: str) -> None:
        """Mark the end of a processing stage and record timing."""
        with self._lock:
            if stage_name in self._stage_start_times:
                duration = time.time() - self._stage_start_times[stage_name]
                self.stats.stage_times[stage_name] = duration
                del self._stage_start_times[stage_name]

    def increment_processed(self, count: int = 1) -> None:
        """Increment the count of processed items."""
        with self._lock:
            self.stats.processed_items += count
            self._update_memory_usage()

    def increment_failed(self, count: int = 1) -> None:
        """Increment the count of failed items."""
        with self._lock:
            self.stats.failed_items += count
            self._update_memory_usage()

    def _update_memory_usage(self) -> None:
        """Update current memory usage."""
        try:
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback if psutil fails
            self.stats.memory_usage_mb = 0.0

    def get_progress_summary(self) -> dict[str, Any]:
        """Get a comprehensive progress summary."""
        with self._lock:
            elapsed_time = time.time() - self.stats.start_time

            return {
                "description": self.description,
                "total_items": self.stats.total_items,
                "processed_items": self.stats.processed_items,
                "failed_items": self.stats.failed_items,
                "completion_percentage": round(self.stats.completion_percentage, 1),
                "processing_rate": round(self.stats.processing_rate, 2),
                "eta": self.stats.eta_formatted,
                "elapsed_time": f"{elapsed_time:.1f}s",
                "memory_usage_mb": round(self.stats.memory_usage_mb, 1),
                "stage_times": {k: f"{v:.2f}s" for k, v in self.stats.stage_times.items()},
            }

    def log_progress(self, logger, level: str = "info") -> None:
        """Log current progress to the provided logger."""
        summary = self.get_progress_summary()

        message = (
            f"{self.description}: {summary['processed_items']}/{summary['total_items']} "
            f"({summary['completion_percentage']}%) - "
            f"{summary['processing_rate']} items/s - "
            f"ETA: {summary['eta']} - "
            f"Memory: {summary['memory_usage_mb']}MB"
        )

        if summary["failed_items"] > 0:
            message += f" - Failed: {summary['failed_items']}"

        getattr(logger, level)(message)

    def is_complete(self) -> bool:
        """Check if processing is complete."""
        with self._lock:
            return self.stats.processed_items + self.stats.failed_items >= self.stats.total_items


class CachePerformanceMonitor:
    """Comprehensive cache performance monitoring system."""

    def __init__(self):
        self._cache_metrics: dict[str, CacheMetrics] = {}
        self._cache_types: dict[str, str] = {}  # cache_name -> cache_type mapping
        self._cache_type_aggregated_metrics: dict[str, CacheMetrics] = {}  # cache_type -> aggregated metrics
        self._lock = threading.Lock()
        self._monitoring_enabled = True
        self._alert_thresholds = {
            "error_rate_threshold": 0.05,  # 5% error rate
            "response_time_threshold_ms": 1000,  # 1 second
            "hit_rate_threshold": 0.8,  # 80% hit rate
            "memory_usage_threshold_mb": 500,  # 500MB per cache
        }
        self._alerts: list[dict[str, Any]] = []
        self._max_alerts = 100

        # Cache type definitions for standardized tracking
        self._standard_cache_types = {
            "embedding": "EmbeddingCacheService",
            "search": "SearchCacheService",
            "project": "ProjectCacheService",
            "file": "FileCacheService",
            "redis": "RedisCache",
            "memory": "MemoryCache",
            "hybrid": "HybridCache",
            "unknown": "UnknownCache",
        }

    def register_cache(self, cache_name: str, cache_type: str) -> None:
        """Register a cache service for monitoring."""
        with self._lock:
            if cache_name not in self._cache_metrics:
                self._cache_metrics[cache_name] = CacheMetrics()
                self._cache_types[cache_name] = cache_type

            # Initialize cache type aggregated metrics if not exists
            if cache_type not in self._cache_type_aggregated_metrics:
                self._cache_type_aggregated_metrics[cache_type] = CacheMetrics()

    def record_operation(
        self,
        cache_name: str,
        operation: CacheOperation,
        response_time: float = 0.0,
        hit: bool = False,
        cache_level: str | None = None,
    ) -> None:
        """Record a cache operation with timing and hit/miss information."""
        if not self._monitoring_enabled:
            return

        with self._lock:
            if cache_name not in self._cache_metrics:
                self.register_cache(cache_name, "unknown")

            metrics = self._cache_metrics[cache_name]
            metrics.record_operation(operation, response_time, hit)

            if hit and cache_level:
                metrics.record_cache_level_hit(cache_level)

            # Update cache type aggregated metrics
            cache_type = self._cache_types[cache_name]
            if cache_type in self._cache_type_aggregated_metrics:
                self._cache_type_aggregated_metrics[cache_type].record_operation(operation, response_time, hit)
                if hit and cache_level:
                    self._cache_type_aggregated_metrics[cache_type].record_cache_level_hit(cache_level)

            # Check for alerts
            self._check_alerts(cache_name, metrics)

    def record_error(self, cache_name: str, error_type: str = "general") -> None:
        """Record a cache error."""
        if not self._monitoring_enabled:
            return

        with self._lock:
            if cache_name not in self._cache_metrics:
                self.register_cache(cache_name, "unknown")

            self._cache_metrics[cache_name].record_error(error_type)

            # Update cache type aggregated metrics
            cache_type = self._cache_types[cache_name]
            if cache_type in self._cache_type_aggregated_metrics:
                self._cache_type_aggregated_metrics[cache_type].record_error(error_type)

    def update_cache_size(self, cache_name: str, size_bytes: int) -> None:
        """Update cache size information."""
        if not self._monitoring_enabled:
            return

        with self._lock:
            if cache_name not in self._cache_metrics:
                self.register_cache(cache_name, "unknown")

            self._cache_metrics[cache_name].update_size(size_bytes)

            # Update cache type aggregated metrics
            cache_type = self._cache_types[cache_name]
            if cache_type in self._cache_type_aggregated_metrics:
                # For aggregated metrics, we sum the sizes
                total_size = sum(
                    self._cache_metrics[name].current_size_bytes for name, type_name in self._cache_types.items() if type_name == cache_type
                )
                self._cache_type_aggregated_metrics[cache_type].update_size(total_size)

    def record_eviction(self, cache_name: str) -> None:
        """Record a cache eviction event."""
        if not self._monitoring_enabled:
            return

        with self._lock:
            if cache_name not in self._cache_metrics:
                self.register_cache(cache_name, "unknown")

            self._cache_metrics[cache_name].record_eviction()

            # Update cache type aggregated metrics
            cache_type = self._cache_types[cache_name]
            if cache_type in self._cache_type_aggregated_metrics:
                self._cache_type_aggregated_metrics[cache_type].record_eviction()

    def record_cleanup_event(
        self, cache_name: str, cleanup_type: str, items_removed: int = 0, bytes_freed: int = 0, reason: str = ""
    ) -> None:
        """Record a cache cleanup/eviction event."""
        if not self._monitoring_enabled:
            return

        with self._lock:
            if cache_name not in self._cache_metrics:
                self.register_cache(cache_name, "unknown")

            self._cache_metrics[cache_name].record_cleanup_event(cleanup_type, items_removed, bytes_freed, reason)

            # Update cache type aggregated metrics
            cache_type = self._cache_types[cache_name]
            if cache_type in self._cache_type_aggregated_metrics:
                self._cache_type_aggregated_metrics[cache_type].record_cleanup_event(cleanup_type, items_removed, bytes_freed, reason)

    def get_cache_size_statistics(self, cache_name: str) -> dict[str, Any] | None:
        """Get size statistics for a specific cache."""
        with self._lock:
            if cache_name in self._cache_metrics:
                return self._cache_metrics[cache_name].get_size_statistics()
            return None

    def get_cache_cleanup_statistics(self, cache_name: str) -> dict[str, Any] | None:
        """Get cleanup statistics for a specific cache."""
        with self._lock:
            if cache_name in self._cache_metrics:
                return self._cache_metrics[cache_name].get_cleanup_statistics()
            return None

    def get_all_cache_size_statistics(self) -> dict[str, dict[str, Any]]:
        """Get size statistics for all caches."""
        with self._lock:
            return {cache_name: metrics.get_size_statistics() for cache_name, metrics in self._cache_metrics.items()}

    def get_all_cache_cleanup_statistics(self) -> dict[str, dict[str, Any]]:
        """Get cleanup statistics for all caches."""
        with self._lock:
            return {cache_name: metrics.get_cleanup_statistics() for cache_name, metrics in self._cache_metrics.items()}

    def get_cache_type_size_statistics(self, cache_type: str) -> dict[str, Any] | None:
        """Get aggregated size statistics for a cache type."""
        with self._lock:
            if cache_type in self._cache_type_aggregated_metrics:
                return self._cache_type_aggregated_metrics[cache_type].get_size_statistics()
            return None

    def get_cache_type_cleanup_statistics(self, cache_type: str) -> dict[str, Any] | None:
        """Get aggregated cleanup statistics for a cache type."""
        with self._lock:
            if cache_type in self._cache_type_aggregated_metrics:
                return self._cache_type_aggregated_metrics[cache_type].get_cleanup_statistics()
            return None

    def get_cache_metrics(self, cache_name: str) -> dict[str, Any] | None:
        """Get metrics for a specific cache."""
        with self._lock:
            if cache_name in self._cache_metrics:
                return self._cache_metrics[cache_name].to_dict()
            return None

    def get_all_cache_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all registered caches."""
        with self._lock:
            return {cache_name: metrics.to_dict() for cache_name, metrics in self._cache_metrics.items()}

    def get_cache_type_metrics(self, cache_type: str) -> dict[str, Any] | None:
        """Get aggregated metrics for a specific cache type."""
        with self._lock:
            if cache_type in self._cache_type_aggregated_metrics:
                return self._cache_type_aggregated_metrics[cache_type].to_dict()
            return None

    def get_all_cache_type_metrics(self) -> dict[str, dict[str, Any]]:
        """Get aggregated metrics for all cache types."""
        with self._lock:
            return {cache_type: metrics.to_dict() for cache_type, metrics in self._cache_type_aggregated_metrics.items()}

    def get_cache_type_hit_rates(self) -> dict[str, float]:
        """Get hit rates for all cache types."""
        with self._lock:
            return {cache_type: metrics.hit_rate for cache_type, metrics in self._cache_type_aggregated_metrics.items()}

    def get_cache_type_summary(self) -> dict[str, dict[str, Any]]:
        """Get a summary of cache performance by type."""
        with self._lock:
            summary = {}
            for cache_type, metrics in self._cache_type_aggregated_metrics.items():
                # Count instances of this cache type
                instance_count = sum(1 for t in self._cache_types.values() if t == cache_type)

                summary[cache_type] = {
                    "instance_count": instance_count,
                    "hit_rate": metrics.hit_rate,
                    "miss_rate": metrics.miss_rate,
                    "total_operations": metrics.total_operations,
                    "error_rate": metrics.error_rate,
                    "average_response_time_ms": metrics.average_response_time * 1000,
                    "current_size_mb": metrics.current_size_bytes / 1024 / 1024,
                    "l1_hit_rate": metrics.l1_hit_rate,
                    "l2_hit_rate": metrics.l2_hit_rate,
                    "eviction_count": metrics.eviction_count,
                    "memory_pressure_events": metrics.memory_pressure_events,
                }
            return summary

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """Get aggregated metrics across all caches."""
        with self._lock:
            if not self._cache_metrics:
                return {"error": "No cache metrics available"}

            # Aggregate metrics
            total_hits = sum(m.hit_count for m in self._cache_metrics.values())
            total_misses = sum(m.miss_count for m in self._cache_metrics.values())
            total_operations = sum(m.total_operations for m in self._cache_metrics.values())
            total_errors = sum(m.error_count for m in self._cache_metrics.values())
            total_size = sum(m.current_size_bytes for m in self._cache_metrics.values())
            total_evictions = sum(m.eviction_count for m in self._cache_metrics.values())

            # Calculate aggregated response time
            total_response_time = sum(m.total_response_time for m in self._cache_metrics.values())
            total_samples = sum(m.response_time_samples for m in self._cache_metrics.values())
            avg_response_time = total_response_time / total_samples if total_samples > 0 else 0.0

            # Cache type breakdown
            type_breakdown = defaultdict(lambda: {"count": 0, "operations": 0, "errors": 0})
            for cache_name, metrics in self._cache_metrics.items():
                cache_type = self._cache_types.get(cache_name, "unknown")
                type_breakdown[cache_type]["count"] += 1
                type_breakdown[cache_type]["operations"] += metrics.total_operations
                type_breakdown[cache_type]["errors"] += metrics.error_count

            return {
                "summary": {
                    "total_caches": len(self._cache_metrics),
                    "total_operations": total_operations,
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "total_errors": total_errors,
                    "overall_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
                    "overall_error_rate": total_errors / total_operations if total_operations > 0 else 0.0,
                    "average_response_time_ms": avg_response_time * 1000,
                    "total_size_mb": total_size / 1024 / 1024,
                    "total_evictions": total_evictions,
                },
                "by_cache_type": dict(type_breakdown),
                "cache_type_summary": self.get_cache_type_summary(),
                "cache_type_hit_rates": self.get_cache_type_hit_rates(),
                "individual_caches": list(self._cache_metrics.keys()),
                "active_alerts": len(self._alerts),
                "monitoring_enabled": self._monitoring_enabled,
                "timestamp": time.time(),
            }

    def _check_alerts(self, cache_name: str, metrics: CacheMetrics) -> None:
        """Check if any alert thresholds are exceeded."""
        alerts_to_add = []

        # Check error rate
        if metrics.error_rate > self._alert_thresholds["error_rate_threshold"]:
            alerts_to_add.append(
                {
                    "type": "high_error_rate",
                    "cache_name": cache_name,
                    "value": metrics.error_rate,
                    "threshold": self._alert_thresholds["error_rate_threshold"],
                    "timestamp": time.time(),
                }
            )

        # Check response time
        if metrics.average_response_time * 1000 > self._alert_thresholds["response_time_threshold_ms"]:
            alerts_to_add.append(
                {
                    "type": "slow_response_time",
                    "cache_name": cache_name,
                    "value": metrics.average_response_time * 1000,
                    "threshold": self._alert_thresholds["response_time_threshold_ms"],
                    "timestamp": time.time(),
                }
            )

        # Check hit rate
        if metrics.hit_rate < self._alert_thresholds["hit_rate_threshold"] and metrics.total_operations > 10:
            alerts_to_add.append(
                {
                    "type": "low_hit_rate",
                    "cache_name": cache_name,
                    "value": metrics.hit_rate,
                    "threshold": self._alert_thresholds["hit_rate_threshold"],
                    "timestamp": time.time(),
                }
            )

        # Check memory usage
        cache_size_mb = metrics.current_size_bytes / 1024 / 1024
        if cache_size_mb > self._alert_thresholds["memory_usage_threshold_mb"]:
            alerts_to_add.append(
                {
                    "type": "high_memory_usage",
                    "cache_name": cache_name,
                    "value": cache_size_mb,
                    "threshold": self._alert_thresholds["memory_usage_threshold_mb"],
                    "timestamp": time.time(),
                }
            )

        # Add alerts and maintain max limit
        for alert in alerts_to_add:
            self._alerts.append(alert)

        # Keep only recent alerts
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]

    def get_alerts(self, cache_name: str | None = None, alert_type: str | None = None) -> list[dict[str, Any]]:
        """Get current alerts, optionally filtered by cache name or type."""
        with self._lock:
            alerts = self._alerts

            if cache_name:
                alerts = [a for a in alerts if a.get("cache_name") == cache_name]

            if alert_type:
                alerts = [a for a in alerts if a.get("type") == alert_type]

            return alerts

    def clear_alerts(self, cache_name: str | None = None) -> None:
        """Clear alerts, optionally for a specific cache."""
        with self._lock:
            if cache_name:
                self._alerts = [a for a in self._alerts if a.get("cache_name") != cache_name]
            else:
                self._alerts.clear()

    def set_alert_threshold(self, threshold_name: str, value: float) -> None:
        """Set an alert threshold."""
        with self._lock:
            if threshold_name in self._alert_thresholds:
                self._alert_thresholds[threshold_name] = value

    def reset_cache_metrics(self, cache_name: str) -> None:
        """Reset metrics for a specific cache."""
        with self._lock:
            if cache_name in self._cache_metrics:
                self._cache_metrics[cache_name].reset()
                # Clear alerts for this cache
                self._alerts = [a for a in self._alerts if a.get("cache_name") != cache_name]

    def reset_cache_type_metrics(self, cache_type: str) -> None:
        """Reset aggregated metrics for a specific cache type."""
        with self._lock:
            if cache_type in self._cache_type_aggregated_metrics:
                self._cache_type_aggregated_metrics[cache_type].reset()
                # Clear alerts for caches of this type
                self._alerts = [a for a in self._alerts if self._cache_types.get(a.get("cache_name")) != cache_type]

    def reset_all_metrics(self) -> None:
        """Reset all cache metrics."""
        with self._lock:
            for metrics in self._cache_metrics.values():
                metrics.reset()
            for metrics in self._cache_type_aggregated_metrics.values():
                metrics.reset()
            self._alerts.clear()

    def enable_monitoring(self) -> None:
        """Enable cache monitoring."""
        self._monitoring_enabled = True

    def disable_monitoring(self) -> None:
        """Disable cache monitoring."""
        self._monitoring_enabled = False

    def is_monitoring_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self._monitoring_enabled


class RealTimeCacheStatsReporter:
    """Real-time cache statistics reporter for live monitoring."""

    def __init__(self, cache_monitor: CachePerformanceMonitor):
        self._cache_monitor = cache_monitor
        self._reporting_enabled = False
        self._report_interval = 5.0  # seconds
        self._report_thread = None
        self._stop_event = threading.Event()
        self._subscribers = []  # List of callback functions
        self._report_history = []  # Recent reports for trend analysis
        self._max_history = 100
        self._last_report_time = 0.0
        self._lock = threading.Lock()

    def subscribe(self, callback_func) -> None:
        """Subscribe to real-time cache statistics updates."""
        with self._lock:
            if callback_func not in self._subscribers:
                self._subscribers.append(callback_func)

    def unsubscribe(self, callback_func) -> None:
        """Unsubscribe from real-time cache statistics updates."""
        with self._lock:
            if callback_func in self._subscribers:
                self._subscribers.remove(callback_func)

    def set_report_interval(self, seconds: float) -> None:
        """Set the reporting interval in seconds."""
        self._report_interval = max(0.1, seconds)  # Minimum 100ms

    def start_reporting(self) -> None:
        """Start real-time cache statistics reporting."""
        if self._reporting_enabled:
            return

        self._reporting_enabled = True
        self._stop_event.clear()
        self._report_thread = threading.Thread(target=self._reporting_loop, daemon=True)
        self._report_thread.start()

    def stop_reporting(self) -> None:
        """Stop real-time cache statistics reporting."""
        if not self._reporting_enabled:
            return

        self._reporting_enabled = False
        self._stop_event.set()

        if self._report_thread and self._report_thread.is_alive():
            self._report_thread.join(timeout=1.0)

    def _reporting_loop(self) -> None:
        """Main reporting loop that runs in a separate thread."""
        while self._reporting_enabled and not self._stop_event.wait(self._report_interval):
            try:
                report = self._generate_report()
                self._distribute_report(report)
                self._store_report(report)
            except Exception:
                # Log error but continue reporting
                pass

    def _generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive real-time cache statistics report."""
        current_time = time.time()

        # Get basic metrics
        aggregated_metrics = self._cache_monitor.get_aggregated_metrics()
        cache_type_summary = self._cache_monitor.get_cache_type_summary()
        all_cache_metrics = self._cache_monitor.get_all_cache_metrics()

        # Get size and cleanup statistics
        size_stats = self._cache_monitor.get_all_cache_size_statistics()
        cleanup_stats = self._cache_monitor.get_all_cache_cleanup_statistics()

        # Calculate deltas from last report
        deltas = self._calculate_deltas(aggregated_metrics, current_time)

        report = {
            "timestamp": current_time,
            "report_id": f"cache_stats_{current_time}",
            "summary": aggregated_metrics.get("summary", {}),
            "cache_types": cache_type_summary,
            "individual_caches": {
                cache_name: {
                    "metrics": metrics,
                    "size_stats": size_stats.get(cache_name, {}),
                    "cleanup_stats": cleanup_stats.get(cache_name, {}),
                }
                for cache_name, metrics in all_cache_metrics.items()
            },
            "deltas": deltas,
            "system_info": {
                "cpu_percent": self._get_cpu_percent(),
                "memory_info": self._get_memory_info(),
                "active_cache_count": len(all_cache_metrics),
                "monitoring_enabled": self._cache_monitor.is_monitoring_enabled(),
            },
            "alerts": self._cache_monitor.get_alerts(),
            "trends": self._calculate_trends(),
        }

        self._last_report_time = current_time
        return report

    def _calculate_deltas(self, current_metrics: dict[str, Any], current_time: float) -> dict[str, Any]:
        """Calculate deltas from the last report."""
        if not self._report_history:
            return {"available": False, "reason": "No previous report"}

        last_report = self._report_history[-1]
        time_delta = current_time - last_report["timestamp"]

        if time_delta <= 0:
            return {"available": False, "reason": "Invalid time delta"}

        current_summary = current_metrics.get("summary", {})
        last_summary = last_report.get("summary", {})

        return {
            "available": True,
            "time_delta_seconds": round(time_delta, 2),
            "operations_per_second": round(
                (current_summary.get("total_operations", 0) - last_summary.get("total_operations", 0)) / time_delta, 2
            ),
            "hits_per_second": round((current_summary.get("total_hits", 0) - last_summary.get("total_hits", 0)) / time_delta, 2),
            "misses_per_second": round((current_summary.get("total_misses", 0) - last_summary.get("total_misses", 0)) / time_delta, 2),
            "errors_per_second": round((current_summary.get("total_errors", 0) - last_summary.get("total_errors", 0)) / time_delta, 2),
            "cache_size_change_mb": round((current_summary.get("total_size_mb", 0) - last_summary.get("total_size_mb", 0)), 2),
            "hit_rate_change": round((current_summary.get("overall_hit_rate", 0) - last_summary.get("overall_hit_rate", 0)) * 100, 2),
        }

    def _calculate_trends(self) -> dict[str, Any]:
        """Calculate trends from recent reports."""
        if len(self._report_history) < 3:
            return {"available": False, "reason": "Insufficient history"}

        # Get last 10 reports for trend analysis
        recent_reports = self._report_history[-10:]

        # Calculate hit rate trend
        hit_rates = [r.get("summary", {}).get("overall_hit_rate", 0) for r in recent_reports]
        hit_rate_trend = self._calculate_linear_trend(hit_rates)

        # Calculate operation rate trend
        operation_counts = [r.get("summary", {}).get("total_operations", 0) for r in recent_reports]
        operation_rate_trend = self._calculate_rate_trend(operation_counts, recent_reports)

        # Calculate cache size trend
        cache_sizes = [r.get("summary", {}).get("total_size_mb", 0) for r in recent_reports]
        size_trend = self._calculate_linear_trend(cache_sizes)

        return {
            "available": True,
            "samples": len(recent_reports),
            "hit_rate_trend": hit_rate_trend,
            "operation_rate_trend": operation_rate_trend,
            "cache_size_trend": size_trend,
        }

    def _calculate_linear_trend(self, values: list[float]) -> dict[str, Any]:
        """Calculate linear trend from a list of values."""
        if len(values) < 2:
            return {"direction": "stable", "slope": 0.0, "confidence": "low"}

        n = len(values)
        x_values = list(range(n))

        # Calculate linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values, strict=False))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Determine direction
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Calculate confidence based on R-squared
        if denominator == 0:
            confidence = "low"
        else:
            predictions = [y_mean + slope * (x - x_mean) for x in x_values]
            ss_res = sum((y - pred) ** 2 for y, pred in zip(values, predictions, strict=False))
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r_squared > 0.8:
                confidence = "high"
            elif r_squared > 0.5:
                confidence = "medium"
            else:
                confidence = "low"

        return {
            "direction": direction,
            "slope": round(slope, 4),
            "confidence": confidence,
        }

    def _calculate_rate_trend(self, values: list[int], reports: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate rate trend from cumulative values."""
        if len(values) < 2:
            return {"direction": "stable", "rate": 0.0, "confidence": "low"}

        # Calculate rates between consecutive reports
        rates = []
        for i in range(1, len(values)):
            time_delta = reports[i]["timestamp"] - reports[i - 1]["timestamp"]
            if time_delta > 0:
                rate = (values[i] - values[i - 1]) / time_delta
                rates.append(rate)

        if not rates:
            return {"direction": "stable", "rate": 0.0, "confidence": "low"}

        return self._calculate_linear_trend(rates)

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percent."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    def _get_memory_info(self) -> dict[str, Any]:
        """Get current memory information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
            }
        except Exception:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

    def _distribute_report(self, report: dict[str, Any]) -> None:
        """Distribute report to all subscribers."""
        with self._lock:
            for callback in self._subscribers[:]:  # Copy to avoid modification during iteration
                try:
                    callback(report)
                except Exception:
                    # Remove broken callbacks
                    self._subscribers.remove(callback)

    def _store_report(self, report: dict[str, Any]) -> None:
        """Store report in history for trend analysis."""
        self._report_history.append(report)

        # Keep only recent history
        if len(self._report_history) > self._max_history:
            self._report_history = self._report_history[-self._max_history :]

    def get_latest_report(self) -> dict[str, Any] | None:
        """Get the latest report."""
        return self._report_history[-1] if self._report_history else None

    def get_report_history(self, count: int = 10) -> list[dict[str, Any]]:
        """Get recent report history."""
        return self._report_history[-count:] if self._report_history else []

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate a summary report of cache performance."""
        latest_report = self.get_latest_report()
        if not latest_report:
            return {"error": "No reports available"}

        # Get report history for comparison
        self.get_report_history(10)  # noqa: F841

        summary = {
            "timestamp": time.time(),
            "reporting_status": {
                "enabled": self._reporting_enabled,
                "interval_seconds": self._report_interval,
                "subscribers": len(self._subscribers),
                "reports_generated": len(self._report_history),
            },
            "current_metrics": latest_report.get("summary", {}),
            "cache_health": {
                "total_caches": len(latest_report.get("individual_caches", {})),
                "active_alerts": len(latest_report.get("alerts", [])),
                "overall_hit_rate": latest_report.get("summary", {}).get("overall_hit_rate", 0),
                "overall_error_rate": latest_report.get("summary", {}).get("overall_error_rate", 0),
            },
            "trends": latest_report.get("trends", {}),
            "top_performing_caches": self._get_top_performing_caches(latest_report),
            "recommendations": self._generate_recommendations(latest_report),
        }

        return summary

    def _get_top_performing_caches(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """Get top performing caches based on hit rate and operations."""
        caches = report.get("individual_caches", {})

        cache_performance = []
        for cache_name, cache_data in caches.items():
            metrics = cache_data.get("metrics", {})
            hit_rate = metrics.get("hit_rate", 0)
            operations = metrics.get("operations", {}).get("total", 0)

            # Calculate performance score (hit rate weighted by operations)
            performance_score = hit_rate * (1 + operations / 1000)  # Bonus for high activity

            cache_performance.append(
                {
                    "cache_name": cache_name,
                    "hit_rate": hit_rate,
                    "operations": operations,
                    "performance_score": performance_score,
                    "cache_type": cache_data.get("metrics", {}).get("cache_type", "unknown"),
                }
            )

        # Sort by performance score
        cache_performance.sort(key=lambda x: x["performance_score"], reverse=True)

        return cache_performance[:5]  # Top 5

    def _generate_recommendations(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate performance recommendations based on current metrics."""
        recommendations = []

        summary = report.get("summary", {})
        overall_hit_rate = summary.get("overall_hit_rate", 0)
        overall_error_rate = summary.get("overall_error_rate", 0)

        # Hit rate recommendations
        if overall_hit_rate < 0.5:
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "high",
                    "title": "Low Cache Hit Rate",
                    "description": f"Overall hit rate is {overall_hit_rate:.1%}. Consider reviewing cache strategies.",
                    "action": "Review cache TTL settings and cache key generation logic",
                }
            )

        # Error rate recommendations
        if overall_error_rate > 0.05:
            recommendations.append(
                {
                    "type": "reliability",
                    "priority": "high",
                    "title": "High Error Rate",
                    "description": f"Error rate is {overall_error_rate:.1%}. Investigate cache reliability issues.",
                    "action": "Check cache connection stability and error handling",
                }
            )

        # Cache type recommendations
        cache_types = report.get("cache_types", {})
        for cache_type, type_data in cache_types.items():
            if type_data.get("hit_rate", 0) < 0.3 and type_data.get("operations", 0) > 100:
                recommendations.append(
                    {
                        "type": "optimization",
                        "priority": "medium",
                        "title": f"Optimize {cache_type} Cache",
                        "description": f"{cache_type} cache has low hit rate ({type_data.get('hit_rate', 0):.1%}) but high activity",
                        "action": f"Review {cache_type} cache configuration and data patterns",
                    }
                )

        return recommendations

    def export_report(self, filepath: str, file_format: str = "json") -> bool:
        """Export the latest report to a file."""
        try:
            latest_report = self.get_latest_report()
            if not latest_report:
                return False

            if file_format.lower() == "json":
                with open(filepath, "w") as f:
                    json.dump(latest_report, f, indent=2)
                return True
            else:
                return False
        except Exception:
            return False


# Global cache performance monitor instance
_cache_performance_monitor = CachePerformanceMonitor()
_real_time_reporter = None


def get_cache_performance_monitor() -> CachePerformanceMonitor:
    """Get the global cache performance monitor instance."""
    return _cache_performance_monitor


def get_real_time_cache_reporter() -> RealTimeCacheStatsReporter:
    """Get the global real-time cache statistics reporter instance."""
    global _real_time_reporter
    if _real_time_reporter is None:
        _real_time_reporter = RealTimeCacheStatsReporter(_cache_performance_monitor)
    return _real_time_reporter


class MemoryMonitor:
    """Memory usage monitor with configurable warning thresholds and detailed cache memory tracking."""

    def __init__(self, warning_threshold_mb: float = 1000.0):
        self.warning_threshold_mb = warning_threshold_mb
        self._last_warning_time = 0.0
        self._warning_cooldown = 30.0  # Seconds between warnings
        self._monitoring = False
        self._monitor_thread = None
        self._cache_monitor = get_cache_performance_monitor()
        self._cache_memory_tracking = {}  # Track memory usage per cache
        self._cache_memory_history = []  # Track memory usage over time
        self._max_history_entries = 100  # Maximum history entries to keep

    def check_memory_usage(self, logger) -> dict[str, Any]:
        """Check current memory usage and log warnings if needed."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()

            memory_stats = {
                "memory_mb": round(memory_mb, 1),
                "memory_percent": round(memory_percent, 1),
                "threshold_mb": self.warning_threshold_mb,
                "above_threshold": memory_mb > self.warning_threshold_mb,
            }

            # Log warning if above threshold and cooldown has passed
            current_time = time.time()
            if memory_mb > self.warning_threshold_mb and current_time - self._last_warning_time > self._warning_cooldown:
                logger.warning(
                    f"Memory usage above threshold: {memory_mb:.1f}MB " f"(threshold: {self.warning_threshold_mb}MB, {memory_percent:.1f}%)"
                )
                self._last_warning_time = current_time

            return memory_stats

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to get memory info: {e}")
            return {
                "memory_mb": 0.0,
                "memory_percent": 0.0,
                "threshold_mb": self.warning_threshold_mb,
                "above_threshold": False,
                "error": str(e),
            }

    def start_monitoring(self):
        """Start memory monitoring."""
        self._monitoring = True

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False

    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def get_system_memory_info(self) -> dict[str, Any]:
        """Get system-wide memory information including cache memory usage."""
        try:
            memory = psutil.virtual_memory()

            # Get cache memory usage from performance monitor
            cache_metrics = self._cache_monitor.get_aggregated_metrics()
            cache_memory_mb = cache_metrics.get("summary", {}).get("total_size_mb", 0.0)

            return {
                "total_mb": round(memory.total / 1024 / 1024, 1),
                "available_mb": round(memory.available / 1024 / 1024, 1),
                "used_mb": round(memory.used / 1024 / 1024, 1),
                "percent_used": round(memory.percent, 1),
                "cache_memory_mb": round(cache_memory_mb, 1),
                "cache_percent_of_system": round((cache_memory_mb / (memory.total / 1024 / 1024)) * 100, 2) if memory.total > 0 else 0.0,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_cache_memory_info(self) -> dict[str, Any]:
        """Get detailed cache memory information per cache and cache type."""
        try:
            # Get all cache metrics
            all_cache_metrics = self._cache_monitor.get_all_cache_metrics()
            cache_type_metrics = self._cache_monitor.get_all_cache_type_metrics()

            # Calculate individual cache memory usage
            cache_memory_details = {}
            for cache_name, metrics in all_cache_metrics.items():
                cache_type = self._cache_monitor._cache_types.get(cache_name, "unknown")
                cache_memory_details[cache_name] = {
                    "type": cache_type,
                    "current_size_mb": round(metrics.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024, 2),
                    "max_size_mb": round(metrics.get("memory", {}).get("max_size_bytes", 0) / 1024 / 1024, 2),
                    "memory_pressure_events": metrics.get("memory", {}).get("memory_pressure_events", 0),
                    "eviction_count": metrics.get("memory", {}).get("eviction_count", 0),
                    "hit_rate": metrics.get("hit_rate", 0.0),
                    "operations": metrics.get("operations", {}).get("total", 0),
                }

            # Calculate cache type memory usage
            cache_type_memory_details = {}
            for cache_type, metrics in cache_type_metrics.items():
                cache_type_memory_details[cache_type] = {
                    "current_size_mb": round(metrics.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024, 2),
                    "max_size_mb": round(metrics.get("memory", {}).get("max_size_bytes", 0) / 1024 / 1024, 2),
                    "memory_pressure_events": metrics.get("memory", {}).get("memory_pressure_events", 0),
                    "eviction_count": metrics.get("memory", {}).get("eviction_count", 0),
                    "hit_rate": metrics.get("hit_rate", 0.0),
                    "operations": metrics.get("operations", {}).get("total", 0),
                    "instance_count": sum(1 for t in self._cache_monitor._cache_types.values() if t == cache_type),
                }

            # Get process memory
            process = psutil.Process()
            process_memory = process.memory_info()

            # Calculate total cache memory
            total_cache_memory_mb = sum(cache_data["current_size_mb"] for cache_data in cache_memory_details.values())

            return {
                "process_memory": {
                    "rss_mb": round(process_memory.rss / 1024 / 1024, 2),
                    "vms_mb": round(process_memory.vms / 1024 / 1024, 2),
                    "percent": round(process.memory_percent(), 2),
                },
                "cache_memory": {
                    "total_mb": round(total_cache_memory_mb, 2),
                    "percent_of_process": (
                        round((total_cache_memory_mb / (process_memory.rss / 1024 / 1024)) * 100, 2) if process_memory.rss > 0 else 0.0
                    ),
                },
                "by_cache": cache_memory_details,
                "by_cache_type": cache_type_memory_details,
                "memory_efficiency": {
                    "total_operations": sum(cache_data["operations"] for cache_data in cache_memory_details.values()),
                    "operations_per_mb": round(
                        sum(cache_data["operations"] for cache_data in cache_memory_details.values()) / max(total_cache_memory_mb, 0.1), 2
                    ),
                    "avg_hit_rate": round(
                        sum(cache_data["hit_rate"] for cache_data in cache_memory_details.values()) / max(len(cache_memory_details), 1), 3
                    ),
                },
                "timestamp": time.time(),
            }
        except Exception as e:
            return {"error": str(e)}

    def record_cache_memory_snapshot(self) -> None:
        """Record a snapshot of cache memory usage for trend analysis."""
        try:
            snapshot = {
                "timestamp": time.time(),
                "cache_memory": self.get_detailed_cache_memory_info(),
                "system_memory": self.get_system_memory_info(),
            }

            self._cache_memory_history.append(snapshot)

            # Keep only recent history
            if len(self._cache_memory_history) > self._max_history_entries:
                self._cache_memory_history = self._cache_memory_history[-self._max_history_entries :]

        except Exception:
            # Log error but don't fail
            pass

    def get_cache_memory_trends(self, minutes: int = 60) -> dict[str, Any]:
        """Get cache memory usage trends over the specified time period."""
        try:
            cutoff_time = time.time() - (minutes * 60)
            recent_history = [snapshot for snapshot in self._cache_memory_history if snapshot["timestamp"] >= cutoff_time]

            if not recent_history:
                return {"error": "No recent memory history available"}

            # Calculate trends
            cache_trends = {}
            for cache_name in self._cache_monitor._cache_metrics.keys():
                cache_sizes = []
                timestamps = []

                for snapshot in recent_history:
                    cache_data = snapshot.get("cache_memory", {}).get("by_cache", {}).get(cache_name, {})
                    if cache_data:
                        cache_sizes.append(cache_data.get("current_size_mb", 0))
                        timestamps.append(snapshot["timestamp"])

                if len(cache_sizes) >= 2:
                    # Calculate trend
                    trend = (cache_sizes[-1] - cache_sizes[0]) / max(len(cache_sizes) - 1, 1)
                    cache_trends[cache_name] = {
                        "size_trend_mb_per_sample": round(trend, 3),
                        "current_size_mb": cache_sizes[-1],
                        "initial_size_mb": cache_sizes[0],
                        "samples": len(cache_sizes),
                        "time_span_minutes": round((timestamps[-1] - timestamps[0]) / 60, 2),
                    }

            return {
                "cache_trends": cache_trends,
                "time_period_minutes": minutes,
                "samples_available": len(recent_history),
                "oldest_sample_age_minutes": round((time.time() - recent_history[0]["timestamp"]) / 60, 2) if recent_history else 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_cache_memory_alerts(self) -> list[dict[str, Any]]:
        """Get cache memory-related alerts and recommendations."""
        alerts = []

        try:
            detailed_info = self.get_detailed_cache_memory_info()

            # Check for high memory usage per cache
            for cache_name, cache_data in detailed_info.get("by_cache", {}).items():
                if cache_data["current_size_mb"] > 100:  # Alert if cache > 100MB
                    alerts.append(
                        {
                            "type": "high_cache_memory",
                            "cache_name": cache_name,
                            "current_size_mb": cache_data["current_size_mb"],
                            "severity": "warning" if cache_data["current_size_mb"] < 500 else "critical",
                            "recommendation": "Consider implementing more aggressive eviction policies or increasing cache size limits",
                            "timestamp": time.time(),
                        }
                    )

            # Check for memory pressure events
            for cache_name, cache_data in detailed_info.get("by_cache", {}).items():
                if cache_data["memory_pressure_events"] > 10:
                    alerts.append(
                        {
                            "type": "memory_pressure",
                            "cache_name": cache_name,
                            "memory_pressure_events": cache_data["memory_pressure_events"],
                            "severity": "warning",
                            "recommendation": "Consider increasing cache size limits or optimizing cache content",
                            "timestamp": time.time(),
                        }
                    )

            # Check for low hit rate with high memory usage
            for cache_name, cache_data in detailed_info.get("by_cache", {}).items():
                if cache_data["hit_rate"] < 0.5 and cache_data["current_size_mb"] > 50:
                    alerts.append(
                        {
                            "type": "inefficient_memory_usage",
                            "cache_name": cache_name,
                            "hit_rate": cache_data["hit_rate"],
                            "current_size_mb": cache_data["current_size_mb"],
                            "severity": "warning",
                            "recommendation": "Cache has low hit rate despite high memory usage. Consider reviewing caching strategy",
                            "timestamp": time.time(),
                        }
                    )

            return alerts

        except Exception as e:
            return [{"type": "error", "message": str(e), "timestamp": time.time()}]


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def format_memory_size(bytes_value: int) -> str:
    """Format bytes as human-readable memory size."""
    if bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f}KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.1f}GB"
