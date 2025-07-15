"""Memory management utilities for MCP tools.

This module provides memory monitoring and management functionality,
including comprehensive cache memory tracking and coordination.
"""

import gc
import logging
import os
import sys
import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

logger = logging.getLogger(__name__)

# Configuration from environment
MEMORY_WARNING_THRESHOLD_MB = int(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "5"))
FORCE_CLEANUP_THRESHOLD_MB = int(os.getenv("FORCE_CLEANUP_THRESHOLD_MB", "1500"))

# Cache memory configuration
CACHE_MEMORY_WARNING_THRESHOLD_MB = int(os.getenv("CACHE_MEMORY_WARNING_THRESHOLD_MB", "500"))
CACHE_MEMORY_CRITICAL_THRESHOLD_MB = int(os.getenv("CACHE_MEMORY_CRITICAL_THRESHOLD_MB", "800"))
CACHE_MEMORY_PRESSURE_THRESHOLD_PERCENT = float(os.getenv("CACHE_MEMORY_PRESSURE_THRESHOLD_PERCENT", "75.0"))
CACHE_MEMORY_TRACKING_INTERVAL = int(os.getenv("CACHE_MEMORY_TRACKING_INTERVAL", "30"))
CACHE_MEMORY_EVICTION_BATCH_SIZE = int(os.getenv("CACHE_MEMORY_EVICTION_BATCH_SIZE", "100"))

# Global cache registry for memory tracking
_cache_registry: dict[str, Any] = {}
_cache_registry_lock = threading.Lock()
_memory_pressure_callbacks: list[Callable] = []
_cache_memory_stats: dict[str, dict[str, Any]] = defaultdict(dict)
_cache_memory_history: list[tuple[float, dict[str, Any]]] = []
_memory_tracking_enabled = True


class MemoryPressureLevel(Enum):
    """Memory pressure levels for cache management."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CacheMemoryEvent(Enum):
    """Types of cache memory events."""

    ALLOCATION = "allocation"
    DEALLOCATION = "deallocation"
    EVICTION = "eviction"
    PRESSURE = "pressure"
    CLEANUP = "cleanup"


@dataclass
class CacheMemoryStats:
    """Cache memory statistics."""

    cache_name: str
    current_size_mb: float = 0.0
    peak_size_mb: float = 0.0
    total_allocated_mb: float = 0.0
    total_deallocated_mb: float = 0.0
    eviction_count: int = 0
    pressure_events: int = 0
    cleanup_events: int = 0
    last_cleanup_time: float = field(default_factory=time.time)
    last_pressure_time: float = 0.0
    memory_efficiency: float = 0.0  # hit_rate * (1 - waste_ratio)
    allocation_rate_mb_per_sec: float = 0.0
    deallocation_rate_mb_per_sec: float = 0.0
    fragmentation_ratio: float = 0.0

    def update_efficiency(self, hit_rate: float, waste_ratio: float) -> None:
        """Update memory efficiency metrics."""
        self.memory_efficiency = hit_rate * (1 - waste_ratio)

    def update_fragmentation(self, fragmentation_ratio: float) -> None:
        """Update fragmentation metrics."""
        self.fragmentation_ratio = fragmentation_ratio


@dataclass
class SystemMemoryPressure:
    """System memory pressure information."""

    level: MemoryPressureLevel
    current_usage_mb: float
    available_mb: float
    total_mb: float
    cache_usage_mb: float
    cache_usage_percent: float
    recommendation: str
    timestamp: float = field(default_factory=time.time)

    @property
    def usage_percent(self) -> float:
        """Calculate total memory usage percentage."""
        return (self.current_usage_mb / self.total_mb) * 100 if self.total_mb > 0 else 0.0


def get_memory_stats() -> dict:
    """Get comprehensive memory statistics for both process and system.

    Returns:
        dict: Memory statistics including process and system memory info
    """
    if not PSUTIL_AVAILABLE:
        return {
            "process_memory_mb": 0.0,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "system_memory": {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "used_mb": 0.0,
                "percent_used": 0.0,
            },
            "psutil_available": False,
        }

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "system_memory": {
                "total_mb": virtual_memory.total / (1024 * 1024),
                "available_mb": virtual_memory.available / (1024 * 1024),
                "used_mb": virtual_memory.used / (1024 * 1024),
                "percent_used": virtual_memory.percent,
            },
            "psutil_available": True,
        }
    except Exception as e:
        logger.warning(f"Failed to get memory stats: {e}")
        return {
            "process_memory_mb": 0.0,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "system_memory": {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "used_mb": 0.0,
                "percent_used": 0.0,
            },
            "psutil_available": False,
            "error": str(e),
        }


def check_memory_usage(context: str = "") -> tuple[float, bool]:
    """Check current memory usage and log warnings if needed.

    Args:
        context: Context string for logging

    Returns:
        Tuple[float, bool]: (memory_mb, needs_cleanup)
    """
    stats = get_memory_stats()
    memory_mb = stats["rss_mb"]

    if memory_mb > 0:
        logger.info(f"Memory usage{' ' + context if context else ''}: {memory_mb:.1f}MB")

        if memory_mb > MEMORY_WARNING_THRESHOLD_MB:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds warning threshold ({MEMORY_WARNING_THRESHOLD_MB}MB)")

        if memory_mb > FORCE_CLEANUP_THRESHOLD_MB:
            return memory_mb, True

    return memory_mb, False


def force_memory_cleanup(context: str = "") -> None:
    """Force comprehensive memory cleanup.

    Args:
        context: Context string for logging
    """
    logger.info(f"Forcing memory cleanup{' for ' + context if context else ''}")

    # Force garbage collection multiple times for thoroughness
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"GC cycle {i + 1}: collected {collected} objects")

    # Clear any cached objects if possible
    if hasattr(gc, "set_threshold"):
        # Temporarily lower GC thresholds to be more aggressive
        original_thresholds = gc.get_threshold()
        gc.set_threshold(100, 10, 10)
        gc.collect()
        gc.set_threshold(*original_thresholds)

    memory_after = log_memory_usage("after cleanup")

    if memory_after > FORCE_CLEANUP_THRESHOLD_MB:
        logger.warning(
            f"Memory still high ({memory_after:.1f}MB) after cleanup. " f"Consider reducing batch sizes or processing fewer files."
        )


def should_cleanup_memory(batch_count: int, force_check: bool = False) -> bool:
    """Determine if memory cleanup should be performed.

    Args:
        batch_count: Number of batches processed
        force_check: Force memory check regardless of batch count

    Returns:
        bool: True if cleanup should be performed
    """
    if force_check or (batch_count > 0 and batch_count % MEMORY_CLEANUP_INTERVAL == 0):
        _, needs_cleanup = check_memory_usage()
        return needs_cleanup
    return False


def get_adaptive_batch_size(base_batch_size: int, memory_usage_mb: float) -> int:
    """Calculate adaptive batch size based on memory usage.

    Args:
        base_batch_size: Base batch size
        memory_usage_mb: Current memory usage in MB

    Returns:
        int: Adjusted batch size
    """
    if memory_usage_mb > FORCE_CLEANUP_THRESHOLD_MB:
        # Critical memory usage - use minimum batch size
        return max(1, base_batch_size // 4)
    elif memory_usage_mb > MEMORY_WARNING_THRESHOLD_MB:
        # High memory usage - reduce batch size
        return max(1, base_batch_size // 2)
    else:
        # Normal memory usage
        return base_batch_size


def clear_processing_variables(*variables) -> None:
    """Clear processing variables to free memory.

    Args:
        *variables: Variables to clear
    """
    for var in variables:
        if var is not None:
            if hasattr(var, "clear") and callable(var.clear):
                var.clear()
            elif hasattr(var, "close") and callable(var.close):
                var.close()
            del var
    gc.collect()


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB.

    Returns:
        float: Current memory usage in MB, or 0.0 if unavailable
    """
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return 0.0


def log_memory_usage(context: str = "") -> float:
    """Log current memory usage and return the value in MB.

    Args:
        context: Context string for logging

    Returns:
        float: Current memory usage in MB
    """
    memory_mb = get_memory_usage_mb()
    if memory_mb > 0:
        logger.info(f"Memory usage{' ' + context if context else ''}: {memory_mb:.1f}MB")
        if memory_mb > MEMORY_WARNING_THRESHOLD_MB:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds warning threshold ({MEMORY_WARNING_THRESHOLD_MB}MB)")
    return memory_mb


# ============================================================================
# Cache Memory Tracking System
# ============================================================================


def register_cache_service(cache_name: str, cache_service: Any) -> None:
    """Register a cache service for memory tracking.

    Args:
        cache_name: Unique name for the cache service
        cache_service: Cache service instance to track
    """
    global _cache_registry, _cache_memory_stats

    with _cache_registry_lock:
        _cache_registry[cache_name] = weakref.ref(cache_service)
        _cache_memory_stats[cache_name] = {"stats": CacheMemoryStats(cache_name), "last_update": time.time(), "events": []}

    logger.info(f"Registered cache service '{cache_name}' for memory tracking")


def unregister_cache_service(cache_name: str) -> None:
    """Unregister a cache service from memory tracking.

    Args:
        cache_name: Name of the cache service to unregister
    """
    global _cache_registry, _cache_memory_stats

    with _cache_registry_lock:
        _cache_registry.pop(cache_name, None)
        _cache_memory_stats.pop(cache_name, None)

    logger.info(f"Unregistered cache service '{cache_name}' from memory tracking")


def track_cache_memory_event(cache_name: str, event_type: CacheMemoryEvent, size_mb: float, metadata: dict[str, Any] | None = None) -> None:
    """Track a cache memory event.

    Args:
        cache_name: Name of the cache service
        event_type: Type of memory event
        size_mb: Size in megabytes affected by the event
        metadata: Additional event metadata
    """
    if not _memory_tracking_enabled or cache_name not in _cache_memory_stats:
        return

    current_time = time.time()
    stats = _cache_memory_stats[cache_name]["stats"]

    # Update statistics based on event type
    if event_type == CacheMemoryEvent.ALLOCATION:
        stats.current_size_mb += size_mb
        stats.total_allocated_mb += size_mb
        stats.peak_size_mb = max(stats.peak_size_mb, stats.current_size_mb)

    elif event_type == CacheMemoryEvent.DEALLOCATION:
        stats.current_size_mb = max(0, stats.current_size_mb - size_mb)
        stats.total_deallocated_mb += size_mb

    elif event_type == CacheMemoryEvent.EVICTION:
        stats.eviction_count += 1
        stats.current_size_mb = max(0, stats.current_size_mb - size_mb)

    elif event_type == CacheMemoryEvent.PRESSURE:
        stats.pressure_events += 1
        stats.last_pressure_time = current_time

    elif event_type == CacheMemoryEvent.CLEANUP:
        stats.cleanup_events += 1
        stats.last_cleanup_time = current_time

    # Record event
    event_data = {"timestamp": current_time, "type": event_type.value, "size_mb": size_mb, "metadata": metadata or {}}

    _cache_memory_stats[cache_name]["events"].append(event_data)
    _cache_memory_stats[cache_name]["last_update"] = current_time

    # Limit event history
    if len(_cache_memory_stats[cache_name]["events"]) > 1000:
        _cache_memory_stats[cache_name]["events"] = _cache_memory_stats[cache_name]["events"][-500:]

    # Calculate allocation/deallocation rates
    _update_cache_memory_rates(cache_name)

    # Check for memory pressure
    _check_cache_memory_pressure(cache_name)


def _update_cache_memory_rates(cache_name: str) -> None:
    """Update cache memory allocation and deallocation rates."""
    if cache_name not in _cache_memory_stats:
        return

    stats = _cache_memory_stats[cache_name]["stats"]
    events = _cache_memory_stats[cache_name]["events"]

    current_time = time.time()
    time_window = 60.0  # 1 minute window

    # Filter events within time window
    recent_events = [e for e in events if current_time - e["timestamp"] <= time_window]

    if not recent_events:
        return

    # Calculate rates
    allocation_mb = sum(e["size_mb"] for e in recent_events if e["type"] == CacheMemoryEvent.ALLOCATION.value)
    deallocation_mb = sum(e["size_mb"] for e in recent_events if e["type"] == CacheMemoryEvent.DEALLOCATION.value)

    stats.allocation_rate_mb_per_sec = allocation_mb / time_window
    stats.deallocation_rate_mb_per_sec = deallocation_mb / time_window


def _check_cache_memory_pressure(cache_name: str) -> None:
    """Check if cache memory pressure handling is needed."""
    if cache_name not in _cache_memory_stats:
        return

    stats = _cache_memory_stats[cache_name]["stats"]

    # Check cache-specific thresholds
    if stats.current_size_mb > CACHE_MEMORY_CRITICAL_THRESHOLD_MB:
        _trigger_memory_pressure_response(cache_name, MemoryPressureLevel.CRITICAL)
    elif stats.current_size_mb > CACHE_MEMORY_WARNING_THRESHOLD_MB:
        _trigger_memory_pressure_response(cache_name, MemoryPressureLevel.HIGH)


def _trigger_memory_pressure_response(cache_name: str, pressure_level: MemoryPressureLevel) -> None:
    """Trigger memory pressure response for a cache."""
    logger.warning(f"Memory pressure {pressure_level.value} detected for cache '{cache_name}'")

    # Record pressure event
    track_cache_memory_event(
        cache_name,
        CacheMemoryEvent.PRESSURE,
        0.0,
        {"pressure_level": pressure_level.value, "current_size_mb": _cache_memory_stats[cache_name]["stats"].current_size_mb},
    )

    # Notify pressure callbacks
    for callback in _memory_pressure_callbacks:
        try:
            callback(cache_name, pressure_level)
        except Exception as e:
            logger.error(f"Error in memory pressure callback: {e}")


def add_memory_pressure_callback(callback: Callable[[str, MemoryPressureLevel], None]) -> None:
    """Add a callback for memory pressure events.

    Args:
        callback: Function to call on memory pressure events
    """
    _memory_pressure_callbacks.append(callback)


def remove_memory_pressure_callback(callback: Callable[[str, MemoryPressureLevel], None]) -> None:
    """Remove a memory pressure callback.

    Args:
        callback: Function to remove from callbacks
    """
    if callback in _memory_pressure_callbacks:
        _memory_pressure_callbacks.remove(callback)


def get_cache_memory_stats(cache_name: str | None = None) -> dict[str, CacheMemoryStats] | CacheMemoryStats:
    """Get cache memory statistics.

    Args:
        cache_name: Specific cache name, or None for all caches

    Returns:
        Cache memory statistics for specified cache or all caches
    """
    if cache_name:
        if cache_name in _cache_memory_stats:
            return _cache_memory_stats[cache_name]["stats"]
        else:
            return CacheMemoryStats(cache_name)
    else:
        return {name: data["stats"] for name, data in _cache_memory_stats.items()}


def get_total_cache_memory_usage() -> float:
    """Get total memory usage across all registered caches.

    Returns:
        Total cache memory usage in MB
    """
    total_mb = 0.0
    for cache_data in _cache_memory_stats.values():
        total_mb += cache_data["stats"].current_size_mb
    return total_mb


def get_system_memory_pressure() -> SystemMemoryPressure:
    """Get current system memory pressure information.

    Returns:
        System memory pressure data
    """
    memory_stats = get_memory_stats()
    cache_memory_mb = get_total_cache_memory_usage()

    current_usage_mb = memory_stats["rss_mb"]
    available_mb = memory_stats["system_memory"]["available_mb"]
    total_mb = memory_stats["system_memory"]["total_mb"]

    if not PSUTIL_AVAILABLE or total_mb == 0:
        return SystemMemoryPressure(
            level=MemoryPressureLevel.LOW,
            current_usage_mb=current_usage_mb,
            available_mb=available_mb,
            total_mb=total_mb,
            cache_usage_mb=cache_memory_mb,
            cache_usage_percent=0.0,
            recommendation="Memory monitoring unavailable",
        )

    usage_percent = (current_usage_mb / total_mb) * 100
    cache_usage_percent = (cache_memory_mb / total_mb) * 100

    # Determine pressure level
    if usage_percent >= 90:
        level = MemoryPressureLevel.CRITICAL
        recommendation = "Critical memory pressure - aggressive cache eviction recommended"
    elif usage_percent >= 80:
        level = MemoryPressureLevel.HIGH
        recommendation = "High memory pressure - moderate cache eviction recommended"
    elif usage_percent >= 70:
        level = MemoryPressureLevel.MODERATE
        recommendation = "Moderate memory pressure - monitor cache growth"
    else:
        level = MemoryPressureLevel.LOW
        recommendation = "Normal memory usage"

    return SystemMemoryPressure(
        level=level,
        current_usage_mb=current_usage_mb,
        available_mb=available_mb,
        total_mb=total_mb,
        cache_usage_mb=cache_memory_mb,
        cache_usage_percent=cache_usage_percent,
        recommendation=recommendation,
    )


def get_cache_memory_recommendations() -> list[dict[str, Any]]:
    """Get memory optimization recommendations for caches.

    Returns:
        List of recommendations for cache memory optimization
    """
    recommendations = []
    system_pressure = get_system_memory_pressure()

    for cache_name, cache_data in _cache_memory_stats.items():
        stats = cache_data["stats"]

        # Check for high memory usage
        if stats.current_size_mb > CACHE_MEMORY_WARNING_THRESHOLD_MB:
            recommendations.append(
                {
                    "cache_name": cache_name,
                    "type": "high_memory_usage",
                    "priority": "high",
                    "message": f"Cache '{cache_name}' using {stats.current_size_mb:.1f}MB "
                    f"(>{CACHE_MEMORY_WARNING_THRESHOLD_MB}MB threshold)",
                    "suggested_action": "Reduce cache size or increase eviction frequency",
                }
            )

        # Check for excessive evictions
        if stats.eviction_count > 100:
            recommendations.append(
                {
                    "cache_name": cache_name,
                    "type": "excessive_evictions",
                    "priority": "medium",
                    "message": f"Cache '{cache_name}' has {stats.eviction_count} evictions",
                    "suggested_action": "Consider increasing cache size or optimizing eviction policy",
                }
            )

        # Check for memory pressure events
        if stats.pressure_events > 10:
            recommendations.append(
                {
                    "cache_name": cache_name,
                    "type": "memory_pressure",
                    "priority": "high",
                    "message": f"Cache '{cache_name}' has {stats.pressure_events} memory pressure events",
                    "suggested_action": "Implement proactive cache cleanup or reduce allocation rate",
                }
            )

        # Check for low memory efficiency
        if stats.memory_efficiency < 0.5:
            recommendations.append(
                {
                    "cache_name": cache_name,
                    "type": "low_efficiency",
                    "priority": "medium",
                    "message": f"Cache '{cache_name}' has low memory efficiency ({stats.memory_efficiency:.2f})",
                    "suggested_action": "Optimize cache key strategy or improve hit rate",
                }
            )

        # Check for high fragmentation
        if stats.fragmentation_ratio > 0.3:
            recommendations.append(
                {
                    "cache_name": cache_name,
                    "type": "high_fragmentation",
                    "priority": "medium",
                    "message": f"Cache '{cache_name}' has high fragmentation ({stats.fragmentation_ratio:.2f})",
                    "suggested_action": "Consider cache defragmentation or reorganization",
                }
            )

    # System-wide recommendations
    if system_pressure.level == MemoryPressureLevel.CRITICAL:
        recommendations.append(
            {
                "cache_name": "system",
                "type": "system_pressure",
                "priority": "critical",
                "message": f"System memory pressure is critical ({system_pressure.usage_percent:.1f}%)",
                "suggested_action": "Immediately reduce cache sizes or restart services",
            }
        )

    return recommendations


def generate_cache_memory_report() -> dict[str, Any]:
    """Generate comprehensive cache memory report.

    Returns:
        Detailed cache memory report
    """
    current_time = time.time()
    system_pressure = get_system_memory_pressure()
    total_cache_memory = get_total_cache_memory_usage()
    recommendations = get_cache_memory_recommendations()

    # Cache statistics
    cache_stats = {}
    for cache_name, cache_data in _cache_memory_stats.items():
        stats = cache_data["stats"]
        recent_events = [e for e in cache_data["events"] if current_time - e["timestamp"] <= 300]  # 5 minutes

        cache_stats[cache_name] = {
            "current_size_mb": stats.current_size_mb,
            "peak_size_mb": stats.peak_size_mb,
            "total_allocated_mb": stats.total_allocated_mb,
            "total_deallocated_mb": stats.total_deallocated_mb,
            "eviction_count": stats.eviction_count,
            "pressure_events": stats.pressure_events,
            "cleanup_events": stats.cleanup_events,
            "memory_efficiency": stats.memory_efficiency,
            "allocation_rate_mb_per_sec": stats.allocation_rate_mb_per_sec,
            "deallocation_rate_mb_per_sec": stats.deallocation_rate_mb_per_sec,
            "fragmentation_ratio": stats.fragmentation_ratio,
            "recent_events_count": len(recent_events),
            "last_cleanup_time": stats.last_cleanup_time,
            "last_pressure_time": stats.last_pressure_time,
        }

    return {
        "timestamp": current_time,
        "system_memory": {
            "pressure_level": system_pressure.level.value,
            "current_usage_mb": system_pressure.current_usage_mb,
            "available_mb": system_pressure.available_mb,
            "total_mb": system_pressure.total_mb,
            "usage_percent": system_pressure.usage_percent,
            "recommendation": system_pressure.recommendation,
        },
        "cache_memory": {
            "total_usage_mb": total_cache_memory,
            "cache_usage_percent": system_pressure.cache_usage_percent,
            "registered_caches": len(_cache_memory_stats),
            "active_caches": len([name for name, ref in _cache_registry.items() if ref() is not None]),
        },
        "cache_statistics": cache_stats,
        "recommendations": recommendations,
        "configuration": {
            "memory_warning_threshold_mb": MEMORY_WARNING_THRESHOLD_MB,
            "cache_memory_warning_threshold_mb": CACHE_MEMORY_WARNING_THRESHOLD_MB,
            "cache_memory_critical_threshold_mb": CACHE_MEMORY_CRITICAL_THRESHOLD_MB,
            "cache_memory_pressure_threshold_percent": CACHE_MEMORY_PRESSURE_THRESHOLD_PERCENT,
            "tracking_enabled": _memory_tracking_enabled,
        },
    }


def enable_cache_memory_tracking() -> None:
    """Enable cache memory tracking."""
    global _memory_tracking_enabled
    _memory_tracking_enabled = True
    logger.info("Cache memory tracking enabled")


def disable_cache_memory_tracking() -> None:
    """Disable cache memory tracking."""
    global _memory_tracking_enabled
    _memory_tracking_enabled = False
    logger.info("Cache memory tracking disabled")


def clear_cache_memory_stats(cache_name: str | None = None) -> None:
    """Clear cache memory statistics.

    Args:
        cache_name: Specific cache name, or None for all caches
    """
    if cache_name:
        if cache_name in _cache_memory_stats:
            _cache_memory_stats[cache_name]["stats"] = CacheMemoryStats(cache_name)
            _cache_memory_stats[cache_name]["events"] = []
            _cache_memory_stats[cache_name]["last_update"] = time.time()
    else:
        for name in _cache_memory_stats:
            _cache_memory_stats[name]["stats"] = CacheMemoryStats(name)
            _cache_memory_stats[name]["events"] = []
            _cache_memory_stats[name]["last_update"] = time.time()

    logger.info(f"Cleared cache memory statistics for {cache_name or 'all caches'}")


def start_cache_memory_monitoring() -> None:
    """Start background cache memory monitoring."""

    def _monitor_loop():
        while _memory_tracking_enabled:
            try:
                # Update system memory history
                current_time = time.time()
                system_stats = get_memory_stats()
                cache_memory = get_total_cache_memory_usage()

                _cache_memory_history.append(
                    (
                        current_time,
                        {
                            "system_memory_mb": system_stats["rss_mb"],
                            "cache_memory_mb": cache_memory,
                            "available_memory_mb": system_stats["system_memory"]["available_mb"],
                        },
                    )
                )

                # Limit history size
                if len(_cache_memory_history) > 1000:
                    _cache_memory_history[:] = _cache_memory_history[-500:]

                # Check for system-wide memory pressure
                system_pressure = get_system_memory_pressure()
                if system_pressure.level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    for cache_name in _cache_memory_stats:
                        _trigger_memory_pressure_response(cache_name, system_pressure.level)

                time.sleep(CACHE_MEMORY_TRACKING_INTERVAL)

            except Exception as e:
                logger.error(f"Error in cache memory monitoring: {e}")
                time.sleep(CACHE_MEMORY_TRACKING_INTERVAL)

    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
    monitor_thread.start()
    logger.info("Started cache memory monitoring background thread")


# Initialize cache memory monitoring
start_cache_memory_monitoring()
