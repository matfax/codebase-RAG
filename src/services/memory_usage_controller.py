"""
Memory Usage Monitoring and Control Service for Wave 6.0 - Subtask 6.3

This module implements comprehensive memory usage monitoring and control mechanisms
to ensure memory usage doesn't exceed set limits, with proactive management,
alerts, and automatic interventions.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Union

import psutil

from src.config.cache_config import CacheConfig, get_global_cache_config
from src.services.cache_memory_profiler import CacheMemoryProfiler, ProfilingLevel
from src.services.intelligent_eviction_coordinator import IntelligentEvictionCoordinator
from src.utils.memory_utils import MemoryPressureLevel, get_system_memory_pressure


class MemoryControlAction(Enum):
    """Types of memory control actions."""

    NONE = "none"
    WARN = "warn"
    THROTTLE = "throttle"
    EVICT = "evict"
    EMERGENCY_CLEANUP = "emergency_cleanup"
    SYSTEM_ALERT = "system_alert"


class MemoryThresholdLevel(Enum):
    """Memory threshold levels for control actions."""

    NORMAL = "normal"  # < 60% memory usage
    ELEVATED = "elevated"  # 60-75% memory usage
    HIGH = "high"  # 75-85% memory usage
    CRITICAL = "critical"  # 85-95% memory usage
    EMERGENCY = "emergency"  # > 95% memory usage


@dataclass
class MemoryLimit:
    """Memory limit configuration."""

    max_memory_mb: float
    warning_threshold: float = 0.75  # 75%
    critical_threshold: float = 0.85  # 85%
    emergency_threshold: float = 0.95  # 95%
    check_interval_seconds: float = 30.0
    enforcement_enabled: bool = True


@dataclass
class MemoryControlEvent:
    """Memory control event record."""

    timestamp: float
    memory_level: MemoryThresholdLevel
    current_usage_mb: float
    threshold_mb: float
    action_taken: MemoryControlAction
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Current memory statistics."""

    timestamp: float
    total_system_memory_mb: float
    available_system_memory_mb: float
    used_system_memory_mb: float
    process_memory_mb: float
    cache_memory_mb: float
    memory_utilization: float
    pressure_level: MemoryPressureLevel
    threshold_level: MemoryThresholdLevel


class MemoryAlertHandler:
    """Handles memory alerts and notifications."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_callbacks: list[Callable] = []
        self.alert_cooldown = 300.0  # 5 minutes
        self.last_alert_time = 0.0

    def register_alert_callback(self, callback: Callable[[MemoryControlEvent], None]) -> None:
        """Register a callback for memory alerts."""
        self.alert_callbacks.append(callback)

    async def send_alert(self, event: MemoryControlEvent) -> None:
        """Send memory alert."""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return

        self.last_alert_time = current_time

        # Log alert
        self.logger.warning(
            f"Memory alert: {event.memory_level.value} - "
            f"Usage: {event.current_usage_mb:.1f}MB / {event.threshold_mb:.1f}MB - "
            f"Action: {event.action_taken.value}"
        )

        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")


class MemoryThrottler:
    """Controls memory allocation throttling."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.throttle_enabled = False
        self.throttle_factor = 1.0  # 1.0 = no throttling, 0.5 = 50% throttling
        self.throttle_lock = Lock()

        # Throttling statistics
        self.throttled_requests = 0
        self.total_requests = 0

    def set_throttle_level(self, factor: float) -> None:
        """Set throttling factor (0.0 to 1.0)."""
        with self.throttle_lock:
            self.throttle_factor = max(0.0, min(1.0, factor))
            self.throttle_enabled = self.throttle_factor < 1.0

        self.logger.info(f"Memory throttling set to {factor:.2f}")

    async def should_allow_allocation(self, size_mb: float) -> bool:
        """Check if allocation should be allowed based on throttling."""
        with self.throttle_lock:
            self.total_requests += 1

            if not self.throttle_enabled:
                return True

            # Simple probabilistic throttling
            import random

            if random.random() > self.throttle_factor:
                self.throttled_requests += 1
                return False

            return True

    def get_throttle_stats(self) -> dict[str, Any]:
        """Get throttling statistics."""
        with self.throttle_lock:
            throttle_rate = self.throttled_requests / max(1, self.total_requests)
            return {
                "throttle_enabled": self.throttle_enabled,
                "throttle_factor": self.throttle_factor,
                "total_requests": self.total_requests,
                "throttled_requests": self.throttled_requests,
                "throttle_rate": throttle_rate,
            }


class MemoryUsageController:
    """
    Comprehensive memory usage monitoring and control service.

    This service provides:
    - Continuous memory monitoring
    - Threshold-based alerts and actions
    - Automatic memory management
    - Throttling and rate limiting
    - Emergency cleanup procedures
    - Memory limit enforcement
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the memory usage controller."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)

        # Memory limits and thresholds
        self.memory_limit = MemoryLimit(
            max_memory_mb=self.config.memory.max_memory_mb,
            warning_threshold=0.75,
            critical_threshold=0.85,
            emergency_threshold=0.95,
            check_interval_seconds=30.0,
            enforcement_enabled=True,
        )

        # Components
        self.alert_handler = MemoryAlertHandler()
        self.throttler = MemoryThrottler()
        self.profiler: CacheMemoryProfiler | None = None
        self.eviction_coordinator: IntelligentEvictionCoordinator | None = None

        # State tracking
        self.current_stats: MemoryStats | None = None
        self.control_events: list[MemoryControlEvent] = []
        self.max_events_history = 1000

        # Control actions mapping
        self.threshold_actions = {
            MemoryThresholdLevel.NORMAL: MemoryControlAction.NONE,
            MemoryThresholdLevel.ELEVATED: MemoryControlAction.WARN,
            MemoryThresholdLevel.HIGH: MemoryControlAction.THROTTLE,
            MemoryThresholdLevel.CRITICAL: MemoryControlAction.EVICT,
            MemoryThresholdLevel.EMERGENCY: MemoryControlAction.EMERGENCY_CLEANUP,
        }

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

        # Thread safety
        self._lock = RLock()

        # Statistics
        self._stats = {
            "monitoring_cycles": 0,
            "threshold_violations": 0,
            "control_actions_taken": 0,
            "memory_recovered_mb": 0.0,
            "emergency_cleanups": 0,
        }

        # Cache registry for memory control
        self.managed_caches: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the memory usage controller."""
        try:
            self.logger.info("Initializing Memory Usage Controller...")

            # Initialize profiler
            self.profiler = CacheMemoryProfiler(ProfilingLevel.DETAILED)
            await self.profiler.initialize()

            # Start monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            # Initial memory assessment
            await self._update_memory_stats()

            self.logger.info("Memory Usage Controller initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Memory Usage Controller: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the memory usage controller."""
        try:
            self.logger.info("Shutting down Memory Usage Controller...")

            # Cancel monitoring tasks
            for task in [self._monitoring_task, self._cleanup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Shutdown profiler
            if self.profiler:
                await self.profiler.shutdown()

            self.logger.info("Memory Usage Controller shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def configure_memory_limits(
        self,
        max_memory_mb: float,
        warning_threshold: float = 0.75,
        critical_threshold: float = 0.85,
        emergency_threshold: float = 0.95,
        enforcement_enabled: bool = True,
    ) -> dict[str, Any]:
        """
        Configure memory limits and thresholds.

        Args:
            max_memory_mb: Maximum memory limit in MB
            warning_threshold: Warning threshold (0.0-1.0)
            critical_threshold: Critical threshold (0.0-1.0)
            emergency_threshold: Emergency threshold (0.0-1.0)
            enforcement_enabled: Enable/disable enforcement

        Returns:
            Configuration result
        """
        try:
            with self._lock:
                self.memory_limit.max_memory_mb = max_memory_mb
                self.memory_limit.warning_threshold = warning_threshold
                self.memory_limit.critical_threshold = critical_threshold
                self.memory_limit.emergency_threshold = emergency_threshold
                self.memory_limit.enforcement_enabled = enforcement_enabled

            self.logger.info(f"Memory limits configured: {max_memory_mb}MB max")

            return {
                "success": True,
                "configuration": {
                    "max_memory_mb": max_memory_mb,
                    "warning_threshold": warning_threshold,
                    "critical_threshold": critical_threshold,
                    "emergency_threshold": emergency_threshold,
                    "enforcement_enabled": enforcement_enabled,
                },
            }

        except Exception as e:
            self.logger.error(f"Error configuring memory limits: {e}")
            return {"success": False, "error": str(e)}

    def register_cache(self, cache_name: str, cache_instance: Any) -> None:
        """Register a cache for memory management."""
        with self._lock:
            self.managed_caches[cache_name] = cache_instance

        self.logger.info(f"Registered cache '{cache_name}' for memory management")

    def unregister_cache(self, cache_name: str) -> None:
        """Unregister a cache from memory management."""
        with self._lock:
            self.managed_caches.pop(cache_name, None)

        self.logger.info(f"Unregistered cache '{cache_name}' from memory management")

    async def check_memory_compliance(self) -> dict[str, Any]:
        """
        Check current memory compliance against limits.

        Returns:
            Compliance report
        """
        try:
            await self._update_memory_stats()

            if not self.current_stats:
                return {"error": "No memory stats available"}

            stats = self.current_stats
            limit = self.memory_limit

            # Calculate threshold values
            warning_mb = limit.max_memory_mb * limit.warning_threshold
            critical_mb = limit.max_memory_mb * limit.critical_threshold
            emergency_mb = limit.max_memory_mb * limit.emergency_threshold

            # Determine compliance status
            current_cache_mb = stats.cache_memory_mb

            compliance_status = "compliant"
            violation_level = None

            if current_cache_mb >= emergency_mb:
                compliance_status = "emergency_violation"
                violation_level = MemoryThresholdLevel.EMERGENCY
            elif current_cache_mb >= critical_mb:
                compliance_status = "critical_violation"
                violation_level = MemoryThresholdLevel.CRITICAL
            elif current_cache_mb >= warning_mb:
                compliance_status = "warning_violation"
                violation_level = MemoryThresholdLevel.HIGH

            return {
                "compliance_status": compliance_status,
                "violation_level": violation_level.value if violation_level else None,
                "current_usage": {
                    "cache_memory_mb": current_cache_mb,
                    "total_process_mb": stats.process_memory_mb,
                    "utilization_percentage": (current_cache_mb / limit.max_memory_mb) * 100,
                },
                "limits": {
                    "max_memory_mb": limit.max_memory_mb,
                    "warning_mb": warning_mb,
                    "critical_mb": critical_mb,
                    "emergency_mb": emergency_mb,
                },
                "enforcement_enabled": limit.enforcement_enabled,
                "timestamp": stats.timestamp,
            }

        except Exception as e:
            self.logger.error(f"Error checking memory compliance: {e}")
            return {"error": str(e)}

    async def enforce_memory_limits(self, force: bool = False) -> dict[str, Any]:
        """
        Enforce memory limits by taking appropriate actions.

        Args:
            force: Force enforcement even if disabled

        Returns:
            Enforcement result
        """
        try:
            if not self.memory_limit.enforcement_enabled and not force:
                return {"success": False, "reason": "Enforcement is disabled"}

            compliance = await self.check_memory_compliance()

            if compliance.get("compliance_status") == "compliant":
                return {"success": True, "action": "none_required", "status": "compliant"}

            # Determine required action
            violation_level = compliance.get("violation_level")
            if not violation_level:
                return {"success": True, "action": "none_required"}

            threshold_level = MemoryThresholdLevel(violation_level)
            action = self.threshold_actions.get(threshold_level, MemoryControlAction.NONE)

            # Execute action
            result = await self._execute_memory_action(action, threshold_level, compliance)

            # Record event
            event = MemoryControlEvent(
                timestamp=time.time(),
                memory_level=threshold_level,
                current_usage_mb=compliance["current_usage"]["cache_memory_mb"],
                threshold_mb=compliance["limits"]["max_memory_mb"],
                action_taken=action,
                success=result.get("success", False),
                details=result,
            )

            self._record_control_event(event)

            return result

        except Exception as e:
            self.logger.error(f"Error enforcing memory limits: {e}")
            return {"success": False, "error": str(e)}

    async def trigger_emergency_cleanup(self) -> dict[str, Any]:
        """
        Trigger emergency memory cleanup procedures.

        Returns:
            Cleanup result
        """
        try:
            self.logger.warning("Triggering emergency memory cleanup")
            start_time = time.time()

            initial_stats = await self._get_current_memory_usage()
            cleanup_results = []

            # 1. Force garbage collection
            import gc

            before_gc = gc.get_count()
            collected = gc.collect()
            after_gc = gc.get_count()
            cleanup_results.append(
                {"action": "garbage_collection", "objects_collected": collected, "before_counts": before_gc, "after_counts": after_gc}
            )

            # 2. Emergency cache eviction
            if self.eviction_coordinator:
                for cache_name in self.managed_caches:
                    try:
                        eviction_result = await self.eviction_coordinator.make_eviction_decisions(
                            cache_name, target_count=100, memory_pressure=MemoryPressureLevel.CRITICAL
                        )
                        cleanup_results.append(
                            {"action": "emergency_eviction", "cache_name": cache_name, "evicted_count": len(eviction_result)}
                        )
                    except Exception as e:
                        self.logger.error(f"Emergency eviction failed for {cache_name}: {e}")

            # 3. Clear temporary data
            if self.profiler:
                self.profiler.reset_profiling_data()
                cleanup_results.append({"action": "clear_profiling_data"})

            # 4. Aggressive throttling
            self.throttler.set_throttle_level(0.1)  # 90% throttling
            cleanup_results.append({"action": "aggressive_throttling", "factor": 0.1})

            final_stats = await self._get_current_memory_usage()
            memory_recovered = initial_stats - final_stats

            duration = time.time() - start_time
            self._stats["emergency_cleanups"] += 1
            self._stats["memory_recovered_mb"] += memory_recovered

            return {
                "success": True,
                "memory_recovered_mb": memory_recovered,
                "initial_usage_mb": initial_stats,
                "final_usage_mb": final_stats,
                "cleanup_actions": cleanup_results,
                "duration_seconds": duration,
            }

        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_memory_status(self) -> dict[str, Any]:
        """
        Get comprehensive memory status report.

        Returns:
            Memory status report
        """
        try:
            await self._update_memory_stats()
            compliance = await self.check_memory_compliance()
            throttle_stats = self.throttler.get_throttle_stats()

            return {
                "timestamp": time.time(),
                "memory_stats": self.current_stats.__dict__ if self.current_stats else {},
                "compliance": compliance,
                "throttling": throttle_stats,
                "configuration": {
                    "max_memory_mb": self.memory_limit.max_memory_mb,
                    "enforcement_enabled": self.memory_limit.enforcement_enabled,
                    "check_interval": self.memory_limit.check_interval_seconds,
                },
                "managed_caches": list(self.managed_caches.keys()),
                "recent_events": [event.__dict__ for event in self.control_events[-10:]],
                "statistics": self._stats,
            }

        except Exception as e:
            self.logger.error(f"Error getting memory status: {e}")
            return {"error": str(e)}

    async def optimize_memory_usage(self) -> dict[str, Any]:
        """
        Optimize memory usage across all managed components.

        Returns:
            Optimization result
        """
        try:
            self.logger.info("Starting memory usage optimization")
            start_time = time.time()

            initial_usage = await self._get_current_memory_usage()
            optimization_actions = []

            # 1. Analyze current usage patterns
            if self.profiler:
                patterns = self.profiler.get_allocation_patterns(window_minutes=30)
                optimization_actions.append({"action": "pattern_analysis", "result": patterns})

            # 2. Intelligent cache size adjustment
            for cache_name, cache_instance in self.managed_caches.items():
                try:
                    if hasattr(cache_instance, "optimize_size"):
                        result = await cache_instance.optimize_size()
                        optimization_actions.append({"action": "cache_optimization", "cache_name": cache_name, "result": result})
                except Exception as e:
                    self.logger.error(f"Cache optimization failed for {cache_name}: {e}")

            # 3. Adjust throttling based on current state
            current_utilization = initial_usage / self.memory_limit.max_memory_mb
            if current_utilization > 0.8:
                self.throttler.set_throttle_level(0.7)  # 30% throttling
            elif current_utilization > 0.6:
                self.throttler.set_throttle_level(0.85)  # 15% throttling
            else:
                self.throttler.set_throttle_level(1.0)  # No throttling

            optimization_actions.append(
                {"action": "throttle_adjustment", "utilization": current_utilization, "throttle_factor": self.throttler.throttle_factor}
            )

            # 4. Proactive cleanup
            import gc

            gc.collect()
            optimization_actions.append({"action": "garbage_collection"})

            final_usage = await self._get_current_memory_usage()
            memory_saved = initial_usage - final_usage
            duration = time.time() - start_time

            return {
                "success": True,
                "memory_saved_mb": memory_saved,
                "initial_usage_mb": initial_usage,
                "final_usage_mb": final_usage,
                "optimization_actions": optimization_actions,
                "duration_seconds": duration,
            }

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {"success": False, "error": str(e)}

    # Helper Methods

    async def _monitoring_loop(self) -> None:
        """Main memory monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.memory_limit.check_interval_seconds)

                self._stats["monitoring_cycles"] += 1

                # Update memory stats
                await self._update_memory_stats()

                # Check compliance and take action if needed
                if self.memory_limit.enforcement_enabled:
                    await self.enforce_memory_limits()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes

                # Cleanup old events
                current_time = time.time()
                cutoff_time = current_time - 3600  # Keep 1 hour of history

                self.control_events = [event for event in self.control_events if event.timestamp >= cutoff_time]

                # Limit event history size
                if len(self.control_events) > self.max_events_history:
                    self.control_events = self.control_events[-self.max_events_history :]

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _update_memory_stats(self) -> None:
        """Update current memory statistics."""
        try:
            # Get system memory info
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            # Get cache memory usage
            cache_memory_mb = 0.0
            for cache_name, cache_instance in self.managed_caches.items():
                try:
                    if hasattr(cache_instance, "get_memory_usage"):
                        cache_memory_mb += await cache_instance.get_memory_usage()
                except Exception:
                    pass

            # Get pressure level
            pressure_info = get_system_memory_pressure()

            # Determine threshold level
            utilization = memory_info.percent / 100.0
            if utilization >= 0.95:
                threshold_level = MemoryThresholdLevel.EMERGENCY
            elif utilization >= 0.85:
                threshold_level = MemoryThresholdLevel.CRITICAL
            elif utilization >= 0.75:
                threshold_level = MemoryThresholdLevel.HIGH
            elif utilization >= 0.60:
                threshold_level = MemoryThresholdLevel.ELEVATED
            else:
                threshold_level = MemoryThresholdLevel.NORMAL

            self.current_stats = MemoryStats(
                timestamp=time.time(),
                total_system_memory_mb=memory_info.total / (1024 * 1024),
                available_system_memory_mb=memory_info.available / (1024 * 1024),
                used_system_memory_mb=memory_info.used / (1024 * 1024),
                process_memory_mb=process_memory.rss / (1024 * 1024),
                cache_memory_mb=cache_memory_mb,
                memory_utilization=utilization,
                pressure_level=pressure_info.level,
                threshold_level=threshold_level,
            )

        except Exception as e:
            self.logger.error(f"Error updating memory stats: {e}")

    async def _get_current_memory_usage(self) -> float:
        """Get current cache memory usage in MB."""
        cache_memory_mb = 0.0
        for cache_name, cache_instance in self.managed_caches.items():
            try:
                if hasattr(cache_instance, "get_memory_usage"):
                    cache_memory_mb += await cache_instance.get_memory_usage()
            except Exception:
                pass
        return cache_memory_mb

    async def _execute_memory_action(
        self, action: MemoryControlAction, threshold_level: MemoryThresholdLevel, compliance_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute memory control action."""
        try:
            if action == MemoryControlAction.NONE:
                return {"success": True, "action": "none"}

            elif action == MemoryControlAction.WARN:
                await self.alert_handler.send_alert(
                    MemoryControlEvent(
                        timestamp=time.time(),
                        memory_level=threshold_level,
                        current_usage_mb=compliance_info["current_usage"]["cache_memory_mb"],
                        threshold_mb=compliance_info["limits"]["max_memory_mb"],
                        action_taken=action,
                        success=True,
                    )
                )
                return {"success": True, "action": "warning_sent"}

            elif action == MemoryControlAction.THROTTLE:
                # Adjust throttling based on severity
                if threshold_level == MemoryThresholdLevel.HIGH:
                    self.throttler.set_throttle_level(0.8)  # 20% throttling
                else:
                    self.throttler.set_throttle_level(0.6)  # 40% throttling

                return {"success": True, "action": "throttling_enabled", "throttle_factor": self.throttler.throttle_factor}

            elif action == MemoryControlAction.EVICT:
                evicted_total = 0
                if self.eviction_coordinator:
                    for cache_name in self.managed_caches:
                        try:
                            decisions = await self.eviction_coordinator.make_eviction_decisions(
                                cache_name, target_count=50, memory_pressure=MemoryPressureLevel.HIGH
                            )
                            evicted_total += len(decisions)
                        except Exception as e:
                            self.logger.error(f"Eviction failed for {cache_name}: {e}")

                return {"success": True, "action": "eviction_triggered", "total_evicted": evicted_total}

            elif action == MemoryControlAction.EMERGENCY_CLEANUP:
                return await self.trigger_emergency_cleanup()

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.error(f"Error executing memory action {action}: {e}")
            return {"success": False, "error": str(e)}

    def _record_control_event(self, event: MemoryControlEvent) -> None:
        """Record a memory control event."""
        with self._lock:
            self.control_events.append(event)

            # Update statistics
            if event.memory_level != MemoryThresholdLevel.NORMAL:
                self._stats["threshold_violations"] += 1

            if event.action_taken != MemoryControlAction.NONE:
                self._stats["control_actions_taken"] += 1


# Global controller instance
_memory_controller: MemoryUsageController | None = None


async def get_memory_controller(config: CacheConfig | None = None) -> MemoryUsageController:
    """Get the global memory usage controller instance."""
    global _memory_controller
    if _memory_controller is None:
        _memory_controller = MemoryUsageController(config)
        await _memory_controller.initialize()
    return _memory_controller


async def shutdown_memory_controller() -> None:
    """Shutdown the global memory usage controller."""
    global _memory_controller
    if _memory_controller:
        await _memory_controller.shutdown()
        _memory_controller = None
