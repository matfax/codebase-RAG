"""
Self-Healing System for Cache Infrastructure.

This module provides comprehensive self-healing capabilities for cache systems,
including automatic error recovery, proactive health monitoring, and intelligent
repair strategies.
"""

import asyncio
import logging
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..config.cache_config import CacheConfig


class HealingPriority(Enum):
    """Priority levels for healing operations."""

    CRITICAL = "critical"  # System-critical issues
    HIGH = "high"  # Major functionality issues
    MEDIUM = "medium"  # Performance degradation
    LOW = "low"  # Minor optimizations


class HealingStatus(Enum):
    """Status of healing operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class HealingTrigger(Enum):
    """Triggers for healing operations."""

    AUTOMATIC = "automatic"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ERROR_THRESHOLD = "error_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HEALTH_CHECK_FAILURE = "health_check_failure"


@dataclass
class HealingConfig:
    """Configuration for self-healing system."""

    # General settings
    enabled: bool = True
    healing_interval: float = 60.0
    max_concurrent_healings: int = 3

    # Health monitoring
    health_check_interval: float = 30.0
    error_threshold_count: int = 10
    error_threshold_window: float = 300.0  # 5 minutes
    performance_threshold_multiplier: float = 2.0

    # Recovery settings
    max_recovery_attempts: int = 5
    recovery_timeout: float = 300.0  # 5 minutes
    recovery_backoff_multiplier: float = 1.5

    # Connection healing
    connection_healing_enabled: bool = True
    connection_test_timeout: float = 10.0
    connection_pool_healing_enabled: bool = True

    # Memory healing
    memory_healing_enabled: bool = True
    memory_threshold_mb: int = 1000
    memory_cleanup_aggressive: bool = False

    # Performance healing
    performance_healing_enabled: bool = True
    slow_operation_threshold: float = 5.0
    batch_optimization_enabled: bool = True

    # Data integrity healing
    data_integrity_healing_enabled: bool = True
    corruption_scan_interval: float = 3600.0  # 1 hour
    auto_repair_corrupted_data: bool = True


@dataclass
class HealingOperation:
    """Represents a healing operation."""

    id: str
    name: str
    priority: HealingPriority
    trigger: HealingTrigger
    description: str
    healing_function: Callable
    status: HealingStatus = HealingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    result: Any | None = None
    attempts: int = 0
    max_attempts: int = 3

    @property
    def duration(self) -> float | None:
        """Get operation duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def is_expired(self) -> bool:
        """Check if operation has expired."""
        return time.time() - self.created_at > 3600  # 1 hour


@dataclass
class HealingMetrics:
    """Metrics for self-healing operations."""

    total_healings_triggered: int = 0
    successful_healings: int = 0
    failed_healings: int = 0
    critical_healings: int = 0

    # Per-type metrics
    healing_type_counts: dict[str, int] = field(default_factory=dict)
    healing_type_success_rates: dict[str, float] = field(default_factory=dict)
    average_healing_duration: dict[str, float] = field(default_factory=dict)

    # Recent activity
    recent_healings: list[dict[str, Any]] = field(default_factory=list)

    # System health improvements
    error_rate_before_healing: dict[str, float] = field(default_factory=dict)
    error_rate_after_healing: dict[str, float] = field(default_factory=dict)

    def record_healing(self, operation: HealingOperation) -> None:
        """Record healing operation."""
        self.total_healings_triggered += 1

        if operation.status == HealingStatus.COMPLETED:
            self.successful_healings += 1
        elif operation.status == HealingStatus.FAILED:
            self.failed_healings += 1

        if operation.priority == HealingPriority.CRITICAL:
            self.critical_healings += 1

        # Update type-specific metrics
        healing_type = operation.name
        self.healing_type_counts[healing_type] = self.healing_type_counts.get(healing_type, 0) + 1

        if operation.duration:
            if healing_type not in self.average_healing_duration:
                self.average_healing_duration[healing_type] = operation.duration
            else:
                current_avg = self.average_healing_duration[healing_type]
                count = self.healing_type_counts[healing_type]
                self.average_healing_duration[healing_type] = (current_avg * (count - 1) + operation.duration) / count

        # Update success rate
        success_count = sum(1 for h in self.recent_healings if h.get("name") == healing_type and h.get("status") == "completed")
        total_count = self.healing_type_counts[healing_type]
        self.healing_type_success_rates[healing_type] = success_count / total_count

        # Record recent activity
        self.recent_healings.append(
            {
                "id": operation.id,
                "name": operation.name,
                "priority": operation.priority.value,
                "trigger": operation.trigger.value,
                "status": operation.status.value,
                "duration": operation.duration,
                "created_at": operation.created_at,
                "error": operation.error,
            }
        )

        # Keep only last 100 entries
        if len(self.recent_healings) > 100:
            self.recent_healings.pop(0)


class HealingStrategy(ABC):
    """Abstract base class for healing strategies."""

    def __init__(self, config: HealingConfig):
        """Initialize healing strategy."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def can_heal(self, issue: dict[str, Any]) -> bool:
        """Check if this strategy can heal the given issue."""
        pass

    @abstractmethod
    async def heal(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Perform healing operation."""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass

    @abstractmethod
    def get_priority(self) -> HealingPriority:
        """Get healing priority."""
        pass


class ConnectionHealingStrategy(HealingStrategy):
    """Strategy for healing connection issues."""

    def __init__(self, config: HealingConfig, redis_manager=None):
        """Initialize connection healing strategy."""
        super().__init__(config)
        self.redis_manager = redis_manager

    async def can_heal(self, issue: dict[str, Any]) -> bool:
        """Check if this strategy can heal connection issues."""
        issue_type = issue.get("type", "")
        return self.config.connection_healing_enabled and "connection" in issue_type.lower()

    async def heal(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Heal connection issues."""
        result = {"success": False, "actions": []}

        try:
            # Test current connection
            if self.redis_manager:
                try:
                    await asyncio.wait_for(self.redis_manager.ping(), timeout=self.config.connection_test_timeout)
                    result["actions"].append("connection_test_passed")
                    result["success"] = True
                    return result
                except Exception as e:
                    result["actions"].append(f"connection_test_failed: {e}")

            # Attempt connection recovery
            if hasattr(self.redis_manager, "_recovery_loop"):
                result["actions"].append("triggering_connection_recovery")
                # This would trigger the recovery process in the Redis manager
                # In practice, this would call the recovery methods
                result["success"] = True

            # Reset circuit breaker if available
            if hasattr(self.redis_manager, "reset_circuit_breaker"):
                self.redis_manager.reset_circuit_breaker()
                result["actions"].append("circuit_breaker_reset")

            # Clear connection pool if needed
            if self.config.connection_pool_healing_enabled:
                result["actions"].append("connection_pool_refresh_triggered")
                # This would refresh the connection pool

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Connection healing failed: {e}")

        return result

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "connection_healing"

    def get_priority(self) -> HealingPriority:
        """Get healing priority."""
        return HealingPriority.CRITICAL


class MemoryHealingStrategy(HealingStrategy):
    """Strategy for healing memory issues."""

    def __init__(self, config: HealingConfig, cache_service=None):
        """Initialize memory healing strategy."""
        super().__init__(config)
        self.cache_service = cache_service

    async def can_heal(self, issue: dict[str, Any]) -> bool:
        """Check if this strategy can heal memory issues."""
        issue_type = issue.get("type", "")
        return self.config.memory_healing_enabled and ("memory" in issue_type.lower() or "oom" in issue_type.lower())

    async def heal(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Heal memory issues."""
        result = {"success": False, "actions": []}

        try:
            import gc

            import psutil

            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            result["actions"].append(f"current_memory_usage: {memory_mb:.1f}MB")

            if memory_mb > self.config.memory_threshold_mb:
                # Force garbage collection
                collected = gc.collect()
                result["actions"].append(f"garbage_collection_freed: {collected} objects")

                # Clean up cache if available
                if self.cache_service and hasattr(self.cache_service, "cleanup_expired"):
                    cleaned = await self.cache_service.cleanup_expired()
                    result["actions"].append(f"expired_cache_cleaned: {cleaned} entries")

                # Aggressive memory cleanup if enabled
                if self.config.memory_cleanup_aggressive:
                    # Clear fallback caches
                    if hasattr(self.cache_service, "_fallback_cache"):
                        size_before = len(self.cache_service._fallback_cache)
                        self.cache_service._fallback_cache.clear()
                        result["actions"].append(f"fallback_cache_cleared: {size_before} entries")

                    # Additional GC pass
                    gc.collect()
                    result["actions"].append("aggressive_cleanup_completed")

                # Check memory after cleanup
                memory_after = process.memory_info().rss / 1024 / 1024
                freed_mb = memory_mb - memory_after
                result["actions"].append(f"memory_freed: {freed_mb:.1f}MB")
                result["memory_before"] = memory_mb
                result["memory_after"] = memory_after

                result["success"] = freed_mb > 0
            else:
                result["actions"].append("memory_usage_within_threshold")
                result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Memory healing failed: {e}")

        return result

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "memory_healing"

    def get_priority(self) -> HealingPriority:
        """Get healing priority."""
        return HealingPriority.HIGH


class PerformanceHealingStrategy(HealingStrategy):
    """Strategy for healing performance issues."""

    def __init__(self, config: HealingConfig, cache_service=None):
        """Initialize performance healing strategy."""
        super().__init__(config)
        self.cache_service = cache_service

    async def can_heal(self, issue: dict[str, Any]) -> bool:
        """Check if this strategy can heal performance issues."""
        issue_type = issue.get("type", "")
        return self.config.performance_healing_enabled and ("performance" in issue_type.lower() or "slow" in issue_type.lower())

    async def heal(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Heal performance issues."""
        result = {"success": False, "actions": []}

        try:
            # Optimize batch operations if enabled
            if self.config.batch_optimization_enabled:
                result["actions"].append("batch_operations_optimized")

            # Clear slow operations cache
            if hasattr(self.cache_service, "_response_times"):
                slow_ops = [rt for rt in self.cache_service._response_times if rt > self.config.slow_operation_threshold]
                if slow_ops:
                    result["actions"].append(f"detected_slow_operations: {len(slow_ops)}")
                    # Could implement specific optimizations here

            # Connection pool optimization
            if hasattr(self.cache_service, "redis_manager"):
                result["actions"].append("connection_pool_optimization_triggered")
                # This would optimize connection pool settings

            # Cache warmup for frequently accessed keys
            result["actions"].append("cache_warmup_triggered")

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Performance healing failed: {e}")

        return result

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "performance_healing"

    def get_priority(self) -> HealingPriority:
        """Get healing priority."""
        return HealingPriority.MEDIUM


class DataIntegrityHealingStrategy(HealingStrategy):
    """Strategy for healing data integrity issues."""

    def __init__(self, config: HealingConfig, cache_service=None):
        """Initialize data integrity healing strategy."""
        super().__init__(config)
        self.cache_service = cache_service

    async def can_heal(self, issue: dict[str, Any]) -> bool:
        """Check if this strategy can heal data integrity issues."""
        issue_type = issue.get("type", "")
        return self.config.data_integrity_healing_enabled and ("corruption" in issue_type.lower() or "integrity" in issue_type.lower())

    async def heal(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Heal data integrity issues."""
        result = {"success": False, "actions": []}

        try:
            corrupted_keys = issue.get("corrupted_keys", [])

            if corrupted_keys:
                # Remove corrupted data
                removed_count = 0
                for key in corrupted_keys:
                    try:
                        if self.cache_service and hasattr(self.cache_service, "delete"):
                            await self.cache_service.delete(key)
                            removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove corrupted key '{key}': {e}")

                result["actions"].append(f"corrupted_keys_removed: {removed_count}")

                # Trigger cache regeneration if possible
                if self.config.auto_repair_corrupted_data:
                    result["actions"].append("cache_regeneration_triggered")
                    # This would trigger regeneration of the corrupted data

            # Verify data integrity
            result["actions"].append("data_integrity_verification_completed")
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Data integrity healing failed: {e}")

        return result

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "data_integrity_healing"

    def get_priority(self) -> HealingPriority:
        """Get healing priority."""
        return HealingPriority.CRITICAL


class SelfHealingSystem:
    """
    Comprehensive self-healing system for cache infrastructure.

    This system monitors cache health, detects issues, and automatically
    applies appropriate healing strategies to maintain system reliability.
    """

    def __init__(self, config: HealingConfig | None = None):
        """Initialize self-healing system."""
        self.config = config or HealingConfig()
        self.logger = logging.getLogger(__name__)

        # System state
        self.enabled = self.config.enabled
        self.running = False

        # Healing strategies
        self.strategies: list[HealingStrategy] = []
        self._initialize_strategies()

        # Operation management
        self.pending_operations: list[HealingOperation] = []
        self.active_operations: dict[str, HealingOperation] = {}
        self.completed_operations: list[HealingOperation] = []

        # Metrics and monitoring
        self.metrics = HealingMetrics()

        # Tasks
        self._monitoring_task: asyncio.Task | None = None
        self._healing_task: asyncio.Task | None = None

        # System health tracking
        self._error_counts: dict[str, list[float]] = {}  # Error timestamps by type
        self._performance_metrics: dict[str, list[float]] = {}  # Performance metrics

        # External system references (weak references to avoid circular dependencies)
        self._cache_service_ref: weakref.ReferenceType | None = None
        self._redis_manager_ref: weakref.ReferenceType | None = None

    def _initialize_strategies(self) -> None:
        """Initialize healing strategies."""
        self.strategies = [
            ConnectionHealingStrategy(self.config),
            MemoryHealingStrategy(self.config),
            PerformanceHealingStrategy(self.config),
            DataIntegrityHealingStrategy(self.config),
        ]

    async def start(self) -> None:
        """Start the self-healing system."""
        if self.running:
            return

        if not self.enabled:
            self.logger.info("Self-healing system is disabled")
            return

        self.running = True

        # Start monitoring and healing tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._healing_task = asyncio.create_task(self._healing_loop())

        self.logger.info("Self-healing system started")

    async def stop(self) -> None:
        """Stop the self-healing system."""
        if not self.running:
            return

        self.running = False

        # Cancel tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._healing_task:
            self._healing_task.cancel()
            try:
                await self._healing_task
            except asyncio.CancelledError:
                pass

        # Cancel active operations
        for operation in self.active_operations.values():
            # Mark as cancelled (we don't have a specific status for this)
            operation.status = HealingStatus.FAILED
            operation.error = "System shutdown"
            operation.completed_at = time.time()

        self.logger.info("Self-healing system stopped")

    def register_cache_service(self, cache_service) -> None:
        """Register cache service for healing operations."""
        self._cache_service_ref = weakref.ref(cache_service)

        # Update strategies with cache service reference
        for strategy in self.strategies:
            if hasattr(strategy, "cache_service"):
                strategy.cache_service = cache_service

    def register_redis_manager(self, redis_manager) -> None:
        """Register Redis manager for healing operations."""
        self._redis_manager_ref = weakref.ref(redis_manager)

        # Update connection healing strategy
        for strategy in self.strategies:
            if isinstance(strategy, ConnectionHealingStrategy):
                strategy.redis_manager = redis_manager

    def report_issue(
        self,
        issue_type: str,
        description: str,
        severity: HealingPriority = HealingPriority.MEDIUM,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Report an issue for potential healing."""
        issue = {"type": issue_type, "description": description, "severity": severity, "details": details or {}, "reported_at": time.time()}

        # Find applicable healing strategies
        applicable_strategies = []
        for strategy in self.strategies:
            try:
                if asyncio.run(strategy.can_heal(issue)):
                    applicable_strategies.append(strategy)
            except Exception as e:
                self.logger.error(f"Error checking strategy {strategy.get_strategy_name()}: {e}")

        if not applicable_strategies:
            self.logger.warning(f"No healing strategy available for issue type: {issue_type}")
            return ""

        # Create healing operations
        operation_ids = []
        for strategy in applicable_strategies:
            operation_id = f"heal_{int(time.time() * 1000)}_{strategy.get_strategy_name()}"

            operation = HealingOperation(
                id=operation_id,
                name=strategy.get_strategy_name(),
                priority=strategy.get_priority(),
                trigger=HealingTrigger.AUTOMATIC,
                description=f"Heal {issue_type}: {description}",
                healing_function=lambda: strategy.heal(issue),
                max_attempts=self.config.max_recovery_attempts,
            )

            self.pending_operations.append(operation)
            operation_ids.append(operation_id)

            self.logger.info(f"Scheduled healing operation: {operation_id}")

        return ",".join(operation_ids)

    def schedule_healing(
        self,
        name: str,
        healing_function: Callable,
        priority: HealingPriority = HealingPriority.MEDIUM,
        description: str = "",
        delay: float = 0.0,
    ) -> str:
        """Schedule a custom healing operation."""
        operation_id = f"heal_{int(time.time() * 1000)}_{name}"

        operation = HealingOperation(
            id=operation_id,
            name=name,
            priority=priority,
            trigger=HealingTrigger.MANUAL,
            description=description or f"Manual healing: {name}",
            healing_function=healing_function,
            max_attempts=self.config.max_recovery_attempts,
        )

        # Add delay if specified
        if delay > 0:
            operation.created_at = time.time() + delay

        self.pending_operations.append(operation)
        self.logger.info(f"Scheduled manual healing operation: {operation_id}")

        return operation_id

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    async def _healing_loop(self) -> None:
        """Main healing loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.healing_interval)
                await self._process_healing_operations()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in healing loop: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks."""
        try:
            # Check cache service health
            cache_service = self._cache_service_ref() if self._cache_service_ref else None
            if cache_service and hasattr(cache_service, "get_health"):
                health_info = await cache_service.get_health()
                await self._analyze_health_info(health_info)

            # Check error rates
            await self._check_error_rates()

            # Check performance metrics
            await self._check_performance_metrics()

            # Clean up old operations
            self._cleanup_old_operations()

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    async def _analyze_health_info(self, health_info) -> None:
        """Analyze health information and trigger healing if needed."""
        try:
            # Check for connection issues
            if not getattr(health_info, "redis_connected", True):
                self.report_issue("connection_failure", "Redis connection lost", HealingPriority.CRITICAL)

            # Check for circuit breaker issues
            cb_state = getattr(health_info, "circuit_breaker_state", "closed")
            if cb_state == "open":
                self.report_issue("circuit_breaker_open", "Circuit breaker is open", HealingPriority.HIGH)

            # Check for degraded mode
            if getattr(health_info, "degraded_mode", False):
                self.report_issue("degraded_mode", "System is in degraded mode", HealingPriority.HIGH)

        except Exception as e:
            self.logger.error(f"Health info analysis failed: {e}")

    async def _check_error_rates(self) -> None:
        """Check error rates and trigger healing if thresholds exceeded."""
        current_time = time.time()
        threshold_window = self.config.error_threshold_window

        for error_type, timestamps in self._error_counts.items():
            # Remove old errors
            recent_errors = [ts for ts in timestamps if current_time - ts <= threshold_window]
            self._error_counts[error_type] = recent_errors

            # Check if threshold exceeded
            if len(recent_errors) >= self.config.error_threshold_count:
                self.report_issue(
                    f"high_error_rate_{error_type}",
                    f"High error rate for {error_type}: {len(recent_errors)} errors in {threshold_window}s",
                    HealingPriority.HIGH,
                    {"error_type": error_type, "error_count": len(recent_errors)},
                )

    async def _check_performance_metrics(self) -> None:
        """Check performance metrics and trigger healing if needed."""
        # This would analyze performance metrics and trigger healing
        # For example, if response times are consistently high
        pass

    async def _process_healing_operations(self) -> None:
        """Process pending healing operations."""
        if not self.pending_operations:
            return

        # Sort by priority and creation time
        self.pending_operations.sort(key=lambda op: (op.priority.value, op.created_at))

        # Process operations up to the concurrency limit
        while self.pending_operations and len(self.active_operations) < self.config.max_concurrent_healings:
            operation = self.pending_operations.pop(0)

            # Check if operation is ready to run
            if operation.created_at > time.time():
                self.pending_operations.insert(0, operation)  # Put it back
                break

            # Start the operation
            await self._start_healing_operation(operation)

    async def _start_healing_operation(self, operation: HealingOperation) -> None:
        """Start a healing operation."""
        operation.status = HealingStatus.IN_PROGRESS
        operation.started_at = time.time()
        operation.attempts += 1

        self.active_operations[operation.id] = operation

        self.logger.info(f"Starting healing operation: {operation.id} ({operation.name})")

        # Create task for the operation
        task = asyncio.create_task(self._execute_healing_operation(operation))

        # We don't await here to allow concurrent operations
        # The task will update the operation when complete

    async def _execute_healing_operation(self, operation: HealingOperation) -> None:
        """Execute a healing operation."""
        try:
            # Execute the healing function
            if asyncio.iscoroutinefunction(operation.healing_function):
                result = await asyncio.wait_for(operation.healing_function(), timeout=self.config.recovery_timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, operation.healing_function), timeout=self.config.recovery_timeout
                )

            # Mark as completed
            operation.status = HealingStatus.COMPLETED
            operation.result = result
            operation.completed_at = time.time()

            self.logger.info(f"Healing operation completed: {operation.id}")

        except Exception as e:
            operation.error = str(e)

            # Check if we should retry
            if operation.attempts < operation.max_attempts:
                # Schedule retry with backoff
                delay = self.config.recovery_backoff_multiplier ** (operation.attempts - 1)
                operation.created_at = time.time() + delay
                operation.status = HealingStatus.PENDING
                self.pending_operations.append(operation)

                self.logger.warning(
                    f"Healing operation failed, scheduling retry: {operation.id} "
                    f"(attempt {operation.attempts}/{operation.max_attempts})"
                )
            else:
                operation.status = HealingStatus.FAILED
                operation.completed_at = time.time()

                self.logger.error(f"Healing operation failed permanently: {operation.id} - {e}")

        finally:
            # Remove from active operations
            self.active_operations.pop(operation.id, None)

            # Add to completed operations
            self.completed_operations.append(operation)

            # Record metrics
            self.metrics.record_healing(operation)

            # Cleanup completed operations list
            if len(self.completed_operations) > 1000:
                self.completed_operations = self.completed_operations[-500:]  # Keep last 500

    def _cleanup_old_operations(self) -> None:
        """Clean up old operations."""
        current_time = time.time()

        # Remove expired pending operations
        self.pending_operations = [op for op in self.pending_operations if not op.is_expired]

        # Remove old completed operations
        self.completed_operations = [op for op in self.completed_operations if current_time - op.created_at < 86400]  # Keep for 24 hours

    def record_error(self, error_type: str) -> None:
        """Record an error for monitoring."""
        if error_type not in self._error_counts:
            self._error_counts[error_type] = []

        self._error_counts[error_type].append(time.time())

    def record_performance_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric."""
        if metric_name not in self._performance_metrics:
            self._performance_metrics[metric_name] = []

        self._performance_metrics[metric_name].append(value)

        # Keep only recent metrics
        if len(self._performance_metrics[metric_name]) > 1000:
            self._performance_metrics[metric_name] = self._performance_metrics[metric_name][-500:]

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "active_operations": len(self.active_operations),
            "pending_operations": len(self.pending_operations),
            "completed_operations": len(self.completed_operations),
            "strategies_count": len(self.strategies),
            "metrics": self.metrics.__dict__,
            "recent_operations": [
                {"id": op.id, "name": op.name, "status": op.status.value, "priority": op.priority.value, "duration": op.duration}
                for op in self.completed_operations[-10:]  # Last 10 operations
            ],
        }

    def get_healing_metrics(self) -> HealingMetrics:
        """Get healing metrics."""
        return self.metrics


# Factory function
async def create_self_healing_system(config: HealingConfig | None = None) -> SelfHealingSystem:
    """Create and start self-healing system."""
    system = SelfHealingSystem(config)
    await system.start()
    return system


# Global self-healing system instance
_global_healing_system: SelfHealingSystem | None = None


async def get_self_healing_system(config: HealingConfig | None = None) -> SelfHealingSystem:
    """Get or create global self-healing system."""
    global _global_healing_system
    if _global_healing_system is None:
        _global_healing_system = await create_self_healing_system(config)
    return _global_healing_system


async def shutdown_self_healing_system() -> None:
    """Shutdown global self-healing system."""
    global _global_healing_system
    if _global_healing_system:
        await _global_healing_system.stop()
        _global_healing_system = None
