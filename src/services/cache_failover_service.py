"""
Cache failover service for the Codebase RAG MCP Server.

This service provides automatic failover capabilities for cache services,
including health monitoring, failover detection, and seamless cache switching.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..services.cache_service import BaseCacheService, get_cache_service
from ..utils.telemetry import get_telemetry_manager, trace_cache_operation


class FailoverTrigger(Enum):
    """Types of failover triggers."""

    HEALTH_CHECK_FAILURE = "health_check_failure"
    CONNECTION_FAILURE = "connection_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL_TRIGGER = "manual_trigger"
    CASCADE_FAILURE = "cascade_failure"
    TIMEOUT_EXCEEDED = "timeout_exceeded"


class FailoverStatus(Enum):
    """Failover operation status."""

    ACTIVE = "active"  # Primary service is active
    FAILING_OVER = "failing_over"  # In process of failing over
    FAILED_OVER = "failed_over"  # Operating on failover service
    RECOVERING = "recovering"  # Attempting to recover to primary
    RECOVERED = "recovered"  # Successfully recovered to primary


class ServiceHealth(Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class FailoverEvent:
    """Represents a failover event."""

    event_id: str
    trigger: FailoverTrigger
    timestamp: datetime
    primary_service_id: str
    failover_service_id: str
    trigger_details: dict[str, Any]
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str | None = None
    recovery_timestamp: datetime | None = None


@dataclass
class ServiceHealthStatus:
    """Health status of a cache service."""

    service_id: str
    health: ServiceHealth
    last_check: datetime
    response_time_ms: float
    error_count: int
    success_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverConfiguration:
    """Configuration for failover behavior."""

    health_check_interval_seconds: int = 30
    failure_threshold: int = 3
    recovery_threshold: int = 5
    timeout_threshold_ms: float = 5000.0
    performance_degradation_threshold: float = 0.5  # 50% slower than baseline
    auto_recovery_enabled: bool = True
    auto_recovery_delay_seconds: int = 300  # 5 minutes
    cascade_failover_enabled: bool = False
    max_failover_attempts: int = 3
    health_check_timeout_seconds: int = 10


class CacheFailoverService:
    """Service for managing cache failover and high availability."""

    def __init__(self, config: CacheConfig | None = None, failover_config: FailoverConfiguration | None = None):
        """Initialize the cache failover service."""
        self.config = config or get_global_cache_config()
        self.failover_config = failover_config or FailoverConfiguration()
        self.logger = logging.getLogger(__name__)
        self._telemetry = get_telemetry_manager()

        # Service registry
        self._primary_service: BaseCacheService | None = None
        self._failover_services: list[BaseCacheService] = []
        self._current_service: BaseCacheService | None = None
        self._service_health: dict[str, ServiceHealthStatus] = {}

        # Failover state
        self._failover_status = FailoverStatus.ACTIVE
        self._failover_events: list[FailoverEvent] = []
        self._consecutive_failures: dict[str, int] = {}
        self._consecutive_successes: dict[str, int] = {}

        # Background tasks
        self._health_monitor_task: asyncio.Task | None = None
        self._auto_recovery_task: asyncio.Task | None = None

        # Callbacks
        self._failover_callbacks: list[Callable[[FailoverEvent], None]] = []
        self._recovery_callbacks: list[Callable[[FailoverEvent], None]] = []

        # Performance baselines
        self._performance_baselines: dict[str, float] = {}
        self._baseline_calculation_window = 100  # operations

    async def initialize(self):
        """Initialize the failover service."""
        try:
            # Get primary cache service
            self._primary_service = await get_cache_service()
            self._current_service = self._primary_service

            # Initialize health status
            primary_id = self._get_service_id(self._primary_service)
            self._service_health[primary_id] = ServiceHealthStatus(
                service_id=primary_id,
                health=ServiceHealth.UNKNOWN,
                last_check=datetime.now(),
                response_time_ms=0.0,
                error_count=0,
                success_rate=1.0,
            )

            # Start health monitoring
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

            # Start auto-recovery if enabled
            if self.failover_config.auto_recovery_enabled:
                self._auto_recovery_task = asyncio.create_task(self._auto_recovery_loop())

            self.logger.info("Cache failover service initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize failover service: {e}")
            raise

    async def shutdown(self):
        """Shutdown the failover service."""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        if self._auto_recovery_task:
            self._auto_recovery_task.cancel()
            try:
                await self._auto_recovery_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Cache failover service shutdown")

    def register_failover_service(self, service: BaseCacheService):
        """Register a failover cache service."""
        self._failover_services.append(service)

        service_id = self._get_service_id(service)
        self._service_health[service_id] = ServiceHealthStatus(
            service_id=service_id,
            health=ServiceHealth.UNKNOWN,
            last_check=datetime.now(),
            response_time_ms=0.0,
            error_count=0,
            success_rate=1.0,
        )

        self.logger.info(f"Registered failover service: {service_id}")

    def add_failover_callback(self, callback: Callable[[FailoverEvent], None]):
        """Add a callback to be called when failover occurs."""
        self._failover_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable[[FailoverEvent], None]):
        """Add a callback to be called when recovery occurs."""
        self._recovery_callbacks.append(callback)

    @trace_cache_operation("failover_get")
    async def get(self, key: str) -> Any:
        """Get value with automatic failover support."""
        return await self._execute_with_failover("get", key)

    @trace_cache_operation("failover_set")
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value with automatic failover support."""
        return await self._execute_with_failover("set", key, value, ttl)

    @trace_cache_operation("failover_delete")
    async def delete(self, key: str) -> bool:
        """Delete value with automatic failover support."""
        return await self._execute_with_failover("delete", key)

    @trace_cache_operation("failover_exists")
    async def exists(self, key: str) -> bool:
        """Check existence with automatic failover support."""
        return await self._execute_with_failover("exists", key)

    async def _execute_with_failover(self, operation: str, *args, **kwargs) -> Any:
        """Execute cache operation with automatic failover on failure."""
        max_attempts = self.failover_config.max_failover_attempts
        last_exception = None

        for attempt in range(max_attempts):
            try:
                start_time = time.time()
                current_service = self._current_service

                if not current_service:
                    raise RuntimeError("No cache service available")

                # Execute operation
                method = getattr(current_service, operation)
                result = await asyncio.wait_for(method(*args, **kwargs), timeout=self.failover_config.health_check_timeout_seconds)

                # Track performance
                response_time = (time.time() - start_time) * 1000
                await self._update_service_health(current_service, response_time, True)

                return result

            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                last_exception = e
                self.logger.warning(f"Cache operation {operation} failed on attempt {attempt + 1}: {e}")

                # Track failure
                await self._update_service_health(self._current_service, 0, False)

                # Trigger failover if this is a critical failure
                if await self._should_trigger_failover(FailoverTrigger.CONNECTION_FAILURE):
                    await self._trigger_failover(FailoverTrigger.CONNECTION_FAILURE, {"error": str(e)})

            except Exception as e:
                last_exception = e
                self.logger.error(f"Unexpected error in cache operation {operation}: {e}")

                # Track failure but don't necessarily trigger failover for all exceptions
                await self._update_service_health(self._current_service, 0, False)

                # Only trigger failover for specific error types
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    if await self._should_trigger_failover(FailoverTrigger.CONNECTION_FAILURE):
                        await self._trigger_failover(FailoverTrigger.CONNECTION_FAILURE, {"error": str(e)})

        # If all attempts failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Cache operation {operation} failed after {max_attempts} attempts")

    async def manual_failover(self, reason: str = "Manual failover requested") -> FailoverEvent:
        """Manually trigger failover to the next available service."""
        return await self._trigger_failover(FailoverTrigger.MANUAL_TRIGGER, {"reason": reason})

    async def manual_recovery(self) -> bool:
        """Manually attempt recovery to primary service."""
        if self._failover_status == FailoverStatus.ACTIVE:
            self.logger.info("Already running on primary service, no recovery needed")
            return True

        return await self._attempt_recovery()

    async def get_failover_status(self) -> dict[str, Any]:
        """Get current failover status and health information."""
        current_service_id = self._get_service_id(self._current_service) if self._current_service else None
        primary_service_id = self._get_service_id(self._primary_service) if self._primary_service else None

        return {
            "status": self._failover_status.value,
            "current_service": current_service_id,
            "primary_service": primary_service_id,
            "is_failed_over": self._failover_status != FailoverStatus.ACTIVE,
            "service_health": {
                service_id: {
                    "health": status.health.value,
                    "last_check": status.last_check.isoformat(),
                    "response_time_ms": status.response_time_ms,
                    "error_count": status.error_count,
                    "success_rate": status.success_rate,
                    "metadata": status.metadata,
                }
                for service_id, status in self._service_health.items()
            },
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "trigger": event.trigger.value,
                    "timestamp": event.timestamp.isoformat(),
                    "success": event.success,
                    "duration_seconds": event.duration_seconds,
                    "recovery_timestamp": event.recovery_timestamp.isoformat() if event.recovery_timestamp else None,
                }
                for event in self._failover_events[-10:]  # Last 10 events
            ],
        }

    async def _health_monitor_loop(self):
        """Background loop for monitoring service health."""
        while True:
            try:
                await asyncio.sleep(self.failover_config.health_check_interval_seconds)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")

    async def _auto_recovery_loop(self):
        """Background loop for automatic recovery attempts."""
        while True:
            try:
                await asyncio.sleep(self.failover_config.auto_recovery_delay_seconds)

                if self._failover_status in [FailoverStatus.FAILED_OVER]:
                    await self._attempt_recovery()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto recovery loop: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all registered services."""
        for service_id, health_status in self._service_health.items():
            try:
                service = self._get_service_by_id(service_id)
                if service:
                    await self._check_service_health(service)
            except Exception as e:
                self.logger.error(f"Error checking health for service {service_id}: {e}")

    async def _check_service_health(self, service: BaseCacheService):
        """Check health of a specific service."""
        service_id = self._get_service_id(service)

        try:
            start_time = time.time()

            # Perform a simple health check operation
            test_key = f"_health_check_{int(time.time())}"
            await asyncio.wait_for(service.set(test_key, "health_check"), timeout=self.failover_config.health_check_timeout_seconds)

            # Clean up test key
            await service.delete(test_key)

            response_time = (time.time() - start_time) * 1000
            await self._update_service_health(service, response_time, True)

        except Exception as e:
            self.logger.warning(f"Health check failed for service {service_id}: {e}")
            await self._update_service_health(service, 0, False)

            # Check if we should trigger failover
            if service == self._current_service:
                if await self._should_trigger_failover(FailoverTrigger.HEALTH_CHECK_FAILURE):
                    await self._trigger_failover(FailoverTrigger.HEALTH_CHECK_FAILURE, {"health_check_error": str(e)})

    async def _update_service_health(self, service: BaseCacheService, response_time: float, success: bool):
        """Update health status for a service."""
        service_id = self._get_service_id(service)

        if service_id not in self._service_health:
            return

        health_status = self._service_health[service_id]
        health_status.last_check = datetime.now()
        health_status.response_time_ms = response_time

        if success:
            self._consecutive_failures[service_id] = 0
            self._consecutive_successes[service_id] = self._consecutive_successes.get(service_id, 0) + 1

            # Update baseline if we have enough successful operations
            if self._consecutive_successes[service_id] <= self._baseline_calculation_window:
                current_baseline = self._performance_baselines.get(service_id, response_time)
                self._performance_baselines[service_id] = (current_baseline + response_time) / 2
        else:
            health_status.error_count += 1
            self._consecutive_failures[service_id] = self._consecutive_failures.get(service_id, 0) + 1
            self._consecutive_successes[service_id] = 0

        # Calculate success rate (over last 100 operations)
        total_ops = health_status.error_count + self._consecutive_successes.get(service_id, 0)
        if total_ops > 0:
            health_status.success_rate = self._consecutive_successes.get(service_id, 0) / min(total_ops, 100)

        # Determine health status
        consecutive_failures = self._consecutive_failures.get(service_id, 0)
        if consecutive_failures >= self.failover_config.failure_threshold:
            health_status.health = ServiceHealth.UNHEALTHY
        elif consecutive_failures > 0 or health_status.success_rate < 0.9:
            health_status.health = ServiceHealth.DEGRADED
        else:
            health_status.health = ServiceHealth.HEALTHY

        # Check for performance degradation
        baseline = self._performance_baselines.get(service_id)
        if baseline and response_time > 0:
            degradation_ratio = response_time / baseline
            if degradation_ratio > (1 + self.failover_config.performance_degradation_threshold):
                health_status.metadata["performance_degraded"] = True

                if service == self._current_service:
                    if await self._should_trigger_failover(FailoverTrigger.PERFORMANCE_DEGRADATION):
                        await self._trigger_failover(
                            FailoverTrigger.PERFORMANCE_DEGRADATION,
                            {"baseline_ms": baseline, "current_ms": response_time, "degradation_ratio": degradation_ratio},
                        )

    async def _should_trigger_failover(self, trigger: FailoverTrigger) -> bool:
        """Determine if failover should be triggered."""
        if self._failover_status != FailoverStatus.ACTIVE:
            return False  # Already failed over

        if not self._failover_services:
            return False  # No failover services available

        current_service_id = self._get_service_id(self._current_service)
        consecutive_failures = self._consecutive_failures.get(current_service_id, 0)

        return consecutive_failures >= self.failover_config.failure_threshold

    async def _trigger_failover(self, trigger: FailoverTrigger, details: dict[str, Any]) -> FailoverEvent:
        """Trigger failover to next available service."""
        start_time = time.time()
        event_id = f"failover_{int(start_time)}"

        primary_service_id = self._get_service_id(self._primary_service)
        current_service_id = self._get_service_id(self._current_service)

        # Find best failover service
        best_service = await self._select_best_failover_service()

        if not best_service:
            error_msg = "No healthy failover services available"
            event = FailoverEvent(
                event_id=event_id,
                trigger=trigger,
                timestamp=datetime.now(),
                primary_service_id=primary_service_id,
                failover_service_id="none",
                trigger_details=details,
                success=False,
                error_message=error_msg,
            )
            self._failover_events.append(event)
            self.logger.error(error_msg)
            return event

        try:
            self._failover_status = FailoverStatus.FAILING_OVER

            # Switch to failover service
            old_service = self._current_service
            self._current_service = best_service

            # Test the new service
            test_key = f"_failover_test_{int(time.time())}"
            await self._current_service.set(test_key, "failover_test")
            await self._current_service.delete(test_key)

            self._failover_status = FailoverStatus.FAILED_OVER

            failover_service_id = self._get_service_id(best_service)
            event = FailoverEvent(
                event_id=event_id,
                trigger=trigger,
                timestamp=datetime.now(),
                primary_service_id=primary_service_id,
                failover_service_id=failover_service_id,
                trigger_details=details,
                duration_seconds=time.time() - start_time,
                success=True,
            )

            self._failover_events.append(event)

            self.logger.warning(f"Failover triggered: {trigger.value}, " f"switched from {current_service_id} to {failover_service_id}")

            # Notify callbacks
            for callback in self._failover_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in failover callback: {e}")

            return event

        except Exception as e:
            self._failover_status = FailoverStatus.ACTIVE
            self._current_service = old_service

            error_msg = f"Failover failed: {e}"
            event = FailoverEvent(
                event_id=event_id,
                trigger=trigger,
                timestamp=datetime.now(),
                primary_service_id=primary_service_id,
                failover_service_id=self._get_service_id(best_service),
                trigger_details=details,
                duration_seconds=time.time() - start_time,
                success=False,
                error_message=error_msg,
            )

            self._failover_events.append(event)
            self.logger.error(error_msg)
            return event

    async def _select_best_failover_service(self) -> BaseCacheService | None:
        """Select the best available failover service."""
        if not self._failover_services:
            return None

        # Score services based on health
        service_scores = []

        for service in self._failover_services:
            service_id = self._get_service_id(service)
            health_status = self._service_health.get(service_id)

            if not health_status:
                continue

            score = 0

            # Health score
            if health_status.health == ServiceHealth.HEALTHY:
                score += 100
            elif health_status.health == ServiceHealth.DEGRADED:
                score += 50
            elif health_status.health == ServiceHealth.UNHEALTHY:
                score += 0
            else:
                score += 25  # UNKNOWN

            # Success rate score
            score += health_status.success_rate * 50

            # Response time score (lower is better)
            if health_status.response_time_ms > 0:
                score += max(0, 50 - (health_status.response_time_ms / 100))

            service_scores.append((service, score))

        if not service_scores:
            return None

        # Sort by score (highest first)
        service_scores.sort(key=lambda x: x[1], reverse=True)

        # Return the best service if it has a reasonable score
        best_service, best_score = service_scores[0]
        if best_score >= 50:  # Minimum acceptable score
            return best_service

        return None

    async def _attempt_recovery(self) -> bool:
        """Attempt to recover to primary service."""
        if self._failover_status == FailoverStatus.ACTIVE:
            return True

        if not self._primary_service:
            return False

        try:
            self._failover_status = FailoverStatus.RECOVERING

            # Test primary service
            test_key = f"_recovery_test_{int(time.time())}"
            await self._primary_service.set(test_key, "recovery_test")
            await self._primary_service.delete(test_key)

            # Check if primary service is consistently healthy
            primary_id = self._get_service_id(self._primary_service)
            consecutive_successes = self._consecutive_successes.get(primary_id, 0)

            if consecutive_successes >= self.failover_config.recovery_threshold:
                # Switch back to primary
                old_service = self._current_service
                self._current_service = self._primary_service
                self._failover_status = FailoverStatus.ACTIVE

                # Create recovery event
                recovery_event = FailoverEvent(
                    event_id=f"recovery_{int(time.time())}",
                    trigger=FailoverTrigger.MANUAL_TRIGGER,  # or AUTO_RECOVERY
                    timestamp=datetime.now(),
                    primary_service_id=primary_id,
                    failover_service_id=self._get_service_id(old_service),
                    trigger_details={"recovery": True},
                    success=True,
                    recovery_timestamp=datetime.now(),
                )

                self._failover_events.append(recovery_event)

                self.logger.info(f"Successfully recovered to primary service: {primary_id}")

                # Notify callbacks
                for callback in self._recovery_callbacks:
                    try:
                        callback(recovery_event)
                    except Exception as e:
                        self.logger.error(f"Error in recovery callback: {e}")

                return True
            else:
                self._failover_status = FailoverStatus.FAILED_OVER
                self.logger.info(
                    f"Primary service not yet ready for recovery, need {self.failover_config.recovery_threshold - consecutive_successes} more successful checks"
                )
                return False

        except Exception as e:
            self._failover_status = FailoverStatus.FAILED_OVER
            self.logger.warning(f"Recovery attempt failed: {e}")
            return False

    def _get_service_id(self, service: BaseCacheService) -> str:
        """Get identifier for a service."""
        if not service:
            return "none"
        return f"{service.__class__.__name__}_{id(service)}"

    def _get_service_by_id(self, service_id: str) -> BaseCacheService | None:
        """Get service by ID."""
        for service in [self._primary_service] + self._failover_services:
            if service and self._get_service_id(service) == service_id:
                return service
        return None


# Global service instance
_failover_service = None


async def get_cache_failover_service() -> CacheFailoverService:
    """Get the global cache failover service instance."""
    global _failover_service
    if _failover_service is None:
        _failover_service = CacheFailoverService()
        await _failover_service.initialize()
    return _failover_service


async def cleanup_failover_service():
    """Clean up the failover service instance."""
    global _failover_service
    if _failover_service is not None:
        await _failover_service.shutdown()
        _failover_service = None
