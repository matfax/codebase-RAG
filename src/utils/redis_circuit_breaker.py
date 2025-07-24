"""
Redis Circuit Breaker Implementation.

This module provides specialized circuit breaker functionality for Redis connections,
with Redis-specific error handling, connection pooling awareness, and health monitoring.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import redis.asyncio as redis
from redis.exceptions import (
    AuthenticationError,
    BusyLoadingError,
    NoScriptError,
    ReadOnlyError,
    ResponseError,
)
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import (
    TimeoutError as RedisTimeoutError,
)

from ..config.cache_config import CacheConfig
from .resilience_framework import CircuitBreakerState, HealthMetrics

T = TypeVar("T")


class RedisErrorType(Enum):
    """Types of Redis errors for circuit breaker logic."""

    CONNECTION = "connection"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    MEMORY = "memory"
    READONLY = "readonly"
    BUSY = "busy"
    SCRIPT = "script"
    RESPONSE = "response"
    UNKNOWN = "unknown"


@dataclass
class RedisHealthMetrics(HealthMetrics):
    """Extended health metrics for Redis connections."""

    # Redis-specific metrics
    connection_pool_exhausted: int = 0
    redis_memory_errors: int = 0
    redis_timeout_errors: int = 0
    redis_auth_errors: int = 0
    redis_readonly_errors: int = 0
    redis_busy_errors: int = 0

    # Connection pool metrics
    active_connections: int = 0
    available_connections: int = 0
    max_connections: int = 0

    # Performance metrics
    average_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float("inf")

    # Redis server metrics
    redis_memory_usage: int = 0
    redis_memory_peak: int = 0
    redis_connected_clients: int = 0
    redis_blocked_clients: int = 0

    def update_response_time(self, response_time: float) -> None:
        """Update response time metrics."""
        self.max_response_time = max(self.max_response_time, response_time)
        self.min_response_time = min(self.min_response_time, response_time)

        # Calculate running average
        if self.total_requests > 0:
            self.average_response_time = (self.average_response_time * (self.total_requests - 1) + response_time) / self.total_requests
        else:
            self.average_response_time = response_time


@dataclass
class RedisCircuitBreakerConfig:
    """Configuration for Redis circuit breaker."""

    # Basic circuit breaker settings
    failure_threshold: int = 10
    success_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3

    # Redis-specific settings
    connection_timeout: float = 5.0
    command_timeout: float = 10.0
    health_check_interval: float = 30.0
    health_check_timeout: float = 3.0

    # Error handling
    ignore_readonly_errors: bool = True
    ignore_auth_errors: bool = False
    ignore_script_errors: bool = True

    # Performance thresholds
    max_response_time_threshold: float = 5.0
    connection_pool_threshold: float = 0.8  # 80% utilization

    # Recovery settings
    recovery_check_commands: list[str] = field(default_factory=lambda: ["PING", "INFO"])
    recovery_timeout: float = 30.0


class RedisErrorClassifier:
    """Classifies Redis errors for circuit breaker decisions."""

    def __init__(self, config: RedisCircuitBreakerConfig):
        """Initialize Redis error classifier."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    def classify_error(self, error: Exception) -> RedisErrorType:
        """Classify Redis error type."""
        if isinstance(error, RedisConnectionError):
            return RedisErrorType.CONNECTION
        elif isinstance(error, RedisTimeoutError):
            return RedisErrorType.TIMEOUT
        elif isinstance(error, AuthenticationError):
            return RedisErrorType.AUTHENTICATION
        elif isinstance(error, ReadOnlyError):
            return RedisErrorType.READONLY
        elif isinstance(error, BusyLoadingError):
            return RedisErrorType.BUSY
        elif isinstance(error, NoScriptError):
            return RedisErrorType.SCRIPT
        elif isinstance(error, ResponseError):
            # Check for memory-related errors in response
            error_msg = str(error).lower()
            if "oom" in error_msg or "memory" in error_msg:
                return RedisErrorType.MEMORY
            return RedisErrorType.RESPONSE
        else:
            return RedisErrorType.UNKNOWN

    def should_trigger_circuit_breaker(self, error_type: RedisErrorType) -> bool:
        """Determine if error should trigger circuit breaker."""
        if error_type == RedisErrorType.CONNECTION:
            return True
        elif error_type == RedisErrorType.TIMEOUT:
            return True
        elif error_type == RedisErrorType.AUTHENTICATION and not self.config.ignore_auth_errors:
            return True
        elif error_type == RedisErrorType.READONLY and not self.config.ignore_readonly_errors:
            return False  # Readonly errors shouldn't trigger circuit breaker
        elif error_type == RedisErrorType.SCRIPT and self.config.ignore_script_errors:
            return False
        elif error_type == RedisErrorType.MEMORY:
            return True  # Memory errors are serious
        elif error_type == RedisErrorType.BUSY:
            return False  # Busy errors are temporary
        else:
            return True  # Default to triggering for unknown errors


class RedisCircuitBreaker:
    """
    Advanced circuit breaker specifically designed for Redis connections.

    This circuit breaker provides Redis-specific error handling, connection pool
    monitoring, performance tracking, and intelligent recovery strategies.
    """

    def __init__(self, config: RedisCircuitBreakerConfig, redis_client: redis.Redis | None = None):
        """Initialize Redis circuit breaker."""
        self.config = config
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)

        # Circuit breaker state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0

        # Error classification
        self.error_classifier = RedisErrorClassifier(config)

        # Metrics and monitoring
        self.metrics = RedisHealthMetrics()
        self._response_times: list[float] = []
        self._error_history: list[tuple] = []  # (timestamp, error_type, error_message)

        # Health monitoring
        self._health_check_task: asyncio.Task | None = None
        self._last_health_check = 0.0

        # Recovery tracking
        self._recovery_attempts = 0
        self._last_recovery_attempt = 0.0

    async def initialize(self) -> None:
        """Initialize the circuit breaker and start monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
        self.logger.info("Redis circuit breaker initialized")

    async def shutdown(self) -> None:
        """Shutdown the circuit breaker."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Redis circuit breaker shutdown")

    async def execute(self, operation: Callable[..., T], operation_name: str = "redis_operation", *args, **kwargs) -> T:
        """
        Execute Redis operation through circuit breaker.

        Args:
            operation: The Redis operation to execute
            operation_name: Name of the operation for logging
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            RedisCircuitBreakerError: When circuit breaker is open
            Exception: Original exception if operation fails
        """
        self.metrics.total_requests += 1

        # Check circuit breaker state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                await self._move_to_half_open()
            else:
                self.metrics.circuit_breaker_trips += 1
                raise RedisCircuitBreakerError(f"Redis circuit breaker is OPEN for operation: {operation_name}")

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.metrics.circuit_breaker_trips += 1
                raise RedisCircuitBreakerError(f"Redis circuit breaker HALF_OPEN limit exceeded for operation: {operation_name}")
            self.half_open_calls += 1

        # Execute operation with timing
        start_time = time.time()
        try:
            # Set timeout for the operation
            result = await asyncio.wait_for(operation(*args, **kwargs), timeout=self.config.command_timeout)

            # Record success
            response_time = time.time() - start_time
            await self._on_success(response_time)

            return result

        except Exception as e:
            # Record failure
            response_time = time.time() - start_time
            await self._on_failure(e, operation_name, response_time)
            raise e

    async def _on_success(self, response_time: float) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()
        self.metrics.update_response_time(response_time)

        # Track response times for performance monitoring
        self._response_times.append(response_time)
        if len(self._response_times) > 100:  # Keep last 100 response times
            self._response_times.pop(0)

        # Check for performance degradation
        if response_time > self.config.max_response_time_threshold:
            self.logger.warning(f"Redis operation slow: {response_time:.2f}s (threshold: {self.config.max_response_time_threshold}s)")

        # Handle state transitions
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                await self._move_to_closed()

    async def _on_failure(self, error: Exception, operation_name: str, response_time: float) -> None:
        """Handle failed operation."""
        error_type = self.error_classifier.classify_error(error)

        # Record error in history
        self._error_history.append((time.time(), error_type, str(error)))
        if len(self._error_history) > 1000:  # Keep last 1000 errors
            self._error_history.pop(0)

        # Update type-specific metrics
        self._update_error_metrics(error_type)

        # General failure metrics
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = time.time()
        self.last_failure_time = time.time()

        # Check if this error should trigger circuit breaker
        if self.error_classifier.should_trigger_circuit_breaker(error_type):
            self.failure_count += 1
            self.success_count = 0

            self.logger.warning(
                f"Redis operation failed: {operation_name} - {error} " f"(type: {error_type.value}, failures: {self.failure_count})"
            )

            # Check state transitions
            if self.state == CircuitBreakerState.HALF_OPEN:
                await self._move_to_open()
            elif self.failure_count >= self.config.failure_threshold:
                await self._move_to_open()
        else:
            self.logger.debug(f"Redis error ignored by circuit breaker: {operation_name} - {error} " f"(type: {error_type.value})")

    def _update_error_metrics(self, error_type: RedisErrorType) -> None:
        """Update error-specific metrics."""
        if error_type == RedisErrorType.TIMEOUT:
            self.metrics.redis_timeout_errors += 1
        elif error_type == RedisErrorType.MEMORY:
            self.metrics.redis_memory_errors += 1
        elif error_type == RedisErrorType.AUTHENTICATION:
            self.metrics.redis_auth_errors += 1
        elif error_type == RedisErrorType.READONLY:
            self.metrics.redis_readonly_errors += 1
        elif error_type == RedisErrorType.BUSY:
            self.metrics.redis_busy_errors += 1

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return time.time() - self.last_failure_time >= self.config.timeout_seconds

    async def _move_to_half_open(self) -> None:
        """Move circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self.logger.info("Redis circuit breaker moved to HALF_OPEN state")

        # Attempt basic health check
        await self._perform_recovery_check()

    async def _move_to_closed(self) -> None:
        """Move circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0
        self.half_open_calls = 0
        self.failure_count = 0
        self._recovery_attempts = 0
        self.logger.info("Redis circuit breaker moved to CLOSED state")

    async def _move_to_open(self) -> None:
        """Move circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.half_open_calls = 0
        self.logger.error(f"Redis circuit breaker moved to OPEN state after {self.failure_count} failures")

    async def _perform_recovery_check(self) -> None:
        """Perform recovery health check."""
        if not self.redis_client:
            return

        try:
            for command in self.config.recovery_check_commands:
                if command.upper() == "PING":
                    await asyncio.wait_for(self.redis_client.ping(), timeout=self.config.health_check_timeout)
                elif command.upper() == "INFO":
                    await asyncio.wait_for(self.redis_client.info(), timeout=self.config.health_check_timeout)

            self.logger.info("Redis recovery check successful")

        except Exception as e:
            self.logger.warning(f"Redis recovery check failed: {e}")
            # Don't update circuit breaker state here, let normal operation flow handle it

    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in Redis health monitoring: {e}")

    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        if not self.redis_client:
            return

        current_time = time.time()
        self._last_health_check = current_time

        try:
            # Basic connectivity check
            start_time = time.time()
            await asyncio.wait_for(self.redis_client.ping(), timeout=self.config.health_check_timeout)
            ping_time = time.time() - start_time

            # Get Redis info
            info = await asyncio.wait_for(self.redis_client.info(), timeout=self.config.health_check_timeout)

            # Update metrics with Redis server info
            self._update_redis_metrics(info)

            # Update connection pool metrics
            await self._update_connection_pool_metrics()

            self.logger.debug(f"Redis health check successful (ping: {ping_time:.3f}s)")

        except Exception as e:
            self.logger.warning(f"Redis health check failed: {e}")
            # This will be handled by the normal operation flow

    def _update_redis_metrics(self, info: dict[str, Any]) -> None:
        """Update metrics from Redis INFO command."""
        try:
            # Memory metrics
            self.metrics.redis_memory_usage = info.get("used_memory", 0)
            self.metrics.redis_memory_peak = info.get("used_memory_peak", 0)

            # Client metrics
            self.metrics.redis_connected_clients = info.get("connected_clients", 0)
            self.metrics.redis_blocked_clients = info.get("blocked_clients", 0)

        except Exception as e:
            self.logger.debug(f"Failed to update Redis metrics: {e}")

    async def _update_connection_pool_metrics(self) -> None:
        """Update connection pool metrics."""
        try:
            if hasattr(self.redis_client, "connection_pool"):
                pool = self.redis_client.connection_pool
                self.metrics.max_connections = getattr(pool, "max_connections", 0)

                # Get current connection counts
                if hasattr(pool, "_available_connections"):
                    self.metrics.available_connections = len(pool._available_connections)
                if hasattr(pool, "_in_use_connections"):
                    self.metrics.active_connections = len(pool._in_use_connections)

                # Check for pool exhaustion
                utilization = self.metrics.active_connections / max(1, self.metrics.max_connections)
                if utilization > self.config.connection_pool_threshold:
                    self.metrics.connection_pool_exhausted += 1
                    self.logger.warning(
                        f"Redis connection pool utilization high: {utilization:.1%} "
                        f"({self.metrics.active_connections}/{self.metrics.max_connections})"
                    )

        except Exception as e:
            self.logger.debug(f"Failed to update connection pool metrics: {e}")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> RedisHealthMetrics:
        """Get comprehensive Redis metrics."""
        return self.metrics

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of recent errors."""
        current_time = time.time()
        recent_errors = [
            (timestamp, error_type, message)
            for timestamp, error_type, message in self._error_history
            if current_time - timestamp < 3600  # Last hour
        ]

        error_counts = {}
        for _, error_type, _ in recent_errors:
            error_counts[error_type.value] = error_counts.get(error_type.value, 0) + 1

        return {
            "total_recent_errors": len(recent_errors),
            "error_type_distribution": error_counts,
            "recent_errors": recent_errors[-10:] if recent_errors else [],  # Last 10 errors
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        if not self._response_times:
            return {"no_data": True}

        return {
            "average_response_time": self.metrics.average_response_time,
            "max_response_time": self.metrics.max_response_time,
            "min_response_time": self.metrics.min_response_time,
            "recent_response_times": self._response_times[-10:],
            "slow_operations": sum(1 for rt in self._response_times if rt > self.config.max_response_time_threshold),
        }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self._recovery_attempts = 0
        self.metrics.reset()
        self._response_times.clear()
        self._error_history.clear()
        self.logger.info("Redis circuit breaker manually reset")

    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.logger.warning("Redis circuit breaker manually forced to OPEN state")

    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.logger.info("Redis circuit breaker manually forced to CLOSED state")


class RedisCircuitBreakerError(Exception):
    """Exception raised when Redis circuit breaker prevents operation."""

    pass


# Decorator for Redis operations
def redis_circuit_breaker(config: RedisCircuitBreakerConfig | None = None, operation_name: str = None):
    """
    Decorator to apply Redis circuit breaker to operations.

    Args:
        config: Circuit breaker configuration
        operation_name: Name of the operation for logging
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        circuit_breaker = None

        async def wrapper(*args, **kwargs) -> T:
            nonlocal circuit_breaker

            # Initialize circuit breaker if needed
            if circuit_breaker is None:
                cb_config = config or RedisCircuitBreakerConfig()
                redis_client = kwargs.get("redis_client") or (args[0] if args else None)
                circuit_breaker = RedisCircuitBreaker(cb_config, redis_client)
                await circuit_breaker.initialize()

            op_name = operation_name or func.__name__
            return await circuit_breaker.execute(func, op_name, *args, **kwargs)

        return wrapper

    return decorator


# Factory function
async def create_redis_circuit_breaker(redis_client: redis.Redis, config: RedisCircuitBreakerConfig | None = None) -> RedisCircuitBreaker:
    """Create and initialize a Redis circuit breaker."""
    config = config or RedisCircuitBreakerConfig()
    circuit_breaker = RedisCircuitBreaker(config, redis_client)
    await circuit_breaker.initialize()
    return circuit_breaker
