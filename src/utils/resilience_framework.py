"""
Resilience Framework for Cache System.

This module provides comprehensive error handling, resilience patterns, and
graceful degradation capabilities for the cache system.
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeVar

from .cache_retry_service import CacheRetryService
from .cache_retry_service import RetryConfig as CacheRetryConfig

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, not allowing requests
    HALF_OPEN = "half_open"  # Testing if service is back


class FallbackStrategy(Enum):
    """Fallback strategies for cache failures."""

    NONE = "none"
    CACHED_DATA = "cached_data"
    DEFAULT_VALUE = "default_value"
    COMPUTE_FRESH = "compute_fresh"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class DegradationLevel(Enum):
    """Levels of system degradation."""

    NORMAL = "normal"
    MINOR = "minor"  # Some features disabled
    MODERATE = "moderate"  # Significant features disabled
    SEVERE = "severe"  # Critical features only
    EMERGENCY = "emergency"  # Minimal operation mode


@dataclass
class ResilienceConfig:
    """Configuration for resilience features."""

    # Circuit Breaker Settings
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 5

    # Retry Settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: tuple = (0.1, 0.3)

    # Degradation Settings
    degradation_enabled: bool = True
    auto_recovery_enabled: bool = True
    health_check_interval: float = 30.0

    # Fallback Settings
    default_fallback_strategy: FallbackStrategy = FallbackStrategy.GRACEFUL_DEGRADATION
    fallback_cache_ttl: int = 300

    # Self-Healing Settings
    self_healing_enabled: bool = True
    healing_check_interval: float = 60.0
    consecutive_success_threshold: int = 5


@dataclass
class HealthMetrics:
    """Health metrics for monitoring system state."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_trips: int = 0
    fallback_activations: int = 0
    degradation_activations: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.circuit_breaker_trips = 0
        self.fallback_activations = 0
        self.degradation_activations = 0
        self.last_failure_time = None
        self.last_success_time = None


class CircuitBreaker:
    """
    Circuit breaker implementation for preventing cascading failures.

    The circuit breaker monitors failures and prevents calls to failing services
    when a threshold is exceeded, allowing the service time to recover.
    """

    def __init__(self, config: ResilienceConfig):
        """Initialize circuit breaker."""
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.logger = logging.getLogger(__name__)
        self.metrics = HealthMetrics()

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        self.metrics.total_requests += 1

        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._move_to_half_open()
            else:
                self.metrics.circuit_breaker_trips += 1
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.metrics.circuit_breaker_trips += 1
                raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN limit exceeded")
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return time.time() - self.last_failure_time >= self.config.timeout_seconds

    def _move_to_half_open(self) -> None:
        """Move circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        self.logger.info("Circuit breaker moved to HALF_OPEN state")

    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._move_to_closed()

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.success_count = 0
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = time.time()
        self.last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self._move_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._move_to_open()

    def _move_to_closed(self) -> None:
        """Move circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0
        self.half_open_calls = 0
        self.logger.info("Circuit breaker moved to CLOSED state")

    def _move_to_open(self) -> None:
        """Move circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.half_open_calls = 0
        self.logger.warning("Circuit breaker moved to OPEN state")

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_metrics(self) -> HealthMetrics:
        """Get circuit breaker metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.metrics.reset()
        self.logger.info("Circuit breaker manually reset to CLOSED state")


class RetryHandler:
    """
    Enhanced retry handler with advanced exponential backoff and intelligent error handling.

    Integrates with the advanced CacheRetryService for sophisticated retry logic
    including error classification, adaptive strategies, and comprehensive metrics.
    """

    def __init__(self, config: ResilienceConfig):
        """Initialize retry handler."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create cache retry configuration
        cache_retry_config = CacheRetryConfig(
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            backoff_multiplier=config.backoff_multiplier,
            jitter_range=config.jitter_range,
            operation_timeout=30.0,
            total_timeout=300.0,
            adaptive_scaling=True,
        )

        # Initialize advanced retry service
        self.cache_retry_service = CacheRetryService(cache_retry_config)

    async def execute_with_retry(
        self, func: Callable[..., T], *args, retryable_exceptions: tuple = (Exception,), operation_id: str = None, **kwargs
    ) -> T:
        """Execute function with advanced retry logic."""
        operation_id = operation_id or f"{func.__name__}_{id(func)}"

        try:
            return await self.cache_retry_service.execute_with_retry(func, operation_id, *args, **kwargs)
        except Exception as e:
            # Check if exception type is in retryable_exceptions
            if not any(isinstance(e, exc_type) for exc_type in retryable_exceptions):
                self.logger.error(f"Non-retryable exception: {e}")
                raise e

            # If it's retryable but still failed, re-raise
            raise e

    def get_metrics(self) -> dict[str, Any]:
        """Get retry metrics."""
        metrics = self.cache_retry_service.get_metrics()
        return {
            "total_attempts": metrics.total_attempts,
            "successful_retries": metrics.successful_retries,
            "failed_retries": metrics.failed_retries,
            "retry_success_rate": metrics.retry_success_rate,
            "average_delay": metrics.average_delay,
            "max_delay_encountered": metrics.max_delay_encountered,
            "error_distribution": metrics.error_distribution,
            "active_retries": self.cache_retry_service.get_active_retries(),
        }

    def reset_metrics(self) -> None:
        """Reset retry metrics."""
        self.cache_retry_service.reset_metrics()

    def _calculate_delay(self, attempt: int) -> float:
        """Legacy method for backward compatibility."""
        base_delay = min(self.config.base_delay * (self.config.backoff_multiplier**attempt), self.config.max_delay)

        # Add jitter to prevent thundering herd
        jitter_min, jitter_max = self.config.jitter_range
        jitter = random.uniform(jitter_min, jitter_max) * base_delay

        return base_delay + jitter


class FallbackManager:
    """
    Fallback manager for handling cache failures gracefully.

    Provides various fallback strategies when cache operations fail,
    including cached data, default values, and computed fallbacks.
    """

    def __init__(self, config: ResilienceConfig):
        """Initialize fallback manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._fallback_cache: dict[str, Any] = {}
        self._fallback_cache_timestamps: dict[str, float] = {}

    async def execute_with_fallback(
        self,
        primary_func: Callable[..., T],
        fallback_strategy: FallbackStrategy = None,
        fallback_value: Any = None,
        cache_key: str = None,
        *args,
        **kwargs,
    ) -> T:
        """Execute function with fallback strategy."""
        strategy = fallback_strategy or self.config.default_fallback_strategy

        try:
            result = await primary_func(*args, **kwargs) if asyncio.iscoroutinefunction(primary_func) else primary_func(*args, **kwargs)

            # Cache successful result for future fallback
            if cache_key and result is not None:
                self._store_fallback_data(cache_key, result)

            return result
        except Exception as e:
            self.logger.warning(f"Primary function failed: {e}. Using fallback strategy: {strategy}")
            return await self._apply_fallback_strategy(strategy, fallback_value, cache_key, e)

    async def _apply_fallback_strategy(self, strategy: FallbackStrategy, fallback_value: Any, cache_key: str, exception: Exception) -> Any:
        """Apply the specified fallback strategy."""
        if strategy == FallbackStrategy.NONE:
            raise exception
        elif strategy == FallbackStrategy.CACHED_DATA:
            return self._get_cached_fallback_data(cache_key)
        elif strategy == FallbackStrategy.DEFAULT_VALUE:
            return fallback_value
        elif strategy == FallbackStrategy.GRACEFUL_DEGRADATION:
            return self._handle_graceful_degradation(cache_key, exception)
        else:
            self.logger.error(f"Unknown fallback strategy: {strategy}")
            raise exception

    def _store_fallback_data(self, key: str, data: Any) -> None:
        """Store data for fallback use."""
        self._fallback_cache[key] = data
        self._fallback_cache_timestamps[key] = time.time()

        # Cleanup old entries
        self._cleanup_fallback_cache()

    def _get_cached_fallback_data(self, key: str) -> Any:
        """Get cached fallback data."""
        if key not in self._fallback_cache:
            return None

        # Check if data is still valid
        timestamp = self._fallback_cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.config.fallback_cache_ttl:
            # Data is too old, remove it
            self._fallback_cache.pop(key, None)
            self._fallback_cache_timestamps.pop(key, None)
            return None

        return self._fallback_cache[key]

    def _handle_graceful_degradation(self, key: str, exception: Exception) -> Any:
        """Handle graceful degradation scenario."""
        # Try cached data first
        cached_data = self._get_cached_fallback_data(key)
        if cached_data is not None:
            self.logger.info(f"Using cached fallback data for key: {key}")
            return cached_data

        # If no cached data, return a degraded response
        self.logger.warning(f"No cached data available for key: {key}. Returning degraded response.")
        return self._create_degraded_response(key, exception)

    def _create_degraded_response(self, key: str, exception: Exception) -> Any:
        """Create a degraded response when cache is unavailable."""
        # This can be customized based on the specific use case
        return {
            "error": "cache_unavailable",
            "message": "Cache service is temporarily unavailable",
            "key": key,
            "degraded": True,
            "timestamp": time.time(),
        }

    def _cleanup_fallback_cache(self) -> None:
        """Clean up expired fallback cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._fallback_cache_timestamps.items() if current_time - timestamp > self.config.fallback_cache_ttl
        ]

        for key in expired_keys:
            self._fallback_cache.pop(key, None)
            self._fallback_cache_timestamps.pop(key, None)


class DegradationManager:
    """
    System degradation manager for handling performance issues.

    Monitors system health and applies appropriate degradation levels
    to maintain system stability during high load or failures.
    """

    def __init__(self, config: ResilienceConfig):
        """Initialize degradation manager."""
        self.config = config
        self.current_level = DegradationLevel.NORMAL
        self.degraded_features: set[str] = set()
        self.logger = logging.getLogger(__name__)
        self._last_health_check = 0.0
        self._consecutive_failures = 0
        self._consecutive_successes = 0

    def update_health_status(self, success: bool, error_rate: float = 0.0) -> None:
        """Update health status and adjust degradation level."""
        current_time = time.time()

        if success:
            self._consecutive_successes += 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

        # Check if we need to change degradation level
        new_level = self._calculate_degradation_level(error_rate)
        if new_level != self.current_level:
            self._apply_degradation_level(new_level)

    def _calculate_degradation_level(self, error_rate: float) -> DegradationLevel:
        """Calculate appropriate degradation level based on metrics."""
        if error_rate >= 0.8 or self._consecutive_failures >= 10:
            return DegradationLevel.EMERGENCY
        elif error_rate >= 0.5 or self._consecutive_failures >= 7:
            return DegradationLevel.SEVERE
        elif error_rate >= 0.3 or self._consecutive_failures >= 5:
            return DegradationLevel.MODERATE
        elif error_rate >= 0.1 or self._consecutive_failures >= 3:
            return DegradationLevel.MINOR
        elif self._consecutive_successes >= self.config.consecutive_success_threshold:
            return DegradationLevel.NORMAL
        else:
            return self.current_level

    def _apply_degradation_level(self, level: DegradationLevel) -> None:
        """Apply the specified degradation level."""
        old_level = self.current_level
        self.current_level = level

        self.logger.warning(f"Degradation level changed from {old_level.value} to {level.value}")

        # Configure features based on degradation level
        if level == DegradationLevel.EMERGENCY:
            self.degraded_features.update(
                ["batch_operations", "analytics", "compression", "encryption", "background_tasks", "non_essential_logging"]
            )
        elif level == DegradationLevel.SEVERE:
            self.degraded_features.update(["batch_operations", "analytics", "compression", "background_tasks"])
        elif level == DegradationLevel.MODERATE:
            self.degraded_features.update(["batch_operations", "analytics"])
        elif level == DegradationLevel.MINOR:
            self.degraded_features.update(["analytics"])
        else:  # NORMAL
            self.degraded_features.clear()

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at current degradation level."""
        return feature not in self.degraded_features

    def get_current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self.current_level

    def get_disabled_features(self) -> set[str]:
        """Get set of currently disabled features."""
        return self.degraded_features.copy()


class SelfHealingManager:
    """
    Self-healing manager for automatic error recovery.

    Monitors system health and automatically attempts to recover
    from failures through various healing strategies.
    """

    def __init__(self, config: ResilienceConfig):
        """Initialize self-healing manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._healing_task: asyncio.Task | None = None
        self._healing_strategies: list[Callable] = []
        self._last_healing_attempt = 0.0
        self._healing_attempts = 0

    async def start_healing_process(self) -> None:
        """Start the self-healing process."""
        if not self.config.self_healing_enabled:
            return

        if self._healing_task and not self._healing_task.done():
            self.logger.warning("Self-healing process already running")
            return

        self._healing_task = asyncio.create_task(self._healing_loop())
        self.logger.info("Self-healing process started")

    async def stop_healing_process(self) -> None:
        """Stop the self-healing process."""
        if self._healing_task:
            self._healing_task.cancel()
            try:
                await self._healing_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Self-healing process stopped")

    def register_healing_strategy(self, strategy: Callable) -> None:
        """Register a healing strategy."""
        self._healing_strategies.append(strategy)
        self.logger.debug(f"Registered healing strategy: {strategy.__name__}")

    async def _healing_loop(self) -> None:
        """Main healing loop."""
        while True:
            try:
                await asyncio.sleep(self.config.healing_check_interval)
                await self._attempt_healing()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in healing loop: {e}")

    async def _attempt_healing(self) -> None:
        """Attempt to heal system issues."""
        current_time = time.time()

        # Rate limit healing attempts
        if current_time - self._last_healing_attempt < self.config.healing_check_interval:
            return

        self._last_healing_attempt = current_time
        self._healing_attempts += 1

        for strategy in self._healing_strategies:
            try:
                self.logger.info(f"Attempting healing strategy: {strategy.__name__}")

                if asyncio.iscoroutinefunction(strategy):
                    await strategy()
                else:
                    strategy()

                self.logger.info(f"Healing strategy {strategy.__name__} completed successfully")
            except Exception as e:
                self.logger.error(f"Healing strategy {strategy.__name__} failed: {e}")


class ResilienceFramework:
    """
    Main resilience framework that coordinates all resilience components.

    This class provides a unified interface for all resilience features
    including circuit breakers, retries, fallbacks, and self-healing.
    """

    def __init__(self, config: ResilienceConfig | None = None):
        """Initialize resilience framework."""
        self.config = config or ResilienceConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.circuit_breaker = CircuitBreaker(self.config)
        self.retry_handler = RetryHandler(self.config)
        self.fallback_manager = FallbackManager(self.config)
        self.degradation_manager = DegradationManager(self.config)
        self.self_healing_manager = SelfHealingManager(self.config)

        # Framework state
        self._initialized = False
        self._monitoring_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the resilience framework."""
        if self._initialized:
            return

        # Start self-healing process
        await self.self_healing_manager.start_healing_process()

        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        self._initialized = True
        self.logger.info("Resilience framework initialized")

    async def shutdown(self) -> None:
        """Shutdown the resilience framework."""
        if not self._initialized:
            return

        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop self-healing
        await self.self_healing_manager.stop_healing_process()

        self._initialized = False
        self.logger.info("Resilience framework shutdown")

    async def execute_resilient_operation(
        self,
        operation: Callable[..., T],
        fallback_strategy: FallbackStrategy = None,
        fallback_value: Any = None,
        cache_key: str = None,
        retryable_exceptions: tuple = (Exception,),
        use_circuit_breaker: bool = True,
        *args,
        **kwargs,
    ) -> T:
        """
        Execute an operation with full resilience features.

        This method combines circuit breaker, retry, and fallback logic
        to provide comprehensive error handling and resilience.
        """

        async def resilient_operation():
            if use_circuit_breaker:
                return await self.circuit_breaker.call(
                    self.retry_handler.execute_with_retry, operation, *args, retryable_exceptions=retryable_exceptions, **kwargs
                )
            else:
                return await self.retry_handler.execute_with_retry(operation, *args, retryable_exceptions=retryable_exceptions, **kwargs)

        return await self.fallback_manager.execute_with_fallback(
            resilient_operation, fallback_strategy=fallback_strategy, fallback_value=fallback_value, cache_key=cache_key
        )

    async def _monitoring_loop(self) -> None:
        """Monitor system health and update degradation status."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Get metrics from circuit breaker
                metrics = self.circuit_breaker.get_metrics()

                # Update degradation manager
                success = metrics.total_requests > 0 and metrics.success_rate > 0.5
                self.degradation_manager.update_health_status(success, metrics.failure_rate)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "circuit_breaker_state": self.circuit_breaker.get_state().value,
            "circuit_breaker_metrics": self.circuit_breaker.get_metrics().__dict__,
            "degradation_level": self.degradation_manager.get_current_level().value,
            "disabled_features": list(self.degradation_manager.get_disabled_features()),
            "self_healing_active": (
                self.self_healing_manager._healing_task is not None and not self.self_healing_manager._healing_task.done()
                if self.self_healing_manager._healing_task
                else False
            ),
            "framework_initialized": self._initialized,
        }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class ResilienceError(Exception):
    """Base exception for resilience framework errors."""

    pass


class FallbackError(Exception):
    """Exception raised when fallback strategy fails."""

    pass


# Decorators for easy integration
def resilient(
    config: ResilienceConfig | None = None,
    fallback_strategy: FallbackStrategy = FallbackStrategy.GRACEFUL_DEGRADATION,
    fallback_value: Any = None,
    cache_key: str = None,
    retryable_exceptions: tuple = (Exception,),
    use_circuit_breaker: bool = True,
):
    """
    Decorator to make any function resilient.

    This decorator wraps a function with resilience features including
    circuit breaker, retry logic, and fallback strategies.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        framework = ResilienceFramework(config)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not framework._initialized:
                await framework.initialize()

            return await framework.execute_resilient_operation(
                func,
                fallback_strategy=fallback_strategy,
                fallback_value=fallback_value,
                cache_key=cache_key,
                retryable_exceptions=retryable_exceptions,
                use_circuit_breaker=use_circuit_breaker,
                *args,
                **kwargs,
            )

        return wrapper

    return decorator


# Global resilience framework instance
_global_framework: ResilienceFramework | None = None


async def get_resilience_framework() -> ResilienceFramework:
    """Get or create global resilience framework instance."""
    global _global_framework
    if _global_framework is None:
        _global_framework = ResilienceFramework()
        await _global_framework.initialize()
    return _global_framework


async def shutdown_resilience_framework() -> None:
    """Shutdown global resilience framework."""
    global _global_framework
    if _global_framework:
        await _global_framework.shutdown()
        _global_framework = None
