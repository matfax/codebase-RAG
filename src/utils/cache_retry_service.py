"""
Cache Retry Service with Advanced Exponential Backoff.

This module provides sophisticated retry logic specifically designed for cache operations,
with intelligent error classification, adaptive backoff strategies, and comprehensive metrics.
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from redis.exceptions import (
    BusyLoadingError,
    ReadOnlyError,
    ResponseError,
)
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
)
from redis.exceptions import (
    TimeoutError as RedisTimeoutError,
)

from ..services.cache_service import CacheConnectionError, CacheError, CacheOperationError

T = TypeVar("T")


class RetryStrategy(Enum):
    """Retry strategy types."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    ADAPTIVE = "adaptive"


class ErrorSeverity(Enum):
    """Error severity levels for retry decision making."""

    TRANSIENT = "transient"  # Temporary errors, safe to retry
    INTERMITTENT = "intermittent"  # Periodic errors, retry with caution
    PERSISTENT = "persistent"  # Likely permanent errors, limited retries
    FATAL = "fatal"  # Permanent errors, don't retry


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time: float = 0.0
    max_delay_encountered: float = 0.0
    error_distribution: dict[str, int] = field(default_factory=dict)
    last_reset: float = field(default_factory=time.time)

    @property
    def retry_success_rate(self) -> float:
        """Calculate retry success rate."""
        total_retries = self.successful_retries + self.failed_retries
        return self.successful_retries / total_retries if total_retries > 0 else 0.0

    @property
    def average_delay(self) -> float:
        """Calculate average delay per retry."""
        return self.total_delay_time / max(1, self.total_attempts)

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.total_delay_time = 0.0
        self.max_delay_encountered = 0.0
        self.error_distribution.clear()
        self.last_reset = time.time()


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Basic retry settings
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0

    # Backoff settings
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter_range: tuple[float, float] = (0.1, 0.3)

    # Adaptive settings
    adaptive_scaling: bool = True
    success_rate_threshold: float = 0.7
    error_rate_window: int = 100

    # Timeout settings
    operation_timeout: float = 30.0
    total_timeout: float = 300.0

    # Error classification
    retryable_exceptions: set[type[Exception]] = field(
        default_factory=lambda: {
            RedisConnectionError,
            RedisTimeoutError,
            BusyLoadingError,
            CacheConnectionError,
            ConnectionError,
            TimeoutError,
        }
    )

    intermittent_exceptions: set[type[Exception]] = field(
        default_factory=lambda: {
            ResponseError,
            CacheOperationError,
        }
    )

    fatal_exceptions: set[type[Exception]] = field(
        default_factory=lambda: {
            ReadOnlyError,
            PermissionError,
            ValueError,
            TypeError,
        }
    )


class ErrorClassifier:
    """Classifies errors to determine retry behavior."""

    def __init__(self, config: RetryConfig):
        """Initialize error classifier."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._error_history: list[tuple[str, float]] = []
        self._error_patterns: dict[str, int] = {}

    def classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity for retry decisions."""
        error_type = type(error)
        error_message = str(error).lower()

        # Record error for pattern analysis
        self._record_error(error_type.__name__, error_message)

        # Check for fatal errors first
        if error_type in self.config.fatal_exceptions:
            return ErrorSeverity.FATAL

        # Check for clearly retryable errors
        if error_type in self.config.retryable_exceptions:
            return ErrorSeverity.TRANSIENT

        # Check for intermittent errors
        if error_type in self.config.intermittent_exceptions:
            return ErrorSeverity.INTERMITTENT

        # Analyze error message for known patterns
        if self._is_connection_related(error_message):
            return ErrorSeverity.TRANSIENT
        elif self._is_timeout_related(error_message):
            return ErrorSeverity.TRANSIENT
        elif self._is_resource_related(error_message):
            return ErrorSeverity.INTERMITTENT
        elif self._is_permission_related(error_message):
            return ErrorSeverity.FATAL

        # Check error frequency to determine if it's persistent
        if self._is_persistent_error(error_type.__name__):
            return ErrorSeverity.PERSISTENT

        # Default to intermittent for unknown errors
        return ErrorSeverity.INTERMITTENT

    def _record_error(self, error_type: str, error_message: str) -> None:
        """Record error for pattern analysis."""
        current_time = time.time()
        self._error_history.append((error_type, current_time))
        self._error_patterns[error_type] = self._error_patterns.get(error_type, 0) + 1

        # Clean old history (keep last hour)
        cutoff_time = current_time - 3600
        self._error_history = [(err_type, timestamp) for err_type, timestamp in self._error_history if timestamp > cutoff_time]

    def _is_connection_related(self, error_message: str) -> bool:
        """Check if error is connection-related."""
        connection_keywords = ["connection", "connect", "refused", "reset", "timeout", "network", "unreachable", "host", "socket"]
        return any(keyword in error_message for keyword in connection_keywords)

    def _is_timeout_related(self, error_message: str) -> bool:
        """Check if error is timeout-related."""
        timeout_keywords = ["timeout", "timed out", "deadline", "expired"]
        return any(keyword in error_message for keyword in timeout_keywords)

    def _is_resource_related(self, error_message: str) -> bool:
        """Check if error is resource-related."""
        resource_keywords = ["memory", "busy", "overload", "limit", "quota", "throttle", "rate", "capacity"]
        return any(keyword in error_message for keyword in resource_keywords)

    def _is_permission_related(self, error_message: str) -> bool:
        """Check if error is permission-related."""
        permission_keywords = ["permission", "denied", "forbidden", "unauthorized", "authentication", "access", "privilege"]
        return any(keyword in error_message for keyword in permission_keywords)

    def _is_persistent_error(self, error_type: str) -> bool:
        """Check if error type appears to be persistent."""
        # Consider error persistent if it occurred more than 5 times in the last hour
        recent_count = sum(1 for err_type, timestamp in self._error_history if err_type == error_type and time.time() - timestamp < 3600)
        return recent_count > 5


class BackoffCalculator:
    """Calculates backoff delays based on retry strategy."""

    def __init__(self, config: RetryConfig):
        """Initialize backoff calculator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._adaptive_multiplier = 1.0
        self._recent_success_rate = 1.0

    def calculate_delay(self, attempt: int, error_severity: ErrorSeverity, previous_delays: list[float] = None) -> float:
        """Calculate delay for next retry attempt."""
        previous_delays = previous_delays or []

        # Base delay calculation based on strategy
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            base_delay = self._calculate_exponential_delay(attempt)
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            base_delay = self._calculate_linear_delay(attempt)
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            base_delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            base_delay = self._calculate_adaptive_delay(attempt, previous_delays)
        else:
            base_delay = self._calculate_exponential_delay(attempt)

        # Apply severity modifier
        severity_modifier = self._get_severity_modifier(error_severity)
        adjusted_delay = base_delay * severity_modifier

        # Apply adaptive scaling
        if self.config.adaptive_scaling:
            adjusted_delay *= self._adaptive_multiplier

        # Add jitter
        jittered_delay = self._add_jitter(adjusted_delay)

        # Ensure within bounds
        final_delay = max(0.1, min(jittered_delay, self.config.max_delay))

        self.logger.debug(
            f"Calculated retry delay: attempt={attempt}, severity={error_severity.value}, "
            f"base={base_delay:.2f}s, adjusted={adjusted_delay:.2f}s, final={final_delay:.2f}s"
        )

        return final_delay

    def _calculate_exponential_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return min(self.config.base_delay * (self.config.backoff_multiplier**attempt), self.config.max_delay)

    def _calculate_linear_delay(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        return min(self.config.base_delay * (1 + attempt), self.config.max_delay)

    def _calculate_adaptive_delay(self, attempt: int, previous_delays: list[float]) -> float:
        """Calculate adaptive delay based on previous attempts."""
        if not previous_delays:
            return self.config.base_delay

        # Use average of previous delays as base, then apply exponential growth
        avg_previous = sum(previous_delays) / len(previous_delays)
        return min(avg_previous * self.config.backoff_multiplier, self.config.max_delay)

    def _get_severity_modifier(self, severity: ErrorSeverity) -> float:
        """Get delay modifier based on error severity."""
        if severity == ErrorSeverity.TRANSIENT:
            return 0.5  # Shorter delay for transient errors
        elif severity == ErrorSeverity.INTERMITTENT:
            return 1.0  # Normal delay
        elif severity == ErrorSeverity.PERSISTENT:
            return 2.0  # Longer delay for persistent errors
        else:  # FATAL
            return 0.0  # No retry

    def _add_jitter(self, delay: float) -> float:
        """Add jitter to prevent thundering herd."""
        jitter_min, jitter_max = self.config.jitter_range
        jitter_factor = random.uniform(jitter_min, jitter_max)
        jitter = delay * jitter_factor
        return delay + jitter

    def update_adaptive_parameters(self, success_rate: float) -> None:
        """Update adaptive parameters based on recent success rate."""
        self._recent_success_rate = success_rate

        if success_rate < self.config.success_rate_threshold:
            # Increase delays when success rate is low
            self._adaptive_multiplier = min(2.0, self._adaptive_multiplier * 1.1)
        else:
            # Decrease delays when success rate is good
            self._adaptive_multiplier = max(0.5, self._adaptive_multiplier * 0.95)


class CacheRetryService:
    """
    Advanced retry service for cache operations.

    Provides intelligent retry logic with exponential backoff, error classification,
    adaptive strategies, and comprehensive metrics.
    """

    def __init__(self, config: RetryConfig | None = None):
        """Initialize cache retry service."""
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.error_classifier = ErrorClassifier(self.config)
        self.backoff_calculator = BackoffCalculator(self.config)
        self.metrics = RetryMetrics()

        # Active retry tracking
        self._active_retries: dict[str, list[float]] = {}
        self._retry_locks: dict[str, asyncio.Lock] = {}

    async def execute_with_retry(self, operation: Callable[..., T], operation_id: str = None, *args, **kwargs) -> T:
        """
        Execute operation with intelligent retry logic.

        Args:
            operation: The operation to execute
            operation_id: Unique identifier for operation (for tracking)
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        operation_id = operation_id or f"op_{id(operation)}"
        start_time = time.time()
        last_exception = None
        delays = []

        # Initialize tracking
        if operation_id not in self._retry_locks:
            self._retry_locks[operation_id] = asyncio.Lock()

        async with self._retry_locks[operation_id]:
            self._active_retries[operation_id] = []

            try:
                for attempt in range(self.config.max_retries + 1):
                    self.metrics.total_attempts += 1

                    try:
                        # Check total timeout
                        if time.time() - start_time > self.config.total_timeout:
                            raise TimeoutError(f"Total retry timeout exceeded: {self.config.total_timeout}s")

                        # Execute operation with timeout
                        result = await self._execute_with_timeout(operation, *args, **kwargs)

                        # Success - update metrics and return
                        if attempt > 0:
                            self.metrics.successful_retries += 1
                            self.logger.info(f"Operation succeeded after {attempt} retries: {operation_id}")

                        return result

                    except Exception as e:
                        last_exception = e

                        # Classify error
                        severity = self.error_classifier.classify_error(e)

                        # Record error in metrics
                        error_type = type(e).__name__
                        self.metrics.error_distribution[error_type] = self.metrics.error_distribution.get(error_type, 0) + 1

                        # Check if we should retry
                        if not self._should_retry(attempt, severity, e):
                            break

                        # Calculate delay
                        delay = self.backoff_calculator.calculate_delay(attempt, severity, delays)
                        delays.append(delay)

                        # Update metrics
                        self.metrics.total_delay_time += delay
                        self.metrics.max_delay_encountered = max(self.metrics.max_delay_encountered, delay)

                        # Log retry attempt
                        self.logger.warning(
                            f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed for {operation_id}: "
                            f"{e} (severity: {severity.value}). Retrying in {delay:.2f}s..."
                        )

                        # Wait before retry
                        await asyncio.sleep(delay)

                # All retries failed
                self.metrics.failed_retries += 1
                self.logger.error(
                    f"All {self.config.max_retries + 1} attempts failed for {operation_id}. " f"Final error: {last_exception}"
                )

                # Update adaptive parameters
                self._update_adaptive_parameters()

                raise last_exception

            finally:
                # Cleanup tracking
                self._active_retries.pop(operation_id, None)

    async def _execute_with_timeout(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """Execute operation with timeout."""
        if asyncio.iscoroutinefunction(operation):
            return await asyncio.wait_for(operation(*args, **kwargs), timeout=self.config.operation_timeout)
        else:
            # For sync operations, run in executor with timeout
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(loop.run_in_executor(None, operation, *args, **kwargs), timeout=self.config.operation_timeout)

    def _should_retry(self, attempt: int, severity: ErrorSeverity, error: Exception) -> bool:
        """Determine if operation should be retried."""
        # Check max retries
        if attempt >= self.config.max_retries:
            return False

        # Don't retry fatal errors
        if severity == ErrorSeverity.FATAL:
            return False

        # Limit retries for persistent errors
        if severity == ErrorSeverity.PERSISTENT and attempt >= 2:
            return False

        return True

    def _update_adaptive_parameters(self) -> None:
        """Update adaptive parameters based on recent performance."""
        if self.config.adaptive_scaling:
            success_rate = self.metrics.retry_success_rate
            self.backoff_calculator.update_adaptive_parameters(success_rate)

    def get_metrics(self) -> RetryMetrics:
        """Get current retry metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset retry metrics."""
        self.metrics.reset()

    def get_active_retries(self) -> dict[str, int]:
        """Get count of active retries by operation."""
        return {op_id: len(delays) for op_id, delays in self._active_retries.items()}


class RetryDecorator:
    """Decorator for adding retry logic to functions."""

    def __init__(self, config: RetryConfig | None = None, operation_id: str = None):
        """Initialize retry decorator."""
        self.config = config or RetryConfig()
        self.operation_id = operation_id
        self.retry_service = CacheRetryService(self.config)

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Apply retry logic to function."""
        operation_id = self.operation_id or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.retry_service.execute_with_retry(func, operation_id, *args, **kwargs)

        return wrapper


# Convenience decorators with common configurations
def cache_retry(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
):
    """Simple cache retry decorator with common settings."""
    config = RetryConfig(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay, strategy=strategy)
    return RetryDecorator(config)


def redis_retry(max_retries: int = 5, max_delay: float = 60.0):
    """Redis-specific retry decorator."""
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=1.0,
        max_delay=max_delay,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions={
            RedisConnectionError,
            RedisTimeoutError,
            BusyLoadingError,
            CacheConnectionError,
            ConnectionError,
            TimeoutError,
        },
    )
    return RetryDecorator(config)


def adaptive_retry(operation_id: str = None):
    """Adaptive retry decorator that learns from past failures."""
    config = RetryConfig(strategy=RetryStrategy.ADAPTIVE, adaptive_scaling=True, max_retries=5)
    return RetryDecorator(config, operation_id)


# Global retry service instance
_global_retry_service: CacheRetryService | None = None


def get_cache_retry_service(config: RetryConfig | None = None) -> CacheRetryService:
    """Get or create global cache retry service."""
    global _global_retry_service
    if _global_retry_service is None or (config and config != _global_retry_service.config):
        _global_retry_service = CacheRetryService(config)
    return _global_retry_service


def reset_cache_retry_service() -> None:
    """Reset global cache retry service."""
    global _global_retry_service
    _global_retry_service = None


# Example usage functions
async def execute_cache_operation_with_retry(
    operation: Callable[..., T], *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs
) -> T:
    """Helper function to execute cache operations with retry."""
    config = RetryConfig(max_retries=max_retries, base_delay=base_delay)
    retry_service = CacheRetryService(config)
    return await retry_service.execute_with_retry(operation, *args, **kwargs)
