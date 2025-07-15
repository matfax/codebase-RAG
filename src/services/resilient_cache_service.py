"""
Resilient Cache Service Implementation.

This module provides enhanced cache services with comprehensive error handling,
graceful degradation, circuit breakers, and self-healing capabilities.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

from redis.exceptions import ConnectionError, TimeoutError

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..utils.resilience_framework import (
    CircuitBreakerOpenError,
    DegradationLevel,
    FallbackStrategy,
    ResilienceConfig,
    ResilienceFramework,
    get_resilience_framework,
)
from .cache_service import (
    BaseCacheService,
    CacheConnectionError,
    CacheError,
    CacheHealthInfo,
    CacheHealthStatus,
    CacheOperationError,
    CacheStats,
    MultiTierCacheService,
    RedisCacheService,
)


class ResilientCacheService(BaseCacheService):
    """
    Resilient cache service with comprehensive error handling and graceful degradation.

    This service wraps the standard cache services with resilience features:
    - Circuit breaker pattern for Redis connections
    - Exponential backoff retry logic
    - Graceful degradation strategies
    - Automatic error recovery
    - Self-healing mechanisms
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize resilient cache service."""
        super().__init__(config)

        # Initialize underlying cache service
        if self.config.cache_level.value == "both":
            self._cache_service = MultiTierCacheService(config)
        else:
            self._cache_service = RedisCacheService(config)

        # Initialize resilience framework
        self._resilience_config = ResilienceConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_seconds=60.0,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            degradation_enabled=True,
            auto_recovery_enabled=True,
            self_healing_enabled=True,
        )

        self._resilience_framework: ResilienceFramework | None = None

        # Graceful degradation state
        self._degraded_mode = False
        self._degradation_start_time = 0.0
        self._fallback_cache: dict[str, Any] = {}
        self._fallback_timestamps: dict[str, float] = {}
        self._fallback_ttl = 300  # 5 minutes

        # Health monitoring
        self._last_successful_operation = time.time()
        self._consecutive_failures = 0
        self._health_check_task: asyncio.Task | None = None

        # Self-healing strategies
        self._healing_strategies_registered = False

    async def initialize(self) -> None:
        """Initialize resilient cache service."""
        try:
            # Initialize resilience framework
            self._resilience_framework = await get_resilience_framework()

            # Register self-healing strategies
            if not self._healing_strategies_registered:
                await self._register_healing_strategies()
                self._healing_strategies_registered = True

            # Initialize underlying cache service
            await self._initialize_with_resilience()

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitoring_loop())

            self.logger.info("Resilient cache service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize resilient cache service: {e}")
            # Enable degraded mode immediately
            await self._enable_degraded_mode()
            # Still mark as "initialized" but in degraded state
            self.logger.warning("Cache service initialized in degraded mode")

    async def shutdown(self) -> None:
        """Shutdown resilient cache service."""
        try:
            # Cancel health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Shutdown underlying cache service
            if self._cache_service:
                await self._cache_service.shutdown()

            self.logger.info("Resilient cache service shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during resilient cache service shutdown: {e}")

    async def get(self, key: str) -> Any | None:
        """Get a value from cache with resilience features."""
        if self._degraded_mode:
            return await self._get_degraded(key)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.get,
                key,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                cache_key=f"get:{key}",
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Store successful result for fallback
            if result is not None:
                self._store_fallback_data(f"get:{key}", result)
                self._on_successful_operation()

            return result

        except CircuitBreakerOpenError:
            self.logger.warning(f"Circuit breaker open for get operation: {key}")
            return await self._get_degraded(key)
        except Exception as e:
            self.logger.error(f"Cache get operation failed for key '{key}': {e}")
            self._on_failed_operation()
            return await self._get_degraded(key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set a value in cache with resilience features."""
        if self._degraded_mode:
            return await self._set_degraded(key, value, ttl)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.set,
                key,
                value,
                ttl,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value=False,
                cache_key=f"set:{key}",
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Store in fallback cache on successful set
            if result:
                self._store_fallback_data(f"get:{key}", value)
                self._on_successful_operation()

            return bool(result)

        except CircuitBreakerOpenError:
            self.logger.warning(f"Circuit breaker open for set operation: {key}")
            return await self._set_degraded(key, value, ttl)
        except Exception as e:
            self.logger.error(f"Cache set operation failed for key '{key}': {e}")
            self._on_failed_operation()
            return await self._set_degraded(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete a value from cache with resilience features."""
        if self._degraded_mode:
            return await self._delete_degraded(key)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.delete,
                key,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value=False,
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Remove from fallback cache
            self._remove_fallback_data(f"get:{key}")
            if result:
                self._on_successful_operation()

            return bool(result)

        except CircuitBreakerOpenError:
            self.logger.warning(f"Circuit breaker open for delete operation: {key}")
            return await self._delete_degraded(key)
        except Exception as e:
            self.logger.error(f"Cache delete operation failed for key '{key}': {e}")
            self._on_failed_operation()
            return await self._delete_degraded(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache with resilience features."""
        if self._degraded_mode:
            return await self._exists_degraded(key)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.exists,
                key,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value=False,
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            if result:
                self._on_successful_operation()

            return bool(result)

        except CircuitBreakerOpenError:
            self.logger.warning(f"Circuit breaker open for exists operation: {key}")
            return await self._exists_degraded(key)
        except Exception as e:
            self.logger.error(f"Cache exists operation failed for key '{key}': {e}")
            self._on_failed_operation()
            return await self._exists_degraded(key)

    async def get_batch(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache with resilience features."""
        if self._degraded_mode:
            return await self._get_batch_degraded(keys)

        # Check if batch operations are allowed in current degradation level
        if not self._resilience_framework.degradation_manager.is_feature_enabled("batch_operations"):
            self.logger.info("Batch operations disabled due to degradation level, falling back to individual gets")
            return await self._get_batch_fallback(keys)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.get_batch,
                keys,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value={},
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Store successful results for fallback
            for key, value in result.items():
                if value is not None:
                    self._store_fallback_data(f"get:{key}", value)

            if result:
                self._on_successful_operation()

            return result

        except CircuitBreakerOpenError:
            self.logger.warning("Circuit breaker open for batch get operation")
            return await self._get_batch_degraded(keys)
        except Exception as e:
            self.logger.error(f"Cache batch get operation failed: {e}")
            self._on_failed_operation()
            return await self._get_batch_degraded(keys)

    async def set_batch(self, items: dict[str, Any], ttl: int | None = None) -> dict[str, bool]:
        """Set multiple values in cache with resilience features."""
        if self._degraded_mode:
            return await self._set_batch_degraded(items, ttl)

        # Check if batch operations are allowed
        if not self._resilience_framework.degradation_manager.is_feature_enabled("batch_operations"):
            self.logger.info("Batch operations disabled due to degradation level, falling back to individual sets")
            return await self._set_batch_fallback(items, ttl)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.set_batch,
                items,
                ttl,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value={key: False for key in items.keys()},
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Store successful sets in fallback cache
            for key, success in result.items():
                if success and key in items:
                    self._store_fallback_data(f"get:{key}", items[key])

            if any(result.values()):
                self._on_successful_operation()

            return result

        except CircuitBreakerOpenError:
            self.logger.warning("Circuit breaker open for batch set operation")
            return await self._set_batch_degraded(items, ttl)
        except Exception as e:
            self.logger.error(f"Cache batch set operation failed: {e}")
            self._on_failed_operation()
            return await self._set_batch_degraded(items, ttl)

    async def delete_batch(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple values from cache with resilience features."""
        if self._degraded_mode:
            return await self._delete_batch_degraded(keys)

        # Check if batch operations are allowed
        if not self._resilience_framework.degradation_manager.is_feature_enabled("batch_operations"):
            self.logger.info("Batch operations disabled due to degradation level, falling back to individual deletes")
            return await self._delete_batch_fallback(keys)

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.delete_batch,
                keys,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value={key: False for key in keys},
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Remove from fallback cache
            for key, success in result.items():
                if success:
                    self._remove_fallback_data(f"get:{key}")

            if any(result.values()):
                self._on_successful_operation()

            return result

        except CircuitBreakerOpenError:
            self.logger.warning("Circuit breaker open for batch delete operation")
            return await self._delete_batch_degraded(keys)
        except Exception as e:
            self.logger.error(f"Cache batch delete operation failed: {e}")
            self._on_failed_operation()
            return await self._delete_batch_degraded(keys)

    async def clear(self) -> bool:
        """Clear all cache entries with resilience features."""
        if self._degraded_mode:
            return await self._clear_degraded()

        try:
            result = await self._resilience_framework.execute_resilient_operation(
                self._cache_service.clear,
                fallback_strategy=FallbackStrategy.GRACEFUL_DEGRADATION,
                fallback_value=False,
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
            )

            # Clear fallback cache as well
            self._fallback_cache.clear()
            self._fallback_timestamps.clear()

            if result:
                self._on_successful_operation()

            return bool(result)

        except CircuitBreakerOpenError:
            self.logger.warning("Circuit breaker open for clear operation")
            return await self._clear_degraded()
        except Exception as e:
            self.logger.error(f"Cache clear operation failed: {e}")
            self._on_failed_operation()
            return await self._clear_degraded()

    async def get_health(self) -> CacheHealthInfo:
        """Get comprehensive health information including resilience status."""
        try:
            # Get base health info
            base_health = (
                await self._cache_service.get_health()
                if not self._degraded_mode
                else CacheHealthInfo(status=CacheHealthStatus.DEGRADED, redis_connected=False, last_error="Service in degraded mode")
            )

            # Get resilience health status
            resilience_health = self._resilience_framework.get_health_status() if self._resilience_framework else {}

            # Combine health information
            base_health.status = self._determine_overall_health_status(base_health.status, resilience_health)

            # Add resilience-specific information
            if hasattr(base_health, "__dict__"):
                base_health.__dict__.update(
                    {
                        "degraded_mode": self._degraded_mode,
                        "degradation_level": resilience_health.get("degradation_level", "normal"),
                        "circuit_breaker_state": resilience_health.get("circuit_breaker_state", "closed"),
                        "disabled_features": resilience_health.get("disabled_features", []),
                        "consecutive_failures": self._consecutive_failures,
                        "last_successful_operation": self._last_successful_operation,
                        "fallback_cache_size": len(self._fallback_cache),
                        "self_healing_active": resilience_health.get("self_healing_active", False),
                    }
                )

            return base_health

        except Exception as e:
            self.logger.error(f"Failed to get health info: {e}")
            return CacheHealthInfo(status=CacheHealthStatus.UNHEALTHY, last_error=str(e), degraded_mode=self._degraded_mode)

    # Degraded mode operations
    async def _get_degraded(self, key: str) -> Any | None:
        """Get operation in degraded mode using fallback cache."""
        fallback_key = f"get:{key}"
        if fallback_key in self._fallback_cache:
            # Check if data is still valid
            timestamp = self._fallback_timestamps.get(fallback_key, 0)
            if time.time() - timestamp <= self._fallback_ttl:
                self.logger.debug(f"Returning fallback data for key: {key}")
                return self._fallback_cache[fallback_key]
            else:
                # Remove expired data
                self._remove_fallback_data(fallback_key)

        self.logger.warning(f"No fallback data available for key: {key}")
        return None

    async def _set_degraded(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set operation in degraded mode using fallback cache."""
        try:
            fallback_key = f"get:{key}"
            self._store_fallback_data(fallback_key, value)
            self.logger.debug(f"Stored value in fallback cache for key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store in fallback cache: {e}")
            return False

    async def _delete_degraded(self, key: str) -> bool:
        """Delete operation in degraded mode."""
        fallback_key = f"get:{key}"
        if fallback_key in self._fallback_cache:
            self._remove_fallback_data(fallback_key)
            self.logger.debug(f"Removed fallback data for key: {key}")
            return True
        return False

    async def _exists_degraded(self, key: str) -> bool:
        """Exists operation in degraded mode."""
        fallback_key = f"get:{key}"
        if fallback_key in self._fallback_cache:
            # Check if data is still valid
            timestamp = self._fallback_timestamps.get(fallback_key, 0)
            return time.time() - timestamp <= self._fallback_ttl
        return False

    async def _get_batch_degraded(self, keys: list[str]) -> dict[str, Any]:
        """Batch get operation in degraded mode."""
        result = {}
        for key in keys:
            value = await self._get_degraded(key)
            if value is not None:
                result[key] = value
        return result

    async def _set_batch_degraded(self, items: dict[str, Any], ttl: int | None = None) -> dict[str, bool]:
        """Batch set operation in degraded mode."""
        result = {}
        for key, value in items.items():
            result[key] = await self._set_degraded(key, value, ttl)
        return result

    async def _delete_batch_degraded(self, keys: list[str]) -> dict[str, bool]:
        """Batch delete operation in degraded mode."""
        result = {}
        for key in keys:
            result[key] = await self._delete_degraded(key)
        return result

    async def _clear_degraded(self) -> bool:
        """Clear operation in degraded mode."""
        try:
            self._fallback_cache.clear()
            self._fallback_timestamps.clear()
            self.logger.info("Cleared fallback cache in degraded mode")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear fallback cache: {e}")
            return False

    # Fallback operations for when batch operations are disabled
    async def _get_batch_fallback(self, keys: list[str]) -> dict[str, Any]:
        """Fallback batch get using individual operations."""
        result = {}
        for key in keys:
            try:
                value = await self.get(key)
                if value is not None:
                    result[key] = value
            except Exception as e:
                self.logger.warning(f"Individual get failed for key {key}: {e}")
        return result

    async def _set_batch_fallback(self, items: dict[str, Any], ttl: int | None = None) -> dict[str, bool]:
        """Fallback batch set using individual operations."""
        result = {}
        for key, value in items.items():
            try:
                result[key] = await self.set(key, value, ttl)
            except Exception as e:
                self.logger.warning(f"Individual set failed for key {key}: {e}")
                result[key] = False
        return result

    async def _delete_batch_fallback(self, keys: list[str]) -> dict[str, bool]:
        """Fallback batch delete using individual operations."""
        result = {}
        for key in keys:
            try:
                result[key] = await self.delete(key)
            except Exception as e:
                self.logger.warning(f"Individual delete failed for key {key}: {e}")
                result[key] = False
        return result

    # Utility methods
    def _store_fallback_data(self, key: str, data: Any) -> None:
        """Store data in fallback cache."""
        try:
            self._fallback_cache[key] = data
            self._fallback_timestamps[key] = time.time()

            # Cleanup old entries periodically
            if len(self._fallback_cache) % 100 == 0:
                self._cleanup_fallback_cache()
        except Exception as e:
            self.logger.error(f"Failed to store fallback data: {e}")

    def _remove_fallback_data(self, key: str) -> None:
        """Remove data from fallback cache."""
        self._fallback_cache.pop(key, None)
        self._fallback_timestamps.pop(key, None)

    def _cleanup_fallback_cache(self) -> None:
        """Clean up expired fallback cache entries."""
        current_time = time.time()
        expired_keys = [key for key, timestamp in self._fallback_timestamps.items() if current_time - timestamp > self._fallback_ttl]

        for key in expired_keys:
            self._remove_fallback_data(key)

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired fallback entries")

    def _on_successful_operation(self) -> None:
        """Handle successful operation."""
        self._last_successful_operation = time.time()
        self._consecutive_failures = 0

        # Try to exit degraded mode if conditions are met
        if self._degraded_mode and self._should_exit_degraded_mode():
            asyncio.create_task(self._disable_degraded_mode())

    def _on_failed_operation(self) -> None:
        """Handle failed operation."""
        self._consecutive_failures += 1

        # Enter degraded mode if failure threshold is reached
        if not self._degraded_mode and self._should_enter_degraded_mode():
            asyncio.create_task(self._enable_degraded_mode())

    def _should_enter_degraded_mode(self) -> bool:
        """Check if service should enter degraded mode."""
        return self._consecutive_failures >= 5

    def _should_exit_degraded_mode(self) -> bool:
        """Check if service should exit degraded mode."""
        # Exit degraded mode after successful health check
        time_in_degraded = time.time() - self._degradation_start_time
        return (
            self._consecutive_failures == 0
            and time_in_degraded >= 60  # At least 1 minute in degraded mode
            and self._resilience_framework
            and self._resilience_framework.circuit_breaker.get_state().value == "closed"
        )

    async def _enable_degraded_mode(self) -> None:
        """Enable degraded mode."""
        if not self._degraded_mode:
            self._degraded_mode = True
            self._degradation_start_time = time.time()
            self.logger.warning("Cache service entered degraded mode")

    async def _disable_degraded_mode(self) -> None:
        """Disable degraded mode."""
        if self._degraded_mode:
            try:
                # Try to reconnect to underlying service
                await self._cache_service.initialize()

                self._degraded_mode = False
                self._degradation_start_time = 0.0
                self.logger.info("Cache service exited degraded mode")

            except Exception as e:
                self.logger.warning(f"Failed to exit degraded mode: {e}")

    def _determine_overall_health_status(self, base_status: CacheHealthStatus, resilience_health: dict) -> CacheHealthStatus:
        """Determine overall health status based on all factors."""
        if self._degraded_mode:
            return CacheHealthStatus.DEGRADED

        circuit_breaker_state = resilience_health.get("circuit_breaker_state", "closed")
        if circuit_breaker_state == "open":
            return CacheHealthStatus.UNHEALTHY
        elif circuit_breaker_state == "half_open":
            return CacheHealthStatus.DEGRADED

        return base_status

    async def _initialize_with_resilience(self) -> None:
        """Initialize underlying cache service with resilience."""
        try:
            await self._resilience_framework.execute_resilient_operation(
                self._cache_service.initialize,
                fallback_strategy=FallbackStrategy.NONE,
                retryable_exceptions=(ConnectionError, TimeoutError, CacheConnectionError),
                use_circuit_breaker=False,  # Don't use circuit breaker for initialization
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize underlying cache service: {e}")
            raise e

    async def _register_healing_strategies(self) -> None:
        """Register self-healing strategies."""
        if not self._resilience_framework:
            return

        # Register connection recovery strategy
        self._resilience_framework.self_healing_manager.register_healing_strategy(self._heal_connection)

        # Register cache cleanup strategy
        self._resilience_framework.self_healing_manager.register_healing_strategy(self._heal_cache_cleanup)

        self.logger.debug("Registered self-healing strategies")

    async def _heal_connection(self) -> None:
        """Healing strategy for connection issues."""
        try:
            if self._degraded_mode:
                self.logger.info("Attempting to heal cache connection")
                await self._disable_degraded_mode()
        except Exception as e:
            self.logger.error(f"Connection healing failed: {e}")

    async def _heal_cache_cleanup(self) -> None:
        """Healing strategy for cache cleanup."""
        try:
            # Clean up fallback cache
            self._cleanup_fallback_cache()

            # Reset consecutive failures if we've been stable
            if time.time() - self._last_successful_operation < 300:  # 5 minutes
                self._consecutive_failures = max(0, self._consecutive_failures - 1)

        except Exception as e:
            self.logger.error(f"Cache cleanup healing failed: {e}")

    async def _health_monitoring_loop(self) -> None:
        """Continuous health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Perform health check
                try:
                    health_info = await self.get_health()

                    # Update resilience framework with health status
                    if self._resilience_framework:
                        success = health_info.status in [CacheHealthStatus.HEALTHY, CacheHealthStatus.DEGRADED]
                        error_rate = self._consecutive_failures / max(10, self._consecutive_failures + 1)
                        self._resilience_framework.degradation_manager.update_health_status(success, error_rate)

                except Exception as e:
                    self.logger.warning(f"Health check failed: {e}")
                    self._on_failed_operation()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")


# Factory function for creating resilient cache service
async def create_resilient_cache_service(config: CacheConfig | None = None) -> ResilientCacheService:
    """Create and initialize a resilient cache service."""
    service = ResilientCacheService(config)
    await service.initialize()
    return service


# Global resilient cache service instance
_resilient_cache_service: ResilientCacheService | None = None


async def get_resilient_cache_service() -> ResilientCacheService:
    """Get or create global resilient cache service instance."""
    global _resilient_cache_service
    if _resilient_cache_service is None:
        _resilient_cache_service = await create_resilient_cache_service()
    return _resilient_cache_service


async def shutdown_resilient_cache_service() -> None:
    """Shutdown global resilient cache service."""
    global _resilient_cache_service
    if _resilient_cache_service:
        await _resilient_cache_service.shutdown()
        _resilient_cache_service = None
