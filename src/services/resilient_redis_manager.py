"""
Resilient Redis Connection Manager with Circuit Breaker.

This module provides an enhanced Redis connection manager that integrates
circuit breaker functionality, advanced error handling, and self-healing capabilities.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from ..config.cache_config import CacheConfig
from ..utils.redis_circuit_breaker import RedisCircuitBreaker, RedisCircuitBreakerConfig, RedisCircuitBreakerError
from .cache_service import CacheHealthInfo, CacheHealthStatus


class ResilientRedisConnectionManager:
    """
    Enhanced Redis connection manager with circuit breaker integration.

    This manager provides robust Redis connection handling with:
    - Circuit breaker pattern for connection failures
    - Automatic connection recovery
    - Health monitoring and metrics
    - Connection pool optimization
    - Self-healing capabilities
    """

    def __init__(self, config: CacheConfig):
        """Initialize resilient Redis connection manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Redis connection components
        self._pool: ConnectionPool | None = None
        self._redis: redis.Redis | None = None
        self._health_check_task: asyncio.Task | None = None

        # Circuit breaker
        self._circuit_breaker: RedisCircuitBreaker | None = None
        self._circuit_breaker_config = RedisCircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=5,
            timeout_seconds=60.0,
            half_open_max_calls=3,
            connection_timeout=self.config.redis.connection_timeout,
            command_timeout=self.config.redis.socket_timeout,
            health_check_interval=self.config.health_check_interval,
            max_response_time_threshold=5.0,
            connection_pool_threshold=0.8,
        )

        # Connection state
        self._health_status = CacheHealthStatus.DISCONNECTED
        self._last_health_check = 0.0
        self._connection_attempts = 0
        self._max_connection_attempts = 5
        self._last_successful_connection = 0.0

        # Self-healing
        self._recovery_task: asyncio.Task | None = None
        self._recovery_attempts = 0
        self._max_recovery_attempts = 10
        self._recovery_interval = 30.0

    async def initialize(self) -> None:
        """Initialize Redis connection with circuit breaker protection."""
        try:
            await self._initialize_connection()
            await self._initialize_circuit_breaker()
            await self._start_monitoring()

            self.logger.info("Resilient Redis connection manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize resilient Redis connection: {e}")
            # Start recovery process even if initial connection fails
            await self._start_recovery_process()
            raise e

    async def shutdown(self) -> None:
        """Shutdown resilient Redis connection manager."""
        try:
            # Stop monitoring tasks
            await self._stop_monitoring()

            # Shutdown circuit breaker
            if self._circuit_breaker:
                await self._circuit_breaker.shutdown()

            # Close Redis connections
            await self._close_connections()

            self.logger.info("Resilient Redis connection manager shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during resilient Redis shutdown: {e}")

    async def get_redis(self) -> redis.Redis:
        """Get Redis client instance with circuit breaker protection."""
        if not self._redis:
            raise ConnectionError("Redis connection not initialized")

        # Check circuit breaker state
        if self._circuit_breaker and self._circuit_breaker.get_state().value == "open":
            raise RedisCircuitBreakerError("Redis circuit breaker is open")

        return self._redis

    async def execute_command(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command through circuit breaker."""
        if not self._circuit_breaker:
            raise ConnectionError("Circuit breaker not initialized")

        redis_client = await self.get_redis()

        # Execute command through circuit breaker
        async def redis_operation():
            return await getattr(redis_client, command.lower())(*args, **kwargs)

        return await self._circuit_breaker.execute(redis_operation, operation_name=f"redis_{command.lower()}")

    async def ping(self) -> bool:
        """Ping Redis server through circuit breaker."""
        try:
            result = await self.execute_command("PING")
            return result == b"PONG" or result == "PONG"
        except Exception:
            return False

    async def info(self, section: str = "default") -> dict[str, Any]:
        """Get Redis info through circuit breaker."""
        return await self.execute_command("INFO", section)

    async def get_health_info(self) -> CacheHealthInfo:
        """Get comprehensive health information."""
        try:
            # Base health info
            redis_connected = False
            ping_time = None
            memory_usage = None
            connection_pool_stats = None
            last_error = None

            # Circuit breaker status
            cb_state = "unknown"
            cb_metrics = {}

            if self._circuit_breaker:
                cb_state = self._circuit_breaker.get_state().value
                cb_metrics = self._circuit_breaker.get_metrics().__dict__

                # Test connection if circuit breaker allows
                if cb_state != "open":
                    try:
                        start_time = time.time()
                        await self.ping()
                        ping_time = time.time() - start_time
                        redis_connected = True

                        # Get memory info
                        info = await self.info("memory")
                        memory_usage = {
                            "used_memory": info.get("used_memory", 0),
                            "used_memory_human": info.get("used_memory_human", "N/A"),
                            "used_memory_peak": info.get("used_memory_peak", 0),
                            "used_memory_peak_human": info.get("used_memory_peak_human", "N/A"),
                        }

                    except Exception as e:
                        last_error = str(e)
                        redis_connected = False

            # Connection pool stats
            if self._pool:
                try:
                    connection_pool_stats = {
                        "created_connections": getattr(self._pool, "created_connections", 0),
                        "available_connections": len(getattr(self._pool, "_available_connections", [])),
                        "in_use_connections": len(getattr(self._pool, "_in_use_connections", [])),
                        "max_connections": getattr(self._pool, "max_connections", 0),
                    }
                except Exception as e:
                    self.logger.debug(f"Failed to get connection pool stats: {e}")

            # Determine overall health status
            status = self._determine_health_status(redis_connected, cb_state)

            health_info = CacheHealthInfo(
                status=status,
                redis_connected=redis_connected,
                redis_ping_time=ping_time,
                memory_usage=memory_usage,
                connection_pool_stats=connection_pool_stats,
                last_error=last_error,
            )

            # Add circuit breaker specific information
            if hasattr(health_info, "__dict__"):
                health_info.__dict__.update(
                    {
                        "circuit_breaker_state": cb_state,
                        "circuit_breaker_metrics": cb_metrics,
                        "connection_attempts": self._connection_attempts,
                        "recovery_attempts": self._recovery_attempts,
                        "last_successful_connection": self._last_successful_connection,
                        "recovery_active": self._recovery_task is not None and not self._recovery_task.done(),
                    }
                )

            return health_info

        except Exception as e:
            self.logger.error(f"Failed to get health info: {e}")
            return CacheHealthInfo(status=CacheHealthStatus.UNHEALTHY, redis_connected=False, last_error=str(e))

    def _determine_health_status(self, redis_connected: bool, cb_state: str) -> CacheHealthStatus:
        """Determine overall health status."""
        if cb_state == "open":
            return CacheHealthStatus.UNHEALTHY
        elif cb_state == "half_open":
            return CacheHealthStatus.DEGRADED
        elif redis_connected:
            return CacheHealthStatus.HEALTHY
        else:
            return CacheHealthStatus.DISCONNECTED

    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy."""
        return (
            self._health_status == CacheHealthStatus.HEALTHY
            and self._circuit_breaker
            and self._circuit_breaker.get_state().value == "closed"
        )

    async def _initialize_connection(self) -> None:
        """Initialize Redis connection pool."""
        try:
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self.config.get_redis_url(),
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout,
                socket_connect_timeout=self.config.redis.connection_timeout,
                retry_on_timeout=self.config.redis.retry_on_timeout,
                retry_on_error=[ConnectionError, TimeoutError],
                health_check_interval=self.config.health_check_interval,
                ssl=self.config.redis.ssl_enabled,
                ssl_cert_reqs=None if not self.config.redis.ssl_enabled else "required",
                ssl_ca_certs=self.config.redis.ssl_ca_cert_path,
                ssl_certfile=self.config.redis.ssl_cert_path,
                ssl_keyfile=self.config.redis.ssl_key_path,
            )

            # Create Redis client
            self._redis = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._test_connection()

            self._health_status = CacheHealthStatus.HEALTHY
            self._last_successful_connection = time.time()
            self._connection_attempts = 0

        except Exception as e:
            self._health_status = CacheHealthStatus.UNHEALTHY
            self._connection_attempts += 1
            raise e

    async def _initialize_circuit_breaker(self) -> None:
        """Initialize circuit breaker."""
        if not self._redis:
            raise ConnectionError("Redis client not available for circuit breaker")

        self._circuit_breaker = RedisCircuitBreaker(self._circuit_breaker_config, self._redis)
        await self._circuit_breaker.initialize()

    async def _start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            return

        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

    async def _close_connections(self) -> None:
        """Close Redis connections."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    async def _test_connection(self) -> None:
        """Test Redis connection."""
        if not self._redis:
            raise ConnectionError("Redis client not initialized")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(self._redis.ping(), timeout=self._circuit_breaker_config.health_check_timeout)
            ping_time = time.time() - start_time

            if result != b"PONG" and result != "PONG":
                raise ConnectionError(f"Invalid ping response: {result}")

            self.logger.debug(f"Redis connection test successful (ping: {ping_time:.3f}s)")

        except Exception as e:
            self.logger.warning(f"Redis connection test failed: {e}")
            raise e

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Perform health check
                try:
                    if await self.ping():
                        self._health_status = CacheHealthStatus.HEALTHY
                        self._last_health_check = time.time()
                    else:
                        self._health_status = CacheHealthStatus.UNHEALTHY

                except Exception as e:
                    self.logger.warning(f"Health check failed: {e}")
                    self._health_status = CacheHealthStatus.UNHEALTHY

                    # Start recovery if not already running
                    if not self._recovery_task or self._recovery_task.done():
                        await self._start_recovery_process()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")

    async def _start_recovery_process(self) -> None:
        """Start connection recovery process."""
        if self._recovery_task and not self._recovery_task.done():
            return

        self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def _recovery_loop(self) -> None:
        """Connection recovery loop."""
        self.logger.info("Starting Redis connection recovery process")

        while self._recovery_attempts < self._max_recovery_attempts:
            try:
                await asyncio.sleep(self._recovery_interval)

                self.logger.info(f"Attempting Redis connection recovery (attempt {self._recovery_attempts + 1})")

                # Close existing connections
                await self._close_connections()

                # Reset circuit breaker
                if self._circuit_breaker:
                    self._circuit_breaker.reset()

                # Attempt to reconnect
                await self._initialize_connection()
                await self._initialize_circuit_breaker()

                self.logger.info("Redis connection recovery successful")
                self._recovery_attempts = 0
                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._recovery_attempts += 1
                self.logger.warning(f"Redis connection recovery attempt {self._recovery_attempts} failed: {e}")

                # Exponential backoff for recovery interval
                self._recovery_interval = min(300, self._recovery_interval * 1.5)

        if self._recovery_attempts >= self._max_recovery_attempts:
            self.logger.error(f"Redis connection recovery failed after {self._max_recovery_attempts} attempts")

    def get_circuit_breaker_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics."""
        if not self._circuit_breaker:
            return {"error": "Circuit breaker not initialized"}

        metrics = self._circuit_breaker.get_metrics()
        return {
            "state": self._circuit_breaker.get_state().value,
            "metrics": metrics.__dict__,
            "error_summary": self._circuit_breaker.get_error_summary(),
            "performance_summary": self._circuit_breaker.get_performance_summary(),
        }

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
            self.logger.info("Redis circuit breaker manually reset")

    def force_circuit_breaker_open(self) -> None:
        """Force circuit breaker to open state."""
        if self._circuit_breaker:
            self._circuit_breaker.force_open()
            self.logger.warning("Redis circuit breaker manually forced open")

    def force_circuit_breaker_closed(self) -> None:
        """Force circuit breaker to closed state."""
        if self._circuit_breaker:
            self._circuit_breaker.force_close()
            self.logger.info("Redis circuit breaker manually forced closed")


# Factory function
async def create_resilient_redis_manager(config: CacheConfig) -> ResilientRedisConnectionManager:
    """Create and initialize a resilient Redis connection manager."""
    manager = ResilientRedisConnectionManager(config)
    await manager.initialize()
    return manager
