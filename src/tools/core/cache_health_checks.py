"""
Cache-specific health check implementations.

This module provides detailed health checks for individual cache services,
including connectivity tests, performance validation, and service-specific metrics.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

from services.cache_service import CacheHealthStatus
from utils.performance_monitor import MemoryMonitor, get_cache_performance_monitor

logger = logging.getLogger(__name__)


class CacheHealthChecker:
    """Comprehensive health checker for individual cache services."""

    def __init__(self):
        self.cache_monitor = get_cache_performance_monitor()
        self.memory_monitor = MemoryMonitor()
        self._cache_services = {}  # Will be populated dynamically
        self._health_thresholds = {
            "response_time_warning_ms": float(os.getenv("CACHE_RESPONSE_TIME_WARNING_MS", "500")),
            "response_time_critical_ms": float(os.getenv("CACHE_RESPONSE_TIME_CRITICAL_MS", "2000")),
            "hit_rate_warning": float(os.getenv("CACHE_HIT_RATE_WARNING", "0.6")),
            "hit_rate_critical": float(os.getenv("CACHE_HIT_RATE_CRITICAL", "0.3")),
            "error_rate_warning": float(os.getenv("CACHE_ERROR_RATE_WARNING", "0.05")),
            "error_rate_critical": float(os.getenv("CACHE_ERROR_RATE_CRITICAL", "0.1")),
            "memory_usage_warning_mb": float(os.getenv("CACHE_MEMORY_WARNING_MB", "100")),
            "memory_usage_critical_mb": float(os.getenv("CACHE_MEMORY_CRITICAL_MB", "500")),
            "connection_timeout_seconds": float(os.getenv("CACHE_CONNECTION_TIMEOUT", "5")),
        }

    def register_cache_service(self, name: str, service: Any) -> None:
        """Register a cache service for health checking."""
        self._cache_services[name] = service
        logger.info(f"Registered cache service '{name}' for health monitoring")

    def unregister_cache_service(self, name: str) -> None:
        """Unregister a cache service from health checking."""
        if name in self._cache_services:
            del self._cache_services[name]
            logger.info(f"Unregistered cache service '{name}' from health monitoring")

    async def check_cache_service_health(self, service_name: str) -> dict[str, Any]:
        """
        Perform comprehensive health check for a specific cache service.

        Args:
            service_name: Name of the cache service to check

        Returns:
            Detailed health check results
        """
        start_time = time.time()

        # Basic service availability check
        service = self._cache_services.get(service_name)
        if not service:
            return {
                "service_name": service_name,
                "status": "error",
                "message": f"Cache service '{service_name}' not registered",
                "timestamp": datetime.now().isoformat(),
                "check_duration_ms": (time.time() - start_time) * 1000,
            }

        health_results = {
            "service_name": service_name,
            "status": "ok",
            "checks": {},
            "metrics": {},
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # 1. Connectivity and Basic Health Check
            connectivity_result = await self._check_connectivity(service_name, service)
            health_results["checks"]["connectivity"] = connectivity_result

            if not connectivity_result["healthy"]:
                health_results["status"] = "error"
                health_results["issues"].append(f"Connectivity check failed: {connectivity_result.get('error', 'Unknown error')}")

            # 2. Performance Metrics Check
            performance_result = await self._check_performance_metrics(service_name)
            health_results["checks"]["performance"] = performance_result

            if performance_result["status"] == "critical":
                health_results["status"] = "error"
                health_results["issues"].extend(performance_result.get("issues", []))
            elif performance_result["status"] == "warning" and health_results["status"] == "ok":
                health_results["status"] = "warning"

            health_results["warnings"].extend(performance_result.get("warnings", []))

            # 3. Memory Usage Check
            memory_result = await self._check_memory_usage(service_name)
            health_results["checks"]["memory"] = memory_result

            if memory_result["status"] == "critical":
                health_results["status"] = "error"
                health_results["issues"].extend(memory_result.get("issues", []))
            elif memory_result["status"] == "warning" and health_results["status"] == "ok":
                health_results["status"] = "warning"

            health_results["warnings"].extend(memory_result.get("warnings", []))

            # 4. Operation Tests (if service is available)
            if connectivity_result["healthy"]:
                operation_result = await self._check_operations(service_name, service)
                health_results["checks"]["operations"] = operation_result

                if not operation_result["all_operations_successful"]:
                    if health_results["status"] == "ok":
                        health_results["status"] = "warning"
                    health_results["warnings"].extend(operation_result.get("warnings", []))

            # 5. Configuration Validation
            config_result = await self._check_configuration(service_name, service)
            health_results["checks"]["configuration"] = config_result

            if not config_result["valid"]:
                if health_results["status"] == "ok":
                    health_results["status"] = "warning"
                health_results["warnings"].extend(config_result.get("warnings", []))

            # 6. Generate Recommendations
            health_results["recommendations"] = self._generate_recommendations(health_results)

            # 7. Aggregate Metrics
            health_results["metrics"] = self._aggregate_service_metrics(service_name)

        except Exception as e:
            logger.error(f"Health check failed for cache service '{service_name}': {e}")
            health_results["status"] = "error"
            health_results["issues"].append(f"Health check failed: {str(e)}")

        health_results["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)
        return health_results

    async def _check_connectivity(self, service_name: str, service: Any) -> dict[str, Any]:
        """Check cache service connectivity and basic health."""
        result = {
            "healthy": False,
            "redis_connected": False,
            "ping_time_ms": None,
            "connection_pool_healthy": False,
            "last_health_check_age_seconds": None,
            "error": None,
        }

        try:
            # Check if service has health info method
            if hasattr(service, "get_health_info"):
                health_info = await service.get_health_info()
                result["redis_connected"] = health_info.redis_connected
                result["ping_time_ms"] = round(health_info.redis_ping_time * 1000, 2) if health_info.redis_ping_time else None
                result["last_health_check_age_seconds"] = round(time.time() - health_info.check_timestamp, 2)

                # Check connection pool stats
                if health_info.connection_pool_stats:
                    pool_stats = health_info.connection_pool_stats
                    available = pool_stats.get("available_connections", 0)
                    max_connections = pool_stats.get("max_connections", 1)
                    result["connection_pool_healthy"] = available > 0 and available <= max_connections

                # Overall health determination
                result["healthy"] = (
                    health_info.status == CacheHealthStatus.HEALTHY
                    and health_info.redis_connected
                    and (health_info.redis_ping_time or 0) < self._health_thresholds["connection_timeout_seconds"]
                )

                if health_info.last_error:
                    result["error"] = health_info.last_error

            # Test basic connectivity if we have direct access
            elif hasattr(service, "_redis") and service._redis:
                start_time = time.time()
                await asyncio.wait_for(service._redis.ping(), timeout=self._health_thresholds["connection_timeout_seconds"])
                result["ping_time_ms"] = round((time.time() - start_time) * 1000, 2)
                result["redis_connected"] = True
                result["healthy"] = True

            else:
                result["error"] = "Service does not provide health check capabilities"

        except asyncio.TimeoutError:
            result["error"] = f"Connection timeout after {self._health_thresholds['connection_timeout_seconds']}s"
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _check_performance_metrics(self, service_name: str) -> dict[str, Any]:
        """Check cache service performance metrics."""
        metrics = self.cache_monitor.get_cache_metrics(service_name)

        result = {
            "status": "ok",
            "hit_rate": 0.0,
            "error_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "total_operations": 0,
            "issues": [],
            "warnings": [],
        }

        if not metrics:
            result["warnings"].append("No performance metrics available")
            return result

        # Extract key metrics
        hit_rate = metrics.get("hit_rate", 0.0)
        error_rate = metrics.get("errors", {}).get("rate", 0.0)
        avg_response_time_ms = metrics.get("performance", {}).get("average_response_time_ms", 0.0)
        total_operations = metrics.get("operations", {}).get("total", 0)

        result.update(
            {
                "hit_rate": hit_rate,
                "error_rate": error_rate,
                "avg_response_time_ms": avg_response_time_ms,
                "total_operations": total_operations,
            }
        )

        # Only evaluate if there are sufficient operations
        if total_operations >= 10:
            # Check hit rate
            if hit_rate < self._health_thresholds["hit_rate_critical"]:
                result["status"] = "critical"
                result["issues"].append(
                    f"Critical hit rate: {hit_rate:.1%} (threshold: {self._health_thresholds['hit_rate_critical']:.1%})"
                )
            elif hit_rate < self._health_thresholds["hit_rate_warning"]:
                result["status"] = "warning"
                result["warnings"].append(f"Low hit rate: {hit_rate:.1%} (threshold: {self._health_thresholds['hit_rate_warning']:.1%})")

            # Check error rate
            if error_rate > self._health_thresholds["error_rate_critical"]:
                result["status"] = "critical"
                result["issues"].append(
                    f"Critical error rate: {error_rate:.1%} (threshold: {self._health_thresholds['error_rate_critical']:.1%})"
                )
            elif error_rate > self._health_thresholds["error_rate_warning"]:
                if result["status"] != "critical":
                    result["status"] = "warning"
                result["warnings"].append(
                    f"Elevated error rate: {error_rate:.1%} (threshold: {self._health_thresholds['error_rate_warning']:.1%})"
                )

            # Check response time
            if avg_response_time_ms > self._health_thresholds["response_time_critical_ms"]:
                result["status"] = "critical"
                result["issues"].append(
                    f"Critical response time: {avg_response_time_ms:.0f}ms "
                    f"(threshold: {self._health_thresholds['response_time_critical_ms']:.0f}ms)"
                )
            elif avg_response_time_ms > self._health_thresholds["response_time_warning_ms"]:
                if result["status"] != "critical":
                    result["status"] = "warning"
                result["warnings"].append(
                    f"Slow response time: {avg_response_time_ms:.0f}ms "
                    f"(threshold: {self._health_thresholds['response_time_warning_ms']:.0f}ms)"
                )

        return result

    async def _check_memory_usage(self, service_name: str) -> dict[str, Any]:
        """Check cache service memory usage."""
        memory_info = self.memory_monitor.get_detailed_cache_memory_info()

        result = {
            "status": "ok",
            "current_size_mb": 0.0,
            "max_size_mb": 0.0,
            "memory_pressure_events": 0,
            "eviction_count": 0,
            "issues": [],
            "warnings": [],
        }

        cache_data = memory_info.get("by_cache", {}).get(service_name, {})
        if not cache_data:
            result["warnings"].append("No memory usage data available")
            return result

        current_size_mb = cache_data.get("current_size_mb", 0.0)
        max_size_mb = cache_data.get("max_size_mb", 0.0)
        memory_pressure_events = cache_data.get("memory_pressure_events", 0)
        eviction_count = cache_data.get("eviction_count", 0)

        result.update(
            {
                "current_size_mb": current_size_mb,
                "max_size_mb": max_size_mb,
                "memory_pressure_events": memory_pressure_events,
                "eviction_count": eviction_count,
            }
        )

        # Check memory usage thresholds
        if current_size_mb > self._health_thresholds["memory_usage_critical_mb"]:
            result["status"] = "critical"
            result["issues"].append(
                f"Critical memory usage: {current_size_mb:.1f}MB (threshold: {self._health_thresholds['memory_usage_critical_mb']:.0f}MB)"
            )
        elif current_size_mb > self._health_thresholds["memory_usage_warning_mb"]:
            result["status"] = "warning"
            result["warnings"].append(
                f"High memory usage: {current_size_mb:.1f}MB (threshold: {self._health_thresholds['memory_usage_warning_mb']:.0f}MB)"
            )

        # Check for memory pressure
        if memory_pressure_events > 5:
            if result["status"] != "critical":
                result["status"] = "warning"
            result["warnings"].append(f"Memory pressure detected: {memory_pressure_events} events")

        # Check for excessive evictions
        if eviction_count > 100:
            if result["status"] != "critical":
                result["status"] = "warning"
            result["warnings"].append(f"High eviction count: {eviction_count} evictions")

        return result

    async def _check_operations(self, service_name: str, service: Any) -> dict[str, Any]:
        """Test basic cache operations."""
        result = {
            "all_operations_successful": True,
            "operations_tested": [],
            "operation_times_ms": {},
            "warnings": [],
            "errors": [],
        }

        test_key = f"__health_check_test_{service_name}_{int(time.time())}"
        test_value = {"test": True, "timestamp": time.time(), "service": service_name}

        try:
            # Test SET operation
            start_time = time.time()
            set_result = await service.set(test_key, test_value, ttl=30)  # 30 second TTL
            set_time = (time.time() - start_time) * 1000

            result["operations_tested"].append("set")
            result["operation_times_ms"]["set"] = round(set_time, 2)

            if not set_result:
                result["all_operations_successful"] = False
                result["errors"].append("SET operation failed")

            # Test GET operation
            start_time = time.time()
            get_result = await service.get(test_key)
            get_time = (time.time() - start_time) * 1000

            result["operations_tested"].append("get")
            result["operation_times_ms"]["get"] = round(get_time, 2)

            if get_result != test_value:
                result["all_operations_successful"] = False
                result["errors"].append("GET operation returned incorrect value")

            # Test EXISTS operation
            if hasattr(service, "exists"):
                start_time = time.time()
                exists_result = await service.exists(test_key)
                exists_time = (time.time() - start_time) * 1000

                result["operations_tested"].append("exists")
                result["operation_times_ms"]["exists"] = round(exists_time, 2)

                if not exists_result:
                    result["all_operations_successful"] = False
                    result["errors"].append("EXISTS operation failed")

            # Test DELETE operation
            start_time = time.time()
            delete_result = await service.delete(test_key)
            delete_time = (time.time() - start_time) * 1000

            result["operations_tested"].append("delete")
            result["operation_times_ms"]["delete"] = round(delete_time, 2)

            if not delete_result:
                result["warnings"].append("DELETE operation returned false (key may not have existed)")

            # Check operation times
            for operation, op_time in result["operation_times_ms"].items():
                if op_time > self._health_thresholds["response_time_warning_ms"]:
                    result["warnings"].append(f"Slow {operation.upper()} operation: {op_time:.0f}ms")

        except Exception as e:
            result["all_operations_successful"] = False
            result["errors"].append(f"Operation test failed: {str(e)}")

            # Clean up test key in case of error
            try:
                await service.delete(test_key)
            except Exception:
                pass  # Ignore cleanup errors

        return result

    async def _check_configuration(self, service_name: str, service: Any) -> dict[str, Any]:
        """Check cache service configuration."""
        result = {
            "valid": True,
            "warnings": [],
            "config_issues": [],
            "recommendations": [],
        }

        try:
            # Check if service has configuration
            if hasattr(service, "config"):
                config = service.config

                # Check TTL configuration
                if hasattr(config, "default_ttl"):
                    if config.default_ttl <= 0:
                        result["warnings"].append("Default TTL is disabled or invalid")
                    elif config.default_ttl < 300:  # Less than 5 minutes
                        result["warnings"].append(f"Short default TTL: {config.default_ttl}s")

                # Check memory limits
                if hasattr(config, "max_memory_mb"):
                    if config.max_memory_mb <= 0:
                        result["warnings"].append("No memory limit configured")
                    elif config.max_memory_mb < 50:  # Less than 50MB
                        result["warnings"].append(f"Low memory limit: {config.max_memory_mb}MB")

                # Check health check interval
                if hasattr(config, "health_check_interval"):
                    if config.health_check_interval > 300:  # More than 5 minutes
                        result["warnings"].append(f"Long health check interval: {config.health_check_interval}s")

                # Check connection pool settings
                if hasattr(config, "redis_pool_size"):
                    if config.redis_pool_size < 5:
                        result["warnings"].append(f"Small connection pool: {config.redis_pool_size} connections")
                    elif config.redis_pool_size > 50:
                        result["warnings"].append(f"Large connection pool: {config.redis_pool_size} connections")

            else:
                result["warnings"].append("No configuration object available")

        except Exception as e:
            result["valid"] = False
            result["config_issues"].append(f"Configuration check failed: {str(e)}")

        return result

    def _generate_recommendations(self, health_results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on health check results."""
        recommendations = []

        # Performance recommendations
        performance = health_results.get("checks", {}).get("performance", {})
        if performance.get("hit_rate", 1.0) < 0.6:
            recommendations.append("Consider reviewing cache key generation and TTL settings to improve hit rate")

        if performance.get("avg_response_time_ms", 0) > 500:
            recommendations.append("Consider optimizing cache operations or scaling cache infrastructure")

        if performance.get("error_rate", 0) > 0.01:
            recommendations.append("Investigate and resolve cache operation errors")

        # Memory recommendations
        memory = health_results.get("checks", {}).get("memory", {})
        if memory.get("current_size_mb", 0) > 100:
            recommendations.append("Monitor cache memory usage and consider implementing more aggressive eviction policies")

        if memory.get("memory_pressure_events", 0) > 0:
            recommendations.append("Increase cache memory limits or optimize cached data size")

        # Connectivity recommendations
        connectivity = health_results.get("checks", {}).get("connectivity", {})
        if not connectivity.get("healthy", True):
            recommendations.append("Check cache service connectivity and Redis server health")

        if connectivity.get("ping_time_ms", 0) > 100:
            recommendations.append("Investigate network latency or Redis server performance")

        return recommendations

    def _aggregate_service_metrics(self, service_name: str) -> dict[str, Any]:
        """Aggregate comprehensive metrics for a cache service."""
        cache_metrics = self.cache_monitor.get_cache_metrics(service_name)
        memory_info = self.memory_monitor.get_detailed_cache_memory_info()

        aggregated = {
            "availability": {
                "uptime_seconds": 0,
                "last_operation_time": None,
                "health_status": "unknown",
            },
            "performance": {
                "hit_rate": 0.0,
                "miss_rate": 0.0,
                "error_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "operations_per_second": 0.0,
            },
            "resource_usage": {
                "memory_mb": 0.0,
                "memory_efficiency": 0.0,
                "eviction_rate": 0.0,
            },
            "activity": {
                "total_operations": 0,
                "recent_operations": 0,
                "operations_by_type": {},
            },
        }

        if cache_metrics:
            # Performance metrics
            aggregated["performance"].update(
                {
                    "hit_rate": cache_metrics.get("hit_rate", 0.0),
                    "miss_rate": cache_metrics.get("miss_rate", 0.0),
                    "error_rate": cache_metrics.get("errors", {}).get("rate", 0.0),
                    "avg_response_time_ms": cache_metrics.get("performance", {}).get("average_response_time_ms", 0.0),
                }
            )

            # Activity metrics
            operations = cache_metrics.get("operations", {})
            aggregated["activity"].update(
                {
                    "total_operations": operations.get("total", 0),
                    "operations_by_type": {
                        "get": operations.get("get", 0),
                        "set": operations.get("set", 0),
                        "delete": operations.get("delete", 0),
                        "exists": operations.get("exists", 0),
                    },
                }
            )

            # Availability metrics
            timestamps = cache_metrics.get("timestamps", {})
            if timestamps.get("uptime_seconds"):
                aggregated["availability"]["uptime_seconds"] = timestamps["uptime_seconds"]
            if timestamps.get("last_operation_time"):
                aggregated["availability"]["last_operation_time"] = timestamps["last_operation_time"]

        # Memory metrics
        cache_data = memory_info.get("by_cache", {}).get(service_name, {})
        if cache_data:
            current_mb = cache_data.get("current_size_mb", 0.0)
            operations = aggregated["activity"]["total_operations"]

            aggregated["resource_usage"].update(
                {
                    "memory_mb": current_mb,
                    "memory_efficiency": operations / max(current_mb, 0.1),  # operations per MB
                    "eviction_rate": cache_data.get("eviction_count", 0) / max(operations, 1),
                }
            )

        return aggregated

    async def check_all_cache_services(self) -> dict[str, Any]:
        """
        Perform health checks on all registered cache services.

        Returns:
            Dictionary containing health check results for all services
        """
        start_time = time.time()
        results = {
            "overall_status": "ok",
            "services_checked": len(self._cache_services),
            "healthy_services": 0,
            "warning_services": 0,
            "error_services": 0,
            "service_results": {},
            "summary": {},
            "recommendations": [],
            "timestamp": datetime.now().isoformat(),
        }

        if not self._cache_services:
            results["overall_status"] = "warning"
            results["summary"]["message"] = "No cache services registered for health checking"
            results["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)
            return results

        # Check each service
        for service_name in self._cache_services:
            try:
                service_result = await self.check_cache_service_health(service_name)
                results["service_results"][service_name] = service_result

                # Count service statuses
                if service_result["status"] == "ok":
                    results["healthy_services"] += 1
                elif service_result["status"] == "warning":
                    results["warning_services"] += 1
                else:
                    results["error_services"] += 1

            except Exception as e:
                logger.error(f"Failed to check health of cache service '{service_name}': {e}")
                results["service_results"][service_name] = {
                    "service_name": service_name,
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
                results["error_services"] += 1

        # Determine overall status
        if results["error_services"] > 0:
            results["overall_status"] = "error"
        elif results["warning_services"] > 0:
            results["overall_status"] = "warning"

        # Generate summary
        results["summary"] = {
            "message": self._generate_overall_summary(results),
            "service_breakdown": {
                "healthy": results["healthy_services"],
                "warning": results["warning_services"],
                "error": results["error_services"],
                "total": results["services_checked"],
            },
            "health_percentage": round((results["healthy_services"] / max(results["services_checked"], 1)) * 100, 1),
        }

        # Generate overall recommendations
        results["recommendations"] = self._generate_overall_recommendations(results)

        results["check_duration_ms"] = round((time.time() - start_time) * 1000, 2)
        return results

    def _generate_overall_summary(self, results: dict[str, Any]) -> str:
        """Generate overall summary message."""
        healthy = results["healthy_services"]
        warning = results["warning_services"]
        error = results["error_services"]
        total = results["services_checked"]

        if error > 0:
            return f"{error} of {total} cache services have critical issues"
        elif warning > 0:
            return f"{warning} of {total} cache services have warnings, {healthy} are healthy"
        else:
            return f"All {total} cache services are healthy"

    def _generate_overall_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate overall system recommendations."""
        recommendations = []

        # Service-specific recommendations
        for service_name, service_result in results["service_results"].items():
            service_recommendations = service_result.get("recommendations", [])
            for rec in service_recommendations:
                recommendations.append(f"{service_name}: {rec}")

        # System-wide recommendations
        if results["error_services"] > 0:
            recommendations.append("Investigate and resolve cache service errors immediately")

        if results["warning_services"] / max(results["services_checked"], 1) > 0.5:
            recommendations.append("More than half of cache services have warnings - review overall cache strategy")

        if results["services_checked"] == 0:
            recommendations.append("Register cache services for comprehensive health monitoring")

        return list(set(recommendations))  # Remove duplicates


# Global instance
_cache_health_checker = CacheHealthChecker()


def get_cache_health_checker() -> CacheHealthChecker:
    """Get the global cache health checker instance."""
    return _cache_health_checker


async def check_individual_cache_health(service_name: str) -> dict[str, Any]:
    """
    Convenience function to check health of a specific cache service.

    Args:
        service_name: Name of the cache service to check

    Returns:
        Health check results for the specific service
    """
    checker = get_cache_health_checker()
    return await checker.check_cache_service_health(service_name)


async def check_all_caches_health() -> dict[str, Any]:
    """
    Convenience function to check health of all registered cache services.

    Returns:
        Health check results for all services
    """
    checker = get_cache_health_checker()
    return await checker.check_all_cache_services()
