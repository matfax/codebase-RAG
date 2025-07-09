"""Health check tool implementation.

This module provides health check functionality for the MCP server.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any

from tools.core.memory_utils import get_memory_stats, get_memory_usage_mb
from utils.performance_monitor import MemoryMonitor, get_cache_performance_monitor

# Import cache health checker (with fallback if not available)
try:
    from tools.core.cache_health_checks import check_all_caches_health, get_cache_health_checker

    CACHE_HEALTH_CHECKS_AVAILABLE = True
except ImportError:
    CACHE_HEALTH_CHECKS_AVAILABLE = False

# Import cache alert service (with fallback if not available)
try:
    from services.cache_alert_service import get_cache_alert_service

    CACHE_ALERT_SERVICE_AVAILABLE = True
except ImportError:
    CACHE_ALERT_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)


async def health_check() -> dict[str, Any]:
    """
    Check the health of the MCP server and its dependencies.

    This is the async version for MCP tool registration.

    Returns:
        Dict[str, Any]: Health status information including:
            - status: Overall health status ("ok", "warning", "error")
            - message: Human-readable status message
            - services: Individual service health statuses
            - memory: Memory usage information
            - cache_health: Detailed cache health check results
            - timestamp: ISO format timestamp
    """
    # Get the synchronous health check results
    sync_results = health_check_sync()

    # Add detailed cache health checks if available
    if CACHE_HEALTH_CHECKS_AVAILABLE:
        try:
            cache_health_results = await check_all_caches_health()
            sync_results["cache_health"] = cache_health_results

            # Update overall status based on cache health
            if cache_health_results["overall_status"] == "error" and sync_results["status"] != "error":
                sync_results["status"] = "error"
            elif cache_health_results["overall_status"] == "warning" and sync_results["status"] == "ok":
                sync_results["status"] = "warning"

            # Add cache issues to overall warnings/issues
            if "warnings" not in sync_results:
                sync_results["warnings"] = []
            if "issues" not in sync_results:
                sync_results["issues"] = []

            # Add cache summary to message
            cache_summary = cache_health_results.get("summary", {}).get("message", "")
            if cache_summary:
                if sync_results["status"] == "ok":
                    sync_results["message"] += f". Cache services: {cache_summary}"
                else:
                    sync_results["message"] += f". Cache: {cache_summary}"

        except Exception as e:
            logger.error(f"Failed to run detailed cache health checks: {e}")
            sync_results["cache_health"] = {
                "error": f"Detailed cache health check failed: {str(e)}",
                "overall_status": "error",
                "timestamp": datetime.now().isoformat(),
            }

    return sync_results


def health_check_sync() -> dict[str, Any]:
    """
    Synchronous health check of the MCP server and its dependencies.

    Checks:
        - Qdrant database connectivity and performance
        - Ollama service availability
        - Memory usage status
        - System resources

    Returns:
        Dict[str, Any]: Comprehensive health status information
    """
    services_status = {}
    overall_status = "ok"
    issues = []
    warnings = []

    start_time = time.time()

    # Check Qdrant connection
    try:
        from qdrant_client import QdrantClient

        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))

        # Create client with connection check
        try:
            client = QdrantClient(host=host, port=port)

            # Import database utils here to avoid circular imports
            from ..database.qdrant_utils import (
                check_qdrant_health,
                retry_qdrant_operation,
            )

            # Use retry wrapper for health check
            qdrant_status = retry_qdrant_operation(
                lambda: check_qdrant_health(client),
                "Qdrant health check",
                max_retries=1,  # Quick check, don't retry too much
            )

            services_status["qdrant"] = qdrant_status

            if not qdrant_status["healthy"]:
                overall_status = "error"
                issues.append("Qdrant database is not healthy")
            elif qdrant_status.get("response_time_ms", 0) > 500:
                warnings.append(f"Qdrant response time is slow: {qdrant_status['response_time_ms']:.0f}ms")
                if overall_status == "ok":
                    overall_status = "warning"

        except Exception as e:
            services_status["qdrant"] = {
                "healthy": False,
                "error": str(e),
                "host": f"{host}:{port}",
                "timestamp": datetime.now().isoformat(),
            }
            overall_status = "error"
            issues.append(f"Cannot connect to Qdrant at {host}:{port}")

    except ImportError:
        services_status["qdrant"] = {
            "healthy": False,
            "error": "Qdrant client not installed",
            "timestamp": datetime.now().isoformat(),
        }
        overall_status = "error"
        issues.append("Qdrant client library not installed")

    # Check Ollama connection
    try:
        import requests

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Basic connectivity check with timeout
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            response_time = response.elapsed.total_seconds() * 1000

            if response.status_code == 200:
                services_status["ollama"] = {
                    "healthy": True,
                    "status": "accessible",
                    "host": ollama_host,
                    "response_time_ms": response_time,
                    "timestamp": datetime.now().isoformat(),
                }

                # Check for slow response
                if response_time > 1000:
                    warnings.append(f"Ollama response time is slow: {response_time:.0f}ms")
                    if overall_status == "ok":
                        overall_status = "warning"
            else:
                services_status["ollama"] = {
                    "healthy": False,
                    "error": f"Unexpected status code: {response.status_code}",
                    "host": ollama_host,
                    "timestamp": datetime.now().isoformat(),
                }
                overall_status = "error"
                issues.append(f"Ollama returned status code {response.status_code}")

        except requests.RequestException as e:
            services_status["ollama"] = {
                "healthy": False,
                "error": f"Connection failed: {str(e)}",
                "host": ollama_host,
                "timestamp": datetime.now().isoformat(),
            }
            overall_status = "error"
            issues.append(f"Cannot connect to Ollama at {ollama_host}")

    except ImportError:
        services_status["ollama"] = {
            "healthy": False,
            "error": "requests library not installed",
            "timestamp": datetime.now().isoformat(),
        }
        overall_status = "error"
        issues.append("requests library not installed")

    # Check memory usage
    try:
        memory_stats = get_memory_stats()
        memory_mb = memory_stats["process_memory_mb"]
        memory_threshold = float(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))

        memory_info = {
            "current_mb": memory_mb,
            "threshold_mb": memory_threshold,
            "system_memory": memory_stats["system_memory"],
            "healthy": memory_mb < memory_threshold,
        }

        services_status["memory"] = memory_info

        if memory_mb > memory_threshold:
            warnings.append(f"Memory usage is high: {memory_mb:.0f}MB (threshold: {memory_threshold:.0f}MB)")
            if overall_status == "ok":
                overall_status = "warning"

        # Check system memory availability
        system_mem = memory_stats["system_memory"]
        if system_mem.get("percent_used", 0) > 90:
            warnings.append(f"System memory usage is critical: {system_mem['percent_used']:.0f}%")
            if overall_status != "error":
                overall_status = "warning"

    except Exception as e:
        services_status["memory"] = {
            "healthy": True,  # Don't fail health check if we can't get memory stats
            "error": f"Could not get memory stats: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }

    # Check cache performance metrics and alert service
    try:
        cache_monitor = get_cache_performance_monitor()
        cache_metrics = cache_monitor.get_aggregated_metrics()
        cache_alerts = cache_monitor.get_alerts()

        # Get alert service information if available
        alert_service_info = None
        if CACHE_ALERT_SERVICE_AVAILABLE:
            try:
                alert_service = get_cache_alert_service()
                alert_service_info = {
                    "monitoring_enabled": alert_service.monitoring_enabled,
                    "active_alerts": len(alert_service.get_active_alerts()),
                    "notification_channels": len(alert_service.notification_channels),
                    "configured_thresholds": len(alert_service.thresholds),
                    "statistics": alert_service.get_alert_statistics(),
                }
            except Exception as e:
                logger.warning(f"Could not get alert service info: {e}")

        # Determine cache health status
        cache_healthy = True
        cache_warnings = []
        cache_errors = []

        if cache_metrics and "summary" in cache_metrics:
            summary = cache_metrics["summary"]

            # Check overall hit rate
            hit_rate = summary.get("overall_hit_rate", 0.0)
            if hit_rate < 0.5 and summary.get("total_operations", 0) > 100:
                cache_warnings.append(f"Low cache hit rate: {hit_rate:.1%}")
                if overall_status == "ok":
                    overall_status = "warning"

            # Check error rate
            error_rate = summary.get("overall_error_rate", 0.0)
            if error_rate > 0.05:  # 5% error rate threshold
                cache_errors.append(f"High cache error rate: {error_rate:.1%}")
                overall_status = "error"
                cache_healthy = False
            elif error_rate > 0.01:  # 1% warning threshold
                cache_warnings.append(f"Elevated cache error rate: {error_rate:.1%}")
                if overall_status == "ok":
                    overall_status = "warning"

            # Check cache memory usage
            cache_memory_mb = summary.get("total_size_mb", 0.0)
            cache_memory_threshold = float(os.getenv("CACHE_MEMORY_WARNING_THRESHOLD_MB", "500"))
            if cache_memory_mb > cache_memory_threshold:
                cache_warnings.append(f"High cache memory usage: {cache_memory_mb:.1f}MB (threshold: {cache_memory_threshold}MB)")
                if overall_status == "ok":
                    overall_status = "warning"

            # Check response times
            cache_types = cache_metrics.get("cache_type_summary", {})
            for cache_type, type_metrics in cache_types.items():
                avg_response_time = type_metrics.get("average_response_time_ms", 0.0)
                if avg_response_time > 1000:  # 1 second threshold
                    cache_warnings.append(f"Slow {cache_type} cache response time: {avg_response_time:.0f}ms")
                    if overall_status == "ok":
                        overall_status = "warning"

        # Count active alerts
        alert_count = len(cache_alerts)
        critical_alerts = [a for a in cache_alerts if a.get("type") in ["high_error_rate", "high_memory_usage"]]

        if critical_alerts:
            cache_healthy = False
            overall_status = "error" if overall_status != "error" else overall_status
            cache_errors.extend([f"Critical cache alert: {alert['type']}" for alert in critical_alerts[:3]])

        cache_status = {
            "healthy": cache_healthy,
            "total_caches": cache_metrics.get("summary", {}).get("total_caches", 0),
            "total_operations": cache_metrics.get("summary", {}).get("total_operations", 0),
            "overall_hit_rate": cache_metrics.get("summary", {}).get("overall_hit_rate", 0.0),
            "overall_error_rate": cache_metrics.get("summary", {}).get("overall_error_rate", 0.0),
            "total_size_mb": cache_metrics.get("summary", {}).get("total_size_mb", 0.0),
            "active_alerts": alert_count,
            "monitoring_enabled": cache_monitor.is_monitoring_enabled(),
            "alert_service": alert_service_info,
            "warnings": cache_warnings if cache_warnings else None,
            "errors": cache_errors if cache_errors else None,
            "timestamp": datetime.now().isoformat(),
        }

        services_status["cache_performance"] = cache_status

        # Add cache warnings and errors to overall health check
        if cache_warnings:
            warnings.extend(cache_warnings)
        if cache_errors:
            issues.extend(cache_errors)

    except Exception as e:
        services_status["cache_performance"] = {
            "healthy": True,  # Don't fail health check if cache monitoring isn't available
            "error": f"Could not get cache metrics: {str(e)}",
            "monitoring_enabled": False,
            "timestamp": datetime.now().isoformat(),
        }

    # Calculate total check time
    total_check_time = (time.time() - start_time) * 1000

    # Build comprehensive message
    if overall_status == "ok":
        message = "All services are operational"
    elif overall_status == "warning":
        message = f"System operational with warnings: {'; '.join(warnings)}"
    else:
        message = f"System has critical issues: {'; '.join(issues)}"
        if warnings:
            message += f". Also: {'; '.join(warnings)}"

    return {
        "status": overall_status,
        "message": message,
        "services": services_status,
        "issues": issues if issues else None,
        "warnings": warnings if warnings else None,
        "dependencies_checked": ["qdrant", "ollama", "memory", "cache_performance"],
        "check_duration_ms": total_check_time,
        "timestamp": datetime.now().isoformat(),
    }


def cache_health_check() -> dict[str, Any]:
    """
    Detailed cache-specific health check with comprehensive metrics.

    Returns:
        Dict[str, Any]: Comprehensive cache health status
    """
    try:
        cache_monitor = get_cache_performance_monitor()
        memory_monitor = MemoryMonitor()

        # Get comprehensive cache metrics
        aggregated_metrics = cache_monitor.get_aggregated_metrics()
        cache_type_metrics = cache_monitor.get_all_cache_type_metrics()
        all_cache_metrics = cache_monitor.get_all_cache_metrics()
        cache_alerts = cache_monitor.get_alerts()

        # Get detailed memory information
        cache_memory_info = memory_monitor.get_detailed_cache_memory_info()
        cache_memory_alerts = memory_monitor.get_cache_memory_alerts()

        # Determine overall cache health
        overall_status = "ok"
        issues = []
        warnings = []
        recommendations = []

        if aggregated_metrics and "summary" in aggregated_metrics:
            summary = aggregated_metrics["summary"]

            # Evaluate cache connectivity and availability
            total_caches = summary.get("total_caches", 0)
            if total_caches == 0:
                overall_status = "warning"
                warnings.append("No active caches detected")
                recommendations.append("Verify cache services are properly initialized")

            # Evaluate performance metrics
            total_operations = summary.get("total_operations", 0)
            if total_operations > 0:
                hit_rate = summary.get("overall_hit_rate", 0.0)
                error_rate = summary.get("overall_error_rate", 0.0)
                avg_response_time = summary.get("average_response_time_ms", 0.0)

                # Hit rate analysis
                if hit_rate < 0.3:
                    overall_status = "error" if overall_status != "error" else overall_status
                    issues.append(f"Critical cache hit rate: {hit_rate:.1%}")
                    recommendations.append("Review cache strategy and TTL settings")
                elif hit_rate < 0.6:
                    if overall_status == "ok":
                        overall_status = "warning"
                    warnings.append(f"Low cache hit rate: {hit_rate:.1%}")
                    recommendations.append("Consider optimizing cache keys and expiration policies")

                # Error rate analysis
                if error_rate > 0.1:  # 10% critical threshold
                    overall_status = "error"
                    issues.append(f"Critical cache error rate: {error_rate:.1%}")
                    recommendations.append("Investigate cache connectivity and error handling")
                elif error_rate > 0.05:  # 5% warning threshold
                    if overall_status == "ok":
                        overall_status = "warning"
                    warnings.append(f"Elevated cache error rate: {error_rate:.1%}")
                    recommendations.append("Monitor cache stability and connections")

                # Response time analysis
                if avg_response_time > 2000:  # 2 seconds critical
                    overall_status = "error" if overall_status != "error" else overall_status
                    issues.append(f"Critical cache response time: {avg_response_time:.0f}ms")
                    recommendations.append("Investigate cache performance bottlenecks")
                elif avg_response_time > 1000:  # 1 second warning
                    if overall_status == "ok":
                        overall_status = "warning"
                    warnings.append(f"Slow cache response time: {avg_response_time:.0f}ms")
                    recommendations.append("Consider cache optimization or resource scaling")

            # Memory usage analysis
            total_memory_mb = summary.get("total_size_mb", 0.0)
            memory_threshold = float(os.getenv("CACHE_MEMORY_CRITICAL_THRESHOLD_MB", "1000"))

            if total_memory_mb > memory_threshold:
                overall_status = "error" if overall_status != "error" else overall_status
                issues.append(f"Critical cache memory usage: {total_memory_mb:.1f}MB")
                recommendations.append("Implement aggressive eviction policies or increase memory limits")
            elif total_memory_mb > memory_threshold * 0.8:  # 80% of threshold
                if overall_status == "ok":
                    overall_status = "warning"
                warnings.append(f"High cache memory usage: {total_memory_mb:.1f}MB")
                recommendations.append("Monitor memory usage and consider cache cleanup")

        # Analyze cache type performance
        cache_type_analysis = {}
        for cache_type, metrics in cache_type_metrics.items():
            type_status = "ok"
            type_issues = []
            type_warnings = []

            hit_rate = metrics.get("hit_rate", 0.0)
            error_rate = metrics.get("error_rate", 0.0)
            operations = metrics.get("operations", {}).get("total", 0)

            if operations > 10:  # Only analyze active caches
                if hit_rate < 0.4:
                    type_status = "warning"
                    type_warnings.append(f"Low hit rate: {hit_rate:.1%}")

                if error_rate > 0.05:
                    type_status = "error"
                    type_issues.append(f"High error rate: {error_rate:.1%}")

            cache_type_analysis[cache_type] = {
                "status": type_status,
                "hit_rate": hit_rate,
                "error_rate": error_rate,
                "operations": operations,
                "memory_mb": metrics.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024,
                "issues": type_issues if type_issues else None,
                "warnings": type_warnings if type_warnings else None,
            }

        # Count alerts by severity
        critical_alerts = [a for a in cache_alerts if a.get("type") in ["high_error_rate", "high_memory_usage"]]
        warning_alerts = [a for a in cache_alerts if a.get("type") in ["low_hit_rate", "slow_response_time"]]

        if critical_alerts:
            overall_status = "error"
            issues.extend([f"Critical alert: {alert['type']}" for alert in critical_alerts[:3]])

        if warning_alerts and overall_status == "ok":
            overall_status = "warning"
            warnings.extend([f"Warning alert: {alert['type']}" for alert in warning_alerts[:3]])

        # Build comprehensive message
        if overall_status == "ok":
            message = "All cache services are performing optimally"
        elif overall_status == "warning":
            message = f"Cache services operational with {len(warnings)} warnings"
        else:
            message = f"Cache services have {len(issues)} critical issues"

        return {
            "status": overall_status,
            "message": message,
            "cache_summary": aggregated_metrics.get("summary", {}),
            "cache_types": cache_type_analysis,
            "individual_caches": {
                name: {
                    "healthy": metrics.get("errors", {}).get("rate", 0) < 0.05,
                    "hit_rate": metrics.get("hit_rate", 0.0),
                    "operations": metrics.get("operations", {}).get("total", 0),
                    "memory_mb": metrics.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024,
                    "avg_response_time_ms": metrics.get("performance", {}).get("average_response_time_ms", 0.0),
                }
                for name, metrics in all_cache_metrics.items()
            },
            "memory_analysis": {
                "total_cache_memory_mb": cache_memory_info.get("cache_memory", {}).get("total_mb", 0.0),
                "cache_memory_alerts": len(cache_memory_alerts),
                "memory_efficiency": cache_memory_info.get("memory_efficiency", {}),
            },
            "alerts": {
                "total": len(cache_alerts),
                "critical": len(critical_alerts),
                "warnings": len(warning_alerts),
                "memory_related": len(cache_memory_alerts),
                "recent_alerts": cache_alerts[-5:] if cache_alerts else [],  # Last 5 alerts
            },
            "connectivity": {
                "monitoring_enabled": cache_monitor.is_monitoring_enabled(),
                "active_caches": len(all_cache_metrics),
                "cache_types_active": len(cache_type_metrics),
            },
            "performance_thresholds": {
                "hit_rate_warning": 0.6,
                "hit_rate_critical": 0.3,
                "error_rate_warning": 0.05,
                "error_rate_critical": 0.1,
                "response_time_warning_ms": 1000,
                "response_time_critical_ms": 2000,
                "memory_critical_mb": memory_threshold,
            },
            "issues": issues if issues else None,
            "warnings": warnings if warnings else None,
            "recommendations": recommendations if recommendations else None,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Cache health check failed: {str(e)}",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def basic_health_check() -> dict[str, Any]:
    """
    Synchronous basic health check without external dependencies.

    This is a minimal health check that only verifies the MCP server
    itself is running, without checking external services.

    Returns:
        Dict[str, Any]: Basic health status
    """
    try:
        # Basic memory check
        memory_mb = get_memory_usage_mb()

        return {
            "status": "ok",
            "message": "MCP server is running",
            "memory_mb": memory_mb,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "ok",  # Server is still running even if we can't get memory
            "message": "MCP server is running",
            "timestamp": datetime.now().isoformat(),
            "note": f"Could not get memory stats: {str(e)}",
        }
