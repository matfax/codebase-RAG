"""
Telemetry management tool for MCP server.

This module provides tools to configure, monitor, and manage OpenTelemetry
integration for distributed tracing and observability.
"""

import logging
import time
from typing import Any, Optional

from src.utils.telemetry import (
    OPENTELEMETRY_AVAILABLE,
    TelemetryConfig,
    configure_telemetry_from_dict,
    get_telemetry_manager,
    get_telemetry_status,
)

logger = logging.getLogger(__name__)


async def get_telemetry_status_info() -> dict[str, Any]:
    """
    Get current telemetry status and configuration.

    Returns:
        Dict containing telemetry status and configuration information.
    """
    try:
        status = get_telemetry_status()
        telemetry_manager = get_telemetry_manager()

        # Get additional runtime information
        runtime_info = {
            "tracer_available": telemetry_manager.tracer is not None,
            "meter_available": telemetry_manager.meter is not None,
            "cache_metrics_created": all(
                [
                    telemetry_manager.cache_operation_counter is not None,
                    telemetry_manager.cache_hit_counter is not None,
                    telemetry_manager.cache_miss_counter is not None,
                    telemetry_manager.cache_error_counter is not None,
                    telemetry_manager.cache_response_time_histogram is not None,
                    telemetry_manager.cache_size_gauge is not None,
                    telemetry_manager.cache_eviction_counter is not None,
                ]
            ),
        }

        return {
            "status": "ok",
            "message": "Telemetry status retrieved successfully",
            "telemetry_status": status,
            "runtime_info": runtime_info,
            "dependencies": {
                "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
                "required_packages": [
                    "opentelemetry-api",
                    "opentelemetry-sdk",
                    "opentelemetry-exporter-otlp",
                    "opentelemetry-exporter-jaeger",
                    "opentelemetry-exporter-zipkin",
                    "opentelemetry-instrumentation-requests",
                    "opentelemetry-instrumentation-urllib3",
                    "opentelemetry-instrumentation-redis",
                ],
            },
        }

    except Exception as e:
        logger.error(f"Failed to get telemetry status: {e}")
        return {
            "status": "error",
            "message": f"Failed to get telemetry status: {str(e)}",
            "error": str(e),
        }


async def configure_telemetry(
    service_name: str | None = None,
    service_version: str | None = None,
    service_namespace: str | None = None,
    tracing_enabled: bool | None = None,
    trace_exporter: str | None = None,
    metrics_enabled: bool | None = None,
    metric_exporter: str | None = None,
    deployment_environment: str | None = None,
    trace_sample_rate: float | None = None,
    **additional_config,
) -> dict[str, Any]:
    """
    Configure telemetry settings.

    Args:
        service_name: Name of the service for telemetry
        service_version: Version of the service
        service_namespace: Namespace for the service
        tracing_enabled: Enable/disable tracing
        trace_exporter: Trace exporter type ("console", "jaeger", "zipkin", "otlp")
        metrics_enabled: Enable/disable metrics
        metric_exporter: Metric exporter type ("console", "otlp")
        deployment_environment: Deployment environment
        trace_sample_rate: Trace sampling rate (0.0 to 1.0)
        **additional_config: Additional configuration parameters

    Returns:
        Dict containing configuration result.
    """
    try:
        # Get current configuration
        current_manager = get_telemetry_manager()
        current_config = current_manager.config

        # Build new configuration
        config_dict = {
            "service_name": service_name or current_config.service_name,
            "service_version": service_version or current_config.service_version,
            "service_namespace": service_namespace or current_config.service_namespace,
            "tracing_enabled": tracing_enabled if tracing_enabled is not None else current_config.tracing_enabled,
            "trace_exporter": trace_exporter or current_config.trace_exporter,
            "metrics_enabled": metrics_enabled if metrics_enabled is not None else current_config.metrics_enabled,
            "metric_exporter": metric_exporter or current_config.metric_exporter,
            "deployment_environment": deployment_environment or current_config.deployment_environment,
            "trace_sample_rate": trace_sample_rate if trace_sample_rate is not None else current_config.trace_sample_rate,
            **additional_config,
        }

        # Validate configuration
        validation_errors = []

        if config_dict["trace_exporter"] not in ["console", "jaeger", "zipkin", "otlp"]:
            validation_errors.append(f"Invalid trace exporter: {config_dict['trace_exporter']}")

        if config_dict["metric_exporter"] not in ["console", "otlp"]:
            validation_errors.append(f"Invalid metric exporter: {config_dict['metric_exporter']}")

        if not (0.0 <= config_dict["trace_sample_rate"] <= 1.0):
            validation_errors.append(f"Invalid trace sample rate: {config_dict['trace_sample_rate']}")

        if validation_errors:
            return {
                "status": "error",
                "message": "Configuration validation failed",
                "errors": validation_errors,
            }

        # Apply new configuration
        new_manager = configure_telemetry_from_dict(config_dict)

        return {
            "status": "ok",
            "message": "Telemetry configuration updated successfully",
            "previous_config": {
                "service_name": current_config.service_name,
                "tracing_enabled": current_config.tracing_enabled,
                "metrics_enabled": current_config.metrics_enabled,
                "trace_exporter": current_config.trace_exporter,
                "metric_exporter": current_config.metric_exporter,
            },
            "new_config": config_dict,
            "telemetry_enabled": new_manager.is_enabled(),
            "requires_restart": "Some changes may require service restart to take full effect",
        }

    except Exception as e:
        logger.error(f"Failed to configure telemetry: {e}")
        return {
            "status": "error",
            "message": f"Failed to configure telemetry: {str(e)}",
            "error": str(e),
        }


async def test_telemetry_exporters() -> dict[str, Any]:
    """
    Test configured telemetry exporters.

    Returns:
        Dict containing test results for each exporter.
    """
    try:
        if not OPENTELEMETRY_AVAILABLE:
            return {
                "status": "error",
                "message": "OpenTelemetry is not available",
                "available": False,
            }

        telemetry_manager = get_telemetry_manager()

        if not telemetry_manager.is_enabled():
            return {
                "status": "warning",
                "message": "Telemetry is not enabled",
                "enabled": False,
            }

        test_results = {
            "tracing": {"tested": False, "success": False, "error": None},
            "metrics": {"tested": False, "success": False, "error": None},
        }

        # Test tracing
        if telemetry_manager.config.tracing_enabled and telemetry_manager.tracer:
            try:
                # Create a test span
                test_span = telemetry_manager.create_span(
                    name="telemetry_test",
                    attributes={
                        "test.type": "exporter_test",
                        "test.component": "tracing",
                        "test.timestamp": time.time(),
                    },
                )

                with test_span:
                    # Add some test events
                    if hasattr(test_span, "add_event"):
                        test_span.add_event("test_event", {"event.type": "test"})

                    # Simulate some work
                    time.sleep(0.001)

                test_results["tracing"]["tested"] = True
                test_results["tracing"]["success"] = True

            except Exception as e:
                test_results["tracing"]["tested"] = True
                test_results["tracing"]["error"] = str(e)

        # Test metrics
        if telemetry_manager.config.metrics_enabled and telemetry_manager.meter:
            try:
                # Record test metrics
                telemetry_manager.record_cache_operation(
                    operation="test",
                    cache_name="test_cache",
                    hit=True,
                    response_time=0.001,
                    cache_size=1024,
                    additional_attributes={"test.metric": "true"},
                )

                telemetry_manager.record_cache_eviction(
                    cache_name="test_cache", eviction_reason="test", items_evicted=1, additional_attributes={"test.eviction": "true"}
                )

                test_results["metrics"]["tested"] = True
                test_results["metrics"]["success"] = True

            except Exception as e:
                test_results["metrics"]["tested"] = True
                test_results["metrics"]["error"] = str(e)

        # Determine overall result
        overall_success = (test_results["tracing"]["success"] if test_results["tracing"]["tested"] else True) and (
            test_results["metrics"]["success"] if test_results["metrics"]["tested"] else True
        )

        return {
            "status": "ok" if overall_success else "error",
            "message": "Telemetry exporter test completed",
            "test_results": test_results,
            "overall_success": overall_success,
            "configuration": {
                "trace_exporter": telemetry_manager.config.trace_exporter,
                "metric_exporter": telemetry_manager.config.metric_exporter,
                "tracing_enabled": telemetry_manager.config.tracing_enabled,
                "metrics_enabled": telemetry_manager.config.metrics_enabled,
            },
        }

    except Exception as e:
        logger.error(f"Failed to test telemetry exporters: {e}")
        return {
            "status": "error",
            "message": f"Failed to test telemetry exporters: {str(e)}",
            "error": str(e),
        }


async def get_telemetry_metrics_summary() -> dict[str, Any]:
    """
    Get a summary of telemetry metrics collected.

    Returns:
        Dict containing metrics summary.
    """
    try:
        telemetry_manager = get_telemetry_manager()

        if not telemetry_manager.is_enabled():
            return {
                "status": "warning",
                "message": "Telemetry is not enabled",
                "enabled": False,
            }

        # Get performance monitor data to supplement telemetry
        from src.utils.performance_monitor import get_cache_performance_monitor

        cache_monitor = get_cache_performance_monitor()
        cache_metrics = cache_monitor.get_aggregated_metrics()

        return {
            "status": "ok",
            "message": "Telemetry metrics summary retrieved successfully",
            "telemetry_enabled": telemetry_manager.is_enabled(),
            "metrics_collection": {
                "cache_operations": "Tracking cache operations (get, set, delete, etc.)",
                "cache_hits_misses": "Tracking cache hit/miss ratios",
                "cache_response_times": "Tracking operation response times",
                "cache_sizes": "Tracking cache memory usage",
                "cache_evictions": "Tracking cache eviction events",
                "cache_errors": "Tracking cache errors and failures",
            },
            "instrumentation": {
                "http_requests": telemetry_manager.config.auto_instrument_requests,
                "redis_operations": telemetry_manager.config.auto_instrument_redis,
                "custom_cache_operations": True,
            },
            "exporters": {
                "trace_exporter": telemetry_manager.config.trace_exporter,
                "metric_exporter": telemetry_manager.config.metric_exporter,
                "metrics_interval": telemetry_manager.config.metrics_interval,
            },
            "cache_performance_data": cache_metrics.get("summary", {}) if cache_metrics else {},
            "observability_platforms": {
                "jaeger": "Supported for distributed tracing",
                "zipkin": "Supported for distributed tracing",
                "otlp": "Supported for traces and metrics",
                "prometheus": "Supported via OTLP metrics",
                "grafana": "Supported via OTLP metrics",
            },
        }

    except Exception as e:
        logger.error(f"Failed to get telemetry metrics summary: {e}")
        return {
            "status": "error",
            "message": f"Failed to get telemetry metrics summary: {str(e)}",
            "error": str(e),
        }


async def export_telemetry_configuration() -> dict[str, Any]:
    """
    Export current telemetry configuration.

    Returns:
        Dict containing telemetry configuration.
    """
    try:
        telemetry_manager = get_telemetry_manager()
        config = telemetry_manager.config

        config_dict = {
            "service_name": config.service_name,
            "service_version": config.service_version,
            "service_namespace": config.service_namespace,
            "tracing_enabled": config.tracing_enabled,
            "trace_exporter": config.trace_exporter,
            "jaeger_endpoint": config.jaeger_endpoint,
            "zipkin_endpoint": config.zipkin_endpoint,
            "otlp_endpoint": config.otlp_endpoint,
            "metrics_enabled": config.metrics_enabled,
            "metric_exporter": config.metric_exporter,
            "metrics_interval": config.metrics_interval,
            "auto_instrument_requests": config.auto_instrument_requests,
            "auto_instrument_redis": config.auto_instrument_redis,
            "trace_sample_rate": config.trace_sample_rate,
            "deployment_environment": config.deployment_environment,
            "resource_attributes": config.resource_attributes,
        }

        return {
            "status": "ok",
            "message": "Telemetry configuration exported successfully",
            "configuration": config_dict,
            "telemetry_enabled": telemetry_manager.is_enabled(),
            "initialized": telemetry_manager.initialized,
            "export_timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Failed to export telemetry configuration: {e}")
        return {
            "status": "error",
            "message": f"Failed to export telemetry configuration: {str(e)}",
            "error": str(e),
        }


async def get_telemetry_health() -> dict[str, Any]:
    """
    Get telemetry system health status.

    Returns:
        Dict containing telemetry health information.
    """
    try:
        telemetry_manager = get_telemetry_manager()

        health_info = {
            "overall_status": "ok",
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check OpenTelemetry availability
        if not OPENTELEMETRY_AVAILABLE:
            health_info["overall_status"] = "error"
            health_info["issues"].append("OpenTelemetry packages are not installed")
            health_info["recommendations"].append("Install OpenTelemetry packages: pip install opentelemetry-api opentelemetry-sdk")

        # Check telemetry initialization
        if not telemetry_manager.initialized:
            health_info["overall_status"] = "error"
            health_info["issues"].append("Telemetry system is not initialized")
            health_info["recommendations"].append("Check telemetry configuration and initialization")

        # Check if telemetry is enabled but not configured
        if telemetry_manager.config.tracing_enabled and not telemetry_manager.tracer:
            health_info["overall_status"] = "warning"
            health_info["warnings"].append("Tracing is enabled but tracer is not available")
            health_info["recommendations"].append("Check trace exporter configuration")

        if telemetry_manager.config.metrics_enabled and not telemetry_manager.meter:
            health_info["overall_status"] = "warning"
            health_info["warnings"].append("Metrics are enabled but meter is not available")
            health_info["recommendations"].append("Check metric exporter configuration")

        # Check cache metrics
        if telemetry_manager.meter:
            cache_metrics_available = all(
                [
                    telemetry_manager.cache_operation_counter is not None,
                    telemetry_manager.cache_hit_counter is not None,
                    telemetry_manager.cache_miss_counter is not None,
                    telemetry_manager.cache_error_counter is not None,
                    telemetry_manager.cache_response_time_histogram is not None,
                    telemetry_manager.cache_size_gauge is not None,
                    telemetry_manager.cache_eviction_counter is not None,
                ]
            )

            if not cache_metrics_available:
                health_info["overall_status"] = "warning"
                health_info["warnings"].append("Some cache metrics are not available")
                health_info["recommendations"].append("Check cache metrics initialization")

        return {
            "status": "ok",
            "message": "Telemetry health check completed",
            "health": health_info,
            "system_info": {
                "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
                "telemetry_enabled": telemetry_manager.is_enabled(),
                "initialized": telemetry_manager.initialized,
                "tracing_configured": telemetry_manager.tracer is not None,
                "metrics_configured": telemetry_manager.meter is not None,
            },
            "configuration": {
                "tracing_enabled": telemetry_manager.config.tracing_enabled,
                "metrics_enabled": telemetry_manager.config.metrics_enabled,
                "trace_exporter": telemetry_manager.config.trace_exporter,
                "metric_exporter": telemetry_manager.config.metric_exporter,
                "service_name": telemetry_manager.config.service_name,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get telemetry health: {e}")
        return {
            "status": "error",
            "message": f"Failed to get telemetry health: {str(e)}",
            "error": str(e),
        }
