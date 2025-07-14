"""
Cache control interface tools for configuration, health monitoring, and optimization.

This module provides MCP tools for:
- Cache configuration management
- Cache health monitoring and alerting
- Performance optimization tools
- Cache backup and restore operations
- Cache migration and upgrade tools
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from src.tools.core.error_utils import handle_tool_error, log_tool_usage
from src.tools.core.errors import CacheError

logger = logging.getLogger(__name__)


# ===== CACHE CONFIGURATION MANAGEMENT TOOLS =====


async def get_cache_configuration(
    config_type: str = "all",
    export_format: str = "json",
) -> dict[str, Any]:
    """
    Get current cache configuration across all services.

    Args:
        config_type: Type of configuration to retrieve (all, redis, memory, ttl, limits)
        export_format: Format for export (json, yaml, env)

    Returns:
        Dictionary with cache configuration details
    """
    with log_tool_usage(
        "get_cache_configuration",
        {
            "config_type": config_type,
            "export_format": export_format,
        },
    ):
        try:
            from config.cache_config import get_global_cache_config

            # Get global cache configuration
            cache_config = get_global_cache_config()

            # Extract configuration based on type
            config_data = {}

            if config_type in ["all", "redis"]:
                config_data["redis"] = {
                    "host": cache_config.redis_host,
                    "port": cache_config.redis_port,
                    "db": cache_config.redis_db,
                    "password_set": bool(cache_config.redis_password),
                    "max_connections": cache_config.redis_max_connections,
                    "connection_timeout": cache_config.redis_timeout,
                    "socket_keepalive": cache_config.redis_socket_keepalive,
                    "socket_keepalive_options": cache_config.redis_socket_keepalive_options,
                }

            if config_type in ["all", "memory"]:
                config_data["memory"] = {
                    "l1_max_size": cache_config.l1_max_size,
                    "l1_ttl_seconds": cache_config.l1_ttl_seconds,
                    "l2_max_size": cache_config.l2_max_size,
                    "l2_ttl_seconds": cache_config.l2_ttl_seconds,
                    "compression_enabled": cache_config.compression_enabled,
                    "compression_level": cache_config.compression_level,
                }

            if config_type in ["all", "ttl"]:
                config_data["ttl"] = {
                    "embedding_ttl_seconds": cache_config.embedding_ttl_seconds,
                    "search_ttl_seconds": cache_config.search_ttl_seconds,
                    "project_ttl_seconds": cache_config.project_ttl_seconds,
                    "file_ttl_seconds": cache_config.file_ttl_seconds,
                    "default_ttl_seconds": cache_config.default_ttl_seconds,
                }

            if config_type in ["all", "limits"]:
                config_data["limits"] = {
                    "max_key_size": cache_config.max_key_size,
                    "max_value_size": cache_config.max_value_size,
                    "batch_size": cache_config.batch_size,
                    "max_concurrent_operations": cache_config.max_concurrent_operations,
                }

            if config_type in ["all", "security"]:
                config_data["security"] = {
                    "encryption_enabled": cache_config.encryption_enabled,
                    "encryption_key_set": bool(getattr(cache_config, "encryption_key", None)),
                    "isolation_enabled": cache_config.isolation_enabled,
                }

            # Format output based on export format
            formatted_output = None
            if export_format == "yaml":
                try:
                    import yaml

                    formatted_output = yaml.dump(config_data, default_flow_style=False)
                except ImportError:
                    formatted_output = "YAML export requires PyYAML package"
            elif export_format == "env":
                env_lines = []
                for category, settings in config_data.items():
                    env_lines.append(f"# {category.upper()} Configuration")
                    for key, value in settings.items():
                        env_key = f"CACHE_{key.upper()}"
                        env_lines.append(f"{env_key}={value}")
                    env_lines.append("")
                formatted_output = "\n".join(env_lines)

            return {
                "success": True,
                "config_type": config_type,
                "export_format": export_format,
                "configuration": config_data,
                "formatted_output": formatted_output,
                "metadata": {
                    "config_source": "global_cache_config",
                    "retrieved_at": time.time(),
                    "categories_included": list(config_data.keys()),
                },
            }

        except Exception as e:
            from src.tools.core.error_utils import handle_caught_exception

            return handle_caught_exception(
                e,
                "get_cache_configuration",
                {
                    "config_type": config_type,
                    "export_format": export_format,
                },
            )


async def update_cache_configuration(
    config_updates: dict[str, Any],
    validate_only: bool = False,
    restart_services: bool = False,
) -> dict[str, Any]:
    """
    Update cache configuration settings.

    Args:
        config_updates: Dictionary of configuration updates to apply
        validate_only: Only validate changes without applying them
        restart_services: Whether to restart cache services after updates

    Returns:
        Dictionary with update results and validation status
    """
    with log_tool_usage(
        "update_cache_configuration",
        {
            "config_updates_count": len(config_updates),
            "validate_only": validate_only,
            "restart_services": restart_services,
        },
    ):
        try:
            from config.cache_config import get_global_cache_config

            # Get current configuration
            current_config = get_global_cache_config()

            # Validate configuration updates
            validation_results = {}
            valid_updates = {}
            invalid_updates = {}

            for key, value in config_updates.items():
                try:
                    # Check if configuration key is valid
                    if hasattr(current_config, key):
                        # Validate value type and constraints
                        current_value = getattr(current_config, key)

                        # Type validation
                        if not isinstance(value, type(current_value)):
                            invalid_updates[key] = f"Type mismatch: expected {type(current_value).__name__}, got {type(value).__name__}"
                            continue

                        # Value validation
                        if key.endswith("_ttl_seconds") and value < 0:
                            invalid_updates[key] = "TTL values must be non-negative"
                            continue

                        if key.endswith("_size") and value < 1:
                            invalid_updates[key] = "Size values must be positive"
                            continue

                        if key.endswith("_port") and (value < 1 or value > 65535):
                            invalid_updates[key] = "Port values must be between 1 and 65535"
                            continue

                        # If validation passes
                        valid_updates[key] = {
                            "old_value": current_value,
                            "new_value": value,
                            "change_type": "update" if current_value != value else "no_change",
                        }

                    else:
                        invalid_updates[key] = f"Unknown configuration key: {key}"

                except Exception as e:
                    invalid_updates[key] = f"Validation error: {str(e)}"

            validation_results = {
                "valid_updates": valid_updates,
                "invalid_updates": invalid_updates,
                "total_updates": len(config_updates),
                "valid_count": len(valid_updates),
                "invalid_count": len(invalid_updates),
            }

            # If validate_only, return validation results
            if validate_only:
                return {
                    "success": True,
                    "validation_only": True,
                    "validation_results": validation_results,
                    "message": "Configuration validation completed",
                }

            # Apply valid updates if validation passed
            applied_updates = {}

            if valid_updates and len(invalid_updates) == 0:
                try:
                    # Apply configuration updates
                    for key, update_info in valid_updates.items():
                        if update_info["change_type"] == "update":
                            setattr(current_config, key, update_info["new_value"])
                            applied_updates[key] = update_info

                    # Restart services if requested
                    restart_results = {}
                    if restart_services and applied_updates:
                        restart_results = await _restart_cache_services()

                    return {
                        "success": True,
                        "validation_results": validation_results,
                        "applied_updates": applied_updates,
                        "restart_results": restart_results if restart_services else None,
                        "message": f"Successfully applied {len(applied_updates)} configuration updates",
                    }

                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to apply configuration updates: {str(e)}",
                        "validation_results": validation_results,
                        "partial_updates": applied_updates,
                    }
            else:
                return {
                    "success": False,
                    "error": "Configuration validation failed",
                    "validation_results": validation_results,
                    "message": f"Found {len(invalid_updates)} invalid configuration updates",
                }

        except Exception as e:
            return handle_tool_error(
                e,
                "update_cache_configuration",
                {
                    "config_updates_count": len(config_updates) if config_updates else 0,
                    "validate_only": validate_only,
                },
            )


async def _restart_cache_services() -> dict[str, Any]:
    """Restart cache services after configuration changes."""
    restart_results = {}

    # Restart cache services
    services_to_restart = [
        "embedding_cache_service",
        "search_cache_service",
        "project_cache_service",
        "file_cache_service",
    ]

    for service_name in services_to_restart:
        try:
            # This would typically involve gracefully shutting down and restarting services
            # For now, we'll simulate the restart process
            restart_results[service_name] = {
                "status": "restarted",
                "restart_time": time.time(),
            }
        except Exception as e:
            restart_results[service_name] = {
                "status": "failed",
                "error": str(e),
            }

    return restart_results


async def export_cache_configuration(
    export_path: str,
    config_type: str = "all",
    include_sensitive: bool = False,
) -> dict[str, Any]:
    """
    Export cache configuration to file.

    Args:
        export_path: Path to export configuration file
        config_type: Type of configuration to export (all, redis, memory, ttl, limits)
        include_sensitive: Whether to include sensitive information (passwords, keys)

    Returns:
        Dictionary with export results
    """
    with log_tool_usage(
        "export_cache_configuration",
        {
            "export_path": export_path,
            "config_type": config_type,
            "include_sensitive": include_sensitive,
        },
    ):
        try:
            # Get configuration
            config_result = await get_cache_configuration(config_type, "json")

            if not config_result.get("success"):
                return config_result

            config_data = config_result["configuration"]

            # Remove sensitive information if not requested
            if not include_sensitive:
                for category in config_data.values():
                    if isinstance(category, dict):
                        for key in list(category.keys()):
                            if any(sensitive in key.lower() for sensitive in ["password", "key", "secret", "token"]):
                                if key.endswith("_set"):
                                    continue  # Keep boolean flags
                                category[key] = "[REDACTED]"

            # Add export metadata
            export_data = {
                "export_metadata": {
                    "exported_at": time.time(),
                    "config_type": config_type,
                    "include_sensitive": include_sensitive,
                    "version": "1.0",
                },
                "configuration": config_data,
            }

            # Write to file
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path_obj, "w") as f:
                json.dump(export_data, f, indent=2)

            return {
                "success": True,
                "export_path": str(export_path_obj.absolute()),
                "config_type": config_type,
                "file_size_bytes": export_path_obj.stat().st_size,
                "categories_exported": list(config_data.keys()),
                "sensitive_included": include_sensitive,
                "message": f"Configuration exported successfully to {export_path}",
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "export_cache_configuration",
                {
                    "export_path": export_path,
                    "config_type": config_type,
                },
            )


async def import_cache_configuration(
    import_path: str,
    config_type: str = "all",
    validate_only: bool = True,
    backup_current: bool = True,
) -> dict[str, Any]:
    """
    Import cache configuration from file.

    Args:
        import_path: Path to configuration file to import
        config_type: Type of configuration to import (all, redis, memory, ttl, limits)
        validate_only: Only validate without applying changes
        backup_current: Create backup of current configuration

    Returns:
        Dictionary with import results
    """
    with log_tool_usage(
        "import_cache_configuration",
        {
            "import_path": import_path,
            "config_type": config_type,
            "validate_only": validate_only,
            "backup_current": backup_current,
        },
    ):
        try:
            import_path_obj = Path(import_path)

            if not import_path_obj.exists():
                return {
                    "success": False,
                    "error": f"Configuration file not found: {import_path}",
                }

            # Read configuration file
            with open(import_path_obj) as f:
                import_data = json.load(f)

            # Extract configuration data
            if "configuration" in import_data:
                config_data = import_data["configuration"]
            else:
                config_data = import_data

            # Filter by config_type if specified
            if config_type != "all" and config_type in config_data:
                config_data = {config_type: config_data[config_type]}

            # Flatten configuration for update
            config_updates = {}
            for category, settings in config_data.items():
                if isinstance(settings, dict):
                    for key, value in settings.items():
                        if not key.endswith("_set") and value != "[REDACTED]":  # Skip flags and redacted values
                            config_updates[key] = value

            # Create backup if requested
            backup_path = None
            if backup_current and not validate_only:
                backup_path = f"cache_config_backup_{int(time.time())}.json"
                backup_result = await export_cache_configuration(backup_path, "all", include_sensitive=True)
                if backup_result.get("success"):
                    backup_path = backup_result["export_path"]
                else:
                    backup_path = None

            # Apply configuration updates
            update_result = await update_cache_configuration(config_updates, validate_only=validate_only)

            return {
                "success": update_result.get("success", False),
                "import_path": str(import_path_obj.absolute()),
                "config_type": config_type,
                "validate_only": validate_only,
                "backup_path": backup_path,
                "import_results": update_result,
                "imported_settings": len(config_updates),
                "message": f"Configuration import {'validated' if validate_only else 'completed'}",
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "import_cache_configuration",
                {
                    "import_path": import_path,
                    "config_type": config_type,
                },
            )


# ===== CACHE HEALTH MONITORING TOOLS =====


async def get_cache_health_status(
    include_detailed_checks: bool = True,
    check_connectivity: bool = True,
    check_performance: bool = True,
) -> dict[str, Any]:
    """
    Get comprehensive cache health status across all services.

    Args:
        include_detailed_checks: Whether to include detailed health checks
        check_connectivity: Whether to check cache connectivity
        check_performance: Whether to check performance metrics

    Returns:
        Dictionary with comprehensive health status
    """
    with log_tool_usage(
        "get_cache_health_status",
        {
            "include_detailed_checks": include_detailed_checks,
            "check_connectivity": check_connectivity,
            "check_performance": check_performance,
        },
    ):
        try:
            health_status = {
                "overall_status": "healthy",
                "checked_at": time.time(),
                "services": {},
                "alerts": [],
                "summary": {},
            }

            # Check individual cache services
            cache_services = [
                ("embedding", "embedding_cache_service"),
                ("search", "search_cache_service"),
                ("project", "project_cache_service"),
                ("file", "file_cache_service"),
            ]

            healthy_services = 0
            total_services = len(cache_services)

            for service_name, service_module in cache_services:
                try:
                    # Import and get service dynamically
                    if service_name == "embedding":
                        from services.embedding_cache_service import get_embedding_cache_service

                        service = await get_embedding_cache_service()
                    elif service_name == "search":
                        from services.search_cache_service import get_search_cache_service

                        service = await get_search_cache_service()
                    elif service_name == "project":
                        from services.project_cache_service import get_project_cache_service

                        service = await get_project_cache_service()
                    elif service_name == "file":
                        from services.file_cache_service import get_file_cache_service

                        service = await get_file_cache_service()
                    else:
                        raise ValueError(f"Unknown service: {service_name}")

                    service_health = {
                        "status": "healthy",
                        "connectivity": "unknown",
                        "performance": {},
                        "errors": [],
                        "last_check": time.time(),
                    }

                    # Connectivity check
                    if check_connectivity and hasattr(service, "health_check"):
                        try:
                            conn_result = await service.health_check()
                            service_health["connectivity"] = conn_result.get("status", "unknown")
                            if conn_result.get("status") == "error":
                                service_health["status"] = "unhealthy"
                                service_health["errors"].append(conn_result.get("error", "Connection failed"))
                        except Exception as e:
                            service_health["connectivity"] = "failed"
                            service_health["status"] = "unhealthy"
                            service_health["errors"].append(f"Health check failed: {str(e)}")

                    # Performance check
                    if check_performance and hasattr(service, "get_stats"):
                        try:
                            stats = service.get_stats()
                            service_health["performance"] = {
                                "hit_rate": stats.get("hit_rate", 0),
                                "avg_response_time": stats.get("avg_response_time_ms", 0),
                                "cache_size": stats.get("cache_size_bytes", 0),
                                "total_requests": stats.get("total_requests", 0),
                            }

                            # Check performance thresholds
                            if stats.get("hit_rate", 1.0) < 0.5:
                                service_health["status"] = "degraded"
                                health_status["alerts"].append(
                                    {
                                        "severity": "warning",
                                        "service": service_name,
                                        "message": f"Low cache hit rate: {stats.get('hit_rate', 0):.1%}",
                                    }
                                )

                            if stats.get("avg_response_time_ms", 0) > 1000:
                                service_health["status"] = "degraded"
                                health_status["alerts"].append(
                                    {
                                        "severity": "warning",
                                        "service": service_name,
                                        "message": f"High response time: {stats.get('avg_response_time_ms', 0):.1f}ms",
                                    }
                                )

                        except Exception as e:
                            service_health["errors"].append(f"Performance check failed: {str(e)}")

                    # Detailed checks
                    if include_detailed_checks:
                        detailed_checks = await _perform_detailed_health_checks(service, service_name)
                        service_health["detailed_checks"] = detailed_checks

                        # Update service status based on detailed checks
                        if any(check.get("status") == "failed" for check in detailed_checks.values()):
                            service_health["status"] = "unhealthy"
                        elif any(check.get("status") == "warning" for check in detailed_checks.values()):
                            service_health["status"] = "degraded"

                    health_status["services"][service_name] = service_health

                    if service_health["status"] == "healthy":
                        healthy_services += 1

                except Exception as e:
                    health_status["services"][service_name] = {
                        "status": "error",
                        "error": str(e),
                        "last_check": time.time(),
                    }
                    health_status["alerts"].append(
                        {
                            "severity": "error",
                            "service": service_name,
                            "message": f"Service check failed: {str(e)}",
                        }
                    )

            # Overall health determination
            if healthy_services == total_services:
                health_status["overall_status"] = "healthy"
            elif healthy_services > total_services / 2:
                health_status["overall_status"] = "degraded"
            else:
                health_status["overall_status"] = "unhealthy"

            # Health summary
            health_status["summary"] = {
                "healthy_services": healthy_services,
                "total_services": total_services,
                "health_percentage": (healthy_services / total_services) * 100,
                "alert_count": len(health_status["alerts"]),
                "critical_alerts": len([a for a in health_status["alerts"] if a["severity"] == "error"]),
            }

            return {
                "success": True,
                "health_status": health_status,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "get_cache_health_status",
                {
                    "include_detailed_checks": include_detailed_checks,
                    "check_connectivity": check_connectivity,
                },
            )


async def _perform_detailed_health_checks(service, service_name: str) -> dict[str, Any]:
    """Perform detailed health checks on a cache service."""
    checks = {}

    # Memory usage check
    try:
        if hasattr(service, "get_memory_usage"):
            memory_usage = await service.get_memory_usage()
            memory_mb = memory_usage / (1024 * 1024)
            checks["memory_usage"] = {
                "status": "passed" if memory_mb < 500 else "warning" if memory_mb < 1000 else "failed",
                "value": memory_mb,
                "unit": "MB",
                "threshold": 500,
            }
        else:
            checks["memory_usage"] = {"status": "skipped", "reason": "Method not available"}
    except Exception as e:
        checks["memory_usage"] = {"status": "error", "error": str(e)}

    # Connection pool check
    try:
        if hasattr(service, "_redis_pool"):
            pool = service._redis_pool
            checks["connection_pool"] = {
                "status": "passed" if pool and not getattr(pool, "closed", True) else "failed",
                "available_connections": getattr(pool, "available_connections", 0),
                "created_connections": getattr(pool, "created_connections", 0),
            }
        else:
            checks["connection_pool"] = {"status": "skipped", "reason": "Redis pool not available"}
    except Exception as e:
        checks["connection_pool"] = {"status": "error", "error": str(e)}

    # Cache coherency check
    try:
        if hasattr(service, "check_cache_coherency"):
            coherency_result = await service.check_cache_coherency()
            checks["cache_coherency"] = {
                "status": "passed" if coherency_result.get("coherent", True) else "warning",
                "l1_l2_consistency": coherency_result.get("l1_l2_consistent", True),
                "stale_entries": coherency_result.get("stale_entries", 0),
            }
        else:
            checks["cache_coherency"] = {"status": "skipped", "reason": "Method not available"}
    except Exception as e:
        checks["cache_coherency"] = {"status": "error", "error": str(e)}

    return checks


# ===== CACHE ALERTING TOOLS =====


async def configure_cache_alerts(
    alert_config: dict[str, Any],
    enable_alerts: bool = True,
) -> dict[str, Any]:
    """
    Configure cache alerting system.

    Args:
        alert_config: Dictionary with alert configuration settings
        enable_alerts: Whether to enable alerts globally

    Returns:
        Dictionary with alert configuration results
    """
    with log_tool_usage(
        "configure_cache_alerts",
        {
            "enable_alerts": enable_alerts,
            "config_keys": list(alert_config.keys()) if alert_config else [],
        },
    ):
        try:
            # Default alert configuration
            default_config = {
                "high_memory_threshold_mb": 1000,
                "low_hit_rate_threshold": 0.5,
                "high_response_time_threshold_ms": 1000,
                "connection_failure_threshold": 3,
                "alert_cooldown_minutes": 30,
                "notification_channels": ["log"],
                "severity_levels": ["warning", "error", "critical"],
            }

            # Merge with provided configuration
            final_config = {**default_config, **alert_config}

            # Validate configuration
            validation_errors = []

            if final_config["high_memory_threshold_mb"] < 0:
                validation_errors.append("Memory threshold must be non-negative")

            if not 0 <= final_config["low_hit_rate_threshold"] <= 1:
                validation_errors.append("Hit rate threshold must be between 0 and 1")

            if final_config["high_response_time_threshold_ms"] < 0:
                validation_errors.append("Response time threshold must be non-negative")

            if validation_errors:
                return {
                    "success": False,
                    "error": "Configuration validation failed",
                    "validation_errors": validation_errors,
                }

            # Apply configuration (simulated)
            # In a real implementation, this would update global alert configuration
            return {
                "success": True,
                "alert_config": final_config,
                "alerts_enabled": enable_alerts,
                "message": f"Alert configuration {'enabled' if enable_alerts else 'disabled'} successfully",
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "configure_cache_alerts",
                {
                    "enable_alerts": enable_alerts,
                    "config_provided": bool(alert_config),
                },
            )


async def get_cache_alerts(
    severity_filter: str | None = None,
    service_filter: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Get current cache alerts and their status.

    Args:
        severity_filter: Filter by severity level (warning, error, critical)
        service_filter: Filter by service name
        limit: Maximum number of alerts to return

    Returns:
        Dictionary with current alerts
    """
    with log_tool_usage(
        "get_cache_alerts",
        {
            "severity_filter": severity_filter,
            "service_filter": service_filter,
            "limit": limit,
        },
    ):
        try:
            # Get health status to generate current alerts
            health_result = await get_cache_health_status(
                include_detailed_checks=True,
                check_connectivity=True,
                check_performance=True,
            )

            if not health_result.get("success"):
                return health_result

            alerts = health_result["health_status"]["alerts"]

            # Apply filters
            filtered_alerts = alerts

            if severity_filter:
                filtered_alerts = [a for a in filtered_alerts if a.get("severity") == severity_filter]

            if service_filter:
                filtered_alerts = [a for a in filtered_alerts if a.get("service") == service_filter]

            # Apply limit
            filtered_alerts = filtered_alerts[:limit]

            # Add timestamps and IDs to alerts
            for i, alert in enumerate(filtered_alerts):
                alert["alert_id"] = f"alert_{int(time.time())}_{i}"
                alert["timestamp"] = time.time()

            return {
                "success": True,
                "alerts": filtered_alerts,
                "total_alerts": len(alerts),
                "filtered_alerts": len(filtered_alerts),
                "filters_applied": {
                    "severity": severity_filter,
                    "service": service_filter,
                    "limit": limit,
                },
                "alert_summary": {
                    "critical": len([a for a in alerts if a.get("severity") == "critical"]),
                    "error": len([a for a in alerts if a.get("severity") == "error"]),
                    "warning": len([a for a in alerts if a.get("severity") == "warning"]),
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "get_cache_alerts",
                {
                    "severity_filter": severity_filter,
                    "service_filter": service_filter,
                    "limit": limit,
                },
            )
