"""
MCP tools for cache performance dashboards.

This module provides MCP tools for generating and accessing various
cache performance dashboards and visualizations.
"""

import json
import logging
import time
from typing import Any, Literal, Optional

from tools.core.cache_dashboard import (
    generate_cache_detailed_dashboard,
    generate_cache_overview_dashboard,
    generate_cache_real_time_dashboard,
    get_cache_dashboard_generator,
)

logger = logging.getLogger(__name__)


async def cache_dashboard_tool(
    dashboard_type: Literal["overview", "detailed", "real_time"] = "overview",
    cache_name: str | None = None,
    format_type: Literal["json", "summary", "report"] = "json",
    include_charts: bool = True,
) -> dict[str, Any]:
    """
    Generate cache performance dashboards with various views and formats.

    Args:
        dashboard_type: Type of dashboard to generate
        cache_name: Required for detailed dashboard, ignored for others
        format_type: Output format for the dashboard data
        include_charts: Whether to include chart data in response

    Returns:
        Dict containing dashboard data in requested format
    """
    try:
        logger.info(f"Generating {dashboard_type} cache dashboard")

        # Generate the appropriate dashboard
        if dashboard_type == "overview":
            dashboard_data = await generate_cache_overview_dashboard()
        elif dashboard_type == "detailed":
            if not cache_name:
                return {
                    "error": "cache_name is required for detailed dashboard",
                    "available_dashboards": ["overview", "real_time"],
                    "tool_info": {
                        "tool_name": "cache_dashboard",
                        "error_type": "missing_parameter",
                    },
                }
            dashboard_data = await generate_cache_detailed_dashboard(cache_name)
        elif dashboard_type == "real_time":
            dashboard_data = await generate_cache_real_time_dashboard()
        else:
            return {
                "error": f"Invalid dashboard type: {dashboard_type}",
                "valid_types": ["overview", "detailed", "real_time"],
                "tool_info": {
                    "tool_name": "cache_dashboard",
                    "error_type": "invalid_parameter",
                },
            }

        # Process based on format type
        if format_type == "summary":
            result = _format_dashboard_summary(dashboard_data, dashboard_type)
        elif format_type == "report":
            result = _format_dashboard_report(dashboard_data, dashboard_type)
        else:  # json format
            result = dashboard_data

        # Remove chart data if not requested
        if not include_charts and "sections" in result:
            sections = result.get("sections", {})
            if "trend_charts" in sections:
                sections["trend_charts"] = {"note": "Chart data excluded from response"}

        # Add tool metadata
        result["tool_info"] = {
            "tool_name": "cache_dashboard",
            "dashboard_type": dashboard_type,
            "format_type": format_type,
            "cache_name": cache_name,
            "include_charts": include_charts,
        }

        return result

    except Exception as e:
        logger.error(f"Cache dashboard generation failed: {e}")
        return {
            "error": f"Dashboard generation failed: {str(e)}",
            "dashboard_type": dashboard_type,
            "cache_name": cache_name,
            "tool_info": {
                "tool_name": "cache_dashboard",
                "error_type": "generation_error",
            },
        }


async def cache_performance_summary_tool() -> dict[str, Any]:
    """
    Generate a concise cache performance summary suitable for quick status checks.

    Returns:
        Dict containing summarized cache performance data
    """
    try:
        logger.info("Generating cache performance summary")

        # Get overview dashboard data
        dashboard_data = await generate_cache_overview_dashboard()

        if "error" in dashboard_data:
            return {
                "error": dashboard_data["error"],
                "tool_info": {
                    "tool_name": "cache_performance_summary",
                    "error_type": "dashboard_error",
                },
            }

        sections = dashboard_data.get("sections", {})

        # Extract key summary data
        system_summary = sections.get("system_summary", {})
        performance_metrics = sections.get("performance_metrics", {})
        health_status = sections.get("health_status", {})
        memory_usage = sections.get("memory_usage", {})
        alerts_section = sections.get("alerts", {})

        summary = {
            "overall_status": health_status.get("overall_status", "unknown"),
            "cache_count": system_summary.get("total_caches", 0),
            "total_operations": system_summary.get("total_operations", 0),
            "performance": {
                "hit_rate_percentage": performance_metrics.get("overall_performance", {}).get("hit_rate_percentage", 0),
                "error_rate_percentage": performance_metrics.get("overall_performance", {}).get("error_rate_percentage", 0),
                "avg_response_time_ms": performance_metrics.get("overall_performance", {}).get("avg_response_time_ms", 0),
            },
            "health": {
                "healthy_caches": health_status.get("status_summary", {}).get("healthy", 0),
                "warning_caches": health_status.get("status_summary", {}).get("warning", 0),
                "critical_caches": health_status.get("status_summary", {}).get("critical", 0),
            },
            "memory": {
                "total_mb": memory_usage.get("total_cache_memory_mb", 0),
                "efficiency": memory_usage.get("memory_efficiency", {}),
            },
            "alerts": {
                "total": alerts_section.get("total_alerts", 0),
                "critical": alerts_section.get("critical_alerts", 0),
                "warning": alerts_section.get("warning_alerts", 0),
            },
            "top_issues": health_status.get("critical_issues", [])[:3],  # Top 3 issues
            "recommendations": _extract_top_recommendations(sections),
            "tool_info": {
                "tool_name": "cache_performance_summary",
                "generated_at": dashboard_data.get("generated_at"),
            },
        }

        return summary

    except Exception as e:
        logger.error(f"Cache performance summary generation failed: {e}")
        return {
            "error": f"Summary generation failed: {str(e)}",
            "tool_info": {
                "tool_name": "cache_performance_summary",
                "error_type": "generation_error",
            },
        }


async def cache_metrics_export_tool(
    cache_name: str | None = None,
    export_format: Literal["json", "csv", "metrics"] = "json",
    include_historical: bool = False,
) -> dict[str, Any]:
    """
    Export cache metrics data in various formats for external analysis.

    Args:
        cache_name: Optional specific cache to export (all if not specified)
        export_format: Format for exported data
        include_historical: Whether to include historical data

    Returns:
        Dict containing exported metrics data
    """
    try:
        logger.info(f"Exporting cache metrics in {export_format} format")

        generator = get_cache_dashboard_generator()

        if cache_name:
            # Export specific cache metrics
            cache_metrics = generator.cache_monitor.get_cache_metrics(cache_name)
            if not cache_metrics:
                return {
                    "error": f"Cache '{cache_name}' not found",
                    "available_caches": list(generator.cache_monitor.get_all_cache_metrics().keys()),
                    "tool_info": {
                        "tool_name": "cache_metrics_export",
                        "error_type": "cache_not_found",
                    },
                }

            export_data = {
                "cache_name": cache_name,
                "metrics": cache_metrics,
                "memory_info": generator.memory_monitor.get_detailed_cache_memory_info().get("by_cache", {}).get(cache_name, {}),
            }
        else:
            # Export all cache metrics
            export_data = {
                "all_caches": generator.cache_monitor.get_all_cache_metrics(),
                "aggregated_metrics": generator.cache_monitor.get_aggregated_metrics(),
                "cache_type_metrics": generator.cache_monitor.get_all_cache_type_metrics(),
                "memory_info": generator.memory_monitor.get_detailed_cache_memory_info(),
            }

        # Add historical data if requested
        if include_historical:
            report_history = generator.real_time_reporter.get_report_history(100)
            export_data["historical_reports"] = report_history
            export_data["memory_trends"] = generator.memory_monitor.get_cache_memory_trends(60)

        # Format according to export_format
        if export_format == "csv":
            result = _format_metrics_as_csv(export_data, cache_name)
        elif export_format == "metrics":
            result = _format_metrics_as_prometheus(export_data, cache_name)
        else:  # json format
            result = {
                "export_format": export_format,
                "export_timestamp": time.time(),
                "data": export_data,
            }

        result["tool_info"] = {
            "tool_name": "cache_metrics_export",
            "export_format": export_format,
            "cache_name": cache_name,
            "include_historical": include_historical,
            "record_count": len(export_data) if isinstance(export_data, dict) else 1,
        }

        return result

    except Exception as e:
        logger.error(f"Cache metrics export failed: {e}")
        return {
            "error": f"Metrics export failed: {str(e)}",
            "tool_info": {
                "tool_name": "cache_metrics_export",
                "error_type": "export_error",
            },
        }


async def cache_dashboard_config_tool(
    action: Literal["get", "update"] = "get",
    config_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Get or update cache dashboard configuration settings.

    Args:
        action: Whether to get or update configuration
        config_updates: Configuration updates (for update action)

    Returns:
        Dict containing configuration data or update results
    """
    try:
        generator = get_cache_dashboard_generator()

        if action == "get":
            return {
                "current_config": generator.config,
                "configurable_options": {
                    "refresh_interval_seconds": "Dashboard refresh rate",
                    "history_hours": "How many hours of history to maintain",
                    "chart_data_points": "Number of data points in trend charts",
                    "top_caches_count": "Number of top performers to show",
                    "alert_thresholds": "Threshold values for various alerts",
                },
                "tool_info": {
                    "tool_name": "cache_dashboard_config",
                    "action": "get",
                },
            }

        elif action == "update":
            if not config_updates:
                return {
                    "error": "config_updates is required for update action",
                    "tool_info": {
                        "tool_name": "cache_dashboard_config",
                        "error_type": "missing_parameter",
                    },
                }

            # Update configuration
            old_config = generator.config.copy()
            updated_keys = []

            for key, value in config_updates.items():
                if key in generator.config:
                    if isinstance(generator.config[key], dict) and isinstance(value, dict):
                        generator.config[key].update(value)
                    else:
                        generator.config[key] = value
                    updated_keys.append(key)

            return {
                "status": "updated",
                "updated_keys": updated_keys,
                "old_config": old_config,
                "new_config": generator.config,
                "tool_info": {
                    "tool_name": "cache_dashboard_config",
                    "action": "update",
                    "changes_made": len(updated_keys),
                },
            }

        else:
            return {
                "error": f"Invalid action: {action}",
                "valid_actions": ["get", "update"],
                "tool_info": {
                    "tool_name": "cache_dashboard_config",
                    "error_type": "invalid_parameter",
                },
            }

    except Exception as e:
        logger.error(f"Cache dashboard config operation failed: {e}")
        return {
            "error": f"Config operation failed: {str(e)}",
            "tool_info": {
                "tool_name": "cache_dashboard_config",
                "error_type": "operation_error",
            },
        }


def _format_dashboard_summary(dashboard_data: dict[str, Any], dashboard_type: str) -> dict[str, Any]:
    """Format dashboard data as a summary."""
    if "error" in dashboard_data:
        return dashboard_data

    sections = dashboard_data.get("sections", {})

    summary = {
        "dashboard_type": dashboard_type,
        "title": dashboard_data.get("title", "Cache Dashboard"),
        "generated_at": dashboard_data.get("generated_at"),
        "status": "unknown",
        "key_metrics": {},
        "main_issues": [],
        "recommendations": [],
    }

    if dashboard_type == "overview":
        system_summary = sections.get("system_summary", {})
        performance = sections.get("performance_metrics", {}).get("overall_performance", {})
        health = sections.get("health_status", {})

        summary.update(
            {
                "status": health.get("overall_status", "unknown"),
                "key_metrics": {
                    "total_caches": system_summary.get("total_caches", 0),
                    "hit_rate_percentage": performance.get("hit_rate_percentage", 0),
                    "error_rate_percentage": performance.get("error_rate_percentage", 0),
                    "total_operations": system_summary.get("total_operations", 0),
                },
                "main_issues": health.get("critical_issues", [])[:3],
                "recommendations": _extract_top_recommendations(sections),
            }
        )

    elif dashboard_type == "detailed":
        cache_overview = sections.get("cache_overview", {})

        summary.update(
            {
                "cache_name": cache_overview.get("cache_name", "unknown"),
                "status": cache_overview.get("status", "unknown"),
                "key_metrics": cache_overview.get("key_metrics", {}),
                "recommendations": sections.get("recommendations", {}).get("recommendations", []),
            }
        )

    elif dashboard_type == "real_time":
        live_metrics = sections.get("live_metrics", {})

        summary.update(
            {
                "status": "live",
                "key_metrics": live_metrics.get("current_metrics", {}),
                "live_rates": live_metrics.get("live_rates", {}),
                "last_update": live_metrics.get("last_update"),
            }
        )

    return summary


def _format_dashboard_report(dashboard_data: dict[str, Any], dashboard_type: str) -> dict[str, Any]:
    """Format dashboard data as a detailed report."""
    if "error" in dashboard_data:
        return dashboard_data

    sections = dashboard_data.get("sections", {})

    report = {
        "report_type": f"{dashboard_type}_dashboard_report",
        "title": dashboard_data.get("title", "Cache Dashboard Report"),
        "generated_at": dashboard_data.get("generated_at"),
        "executive_summary": "",
        "detailed_findings": [],
        "recommendations": [],
        "appendices": {},
    }

    # Generate executive summary
    if dashboard_type == "overview":
        system_summary = sections.get("system_summary", {})
        health = sections.get("health_status", {})

        report["executive_summary"] = (
            f"Cache system overview: {system_summary.get('total_caches', 0)} active caches "
            f"with {health.get('overall_status', 'unknown')} health status. "
            f"Total operations: {system_summary.get('total_operations', 0):,}."
        )

        # Add detailed findings
        for section_name, section_data in sections.items():
            if section_name in ["system_summary", "performance_metrics", "health_status"]:
                report["detailed_findings"].append(
                    {
                        "section": section_name,
                        "summary": f"Analysis of {section_name.replace('_', ' ')}",
                        "data": section_data,
                    }
                )

    # Extract recommendations
    report["recommendations"] = _extract_top_recommendations(sections)

    # Add raw data as appendices
    report["appendices"]["raw_dashboard_data"] = dashboard_data

    return report


def _extract_top_recommendations(sections: dict[str, Any]) -> list[str]:
    """Extract top recommendations from dashboard sections."""
    recommendations = []

    # From health status
    health = sections.get("health_status", {})
    critical_issues = health.get("critical_issues", [])
    for issue in critical_issues[:3]:
        recommendations.append(f"Address critical issue: {issue}")

    # From alerts
    alerts = sections.get("alerts", {})
    if alerts.get("critical_alerts", 0) > 0:
        recommendations.append("Resolve critical alerts immediately")

    # From memory usage
    memory = sections.get("memory_usage", {})
    memory_alerts = memory.get("memory_alerts", [])
    for alert in memory_alerts[:2]:
        recommendations.append(f"Memory: {alert}")

    # Generic recommendations based on performance
    performance = sections.get("performance_metrics", {}).get("overall_performance", {})
    hit_rate = performance.get("hit_rate_percentage", 0)
    if hit_rate < 60:
        recommendations.append("Improve cache hit rate through optimization")

    return recommendations[:5]  # Top 5 recommendations


def _format_metrics_as_csv(export_data: dict[str, Any], cache_name: str | None) -> dict[str, Any]:
    """Format metrics data as CSV structure."""
    csv_data = {
        "format": "csv",
        "note": "CSV export would require pandas or csv module implementation",
        "headers": [],
        "rows": [],
        "preview": "Cache metrics CSV export functionality not fully implemented",
    }

    # This would be implemented with proper CSV formatting
    return csv_data


def _format_metrics_as_prometheus(export_data: dict[str, Any], cache_name: str | None) -> dict[str, Any]:
    """Format metrics data as Prometheus metrics."""
    metrics_data = {
        "format": "prometheus",
        "metrics": [],
        "note": "Prometheus metrics export functionality",
    }

    # This would generate proper Prometheus metrics format
    # Example metrics that would be generated:
    example_metrics = [
        "# HELP cache_hit_rate Cache hit rate percentage",
        "# TYPE cache_hit_rate gauge",
        'cache_hit_rate{cache_name="example"} 0.85',
        "",
        "# HELP cache_operations_total Total cache operations",
        "# TYPE cache_operations_total counter",
        'cache_operations_total{cache_name="example",operation="get"} 1000',
    ]

    metrics_data["example_output"] = example_metrics
    metrics_data["note"] = "Full Prometheus metrics generation would be implemented here"

    return metrics_data
