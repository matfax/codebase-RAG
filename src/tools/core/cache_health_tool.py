"""
MCP tool for cache-specific health checks.

This module provides MCP tools for performing detailed health checks
on individual cache services and all cache services collectively.
"""

import logging
from typing import Any

from tools.core.cache_health_checks import (
    check_all_caches_health,
    check_individual_cache_health,
    get_cache_health_checker,
)

logger = logging.getLogger(__name__)


async def cache_health_check_tool(service_name: str | None = None) -> dict[str, Any]:
    """
    Perform comprehensive health check on cache services.

    Args:
        service_name: Optional specific cache service name to check.
                     If not provided, checks all registered cache services.

    Returns:
        Dict[str, Any]: Comprehensive health check results
    """
    try:
        if service_name:
            # Check specific cache service
            logger.info(f"Performing health check on cache service: {service_name}")
            result = await check_individual_cache_health(service_name)

            # Add tool metadata
            result["tool_info"] = {
                "tool_name": "cache_health_check",
                "target": f"service:{service_name}",
                "mode": "individual",
            }

            return result

        else:
            # Check all cache services
            logger.info("Performing health check on all cache services")
            result = await check_all_caches_health()

            # Add tool metadata
            result["tool_info"] = {
                "tool_name": "cache_health_check",
                "target": "all_services",
                "mode": "comprehensive",
            }

            return result

    except Exception as e:
        logger.error(f"Cache health check tool failed: {e}")
        return {
            "status": "error",
            "message": f"Cache health check failed: {str(e)}",
            "error": str(e),
            "tool_info": {
                "tool_name": "cache_health_check",
                "target": service_name or "all_services",
                "mode": "error",
            },
        }


async def cache_connectivity_test_tool(service_name: str) -> dict[str, Any]:
    """
    Test connectivity and basic operations for a specific cache service.

    Args:
        service_name: Name of the cache service to test

    Returns:
        Dict[str, Any]: Connectivity test results
    """
    try:
        logger.info(f"Testing connectivity for cache service: {service_name}")

        # Get the health checker
        checker = get_cache_health_checker()

        # Check if service is registered
        if service_name not in checker._cache_services:
            return {
                "status": "error",
                "message": f"Cache service '{service_name}' is not registered",
                "available_services": list(checker._cache_services.keys()),
                "tool_info": {
                    "tool_name": "cache_connectivity_test",
                    "target": service_name,
                    "mode": "not_found",
                },
            }

        # Perform full health check and extract connectivity/operations info
        full_health = await checker.check_cache_service_health(service_name)

        # Extract relevant sections
        connectivity = full_health.get("checks", {}).get("connectivity", {})
        operations = full_health.get("checks", {}).get("operations", {})

        result = {
            "service_name": service_name,
            "status": "ok" if connectivity.get("healthy", False) else "error",
            "connectivity": connectivity,
            "operations": operations,
            "summary": {
                "connected": connectivity.get("healthy", False),
                "redis_connected": connectivity.get("redis_connected", False),
                "ping_time_ms": connectivity.get("ping_time_ms"),
                "operations_successful": operations.get("all_operations_successful", False),
                "operations_tested": operations.get("operations_tested", []),
            },
            "tool_info": {
                "tool_name": "cache_connectivity_test",
                "target": service_name,
                "mode": "connectivity",
            },
            "timestamp": full_health.get("timestamp"),
        }

        return result

    except Exception as e:
        logger.error(f"Cache connectivity test failed for {service_name}: {e}")
        return {
            "status": "error",
            "message": f"Connectivity test failed: {str(e)}",
            "service_name": service_name,
            "error": str(e),
            "tool_info": {
                "tool_name": "cache_connectivity_test",
                "target": service_name,
                "mode": "error",
            },
        }


async def cache_service_register_tool(service_name: str, service_type: str) -> dict[str, Any]:
    """
    Register a cache service for health monitoring.

    Args:
        service_name: Name of the cache service
        service_type: Type/class of the cache service

    Returns:
        Dict[str, Any]: Registration result
    """
    try:
        logger.info(f"Attempting to register cache service: {service_name} (type: {service_type})")

        # This is a placeholder - in a real implementation, you would need to
        # import and instantiate the actual service based on service_type
        # For now, we'll just register it with the health checker registry

        checker = get_cache_health_checker()

        # For demonstration, we'll create a mock service entry
        # In practice, this would involve proper service discovery and instantiation

        return {
            "status": "warning",
            "message": f"Service registration placeholder for '{service_name}'",
            "service_name": service_name,
            "service_type": service_type,
            "note": "Actual service registration requires service instance - this is a registration placeholder",
            "registered_services": list(checker._cache_services.keys()),
            "tool_info": {
                "tool_name": "cache_service_register",
                "target": service_name,
                "mode": "placeholder",
            },
        }

    except Exception as e:
        logger.error(f"Cache service registration failed for {service_name}: {e}")
        return {
            "status": "error",
            "message": f"Service registration failed: {str(e)}",
            "service_name": service_name,
            "error": str(e),
            "tool_info": {
                "tool_name": "cache_service_register",
                "target": service_name,
                "mode": "error",
            },
        }


async def cache_health_summary_tool() -> dict[str, Any]:
    """
    Get a summary of cache health across all services.

    Returns:
        Dict[str, Any]: Cache health summary
    """
    try:
        logger.info("Generating cache health summary")

        # Get comprehensive health check
        full_health = await check_all_caches_health()

        # Extract and format summary information
        summary = {
            "overall_status": full_health.get("overall_status", "unknown"),
            "service_count": {
                "total": full_health.get("services_checked", 0),
                "healthy": full_health.get("healthy_services", 0),
                "warning": full_health.get("warning_services", 0),
                "error": full_health.get("error_services", 0),
            },
            "health_percentage": full_health.get("summary", {}).get("health_percentage", 0),
            "summary_message": full_health.get("summary", {}).get("message", ""),
            "key_issues": [],
            "key_recommendations": full_health.get("recommendations", [])[:5],  # Top 5 recommendations
            "service_overview": {},
            "tool_info": {
                "tool_name": "cache_health_summary",
                "target": "all_services",
                "mode": "summary",
            },
            "timestamp": full_health.get("timestamp"),
        }

        # Extract key issues and create service overview
        service_results = full_health.get("service_results", {})
        for service_name, service_data in service_results.items():
            status = service_data.get("status", "unknown")
            summary["service_overview"][service_name] = {
                "status": status,
                "issues_count": len(service_data.get("issues", [])),
                "warnings_count": len(service_data.get("warnings", [])),
            }

            # Collect critical issues
            if status == "error":
                issues = service_data.get("issues", [])
                for issue in issues[:2]:  # Max 2 issues per service
                    summary["key_issues"].append(f"{service_name}: {issue}")

        # Limit key issues
        summary["key_issues"] = summary["key_issues"][:10]

        return summary

    except Exception as e:
        logger.error(f"Cache health summary generation failed: {e}")
        return {
            "status": "error",
            "message": f"Health summary generation failed: {str(e)}",
            "error": str(e),
            "tool_info": {
                "tool_name": "cache_health_summary",
                "target": "all_services",
                "mode": "error",
            },
        }


async def cache_performance_analysis_tool(service_name: str | None = None) -> dict[str, Any]:
    """
    Analyze cache performance metrics for optimization insights.

    Args:
        service_name: Optional specific cache service name to analyze

    Returns:
        Dict[str, Any]: Performance analysis results
    """
    try:
        if service_name:
            logger.info(f"Analyzing performance for cache service: {service_name}")

            # Get detailed health check for specific service
            health_data = await check_individual_cache_health(service_name)

            performance = health_data.get("checks", {}).get("performance", {})
            memory = health_data.get("checks", {}).get("memory", {})

            analysis = {
                "service_name": service_name,
                "performance_score": "unknown",
                "key_metrics": {
                    "hit_rate": performance.get("hit_rate", 0.0),
                    "error_rate": performance.get("error_rate", 0.0),
                    "avg_response_time_ms": performance.get("avg_response_time_ms", 0.0),
                    "memory_usage_mb": memory.get("current_size_mb", 0.0),
                    "total_operations": performance.get("total_operations", 0),
                },
                "bottlenecks": [],
                "optimization_opportunities": [],
                "efficiency_rating": "unknown",
            }

            # Calculate performance score
            hit_rate = performance.get("hit_rate", 0.0)
            error_rate = performance.get("error_rate", 0.0)
            response_time = performance.get("avg_response_time_ms", 0.0)

            score = 100
            if hit_rate < 0.8:
                score -= (0.8 - hit_rate) * 50
            if error_rate > 0.01:
                score -= error_rate * 500
            if response_time > 100:
                score -= min((response_time - 100) / 10, 30)

            analysis["performance_score"] = max(0, round(score, 1))

            # Identify bottlenecks
            if hit_rate < 0.6:
                analysis["bottlenecks"].append("Low cache hit rate indicates poor cache effectiveness")
            if response_time > 500:
                analysis["bottlenecks"].append("High response time indicates performance bottleneck")
            if error_rate > 0.05:
                analysis["bottlenecks"].append("High error rate indicates reliability issues")

            # Identify optimization opportunities
            if hit_rate < 0.8:
                analysis["optimization_opportunities"].append("Optimize cache key generation and TTL settings")
            if memory.get("memory_pressure_events", 0) > 0:
                analysis["optimization_opportunities"].append("Increase cache memory limits or implement better eviction policies")
            if response_time > 200:
                analysis["optimization_opportunities"].append("Optimize cache operations or consider infrastructure scaling")

            # Efficiency rating
            if analysis["performance_score"] >= 80:
                analysis["efficiency_rating"] = "excellent"
            elif analysis["performance_score"] >= 60:
                analysis["efficiency_rating"] = "good"
            elif analysis["performance_score"] >= 40:
                analysis["efficiency_rating"] = "fair"
            else:
                analysis["efficiency_rating"] = "poor"

            analysis["tool_info"] = {
                "tool_name": "cache_performance_analysis",
                "target": service_name,
                "mode": "individual",
            }

            return analysis

        else:
            logger.info("Analyzing performance for all cache services")

            # Get all services health data
            all_health = await check_all_caches_health()

            analysis = {
                "overall_performance_score": 0,
                "service_scores": {},
                "system_bottlenecks": [],
                "top_performers": [],
                "needs_attention": [],
                "optimization_priorities": [],
                "tool_info": {
                    "tool_name": "cache_performance_analysis",
                    "target": "all_services",
                    "mode": "comprehensive",
                },
                "timestamp": all_health.get("timestamp"),
            }

            service_results = all_health.get("service_results", {})
            scores = []

            for service_name, service_data in service_results.items():
                # Get individual performance analysis
                individual_analysis = await cache_performance_analysis_tool(service_name)
                score = individual_analysis.get("performance_score", 0)

                analysis["service_scores"][service_name] = {
                    "score": score,
                    "rating": individual_analysis.get("efficiency_rating", "unknown"),
                    "key_issues": len(service_data.get("issues", [])),
                }

                scores.append(score)

                # Categorize services
                if score >= 75:
                    analysis["top_performers"].append(service_name)
                elif score < 50:
                    analysis["needs_attention"].append(service_name)

            # Calculate overall score
            if scores:
                analysis["overall_performance_score"] = round(sum(scores) / len(scores), 1)

            # System-wide bottlenecks and priorities
            if analysis["overall_performance_score"] < 60:
                analysis["system_bottlenecks"].append("Overall cache system performance is below optimal")

            if len(analysis["needs_attention"]) > len(analysis["top_performers"]):
                analysis["optimization_priorities"].append("Focus on improving underperforming cache services")

            return analysis

    except Exception as e:
        logger.error(f"Cache performance analysis failed: {e}")
        return {
            "status": "error",
            "message": f"Performance analysis failed: {str(e)}",
            "error": str(e),
            "tool_info": {
                "tool_name": "cache_performance_analysis",
                "target": service_name or "all_services",
                "mode": "error",
            },
        }
