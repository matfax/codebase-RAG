"""
Cache performance dashboard implementation.

This module provides comprehensive cache performance dashboards with
real-time metrics, historical trends, and visualization capabilities.
"""

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from utils.performance_monitor import (
    MemoryMonitor,
    get_cache_performance_monitor,
    get_real_time_cache_reporter,
)

logger = logging.getLogger(__name__)


class CacheDashboardGenerator:
    """Generates comprehensive cache performance dashboards and reports."""

    def __init__(self):
        self.cache_monitor = get_cache_performance_monitor()
        self.real_time_reporter = get_real_time_cache_reporter()
        self.memory_monitor = MemoryMonitor()

        # Dashboard configuration
        self.config = {
            "refresh_interval_seconds": 30,
            "history_hours": 24,
            "chart_data_points": 100,
            "top_caches_count": 10,
            "alert_thresholds": {
                "hit_rate_warning": 0.6,
                "hit_rate_critical": 0.3,
                "error_rate_warning": 0.05,
                "error_rate_critical": 0.1,
                "response_time_warning_ms": 500,
                "response_time_critical_ms": 2000,
                "memory_warning_mb": 100,
                "memory_critical_mb": 500,
            },
        }

    def generate_overview_dashboard(self) -> dict[str, Any]:
        """
        Generate a comprehensive overview dashboard with key metrics and status.

        Returns:
            Dict containing dashboard data with metrics, charts, and status information
        """
        dashboard = {
            "dashboard_type": "overview",
            "title": "Cache Performance Overview",
            "generated_at": datetime.now().isoformat(),
            "refresh_interval": self.config["refresh_interval_seconds"],
            "sections": {},
            "alerts": [],
            "metadata": {
                "version": "1.0",
                "generator": "CacheDashboardGenerator",
            },
        }

        try:
            # 1. System Summary Section
            dashboard["sections"]["system_summary"] = self._generate_system_summary()

            # 2. Performance Metrics Section
            dashboard["sections"]["performance_metrics"] = self._generate_performance_metrics()

            # 3. Cache Health Status Section
            dashboard["sections"]["health_status"] = self._generate_health_status()

            # 4. Memory Usage Section
            dashboard["sections"]["memory_usage"] = self._generate_memory_usage()

            # 5. Top Performing Caches Section
            dashboard["sections"]["top_performers"] = self._generate_top_performers()

            # 6. Recent Activity Section
            dashboard["sections"]["recent_activity"] = self._generate_recent_activity()

            # 7. Alerts and Issues Section
            dashboard["sections"]["alerts"] = self._generate_alerts_section()

            # 8. Trend Charts Section
            dashboard["sections"]["trend_charts"] = self._generate_trend_charts()

        except Exception as e:
            logger.error(f"Failed to generate overview dashboard: {e}")
            dashboard["error"] = f"Dashboard generation failed: {str(e)}"

        return dashboard

    def generate_detailed_dashboard(self, cache_name: str) -> dict[str, Any]:
        """
        Generate a detailed dashboard for a specific cache service.

        Args:
            cache_name: Name of the cache service

        Returns:
            Dict containing detailed dashboard data for the specific cache
        """
        dashboard = {
            "dashboard_type": "detailed",
            "title": f"Cache Performance: {cache_name}",
            "cache_name": cache_name,
            "generated_at": datetime.now().isoformat(),
            "refresh_interval": self.config["refresh_interval_seconds"],
            "sections": {},
            "metadata": {
                "version": "1.0",
                "generator": "CacheDashboardGenerator",
                "target_cache": cache_name,
            },
        }

        try:
            # Check if cache exists
            cache_metrics = self.cache_monitor.get_cache_metrics(cache_name)
            if not cache_metrics:
                dashboard["error"] = f"Cache '{cache_name}' not found or has no metrics"
                dashboard["available_caches"] = list(self.cache_monitor.get_all_cache_metrics().keys())
                return dashboard

            # 1. Cache Overview Section
            dashboard["sections"]["cache_overview"] = self._generate_cache_overview(cache_name)

            # 2. Performance Details Section
            dashboard["sections"]["performance_details"] = self._generate_performance_details(cache_name)

            # 3. Memory Analysis Section
            dashboard["sections"]["memory_analysis"] = self._generate_memory_analysis(cache_name)

            # 4. Operation Statistics Section
            dashboard["sections"]["operation_statistics"] = self._generate_operation_statistics(cache_name)

            # 5. Historical Trends Section
            dashboard["sections"]["historical_trends"] = self._generate_historical_trends(cache_name)

            # 6. Error Analysis Section
            dashboard["sections"]["error_analysis"] = self._generate_error_analysis(cache_name)

            # 7. Configuration Info Section
            dashboard["sections"]["configuration"] = self._generate_configuration_info(cache_name)

            # 8. Recommendations Section
            dashboard["sections"]["recommendations"] = self._generate_recommendations(cache_name)

        except Exception as e:
            logger.error(f"Failed to generate detailed dashboard for {cache_name}: {e}")
            dashboard["error"] = f"Dashboard generation failed: {str(e)}"

        return dashboard

    def generate_real_time_dashboard(self) -> dict[str, Any]:
        """
        Generate a real-time dashboard with live updating metrics.

        Returns:
            Dict containing real-time dashboard data
        """
        dashboard = {
            "dashboard_type": "real_time",
            "title": "Real-Time Cache Performance",
            "generated_at": datetime.now().isoformat(),
            "refresh_interval": 5,  # Real-time updates every 5 seconds
            "sections": {},
            "live_data": True,
            "metadata": {
                "version": "1.0",
                "generator": "CacheDashboardGenerator",
                "mode": "real_time",
            },
        }

        try:
            # Get latest real-time report
            latest_report = self.real_time_reporter.get_latest_report()
            if not latest_report:
                dashboard["sections"]["status"] = {
                    "message": "Real-time reporting not active",
                    "action": "Start real-time reporting to see live metrics",
                }
                return dashboard

            # 1. Live Metrics Section
            dashboard["sections"]["live_metrics"] = self._generate_live_metrics(latest_report)

            # 2. Activity Feed Section
            dashboard["sections"]["activity_feed"] = self._generate_activity_feed(latest_report)

            # 3. Performance Gauges Section
            dashboard["sections"]["performance_gauges"] = self._generate_performance_gauges(latest_report)

            # 4. Memory Monitoring Section
            dashboard["sections"]["memory_monitoring"] = self._generate_memory_monitoring(latest_report)

            # 5. Alerts Feed Section
            dashboard["sections"]["alerts_feed"] = self._generate_alerts_feed(latest_report)

            # 6. System Health Section
            dashboard["sections"]["system_health"] = self._generate_system_health(latest_report)

        except Exception as e:
            logger.error(f"Failed to generate real-time dashboard: {e}")
            dashboard["error"] = f"Real-time dashboard generation failed: {str(e)}"

        return dashboard

    def _generate_system_summary(self) -> dict[str, Any]:
        """Generate system summary metrics."""
        aggregated = self.cache_monitor.get_aggregated_metrics()
        summary = aggregated.get("summary", {})

        return {
            "total_caches": summary.get("total_caches", 0),
            "total_operations": summary.get("total_operations", 0),
            "overall_hit_rate": round(summary.get("overall_hit_rate", 0.0) * 100, 1),
            "overall_error_rate": round(summary.get("overall_error_rate", 0.0) * 100, 2),
            "total_memory_mb": round(summary.get("total_size_mb", 0.0), 1),
            "active_alerts": len(self.cache_monitor.get_alerts()),
            "monitoring_enabled": self.cache_monitor.is_monitoring_enabled(),
            "uptime_info": {
                "monitoring_since": aggregated.get("timestamp", time.time()),
                "age_hours": round((time.time() - aggregated.get("timestamp", time.time())) / 3600, 1),
            },
        }

    def _generate_performance_metrics(self) -> dict[str, Any]:
        """Generate performance metrics visualization data."""
        aggregated = self.cache_monitor.get_aggregated_metrics()
        cache_types = self.cache_monitor.get_cache_type_summary()

        metrics = {
            "overall_performance": {
                "hit_rate_percentage": round(aggregated.get("summary", {}).get("overall_hit_rate", 0.0) * 100, 1),
                "miss_rate_percentage": round((1.0 - aggregated.get("summary", {}).get("overall_hit_rate", 0.0)) * 100, 1),
                "error_rate_percentage": round(aggregated.get("summary", {}).get("overall_error_rate", 0.0) * 100, 2),
                "avg_response_time_ms": round(aggregated.get("summary", {}).get("average_response_time_ms", 0.0), 1),
            },
            "cache_type_performance": {},
            "performance_distribution": {
                "excellent": 0,  # >90% hit rate
                "good": 0,  # 70-90% hit rate
                "fair": 0,  # 50-70% hit rate
                "poor": 0,  # <50% hit rate
            },
        }

        # Cache type performance
        for cache_type, type_data in cache_types.items():
            metrics["cache_type_performance"][cache_type] = {
                "hit_rate_percentage": round(type_data.get("hit_rate", 0.0) * 100, 1),
                "error_rate_percentage": round(type_data.get("error_rate", 0.0) * 100, 2),
                "avg_response_time_ms": round(type_data.get("average_response_time_ms", 0.0), 1),
                "operations": type_data.get("total_operations", 0),
                "memory_mb": round(type_data.get("current_size_mb", 0.0), 1),
            }

        # Performance distribution
        all_caches = self.cache_monitor.get_all_cache_metrics()
        for cache_name, cache_data in all_caches.items():
            hit_rate = cache_data.get("hit_rate", 0.0)
            if hit_rate >= 0.9:
                metrics["performance_distribution"]["excellent"] += 1
            elif hit_rate >= 0.7:
                metrics["performance_distribution"]["good"] += 1
            elif hit_rate >= 0.5:
                metrics["performance_distribution"]["fair"] += 1
            else:
                metrics["performance_distribution"]["poor"] += 1

        return metrics

    def _generate_health_status(self) -> dict[str, Any]:
        """Generate health status overview."""
        all_caches = self.cache_monitor.get_all_cache_metrics()
        self.cache_monitor.get_alerts()

        health = {
            "overall_status": "healthy",
            "status_summary": {
                "healthy": 0,
                "warning": 0,
                "critical": 0,
            },
            "cache_statuses": {},
            "critical_issues": [],
            "warning_issues": [],
        }

        # Evaluate each cache health
        for cache_name, cache_data in all_caches.items():
            hit_rate = cache_data.get("hit_rate", 0.0)
            error_rate = cache_data.get("errors", {}).get("rate", 0.0)
            response_time = cache_data.get("performance", {}).get("average_response_time_ms", 0.0)

            status = "healthy"
            issues = []

            # Check critical conditions
            if (
                hit_rate < self.config["alert_thresholds"]["hit_rate_critical"]
                or error_rate > self.config["alert_thresholds"]["error_rate_critical"]
                or response_time > self.config["alert_thresholds"]["response_time_critical_ms"]
            ):
                status = "critical"
                if hit_rate < self.config["alert_thresholds"]["hit_rate_critical"]:
                    issues.append(f"Critical hit rate: {hit_rate:.1%}")
                if error_rate > self.config["alert_thresholds"]["error_rate_critical"]:
                    issues.append(f"High error rate: {error_rate:.1%}")
                if response_time > self.config["alert_thresholds"]["response_time_critical_ms"]:
                    issues.append(f"Slow response: {response_time:.0f}ms")

            # Check warning conditions
            elif (
                hit_rate < self.config["alert_thresholds"]["hit_rate_warning"]
                or error_rate > self.config["alert_thresholds"]["error_rate_warning"]
                or response_time > self.config["alert_thresholds"]["response_time_warning_ms"]
            ):
                status = "warning"
                if hit_rate < self.config["alert_thresholds"]["hit_rate_warning"]:
                    issues.append(f"Low hit rate: {hit_rate:.1%}")
                if error_rate > self.config["alert_thresholds"]["error_rate_warning"]:
                    issues.append(f"Elevated errors: {error_rate:.1%}")
                if response_time > self.config["alert_thresholds"]["response_time_warning_ms"]:
                    issues.append(f"Slow response: {response_time:.0f}ms")

            health["cache_statuses"][cache_name] = {
                "status": status,
                "issues": issues,
                "hit_rate": hit_rate,
                "error_rate": error_rate,
                "response_time_ms": response_time,
            }

            health["status_summary"][status] += 1

            # Collect critical and warning issues
            if status == "critical":
                health["critical_issues"].extend([f"{cache_name}: {issue}" for issue in issues])
            elif status == "warning":
                health["warning_issues"].extend([f"{cache_name}: {issue}" for issue in issues])

        # Determine overall health
        if health["status_summary"]["critical"] > 0:
            health["overall_status"] = "critical"
        elif health["status_summary"]["warning"] > 0:
            health["overall_status"] = "warning"

        return health

    def _generate_memory_usage(self) -> dict[str, Any]:
        """Generate memory usage analysis."""
        memory_info = self.memory_monitor.get_detailed_cache_memory_info()

        memory_data = {
            "total_cache_memory_mb": round(memory_info.get("cache_memory", {}).get("total_mb", 0.0), 1),
            "memory_efficiency": memory_info.get("memory_efficiency", {}),
            "by_cache": {},
            "memory_distribution": [],
            "memory_trends": "stable",  # Would need historical data for real trends
            "memory_alerts": [],
        }

        # Process individual cache memory data
        cache_memory = memory_info.get("by_cache", {})
        for cache_name, cache_data in cache_memory.items():
            current_mb = cache_data.get("current_size_mb", 0.0)
            max_mb = cache_data.get("max_size_mb", 0.0)

            memory_data["by_cache"][cache_name] = {
                "current_mb": round(current_mb, 1),
                "max_mb": round(max_mb, 1),
                "utilization_percentage": round((current_mb / max(max_mb, 0.1)) * 100, 1) if max_mb > 0 else 0,
                "hit_rate": cache_data.get("hit_rate", 0.0),
                "operations": cache_data.get("operations", 0),
                "efficiency": round(cache_data.get("operations", 0) / max(current_mb, 0.1), 1),  # ops per MB
            }

            # Create memory distribution data
            memory_data["memory_distribution"].append(
                {
                    "cache_name": cache_name,
                    "size_mb": round(current_mb, 1),
                    "percentage": round((current_mb / max(memory_data["total_cache_memory_mb"], 0.1)) * 100, 1),
                }
            )

            # Check for memory alerts
            if current_mb > self.config["alert_thresholds"]["memory_critical_mb"]:
                memory_data["memory_alerts"].append(f"{cache_name}: Critical memory usage ({current_mb:.1f}MB)")
            elif current_mb > self.config["alert_thresholds"]["memory_warning_mb"]:
                memory_data["memory_alerts"].append(f"{cache_name}: High memory usage ({current_mb:.1f}MB)")

        # Sort memory distribution by size
        memory_data["memory_distribution"].sort(key=lambda x: x["size_mb"], reverse=True)

        return memory_data

    def _generate_top_performers(self) -> dict[str, Any]:
        """Generate top performing caches analysis."""
        all_caches = self.cache_monitor.get_all_cache_metrics()

        performers = []
        for cache_name, cache_data in all_caches.items():
            hit_rate = cache_data.get("hit_rate", 0.0)
            operations = cache_data.get("operations", {}).get("total", 0)
            error_rate = cache_data.get("errors", {}).get("rate", 0.0)
            response_time = cache_data.get("performance", {}).get("average_response_time_ms", 0.0)

            # Calculate performance score
            score = hit_rate * 100  # Base score from hit rate
            score -= error_rate * 1000  # Penalty for errors
            score -= min(response_time / 10, 50)  # Penalty for slow response (max 50 point penalty)
            score *= 1 + math.log10(max(operations, 1)) / 10  # Bonus for high activity

            performers.append(
                {
                    "cache_name": cache_name,
                    "performance_score": round(max(score, 0), 1),
                    "hit_rate": round(hit_rate * 100, 1),
                    "operations": operations,
                    "error_rate": round(error_rate * 100, 2),
                    "response_time_ms": round(response_time, 1),
                    "memory_mb": round(cache_data.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024, 1),
                }
            )

        # Sort by performance score
        performers.sort(key=lambda x: x["performance_score"], reverse=True)

        return {
            "top_performers": performers[: self.config["top_caches_count"]],
            "bottom_performers": performers[-5:] if len(performers) > 5 else [],
            "performance_stats": {
                "highest_score": performers[0]["performance_score"] if performers else 0,
                "lowest_score": performers[-1]["performance_score"] if performers else 0,
                "average_score": round(sum(p["performance_score"] for p in performers) / max(len(performers), 1), 1),
            },
        }

    def _generate_recent_activity(self) -> dict[str, Any]:
        """Generate recent activity summary."""
        aggregated = self.cache_monitor.get_aggregated_metrics()

        # Get recent report history for activity analysis
        recent_reports = self.real_time_reporter.get_report_history(5)

        activity = {
            "current_activity": {
                "total_operations": aggregated.get("summary", {}).get("total_operations", 0),
                "operations_per_second": 0,  # Would calculate from deltas
                "hits_per_second": 0,
                "misses_per_second": 0,
                "errors_per_second": 0,
            },
            "activity_trends": {
                "operations_trend": "stable",
                "hit_rate_trend": "stable",
                "error_rate_trend": "stable",
            },
            "busiest_caches": [],
            "recent_events": [],
        }

        # Calculate activity rates if we have recent data
        if len(recent_reports) >= 2:
            latest = recent_reports[-1]
            previous = recent_reports[-2]

            time_delta = latest["timestamp"] - previous["timestamp"]
            if time_delta > 0:
                ops_delta = latest.get("summary", {}).get("total_operations", 0) - previous.get("summary", {}).get("total_operations", 0)
                hits_delta = latest.get("summary", {}).get("total_hits", 0) - previous.get("summary", {}).get("total_hits", 0)
                misses_delta = latest.get("summary", {}).get("total_misses", 0) - previous.get("summary", {}).get("total_misses", 0)
                errors_delta = latest.get("summary", {}).get("total_errors", 0) - previous.get("summary", {}).get("total_errors", 0)

                activity["current_activity"].update(
                    {
                        "operations_per_second": round(ops_delta / time_delta, 1),
                        "hits_per_second": round(hits_delta / time_delta, 1),
                        "misses_per_second": round(misses_delta / time_delta, 1),
                        "errors_per_second": round(errors_delta / time_delta, 2),
                    }
                )

        # Find busiest caches
        all_caches = self.cache_monitor.get_all_cache_metrics()
        busiest = []
        for cache_name, cache_data in all_caches.items():
            operations = cache_data.get("operations", {}).get("total", 0)
            busiest.append(
                {
                    "cache_name": cache_name,
                    "operations": operations,
                    "hit_rate": round(cache_data.get("hit_rate", 0.0) * 100, 1),
                }
            )

        busiest.sort(key=lambda x: x["operations"], reverse=True)
        activity["busiest_caches"] = busiest[:5]

        return activity

    def _generate_alerts_section(self) -> dict[str, Any]:
        """Generate alerts and issues section."""
        alerts = self.cache_monitor.get_alerts()
        memory_alerts = self.memory_monitor.get_cache_memory_alerts()

        alerts_data = {
            "total_alerts": len(alerts) + len(memory_alerts),
            "critical_alerts": 0,
            "warning_alerts": 0,
            "recent_alerts": [],
            "alert_categories": defaultdict(int),
            "alert_summary": "No active alerts",
        }

        # Process cache performance alerts
        for alert in alerts:
            alert_type = alert.get("type", "unknown")
            severity = "critical" if alert_type in ["high_error_rate", "high_memory_usage"] else "warning"

            alerts_data[f"{severity}_alerts"] += 1
            alerts_data["alert_categories"][alert_type] += 1

            alerts_data["recent_alerts"].append(
                {
                    "type": alert_type,
                    "severity": severity,
                    "cache_name": alert.get("cache_name", "unknown"),
                    "message": f"{alert_type.replace('_', ' ').title()}: {alert.get('value', 'N/A')}",
                    "timestamp": alert.get("timestamp", time.time()),
                }
            )

        # Process memory alerts
        for memory_alert in memory_alerts:
            alert_type = memory_alert.get("type", "memory_alert")
            severity = memory_alert.get("severity", "warning")

            alerts_data[f"{severity}_alerts"] += 1
            alerts_data["alert_categories"][alert_type] += 1

            alerts_data["recent_alerts"].append(
                {
                    "type": alert_type,
                    "severity": severity,
                    "cache_name": memory_alert.get("cache_name", "unknown"),
                    "message": memory_alert.get("recommendation", "Memory alert"),
                    "timestamp": memory_alert.get("timestamp", time.time()),
                }
            )

        # Sort alerts by timestamp (most recent first)
        alerts_data["recent_alerts"].sort(key=lambda x: x["timestamp"], reverse=True)
        alerts_data["recent_alerts"] = alerts_data["recent_alerts"][:10]  # Keep only recent 10

        # Generate summary
        if alerts_data["critical_alerts"] > 0:
            alerts_data["alert_summary"] = f"{alerts_data['critical_alerts']} critical alerts require immediate attention"
        elif alerts_data["warning_alerts"] > 0:
            alerts_data["alert_summary"] = f"{alerts_data['warning_alerts']} warnings detected"

        return alerts_data

    def _generate_trend_charts(self) -> dict[str, Any]:
        """Generate trend chart data for visualization."""
        # Get recent report history for trend analysis
        recent_reports = self.real_time_reporter.get_report_history(self.config["chart_data_points"])

        charts = {
            "hit_rate_trend": {
                "data_points": [],
                "labels": [],
                "trend_direction": "stable",
            },
            "operations_trend": {
                "data_points": [],
                "labels": [],
                "trend_direction": "stable",
            },
            "memory_trend": {
                "data_points": [],
                "labels": [],
                "trend_direction": "stable",
            },
            "error_rate_trend": {
                "data_points": [],
                "labels": [],
                "trend_direction": "stable",
            },
        }

        if recent_reports:
            for report in recent_reports:
                timestamp = report.get("timestamp", time.time())
                label = datetime.fromtimestamp(timestamp).strftime("%H:%M")
                summary = report.get("summary", {})

                charts["hit_rate_trend"]["data_points"].append(summary.get("overall_hit_rate", 0.0) * 100)
                charts["hit_rate_trend"]["labels"].append(label)

                charts["operations_trend"]["data_points"].append(summary.get("total_operations", 0))
                charts["operations_trend"]["labels"].append(label)

                charts["memory_trend"]["data_points"].append(summary.get("total_size_mb", 0.0))
                charts["memory_trend"]["labels"].append(label)

                charts["error_rate_trend"]["data_points"].append(summary.get("overall_error_rate", 0.0) * 100)
                charts["error_rate_trend"]["labels"].append(label)

            # Calculate trend directions (simplified)
            for chart_name, chart_data in charts.items():
                data_points = chart_data["data_points"]
                if len(data_points) >= 2:
                    recent_avg = sum(data_points[-5:]) / min(len(data_points), 5)
                    older_avg = sum(data_points[:5]) / min(len(data_points), 5)

                    if recent_avg > older_avg * 1.1:
                        chart_data["trend_direction"] = "increasing"
                    elif recent_avg < older_avg * 0.9:
                        chart_data["trend_direction"] = "decreasing"

        return charts

    def _generate_cache_overview(self, cache_name: str) -> dict[str, Any]:
        """Generate overview section for specific cache."""
        cache_metrics = self.cache_monitor.get_cache_metrics(cache_name)
        if not cache_metrics:
            return {"error": "No metrics available"}

        return {
            "cache_name": cache_name,
            "cache_type": cache_metrics.get("cache_type", "unknown"),
            "status": "healthy",  # Would determine from metrics
            "uptime": cache_metrics.get("timestamps", {}).get("uptime_seconds", 0),
            "last_activity": cache_metrics.get("timestamps", {}).get("last_operation_time"),
            "key_metrics": {
                "hit_rate": round(cache_metrics.get("hit_rate", 0.0) * 100, 1),
                "total_operations": cache_metrics.get("operations", {}).get("total", 0),
                "error_rate": round(cache_metrics.get("errors", {}).get("rate", 0.0) * 100, 2),
                "avg_response_time_ms": round(cache_metrics.get("performance", {}).get("average_response_time_ms", 0.0), 1),
                "memory_mb": round(cache_metrics.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024, 1),
            },
        }

    def _generate_performance_details(self, cache_name: str) -> dict[str, Any]:
        """Generate detailed performance metrics for specific cache."""
        cache_metrics = self.cache_monitor.get_cache_metrics(cache_name)
        if not cache_metrics:
            return {"error": "No metrics available"}

        performance = cache_metrics.get("performance", {})
        operations = cache_metrics.get("operations", {})

        return {
            "response_times": {
                "average_ms": round(performance.get("average_response_time_ms", 0.0), 1),
                "min_ms": round(performance.get("min_response_time_ms", 0.0), 1),
                "max_ms": round(performance.get("max_response_time_ms", 0.0), 1),
                "samples": performance.get("samples", 0),
            },
            "hit_miss_analysis": {
                "hit_rate": round(cache_metrics.get("hit_rate", 0.0) * 100, 1),
                "miss_rate": round(cache_metrics.get("miss_rate", 0.0) * 100, 1),
                "hit_count": cache_metrics.get("hit_count", 0),
                "miss_count": cache_metrics.get("miss_count", 0),
            },
            "operation_breakdown": {
                "get": operations.get("get", 0),
                "set": operations.get("set", 0),
                "delete": operations.get("delete", 0),
                "exists": operations.get("exists", 0),
                "batch_get": operations.get("batch_get", 0),
                "batch_set": operations.get("batch_set", 0),
            },
            "cache_levels": cache_metrics.get("cache_levels", {}),
        }

    def _generate_memory_analysis(self, cache_name: str) -> dict[str, Any]:
        """Generate memory analysis for specific cache."""
        memory_info = self.memory_monitor.get_detailed_cache_memory_info()
        cache_data = memory_info.get("by_cache", {}).get(cache_name, {})

        size_stats = self.cache_monitor.get_cache_size_statistics(cache_name)
        cleanup_stats = self.cache_monitor.get_cache_cleanup_statistics(cache_name)

        return {
            "current_usage": {
                "size_mb": round(cache_data.get("current_size_mb", 0.0), 1),
                "max_size_mb": round(cache_data.get("max_size_mb", 0.0), 1),
                "utilization_percentage": round(
                    (cache_data.get("current_size_mb", 0.0) / max(cache_data.get("max_size_mb", 0.1), 0.1)) * 100, 1
                ),
            },
            "memory_events": {
                "eviction_count": cache_data.get("eviction_count", 0),
                "memory_pressure_events": cache_data.get("memory_pressure_events", 0),
            },
            "size_statistics": size_stats or {},
            "cleanup_statistics": cleanup_stats or {},
        }

    def _generate_operation_statistics(self, cache_name: str) -> dict[str, Any]:
        """Generate operation statistics for specific cache."""
        cache_metrics = self.cache_monitor.get_cache_metrics(cache_name)
        if not cache_metrics:
            return {"error": "No metrics available"}

        operations = cache_metrics.get("operations", {})
        errors = cache_metrics.get("errors", {})

        total_ops = operations.get("total", 0)

        return {
            "operation_counts": operations,
            "operation_percentages": {
                op_type: round((count / max(total_ops, 1)) * 100, 1) for op_type, count in operations.items() if op_type != "total"
            },
            "error_statistics": {
                "total_errors": errors.get("total", 0),
                "timeout_errors": errors.get("timeout", 0),
                "connection_errors": errors.get("connection", 0),
                "error_rate": round(errors.get("rate", 0.0) * 100, 2),
            },
            "reliability_metrics": {
                "success_rate": round((1.0 - errors.get("rate", 0.0)) * 100, 2),
                "operations_per_error": round(total_ops / max(errors.get("total", 1), 1), 1),
            },
        }

    def _generate_historical_trends(self, cache_name: str) -> dict[str, Any]:
        """Generate historical trends for specific cache."""
        # This would require historical data storage
        # For now, return placeholder structure
        return {
            "data_available": False,
            "note": "Historical trend data requires time-series storage implementation",
            "placeholder_trends": {
                "hit_rate_trend": "stable",
                "memory_trend": "growing",
                "operations_trend": "increasing",
            },
        }

    def _generate_error_analysis(self, cache_name: str) -> dict[str, Any]:
        """Generate error analysis for specific cache."""
        cache_metrics = self.cache_monitor.get_cache_metrics(cache_name)
        if not cache_metrics:
            return {"error": "No metrics available"}

        errors = cache_metrics.get("errors", {})

        return {
            "error_summary": {
                "total_errors": errors.get("total", 0),
                "error_rate": round(errors.get("rate", 0.0) * 100, 2),
                "error_types": {
                    "timeout": errors.get("timeout", 0),
                    "connection": errors.get("connection", 0),
                    "general": errors.get("total", 0) - errors.get("timeout", 0) - errors.get("connection", 0),
                },
            },
            "error_impact": {
                "operations_affected": errors.get("total", 0),
                "success_rate": round((1.0 - errors.get("rate", 0.0)) * 100, 2),
            },
            "recent_error_patterns": "No pattern analysis available",  # Would need error logging
        }

    def _generate_configuration_info(self, cache_name: str) -> dict[str, Any]:
        """Generate configuration information for specific cache."""
        # This would require access to cache service configuration
        # For now, return basic structure
        return {
            "configuration_available": False,
            "note": "Configuration details require cache service access",
            "expected_config": {
                "ttl_settings": "unknown",
                "memory_limits": "unknown",
                "eviction_policy": "unknown",
                "connection_pool": "unknown",
            },
        }

    def _generate_recommendations(self, cache_name: str) -> dict[str, Any]:
        """Generate optimization recommendations for specific cache."""
        cache_metrics = self.cache_monitor.get_cache_metrics(cache_name)
        if not cache_metrics:
            return {"error": "No metrics available"}

        recommendations = []
        priority_scores = []

        hit_rate = cache_metrics.get("hit_rate", 0.0)
        error_rate = cache_metrics.get("errors", {}).get("rate", 0.0)
        response_time = cache_metrics.get("performance", {}).get("average_response_time_ms", 0.0)

        # Hit rate recommendations
        if hit_rate < 0.5:
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "high",
                    "issue": f"Low hit rate ({hit_rate:.1%})",
                    "recommendation": "Review cache key generation strategy and TTL settings",
                    "impact": "High performance improvement potential",
                }
            )
            priority_scores.append(8)
        elif hit_rate < 0.7:
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "medium",
                    "issue": f"Moderate hit rate ({hit_rate:.1%})",
                    "recommendation": "Optimize cache key patterns and consider increasing cache size",
                    "impact": "Moderate performance improvement",
                }
            )
            priority_scores.append(5)

        # Error rate recommendations
        if error_rate > 0.05:
            recommendations.append(
                {
                    "category": "reliability",
                    "priority": "high",
                    "issue": f"High error rate ({error_rate:.1%})",
                    "recommendation": "Investigate connection stability and error handling",
                    "impact": "Critical for system reliability",
                }
            )
            priority_scores.append(9)

        # Response time recommendations
        if response_time > 1000:
            recommendations.append(
                {
                    "category": "performance",
                    "priority": "high",
                    "issue": f"Slow response time ({response_time:.0f}ms)",
                    "recommendation": "Optimize cache operations or consider infrastructure scaling",
                    "impact": "User experience improvement",
                }
            )
            priority_scores.append(7)

        # Memory recommendations
        memory_mb = cache_metrics.get("memory", {}).get("current_size_bytes", 0) / 1024 / 1024
        if memory_mb > 200:
            recommendations.append(
                {
                    "category": "resource",
                    "priority": "medium",
                    "issue": f"High memory usage ({memory_mb:.1f}MB)",
                    "recommendation": "Implement more aggressive eviction policies or increase memory limits",
                    "impact": "Resource optimization",
                }
            )
            priority_scores.append(4)

        return {
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "priority_summary": {
                "high_priority": len([r for r in recommendations if r["priority"] == "high"]),
                "medium_priority": len([r for r in recommendations if r["priority"] == "medium"]),
                "low_priority": len([r for r in recommendations if r["priority"] == "low"]),
            },
            "overall_health_score": round(100 - sum(priority_scores), 1) if priority_scores else 95,
        }

    def _generate_live_metrics(self, latest_report: dict[str, Any]) -> dict[str, Any]:
        """Generate live metrics section."""
        summary = latest_report.get("summary", {})
        deltas = latest_report.get("deltas", {})

        return {
            "current_metrics": {
                "total_operations": summary.get("total_operations", 0),
                "hit_rate_percentage": round(summary.get("overall_hit_rate", 0.0) * 100, 1),
                "error_rate_percentage": round(summary.get("overall_error_rate", 0.0) * 100, 2),
                "memory_mb": round(summary.get("total_size_mb", 0.0), 1),
                "active_caches": summary.get("total_caches", 0),
            },
            "live_rates": (
                {
                    "operations_per_second": deltas.get("operations_per_second", 0),
                    "hits_per_second": deltas.get("hits_per_second", 0),
                    "misses_per_second": deltas.get("misses_per_second", 0),
                    "errors_per_second": deltas.get("errors_per_second", 0),
                }
                if deltas.get("available")
                else {"status": "No rate data available"}
            ),
            "last_update": latest_report.get("timestamp", time.time()),
        }

    def _generate_activity_feed(self, latest_report: dict[str, Any]) -> dict[str, Any]:
        """Generate activity feed section."""
        return {
            "recent_activity": "Live activity feed would require event streaming implementation",
            "cache_activities": latest_report.get("individual_caches", {}),
            "system_events": "System events would be tracked here",
        }

    def _generate_performance_gauges(self, latest_report: dict[str, Any]) -> dict[str, Any]:
        """Generate performance gauge data."""
        summary = latest_report.get("summary", {})

        return {
            "hit_rate_gauge": {
                "value": round(summary.get("overall_hit_rate", 0.0) * 100, 1),
                "min": 0,
                "max": 100,
                "thresholds": {"warning": 60, "critical": 30},
                "status": "good" if summary.get("overall_hit_rate", 0.0) > 0.6 else "warning",
            },
            "error_rate_gauge": {
                "value": round(summary.get("overall_error_rate", 0.0) * 100, 2),
                "min": 0,
                "max": 20,  # 20% max for gauge display
                "thresholds": {"warning": 5, "critical": 10},
                "status": "critical" if summary.get("overall_error_rate", 0.0) > 0.1 else "good",
            },
            "memory_gauge": {
                "value": round(summary.get("total_size_mb", 0.0), 1),
                "min": 0,
                "max": 1000,  # 1GB max for gauge display
                "thresholds": {"warning": 500, "critical": 800},
                "status": "good",
            },
        }

    def _generate_memory_monitoring(self, latest_report: dict[str, Any]) -> dict[str, Any]:
        """Generate memory monitoring section."""
        system_info = latest_report.get("system_info", {})

        return {
            "system_memory": {
                "cpu_percent": system_info.get("cpu_percent", 0),
                "memory_info": system_info.get("memory_info", {}),
            },
            "cache_memory": "Detailed cache memory tracking from latest report",
            "memory_pressure": "Memory pressure indicators would be shown here",
        }

    def _generate_alerts_feed(self, latest_report: dict[str, Any]) -> dict[str, Any]:
        """Generate alerts feed section."""
        alerts = latest_report.get("alerts", [])

        return {
            "active_alerts": len(alerts),
            "recent_alerts": alerts[-5:] if alerts else [],
            "alert_stream": "Real-time alert stream would be implemented here",
        }

    def _generate_system_health(self, latest_report: dict[str, Any]) -> dict[str, Any]:
        """Generate system health section."""
        return {
            "overall_health": "healthy",  # Would calculate from latest report
            "monitoring_status": "active",
            "last_update": latest_report.get("timestamp", time.time()),
            "health_indicators": {
                "cache_connectivity": "good",
                "performance": "good",
                "memory": "good",
                "errors": "good",
            },
        }


# Global dashboard generator instance
_dashboard_generator = CacheDashboardGenerator()


def get_cache_dashboard_generator() -> CacheDashboardGenerator:
    """Get the global cache dashboard generator instance."""
    return _dashboard_generator


async def generate_cache_overview_dashboard() -> dict[str, Any]:
    """
    Convenience function to generate cache overview dashboard.

    Returns:
        Dict containing overview dashboard data
    """
    generator = get_cache_dashboard_generator()
    return generator.generate_overview_dashboard()


async def generate_cache_detailed_dashboard(cache_name: str) -> dict[str, Any]:
    """
    Convenience function to generate detailed cache dashboard.

    Args:
        cache_name: Name of the cache service

    Returns:
        Dict containing detailed dashboard data for the specific cache
    """
    generator = get_cache_dashboard_generator()
    return generator.generate_detailed_dashboard(cache_name)


async def generate_cache_real_time_dashboard() -> dict[str, Any]:
    """
    Convenience function to generate real-time cache dashboard.

    Returns:
        Dict containing real-time dashboard data
    """
    generator = get_cache_dashboard_generator()
    return generator.generate_real_time_dashboard()
