"""
Performance Dashboard Tool - Wave 5.0 Implementation.

Provides a comprehensive performance monitoring dashboard interface with:
- Real-time performance metrics visualization
- Performance alerts and notifications
- System health monitoring
- Historical trend analysis
- Optimization recommendations display
- Interactive performance controls
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from src.services.performance_monitor import (
    AlertSeverity,
    AlertStatus,
    PerformanceMetricType,
    get_performance_monitor,
)
from src.utils.performance_monitor import (
    get_cache_performance_monitor,
    get_real_time_cache_reporter,
)


class PerformanceDashboard:
    """
    Interactive performance dashboard for monitoring system performance.

    Provides comprehensive performance monitoring capabilities including:
    - Real-time metric visualization
    - Alert management
    - Historical trend analysis
    - Performance optimization guidance
    """

    def __init__(self):
        """Initialize the performance dashboard."""
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = get_performance_monitor()
        self.cache_monitor = get_cache_performance_monitor()
        self.cache_reporter = get_real_time_cache_reporter()

        # Dashboard state
        self._refresh_interval = 5.0  # Default 5 second refresh
        self._is_running = False
        self._dashboard_task: asyncio.Task | None = None

        self.logger.info("PerformanceDashboard initialized")

    async def start_dashboard(self, auto_refresh: bool = True, refresh_interval: float = 5.0):
        """
        Start the performance dashboard.

        Args:
            auto_refresh: Whether to automatically refresh the dashboard
            refresh_interval: Seconds between dashboard refreshes
        """
        if self._is_running:
            return {"status": "already_running", "message": "Dashboard already running"}

        try:
            self._refresh_interval = refresh_interval
            self._is_running = True

            # Ensure performance monitoring is started
            if not self.performance_monitor._is_running:
                await self.performance_monitor.start_monitoring()

            if auto_refresh:
                self._dashboard_task = asyncio.create_task(self._dashboard_loop())

            self.logger.info(f"Performance dashboard started with {refresh_interval}s refresh")
            return {
                "status": "started",
                "message": "Performance dashboard started successfully",
                "refresh_interval": refresh_interval,
                "auto_refresh": auto_refresh,
            }

        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
            self._is_running = False
            return {"status": "error", "message": f"Failed to start dashboard: {e}"}

    async def stop_dashboard(self):
        """Stop the performance dashboard."""
        self._is_running = False

        if self._dashboard_task:
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance dashboard stopped")
        return {"status": "stopped", "message": "Dashboard stopped successfully"}

    async def _dashboard_loop(self):
        """Main dashboard refresh loop."""
        self.logger.info("Dashboard auto-refresh loop started")

        while self._is_running:
            try:
                # Generate and log dashboard update
                dashboard_data = await self.get_dashboard_data()
                self._log_dashboard_summary(dashboard_data)

                # Wait for next refresh
                await asyncio.sleep(self._refresh_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in dashboard loop: {e}")
                await asyncio.sleep(min(self._refresh_interval, 30.0))

        self.logger.info("Dashboard auto-refresh loop stopped")

    def _log_dashboard_summary(self, dashboard_data: dict[str, Any]):
        """Log a summary of the current dashboard state."""
        try:
            summary = dashboard_data.get("summary", {})
            alerts = dashboard_data.get("alerts", {})

            self.logger.info(
                f"Dashboard Update - "
                f"Status: {summary.get('status', 'unknown')}, "
                f"Active Alerts: {alerts.get('total_active', 0)}, "
                f"Critical: {alerts.get('critical', 0)}, "
                f"Warnings: {alerts.get('warnings', 0)}, "
                f"Components: {len(summary.get('components', {}))}"
            )

        except Exception as e:
            self.logger.error(f"Error logging dashboard summary: {e}")

    async def get_dashboard_data(self, include_history: bool = False) -> dict[str, Any]:
        """
        Get comprehensive dashboard data.

        Args:
            include_history: Whether to include historical trend data

        Returns:
            Dictionary containing all dashboard data
        """
        try:
            current_time = time.time()

            # Get performance summary
            performance_summary = self.performance_monitor.get_performance_summary()

            # Get current metrics
            current_metrics = self.performance_monitor.get_current_metrics()

            # Get active alerts
            active_alerts = self.performance_monitor.get_active_alerts()

            # Get optimization recommendations
            recommendations = self.performance_monitor.get_optimization_recommendations()

            # Get cache performance data
            cache_data = await self._get_cache_performance_data()

            # Get system health data
            system_health = await self._get_system_health_data()

            # Organize metrics by component and type
            metrics_by_component = self._organize_metrics_by_component(current_metrics)

            # Generate alert summary
            alert_summary = self._generate_alert_summary(active_alerts)

            # Generate recommendations summary
            recommendations_summary = self._generate_recommendations_summary(recommendations)

            dashboard_data = {
                "timestamp": current_time,
                "status": "active" if self.performance_monitor._is_running else "inactive",
                "summary": {
                    "status": system_health.get("overall_status", "unknown"),
                    "uptime_seconds": current_time - performance_summary.get("start_time", current_time),
                    "total_metrics": performance_summary.get("total_metrics_collected", 0),
                    "components": list(metrics_by_component.keys()),
                    "monitoring_active": self.performance_monitor._is_running,
                    "dashboard_active": self._is_running,
                    "refresh_interval": self._refresh_interval,
                },
                "metrics": {
                    "by_component": metrics_by_component,
                    "recent_count": len(current_metrics),
                    "collection_rate": len(current_metrics) / self._refresh_interval if current_metrics else 0,
                },
                "alerts": alert_summary,
                "recommendations": recommendations_summary,
                "cache_performance": cache_data,
                "system_health": system_health,
                "performance_summary": performance_summary,
            }

            # Add historical data if requested
            if include_history:
                dashboard_data["history"] = await self._get_historical_data()

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {"timestamp": time.time(), "status": "error", "error": str(e), "summary": {"status": "error", "components": []}}

    def _organize_metrics_by_component(self, metrics: list) -> dict[str, dict[str, Any]]:
        """Organize metrics by component and metric type."""
        try:
            organized = defaultdict(lambda: defaultdict(list))

            for metric in metrics:
                organized[metric.component][metric.metric_type.value].append(
                    {
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp,
                        "tags": metric.tags,
                        "metadata": metric.metadata,
                    }
                )

            # Convert to regular dict and add summary statistics
            result = {}
            for component, metric_types in organized.items():
                result[component] = {}
                for metric_type, values in metric_types.items():
                    if values:
                        latest = values[-1]  # Most recent value
                        result[component][metric_type] = {
                            "current": latest["value"],
                            "unit": latest["unit"],
                            "timestamp": latest["timestamp"],
                            "count": len(values),
                            "values": values[-10:] if len(values) > 10 else values,  # Last 10 values
                        }

            return result

        except Exception as e:
            self.logger.error(f"Error organizing metrics: {e}")
            return {}

    def _generate_alert_summary(self, alerts: list) -> dict[str, Any]:
        """Generate a summary of performance alerts."""
        try:
            summary = {
                "total_active": len(alerts),
                "critical": 0,
                "errors": 0,
                "warnings": 0,
                "info": 0,
                "by_component": defaultdict(int),
                "by_type": defaultdict(int),
                "recent_alerts": [],
            }

            for alert in alerts:
                # Count by severity
                if alert.severity == AlertSeverity.CRITICAL:
                    summary["critical"] += 1
                elif alert.severity == AlertSeverity.ERROR:
                    summary["errors"] += 1
                elif alert.severity == AlertSeverity.WARNING:
                    summary["warnings"] += 1
                elif alert.severity == AlertSeverity.INFO:
                    summary["info"] += 1

                # Count by component
                summary["by_component"][alert.component] += 1

                # Count by alert type
                summary["by_type"][alert.alert_type] += 1

                # Add to recent alerts (with serializable data)
                summary["recent_alerts"].append(
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity.value,
                        "component": alert.component,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "message": alert.message,
                        "timestamp": alert.timestamp,
                        "age_seconds": alert.age_seconds,
                        "status": alert.status.value,
                    }
                )

            # Sort recent alerts by timestamp (newest first)
            summary["recent_alerts"].sort(key=lambda x: x["timestamp"], reverse=True)
            summary["recent_alerts"] = summary["recent_alerts"][:20]  # Keep only 20 most recent

            # Convert defaultdicts to regular dicts
            summary["by_component"] = dict(summary["by_component"])
            summary["by_type"] = dict(summary["by_type"])

            return summary

        except Exception as e:
            self.logger.error(f"Error generating alert summary: {e}")
            return {"total_active": 0, "critical": 0, "errors": 0, "warnings": 0, "info": 0}

    def _generate_recommendations_summary(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a summary of optimization recommendations."""
        try:
            summary = {
                "total": len(recommendations),
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0,
                "by_type": defaultdict(int),
                "by_component": defaultdict(int),
                "recent_recommendations": [],
            }

            for rec in recommendations:
                # Count by priority
                priority = rec.get("priority", "medium")
                if priority == "high":
                    summary["high_priority"] += 1
                elif priority == "medium":
                    summary["medium_priority"] += 1
                elif priority == "low":
                    summary["low_priority"] += 1

                # Count by type and component
                summary["by_type"][rec.get("type", "unknown")] += 1
                summary["by_component"][rec.get("component", "unknown")] += 1

                # Add to recent recommendations
                summary["recent_recommendations"].append(
                    {
                        "type": rec.get("type", "unknown"),
                        "component": rec.get("component", "unknown"),
                        "priority": priority,
                        "message": rec.get("message", ""),
                        "current_value": rec.get("current_value"),
                        "recommendations": rec.get("recommendations", []),
                        "timestamp": rec.get("timestamp", time.time()),
                    }
                )

            # Sort by timestamp (newest first)
            summary["recent_recommendations"].sort(key=lambda x: x["timestamp"], reverse=True)
            summary["recent_recommendations"] = summary["recent_recommendations"][:15]  # Keep only 15 most recent

            # Convert defaultdicts to regular dicts
            summary["by_type"] = dict(summary["by_type"])
            summary["by_component"] = dict(summary["by_component"])

            return summary

        except Exception as e:
            self.logger.error(f"Error generating recommendations summary: {e}")
            return {"total": 0, "high_priority": 0, "medium_priority": 0, "low_priority": 0}

    async def _get_cache_performance_data(self) -> dict[str, Any]:
        """Get cache performance data for dashboard."""
        try:
            # Get aggregated cache metrics
            cache_metrics = self.cache_monitor.get_aggregated_metrics()

            # Get latest cache report
            latest_report = self.cache_reporter.get_latest_report()

            cache_data = {
                "status": "active" if self.cache_monitor.is_monitoring_enabled() else "inactive",
                "overall_hit_rate": 0.0,
                "total_operations": 0,
                "error_rate": 0.0,
                "response_time_ms": 0.0,
                "cache_types": {},
                "alerts_count": 0,
            }

            if cache_metrics and "summary" in cache_metrics:
                summary = cache_metrics["summary"]
                cache_data.update(
                    {
                        "overall_hit_rate": summary.get("overall_hit_rate", 0.0) * 100,
                        "total_operations": summary.get("total_operations", 0),
                        "error_rate": summary.get("overall_error_rate", 0.0) * 100,
                        "response_time_ms": summary.get("average_response_time_ms", 0.0),
                        "cache_types": cache_metrics.get("cache_type_summary", {}),
                        "alerts_count": cache_metrics.get("active_alerts", 0),
                    }
                )

            # Add trend data if available
            if latest_report:
                cache_data["trends"] = latest_report.get("trends", {})
                cache_data["deltas"] = latest_report.get("deltas", {})

            return cache_data

        except Exception as e:
            self.logger.error(f"Error getting cache performance data: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_system_health_data(self) -> dict[str, Any]:
        """Get system health data for dashboard."""
        try:
            import psutil

            # System metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage("/")

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            # Determine overall health status
            health_score = 100
            issues = []

            # Memory check
            if memory.percent > 85:
                health_score -= 20
                issues.append("High system memory usage")

            # CPU check
            if cpu_percent > 80:
                health_score -= 15
                issues.append("High CPU usage")

            # Disk check
            if disk.percent > 90:
                health_score -= 25
                issues.append("Low disk space")

            # Process memory check
            process_memory_mb = process_memory.rss / (1024 * 1024)
            if process_memory_mb > 2048:  # 2GB
                health_score -= 10
                issues.append("High process memory usage")

            # Determine status
            if health_score >= 90:
                status = "excellent"
            elif health_score >= 75:
                status = "good"
            elif health_score >= 60:
                status = "fair"
            elif health_score >= 40:
                status = "poor"
            else:
                status = "critical"

            return {
                "overall_status": status,
                "health_score": health_score,
                "issues": issues,
                "system": {
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "cpu_percent": cpu_percent,
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                "process": {
                    "memory_mb": process_memory_mb,
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()) if hasattr(process, "open_files") else 0,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting system health data: {e}")
            return {"overall_status": "unknown", "health_score": 0, "issues": [f"Health check failed: {e}"], "system": {}, "process": {}}

    async def _get_historical_data(self) -> dict[str, Any]:
        """Get historical performance data for trends."""
        try:
            # Get trend data from performance monitor
            historical_data = {"time_range_hours": 1, "data_points": [], "trends": {}}  # Last hour

            # Get metrics from the last hour
            current_time = time.time()
            hour_ago = current_time - 3600

            recent_metrics = [m for m in self.performance_monitor._metrics if m.timestamp > hour_ago]

            # Group by 5-minute intervals
            intervals = {}
            for metric in recent_metrics:
                interval = int((metric.timestamp - hour_ago) // 300) * 300  # 5-minute intervals
                if interval not in intervals:
                    intervals[interval] = defaultdict(list)
                intervals[interval][f"{metric.component}_{metric.metric_type.value}"].append(metric.value)

            # Convert to time series data
            for interval, metrics in sorted(intervals.items()):
                data_point = {"timestamp": hour_ago + interval, "metrics": {}}

                for metric_key, values in metrics.items():
                    if values:
                        data_point["metrics"][metric_key] = {
                            "average": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                        }

                historical_data["data_points"].append(data_point)

            return historical_data

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return {"time_range_hours": 0, "data_points": [], "trends": {}}

    async def get_component_details(self, component: str) -> dict[str, Any]:
        """Get detailed performance information for a specific component."""
        try:
            # Get component metrics
            component_metrics = self.performance_monitor.get_current_metrics(component=component)

            # Get component alerts
            component_alerts = [alert for alert in self.performance_monitor.get_active_alerts() if alert.component == component]

            # Get component recommendations
            component_recommendations = self.performance_monitor.get_optimization_recommendations(component=component)

            # Organize metrics by type
            metrics_by_type = defaultdict(list)
            for metric in component_metrics:
                metrics_by_type[metric.metric_type.value].append(
                    {"value": metric.value, "unit": metric.unit, "timestamp": metric.timestamp, "age_seconds": metric.age_seconds}
                )

            return {
                "component": component,
                "timestamp": time.time(),
                "metrics": dict(metrics_by_type),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "age_seconds": alert.age_seconds,
                    }
                    for alert in component_alerts
                ],
                "recommendations": component_recommendations,
                "performance_score": self._calculate_component_performance_score(component_metrics),
                "status": self._determine_component_status(component_alerts, component_recommendations),
            }

        except Exception as e:
            self.logger.error(f"Error getting component details for {component}: {e}")
            return {"component": component, "error": str(e), "status": "error"}

    def _calculate_component_performance_score(self, metrics: list) -> float:
        """Calculate a performance score for a component based on its metrics."""
        try:
            if not metrics:
                return 50.0  # Default neutral score

            score = 100.0

            # Analyze different metric types
            for metric in metrics:
                if metric.metric_type == PerformanceMetricType.RESPONSE_TIME:
                    # Lower response time is better
                    if metric.value > 30000:  # > 30 seconds
                        score -= 30
                    elif metric.value > 15000:  # > 15 seconds
                        score -= 15
                    elif metric.value > 5000:  # > 5 seconds
                        score -= 5

                elif metric.metric_type == PerformanceMetricType.ERROR_RATE:
                    # Lower error rate is better
                    if metric.value > 10:  # > 10%
                        score -= 25
                    elif metric.value > 5:  # > 5%
                        score -= 15
                    elif metric.value > 1:  # > 1%
                        score -= 5

                elif metric.metric_type == PerformanceMetricType.CACHE_HIT_RATE:
                    # Higher cache hit rate is better
                    if metric.value < 50:  # < 50%
                        score -= 20
                    elif metric.value < 70:  # < 70%
                        score -= 10
                    elif metric.value < 85:  # < 85%
                        score -= 5

                elif metric.metric_type == PerformanceMetricType.MEMORY_USAGE:
                    # Check if memory usage is excessive
                    if metric.value > 2048:  # > 2GB
                        score -= 15
                    elif metric.value > 1024:  # > 1GB
                        score -= 8

                elif metric.metric_type == PerformanceMetricType.CPU_USAGE:
                    # Check if CPU usage is high
                    if metric.value > 90:  # > 90%
                        score -= 20
                    elif metric.value > 80:  # > 80%
                        score -= 10
                    elif metric.value > 70:  # > 70%
                        score -= 5

            return max(0.0, min(100.0, score))

        except Exception as e:
            self.logger.error(f"Error calculating component performance score: {e}")
            return 50.0

    def _determine_component_status(self, alerts: list, recommendations: list[dict[str, Any]]) -> str:
        """Determine the overall status of a component."""
        try:
            # Check for critical alerts
            if any(alert.severity == AlertSeverity.CRITICAL for alert in alerts):
                return "critical"

            # Check for error alerts
            if any(alert.severity == AlertSeverity.ERROR for alert in alerts):
                return "error"

            # Check for warning alerts
            if any(alert.severity == AlertSeverity.WARNING for alert in alerts):
                return "warning"

            # Check for high-priority recommendations
            if any(rec.get("priority") == "high" for rec in recommendations):
                return "needs_attention"

            # Check for any recommendations
            if recommendations:
                return "fair"

            return "good"

        except Exception as e:
            self.logger.error(f"Error determining component status: {e}")
            return "unknown"

    async def acknowledge_alert(self, alert_id: str, user: str = "dashboard_user") -> dict[str, Any]:
        """Acknowledge a performance alert."""
        try:
            success = await self.performance_monitor.acknowledge_alert(alert_id, user)

            return {
                "success": success,
                "message": f"Alert {'acknowledged' if success else 'not found or already acknowledged'}",
                "alert_id": alert_id,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return {"success": False, "message": f"Error acknowledging alert: {e}", "alert_id": alert_id}

    async def resolve_alert(self, alert_id: str, resolution_note: str = "", user: str = "dashboard_user") -> dict[str, Any]:
        """Resolve a performance alert."""
        try:
            success = await self.performance_monitor.resolve_alert(alert_id, user, resolution_note)

            return {
                "success": success,
                "message": f"Alert {'resolved' if success else 'not found or already resolved'}",
                "alert_id": alert_id,
                "resolution_note": resolution_note,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error resolving alert {alert_id}: {e}")
            return {"success": False, "message": f"Error resolving alert: {e}", "alert_id": alert_id}

    def set_refresh_interval(self, interval: float) -> dict[str, Any]:
        """Update the dashboard refresh interval."""
        try:
            old_interval = self._refresh_interval
            self._refresh_interval = max(1.0, interval)  # Minimum 1 second

            return {
                "success": True,
                "message": f"Refresh interval updated from {old_interval}s to {self._refresh_interval}s",
                "old_interval": old_interval,
                "new_interval": self._refresh_interval,
            }

        except Exception as e:
            self.logger.error(f"Error setting refresh interval: {e}")
            return {"success": False, "message": f"Error setting refresh interval: {e}"}

    async def export_dashboard_data(
        self, format_type: str = "json", include_history: bool = True, file_path: str | None = None
    ) -> dict[str, Any]:
        """Export dashboard data to file or return as formatted data."""
        try:
            dashboard_data = await self.get_dashboard_data(include_history=include_history)

            if format_type.lower() == "json":
                exported_data = json.dumps(dashboard_data, indent=2)
            else:
                return {"success": False, "message": f"Unsupported format type: {format_type}"}

            if file_path:
                with open(file_path, "w") as f:
                    f.write(exported_data)

                return {
                    "success": True,
                    "message": f"Dashboard data exported to {file_path}",
                    "file_path": file_path,
                    "format": format_type,
                    "size_bytes": len(exported_data),
                }
            else:
                return {
                    "success": True,
                    "message": "Dashboard data exported successfully",
                    "data": exported_data,
                    "format": format_type,
                    "size_bytes": len(exported_data),
                }

        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")
            return {"success": False, "message": f"Error exporting dashboard data: {e}"}

    async def get_dashboard_status(self) -> dict[str, Any]:
        """Get the current status of the dashboard system."""
        try:
            return {
                "dashboard_active": self._is_running,
                "performance_monitoring_active": self.performance_monitor._is_running,
                "cache_monitoring_active": self.cache_monitor.is_monitoring_enabled(),
                "cache_reporting_active": self.cache_reporter._reporting_enabled,
                "refresh_interval": self._refresh_interval,
                "uptime_seconds": time.time() - getattr(self, "_start_time", time.time()),
                "components_monitored": len(set(m.component for m in self.performance_monitor.get_current_metrics())),
                "active_alerts": len(self.performance_monitor.get_active_alerts()),
                "total_recommendations": len(self.performance_monitor.get_optimization_recommendations()),
                "memory_usage_mb": self._get_dashboard_memory_usage(),
            }

        except Exception as e:
            self.logger.error(f"Error getting dashboard status: {e}")
            return {"error": str(e), "dashboard_active": False}

    def _get_dashboard_memory_usage(self) -> float:
        """Get the memory usage of the dashboard system."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    async def shutdown(self):
        """Shutdown the performance dashboard."""
        self.logger.info("Shutting down PerformanceDashboard")
        await self.stop_dashboard()
        self.logger.info("PerformanceDashboard shutdown complete")


# Global dashboard instance
_dashboard: PerformanceDashboard | None = None


def get_performance_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = PerformanceDashboard()
    return _dashboard
