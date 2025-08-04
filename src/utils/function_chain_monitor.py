"""
Function Chain Monitoring and Logging System.

This module provides comprehensive monitoring, logging, and metrics collection
for Function Chain MCP tools, tracking usage patterns and performance metrics.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class FunctionChainMetric:
    """Metric for function chain tool usage."""

    tool_name: str
    operation_type: str  # "trace", "path_finding", "project_analysis"
    project_name: str
    entry_point: str
    timestamp: datetime
    response_time: float
    success: bool
    result_size: int  # Number of functions/paths found
    error_message: str | None = None
    user_context: str | None = None
    complexity_score: float | None = None
    cache_hit: bool = False


@dataclass
class PerformanceAlert:
    """Performance alert for monitoring."""

    alert_type: str  # "slow_response", "high_error_rate", "cache_miss"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime
    tool_name: str
    metric_value: float
    threshold: float


class FunctionChainMonitor:
    """Comprehensive monitoring system for Function Chain tools."""

    def __init__(
        self,
        enable_file_logging: bool = True,
        log_file_path: str | None = None,
        metrics_retention_hours: int = 24,
        alert_thresholds: dict[str, float] | None = None,
    ):
        """
        Initialize the Function Chain monitor.

        Args:
            enable_file_logging: Whether to enable file-based logging
            log_file_path: Path to log file (default: auto-generate)
            metrics_retention_hours: Hours to retain metrics in memory
            alert_thresholds: Custom alert thresholds
        """
        self.enable_file_logging = enable_file_logging
        self.metrics_retention_hours = metrics_retention_hours

        # Set up log file path
        if log_file_path:
            self.log_file_path = Path(log_file_path)
        else:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.log_file_path = log_dir / f"function_chain_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"

        # Metrics storage
        self.metrics: deque[FunctionChainMetric] = deque()
        self.alerts: deque[PerformanceAlert] = deque()

        # Performance tracking
        self.tool_stats = defaultdict(
            lambda: {"total_calls": 0, "successful_calls": 0, "total_response_time": 0.0, "error_count": 0, "cache_hits": 0}
        )

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "slow_response_time": 3.0,  # seconds
            "high_error_rate": 0.15,  # 15%
            "low_cache_hit_rate": 0.30,  # 30%
            "high_complexity": 0.8,  # complexity score
        }

        # Background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

        logger.info(f"Function Chain monitor initialized. Logging to: {self.log_file_path}")

    def record_tool_usage(
        self,
        tool_name: str,
        operation_type: str,
        project_name: str,
        entry_point: str,
        response_time: float,
        success: bool,
        result_size: int = 0,
        error_message: str | None = None,
        user_context: str | None = None,
        complexity_score: float | None = None,
        cache_hit: bool = False,
    ) -> None:
        """
        Record a tool usage metric.

        Args:
            tool_name: Name of the tool used
            operation_type: Type of operation performed
            project_name: Name of the project analyzed
            entry_point: Entry point or query used
            response_time: Time taken for the operation
            success: Whether the operation was successful
            result_size: Number of results returned
            error_message: Error message if operation failed
            user_context: Additional user context
            complexity_score: Complexity score if calculated
            cache_hit: Whether the result came from cache
        """
        # Create metric
        metric = FunctionChainMetric(
            tool_name=tool_name,
            operation_type=operation_type,
            project_name=project_name,
            entry_point=entry_point,
            timestamp=datetime.now(),
            response_time=response_time,
            success=success,
            result_size=result_size,
            error_message=error_message,
            user_context=user_context,
            complexity_score=complexity_score,
            cache_hit=cache_hit,
        )

        # Store metric
        self.metrics.append(metric)

        # Update tool statistics
        self._update_tool_stats(metric)

        # Check for alerts
        self._check_alerts(metric)

        # Log to file if enabled
        if self.enable_file_logging:
            self._log_metric_to_file(metric)

        logger.debug(f"Recorded metric: {tool_name} - {operation_type} ({response_time:.3f}s)")

    def _update_tool_stats(self, metric: FunctionChainMetric) -> None:
        """Update tool statistics with new metric."""
        stats = self.tool_stats[metric.tool_name]
        stats["total_calls"] += 1
        stats["total_response_time"] += metric.response_time

        if metric.success:
            stats["successful_calls"] += 1
        else:
            stats["error_count"] += 1

        if metric.cache_hit:
            stats["cache_hits"] += 1

    def _check_alerts(self, metric: FunctionChainMetric) -> None:
        """Check if the metric triggers any alerts."""
        alerts = []

        # Check slow response time
        if metric.response_time > self.alert_thresholds["slow_response_time"]:
            severity = "high" if metric.response_time > 5.0 else "medium"
            alerts.append(
                PerformanceAlert(
                    alert_type="slow_response",
                    severity=severity,
                    message=f"Slow response time: {metric.response_time:.2f}s for {metric.tool_name}",
                    timestamp=metric.timestamp,
                    tool_name=metric.tool_name,
                    metric_value=metric.response_time,
                    threshold=self.alert_thresholds["slow_response_time"],
                )
            )

        # Check high complexity
        if metric.complexity_score and metric.complexity_score > self.alert_thresholds["high_complexity"]:
            alerts.append(
                PerformanceAlert(
                    alert_type="high_complexity",
                    severity="medium",
                    message=f"High complexity detected: {metric.complexity_score:.2f} for {metric.entry_point}",
                    timestamp=metric.timestamp,
                    tool_name=metric.tool_name,
                    metric_value=metric.complexity_score,
                    threshold=self.alert_thresholds["high_complexity"],
                )
            )

        # Check error rates (calculated over recent metrics)
        recent_metrics = self._get_recent_metrics_for_tool(metric.tool_name, minutes=10)
        if len(recent_metrics) >= 5:  # Only check if we have enough data
            error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
            if error_rate > self.alert_thresholds["high_error_rate"]:
                alerts.append(
                    PerformanceAlert(
                        alert_type="high_error_rate",
                        severity="high",
                        message=f"High error rate: {error_rate:.1%} for {metric.tool_name}",
                        timestamp=metric.timestamp,
                        tool_name=metric.tool_name,
                        metric_value=error_rate,
                        threshold=self.alert_thresholds["high_error_rate"],
                    )
                )

        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"ALERT [{alert.severity.upper()}] {alert.message}")

    def _log_metric_to_file(self, metric: FunctionChainMetric) -> None:
        """Log metric to file in JSONL format."""
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                metric_data = asdict(metric)
                # Convert datetime to ISO string
                metric_data["timestamp"] = metric.timestamp.isoformat()
                f.write(json.dumps(metric_data) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metric to file: {e}")

    def get_tool_statistics(self, tool_name: str | None = None, time_window_hours: int = 24) -> dict[str, Any]:
        """
        Get statistics for tools within a time window.

        Args:
            tool_name: Specific tool name (None for all tools)
            time_window_hours: Time window in hours

        Returns:
            Dictionary containing tool statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        # Filter metrics by time window
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        if tool_name:
            recent_metrics = [m for m in recent_metrics if m.tool_name == tool_name]

        if not recent_metrics:
            return {"message": "No metrics found for the specified criteria"}

        # Calculate statistics
        stats = {}

        # Group by tool name
        tools = defaultdict(list)
        for metric in recent_metrics:
            tools[metric.tool_name].append(metric)

        for tool, metrics in tools.items():
            successful_metrics = [m for m in metrics if m.success]
            response_times = [m.response_time for m in metrics]

            tool_stats = {
                "total_calls": len(metrics),
                "successful_calls": len(successful_metrics),
                "success_rate": len(successful_metrics) / len(metrics),
                "error_rate": (len(metrics) - len(successful_metrics)) / len(metrics),
                "avg_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "cache_hits": sum(1 for m in metrics if m.cache_hit),
                "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics),
                "avg_result_size": sum(m.result_size for m in metrics) / len(metrics),
            }

            # Add complexity statistics if available
            complexity_scores = [m.complexity_score for m in metrics if m.complexity_score]
            if complexity_scores:
                tool_stats["avg_complexity"] = sum(complexity_scores) / len(complexity_scores)
                tool_stats["max_complexity"] = max(complexity_scores)

            stats[tool] = tool_stats

        return {
            "time_window_hours": time_window_hours,
            "total_metrics": len(recent_metrics),
            "tools": stats,
            "overall": {
                "total_calls": len(recent_metrics),
                "avg_response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
                "success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
            },
        }

    def get_usage_patterns(self, time_window_hours: int = 24) -> dict[str, Any]:
        """
        Analyze usage patterns within a time window.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Dictionary containing usage pattern analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {"message": "No metrics found for the specified time window"}

        # Analyze patterns
        patterns = {
            "most_used_tools": {},
            "most_analyzed_projects": {},
            "common_entry_points": {},
            "operation_types": {},
            "hourly_distribution": defaultdict(int),
            "performance_trends": {"response_times": [], "success_rates": [], "timestamps": []},
        }

        # Count usage patterns
        for metric in recent_metrics:
            # Tool usage
            patterns["most_used_tools"][metric.tool_name] = patterns["most_used_tools"].get(metric.tool_name, 0) + 1

            # Project analysis
            patterns["most_analyzed_projects"][metric.project_name] = patterns["most_analyzed_projects"].get(metric.project_name, 0) + 1

            # Entry points
            patterns["common_entry_points"][metric.entry_point] = patterns["common_entry_points"].get(metric.entry_point, 0) + 1

            # Operation types
            patterns["operation_types"][metric.operation_type] = patterns["operation_types"].get(metric.operation_type, 0) + 1

            # Hourly distribution
            hour = metric.timestamp.hour
            patterns["hourly_distribution"][hour] += 1

            # Performance trends
            patterns["performance_trends"]["response_times"].append(metric.response_time)
            patterns["performance_trends"]["success_rates"].append(1 if metric.success else 0)
            patterns["performance_trends"]["timestamps"].append(metric.timestamp.isoformat())

        # Sort by frequency
        patterns["most_used_tools"] = dict(sorted(patterns["most_used_tools"].items(), key=lambda x: x[1], reverse=True))
        patterns["most_analyzed_projects"] = dict(sorted(patterns["most_analyzed_projects"].items(), key=lambda x: x[1], reverse=True))
        patterns["common_entry_points"] = dict(
            sorted(patterns["common_entry_points"].items(), key=lambda x: x[1], reverse=True)[:10]
        )  # Top 10

        return patterns

    def get_recent_alerts(
        self, severity_filter: str | None = None, alert_type_filter: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Get recent performance alerts.

        Args:
            severity_filter: Filter by severity level
            alert_type_filter: Filter by alert type
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        alerts = list(self.alerts)

        # Apply filters
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]

        if alert_type_filter:
            alerts = [a for a in alerts if a.alert_type == alert_type_filter]

        # Sort by timestamp (most recent first) and limit
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        alerts = alerts[:limit]

        # Convert to dictionaries
        return [
            {
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "tool_name": alert.tool_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
            }
            for alert in alerts
        ]

    def _get_recent_metrics_for_tool(self, tool_name: str, minutes: int = 10) -> list[FunctionChainMetric]:
        """Get recent metrics for a specific tool."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics if m.tool_name == tool_name and m.timestamp >= cutoff_time]

    def _start_cleanup_task(self) -> None:
        """Start background task to clean up old metrics."""

        async def cleanup_old_metrics():
            while True:
                try:
                    # Wait 1 hour between cleanups
                    await asyncio.sleep(3600)

                    cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)

                    # Remove old metrics
                    old_metrics_count = len(self.metrics)
                    self.metrics = deque([m for m in self.metrics if m.timestamp >= cutoff_time])

                    # Remove old alerts (keep for 7 days)
                    alert_cutoff = datetime.now() - timedelta(days=7)
                    old_alerts_count = len(self.alerts)
                    self.alerts = deque([a for a in self.alerts if a.timestamp >= alert_cutoff])

                    removed_metrics = old_metrics_count - len(self.metrics)
                    removed_alerts = old_alerts_count - len(self.alerts)

                    if removed_metrics > 0 or removed_alerts > 0:
                        logger.info(f"Cleaned up {removed_metrics} old metrics and {removed_alerts} old alerts")

                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        # Start the cleanup task
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(cleanup_old_metrics())

    def export_metrics(self, output_path: str, format: str = "json", time_window_hours: int | None = None) -> None:
        """
        Export metrics to file.

        Args:
            output_path: Path to output file
            format: Export format ("json", "csv")
            time_window_hours: Time window in hours (None for all metrics)
        """
        metrics_to_export = list(self.metrics)

        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            metrics_to_export = [m for m in metrics_to_export if m.timestamp >= cutoff_time]

        output_path = Path(output_path)

        if format.lower() == "json":
            data = [{**asdict(metric), "timestamp": metric.timestamp.isoformat()} for metric in metrics_to_export]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elif format.lower() == "csv":
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                if metrics_to_export:
                    fieldnames = list(asdict(metrics_to_export[0]).keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for metric in metrics_to_export:
                        row = asdict(metric)
                        row["timestamp"] = metric.timestamp.isoformat()
                        writer.writerow(row)

        logger.info(f"Exported {len(metrics_to_export)} metrics to {output_path}")

    def generate_monitoring_report(self) -> str:
        """Generate a comprehensive monitoring report."""
        stats = self.get_tool_statistics(time_window_hours=24)
        patterns = self.get_usage_patterns(time_window_hours=24)
        alerts = self.get_recent_alerts(limit=10)

        report_lines = [
            "=" * 80,
            "FUNCTION CHAIN MONITORING REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Time Window: 24 hours",
            "",
            "TOOL USAGE STATISTICS:",
        ]

        if "tools" in stats:
            for tool_name, tool_stats in stats["tools"].items():
                report_lines.extend(
                    [
                        f"  {tool_name}:",
                        f"    Total Calls: {tool_stats['total_calls']}",
                        f"    Success Rate: {tool_stats['success_rate']:.1%}",
                        f"    Avg Response Time: {tool_stats['avg_response_time']:.3f}s",
                        f"    Cache Hit Rate: {tool_stats['cache_hit_rate']:.1%}",
                        "",
                    ]
                )

        report_lines.extend(
            [
                "USAGE PATTERNS:",
                f"  Most Used Tools: {list(patterns.get('most_used_tools', {}).keys())[:3]}",
                f"  Most Analyzed Projects: {list(patterns.get('most_analyzed_projects', {}).keys())[:3]}",
                f"  Common Entry Points: {list(patterns.get('common_entry_points', {}).keys())[:3]}",
                "",
            ]
        )

        if alerts:
            report_lines.extend(
                [
                    "RECENT ALERTS:",
                ]
            )
            for alert in alerts[:5]:  # Show top 5 alerts
                report_lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")

            if len(alerts) > 5:
                report_lines.append(f"  ... and {len(alerts) - 5} more alerts")
        else:
            report_lines.append("RECENT ALERTS: None")

        report_lines.extend(["", "=" * 80])

        return "\n".join(report_lines)

    def close(self) -> None:
        """Clean up the monitor."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        logger.info("Function Chain monitor closed")


# Global monitor instance
_global_monitor: FunctionChainMonitor | None = None


def get_monitor() -> FunctionChainMonitor:
    """Get the global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = FunctionChainMonitor()
    return _global_monitor


def record_function_chain_usage(
    tool_name: str, operation_type: str, project_name: str, entry_point: str, response_time: float, success: bool, **kwargs
) -> None:
    """Convenience function to record tool usage."""
    monitor = get_monitor()
    monitor.record_tool_usage(
        tool_name=tool_name,
        operation_type=operation_type,
        project_name=project_name,
        entry_point=entry_point,
        response_time=response_time,
        success=success,
        **kwargs,
    )


# Decorator for automatic monitoring
def monitor_function_chain_tool(operation_type: str):
    """Decorator to automatically monitor function chain tool usage."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            tool_name = func.__name__
            project_name = kwargs.get("project_name", "unknown")
            entry_point = kwargs.get("entry_point", kwargs.get("start_function", kwargs.get("breadcrumb", "unknown")))

            try:
                result = await func(*args, **kwargs)
                response_time = time.time() - start_time
                success = "error" not in result if isinstance(result, dict) else True

                # Extract additional metrics from result
                result_size = 0
                complexity_score = None

                if isinstance(result, dict):
                    # Try to extract result size
                    if "paths" in result:
                        result_size = len(result["paths"])
                    elif "chains" in result:
                        result_size = len(result["chains"])
                    elif "results" in result:
                        result_size = len(result["results"])

                    # Try to extract complexity score
                    if "complexity" in result:
                        complexity_score = result["complexity"].get("overall_score")

                # Record the usage
                record_function_chain_usage(
                    tool_name=tool_name,
                    operation_type=operation_type,
                    project_name=project_name,
                    entry_point=str(entry_point),
                    response_time=response_time,
                    success=success,
                    result_size=result_size,
                    complexity_score=complexity_score,
                )

                return result

            except Exception as e:
                response_time = time.time() - start_time

                # Record the failed usage
                record_function_chain_usage(
                    tool_name=tool_name,
                    operation_type=operation_type,
                    project_name=project_name,
                    entry_point=str(entry_point),
                    response_time=response_time,
                    success=False,
                    error_message=str(e),
                )

                raise  # Re-raise the exception

        return wrapper

    return decorator


if __name__ == "__main__":
    # Example usage and testing
    async def test_monitoring():
        monitor = FunctionChainMonitor()

        # Simulate some tool usage
        monitor.record_tool_usage(
            tool_name="trace_function_chain",
            operation_type="trace",
            project_name="test_project",
            entry_point="main",
            response_time=1.5,
            success=True,
            result_size=10,
            complexity_score=0.6,
        )

        # Generate report
        monitor.generate_monitoring_report()

        monitor.close()

    asyncio.run(test_monitoring())
