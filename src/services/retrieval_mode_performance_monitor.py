"""
Retrieval Mode Performance Monitor

This service provides comprehensive performance monitoring and analytics
for the multi-modal retrieval system, tracking effectiveness and performance
of different retrieval modes.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..models.query_features import PerformanceMetrics, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalModeStats:
    """Statistics for a specific retrieval mode."""

    mode_name: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # Timing metrics
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float("inf")
    max_execution_time_ms: float = 0.0

    # Quality metrics
    total_results: int = 0
    average_results_per_query: float = 0.0
    average_confidence: float = 0.0
    average_diversity: float = 0.0

    # Success rate
    success_rate: float = 0.0

    # Recent performance (last 100 queries)
    recent_performance: list[float] = None
    performance_trend: str = "stable"  # improving, declining, stable

    def __post_init__(self):
        if self.recent_performance is None:
            self.recent_performance = []


@dataclass
class PerformanceAlert:
    """Performance alert for monitoring system."""

    alert_id: str
    mode_name: str
    alert_type: str  # performance_degradation, high_failure_rate, slow_response
    severity: str  # low, medium, high, critical
    message: str
    timestamp: str
    metrics: dict[str, Any]
    recommendations: list[str]


@dataclass
class PerformanceComparison:
    """Comparison between different retrieval modes."""

    timestamp: str
    modes_compared: list[str]
    metrics_compared: dict[str, dict[str, float]]
    best_performing_mode: dict[str, str]  # metric -> best_mode
    recommendations: list[str]


class RetrievalModePerformanceMonitor:
    """
    Comprehensive performance monitoring service for retrieval modes.

    This service tracks performance metrics, detects performance issues,
    provides analytics, and generates recommendations for optimization.
    """

    def __init__(self, max_history_size: int = 1000):
        """Initialize the performance monitor."""
        self.max_history_size = max_history_size
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.mode_stats = {
            "local": RetrievalModeStats("local"),
            "global": RetrievalModeStats("global"),
            "hybrid": RetrievalModeStats("hybrid"),
            "mix": RetrievalModeStats("mix"),
        }

        # Query history for detailed analysis
        self.query_history = deque(maxlen=max_history_size)

        # Alert system
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)

        # Performance thresholds
        self.thresholds = {
            "max_execution_time_ms": 5000.0,  # 5 seconds
            "min_success_rate": 0.85,  # 85%
            "min_average_confidence": 0.3,  # 30%
            "max_failure_rate": 0.15,  # 15%
        }

        # Comparison cache
        self.last_comparison = None
        self.comparison_interval_minutes = 15

        # Start background monitoring
        self.monitoring_enabled = True
        self._start_background_monitoring()

    def record_query_result(self, result: RetrievalResult):
        """Record a query result for performance tracking."""
        try:
            mode_name = result.mode_used
            if "(" in mode_name:  # Handle mix(local) format
                base_mode = mode_name.split("(")[0]
            else:
                base_mode = mode_name

            # Ensure we have stats for this mode
            if base_mode not in self.mode_stats:
                self.mode_stats[base_mode] = RetrievalModeStats(base_mode)

            stats = self.mode_stats[base_mode]

            # Update basic counters
            stats.total_queries += 1

            # Extract execution time (always available)
            execution_time = result.total_execution_time_ms

            if result.error_message is None:
                stats.successful_queries += 1

                # Update timing metrics
                stats.total_execution_time_ms += execution_time
                stats.min_execution_time_ms = min(stats.min_execution_time_ms, execution_time)
                stats.max_execution_time_ms = max(stats.max_execution_time_ms, execution_time)

                # Update quality metrics
                stats.total_results += result.total_results

                # Update averages
                stats.average_execution_time_ms = stats.total_execution_time_ms / stats.successful_queries
                stats.average_results_per_query = stats.total_results / stats.successful_queries
                stats.average_confidence = (
                    stats.average_confidence * (stats.successful_queries - 1) + result.average_confidence
                ) / stats.successful_queries
                stats.average_diversity = (
                    stats.average_diversity * (stats.successful_queries - 1) + result.result_diversity_score
                ) / stats.successful_queries

                # Update recent performance
                if len(stats.recent_performance) >= 100:
                    stats.recent_performance.pop(0)
                stats.recent_performance.append(execution_time)

                # Update performance trend
                stats.performance_trend = self._calculate_performance_trend(stats.recent_performance)

            else:
                stats.failed_queries += 1

            # Update success rate
            stats.success_rate = stats.successful_queries / stats.total_queries

            # Add to query history
            self.query_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "mode": mode_name,
                    "query": result.query,
                    "execution_time_ms": result.total_execution_time_ms,
                    "results_count": result.total_results,
                    "success": result.error_message is None,
                    "confidence": result.average_confidence,
                    "diversity": result.result_diversity_score,
                }
            )

            # Check for performance issues
            self._check_performance_alerts(base_mode, stats)

            self.logger.debug(f"Recorded performance data for {mode_name}: {execution_time:.2f}ms, {result.total_results} results")

        except Exception as e:
            self.logger.error(f"Error recording query result: {e}")

    def _calculate_performance_trend(self, recent_performance: list[float]) -> str:
        """Calculate performance trend from recent performance data."""
        if len(recent_performance) < 20:
            return "stable"

        # Compare recent 20 with previous 20
        recent_20 = recent_performance[-20:]
        previous_20 = recent_performance[-40:-20] if len(recent_performance) >= 40 else recent_performance[:-20]

        recent_avg = sum(recent_20) / len(recent_20)
        previous_avg = sum(previous_20) / len(previous_20)

        change_ratio = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0

        if change_ratio > 0.2:  # 20% slower
            return "declining"
        elif change_ratio < -0.2:  # 20% faster
            return "improving"
        else:
            return "stable"

    def _check_performance_alerts(self, mode_name: str, stats: RetrievalModeStats):
        """Check for performance issues and generate alerts."""
        alerts = []

        # Check execution time threshold
        if stats.average_execution_time_ms > self.thresholds["max_execution_time_ms"]:
            alerts.append(
                PerformanceAlert(
                    alert_id=f"{mode_name}_slow_response_{int(time.time())}",
                    mode_name=mode_name,
                    alert_type="slow_response",
                    severity="medium" if stats.average_execution_time_ms < self.thresholds["max_execution_time_ms"] * 1.5 else "high",
                    message=f"{mode_name} mode average execution time ({stats.average_execution_time_ms:.2f}ms) exceeds threshold ({self.thresholds['max_execution_time_ms']:.2f}ms)",
                    timestamp=datetime.now().isoformat(),
                    metrics={
                        "average_execution_time_ms": stats.average_execution_time_ms,
                        "threshold_ms": self.thresholds["max_execution_time_ms"],
                    },
                    recommendations=[
                        "Consider optimizing query processing",
                        "Check database performance",
                        "Review graph traversal depth settings",
                    ],
                )
            )

        # Check success rate threshold
        if stats.success_rate < self.thresholds["min_success_rate"]:
            alerts.append(
                PerformanceAlert(
                    alert_id=f"{mode_name}_low_success_{int(time.time())}",
                    mode_name=mode_name,
                    alert_type="high_failure_rate",
                    severity="high" if stats.success_rate < 0.7 else "medium",
                    message=f"{mode_name} mode success rate ({stats.success_rate:.2f}) is below threshold ({self.thresholds['min_success_rate']:.2f})",
                    timestamp=datetime.now().isoformat(),
                    metrics={"success_rate": stats.success_rate, "threshold": self.thresholds["min_success_rate"]},
                    recommendations=[
                        "Review error patterns in failed queries",
                        "Check service dependencies",
                        "Validate configuration parameters",
                    ],
                )
            )

        # Check confidence threshold
        if stats.average_confidence < self.thresholds["min_average_confidence"]:
            alerts.append(
                PerformanceAlert(
                    alert_id=f"{mode_name}_low_confidence_{int(time.time())}",
                    mode_name=mode_name,
                    alert_type="performance_degradation",
                    severity="low",
                    message=f"{mode_name} mode average confidence ({stats.average_confidence:.2f}) is below threshold ({self.thresholds['min_average_confidence']:.2f})",
                    timestamp=datetime.now().isoformat(),
                    metrics={"average_confidence": stats.average_confidence, "threshold": self.thresholds["min_average_confidence"]},
                    recommendations=[
                        "Review query analysis accuracy",
                        "Consider adjusting similarity thresholds",
                        "Evaluate training data quality",
                    ],
                )
            )

        # Check performance trend
        if stats.performance_trend == "declining" and len(stats.recent_performance) >= 50:
            alerts.append(
                PerformanceAlert(
                    alert_id=f"{mode_name}_declining_trend_{int(time.time())}",
                    mode_name=mode_name,
                    alert_type="performance_degradation",
                    severity="medium",
                    message=f"{mode_name} mode shows declining performance trend",
                    timestamp=datetime.now().isoformat(),
                    metrics={"trend": stats.performance_trend, "recent_avg_ms": sum(stats.recent_performance[-20:]) / 20},
                    recommendations=[
                        "Monitor system resources",
                        "Check for data growth impact",
                        "Consider cache optimization",
                    ],
                )
            )

        # Add new alerts
        for alert in alerts:
            # Check if similar alert already exists
            existing_alert = next(
                (a for a in self.active_alerts if a.mode_name == alert.mode_name and a.alert_type == alert.alert_type), None
            )

            if not existing_alert:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                self.logger.warning(f"Performance alert: {alert.message}")

    def get_mode_statistics(self, mode_name: str | None = None) -> dict[str, Any]:
        """Get performance statistics for a specific mode or all modes."""
        if mode_name:
            if mode_name in self.mode_stats:
                return asdict(self.mode_stats[mode_name])
            else:
                return {"error": f"Mode '{mode_name}' not found"}
        else:
            return {mode: asdict(stats) for mode, stats in self.mode_stats.items()}

    def compare_modes(self, modes: list[str] | None = None) -> PerformanceComparison:
        """Compare performance between different retrieval modes."""
        if modes is None:
            modes = list(self.mode_stats.keys())

        # Filter existing modes
        existing_modes = [mode for mode in modes if mode in self.mode_stats]

        if len(existing_modes) < 2:
            raise ValueError("Need at least 2 modes to compare")

        # Prepare comparison metrics
        metrics_compared = {}
        best_performing_mode = {}

        comparison_metrics = [
            "average_execution_time_ms",
            "success_rate",
            "average_confidence",
            "average_diversity",
            "average_results_per_query",
        ]

        for metric in comparison_metrics:
            metrics_compared[metric] = {}
            best_value = None
            best_mode = None

            for mode in existing_modes:
                stats = self.mode_stats[mode]
                value = getattr(stats, metric, 0)
                metrics_compared[metric][mode] = value

                # Determine best (lower is better for execution time, higher for others)
                if metric == "average_execution_time_ms":
                    if best_value is None or (value > 0 and value < best_value):
                        best_value = value
                        best_mode = mode
                else:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_mode = mode

            best_performing_mode[metric] = best_mode or existing_modes[0]

        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(metrics_compared, best_performing_mode)

        comparison = PerformanceComparison(
            timestamp=datetime.now().isoformat(),
            modes_compared=existing_modes,
            metrics_compared=metrics_compared,
            best_performing_mode=best_performing_mode,
            recommendations=recommendations,
        )

        self.last_comparison = comparison
        return comparison

    def _generate_comparison_recommendations(
        self, metrics_compared: dict[str, dict[str, float]], best_performing_mode: dict[str, str]
    ) -> list[str]:
        """Generate recommendations based on mode comparison."""
        recommendations = []

        # Analyze overall best performer
        mode_scores = defaultdict(int)
        for metric, best_mode in best_performing_mode.items():
            mode_scores[best_mode] += 1

        overall_best = max(mode_scores.items(), key=lambda x: x[1])

        if overall_best[1] >= 3:  # Best in at least 3 metrics
            recommendations.append(f"Consider using '{overall_best[0]}' mode as default for better overall performance")

        # Check for significant performance gaps
        execution_times = metrics_compared.get("average_execution_time_ms", {})
        if len(execution_times) > 1:
            times = [t for t in execution_times.values() if t > 0]
            if times:
                max_time = max(times)
                min_time = min(times)
                if max_time > min_time * 2:  # More than 2x difference
                    slowest_mode = max(execution_times.items(), key=lambda x: x[1])[0]
                    recommendations.append(
                        f"'{slowest_mode}' mode shows significantly slower performance - investigate optimization opportunities"
                    )

        # Check success rates
        success_rates = metrics_compared.get("success_rate", {})
        for mode, rate in success_rates.items():
            if rate < 0.8:  # Less than 80% success
                recommendations.append(f"'{mode}' mode has low success rate ({rate:.2f}) - review error patterns")

        return recommendations

    def get_active_alerts(self) -> list[PerformanceAlert]:
        """Get all active performance alerts."""
        return self.active_alerts.copy()

    def get_alert_history(self, limit: int | None = None) -> list[PerformanceAlert]:
        """Get alert history."""
        history = list(self.alert_history)
        if limit:
            history = history[-limit:]
        return history

    def clear_alert(self, alert_id: str) -> bool:
        """Clear a specific alert."""
        for i, alert in enumerate(self.active_alerts):
            if alert.alert_id == alert_id:
                self.active_alerts.pop(i)
                self.logger.info(f"Cleared alert: {alert_id}")
                return True
        return False

    def clear_all_alerts(self):
        """Clear all active alerts."""
        cleared_count = len(self.active_alerts)
        self.active_alerts.clear()
        self.logger.info(f"Cleared {cleared_count} active alerts")

    def get_query_history(
        self,
        mode: str | None = None,
        limit: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get query history with optional filtering."""
        history = list(self.query_history)

        # Filter by mode
        if mode:
            history = [h for h in history if h["mode"].startswith(mode)]

        # Filter by time range
        if start_time or end_time:
            filtered_history = []
            for h in history:
                query_time = datetime.fromisoformat(h["timestamp"])
                if start_time and query_time < start_time:
                    continue
                if end_time and query_time > end_time:
                    continue
                filtered_history.append(h)
            history = filtered_history

        # Apply limit
        if limit:
            history = history[-limit:]

        return history

    def get_performance_report(self) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "monitoring_period": {
                "total_queries": sum(stats.total_queries for stats in self.mode_stats.values()),
                "query_history_size": len(self.query_history),
            },
            "mode_statistics": self.get_mode_statistics(),
            "active_alerts": len(self.active_alerts),
            "alert_summary": self._get_alert_summary(),
        }

        # Add comparison if available
        if self.last_comparison:
            report["last_comparison"] = asdict(self.last_comparison)

        # Add performance trends
        report["performance_trends"] = {mode: stats.performance_trend for mode, stats in self.mode_stats.items()}

        # Add recommendations
        report["recommendations"] = self._generate_overall_recommendations()

        return report

    def _get_alert_summary(self) -> dict[str, Any]:
        """Get summary of alerts."""
        alert_types = defaultdict(int)
        severity_counts = defaultdict(int)

        for alert in self.active_alerts:
            alert_types[alert.alert_type] += 1
            severity_counts[alert.severity] += 1

        return {
            "total_active": len(self.active_alerts),
            "by_type": dict(alert_types),
            "by_severity": dict(severity_counts),
            "most_recent": self.active_alerts[-1].message if self.active_alerts else None,
        }

    def _generate_overall_recommendations(self) -> list[str]:
        """Generate overall system recommendations."""
        recommendations = []

        # Analyze overall system health
        total_queries = sum(stats.total_queries for stats in self.mode_stats.values())
        if total_queries == 0:
            recommendations.append("No queries recorded yet - start using the system to gather performance data")
            return recommendations

        overall_success_rate = sum(stats.successful_queries for stats in self.mode_stats.values()) / total_queries
        if overall_success_rate < 0.9:
            recommendations.append(f"Overall success rate ({overall_success_rate:.2f}) could be improved - review system health")

        # Check for underutilized modes
        mode_usage = {mode: stats.total_queries for mode, stats in self.mode_stats.items()}
        total_usage = sum(mode_usage.values())
        if total_usage > 0:
            for mode, usage in mode_usage.items():
                usage_ratio = usage / total_usage
                if usage_ratio < 0.05:  # Less than 5% usage
                    recommendations.append(
                        f"'{mode}' mode is underutilized ({usage_ratio:.1%}) - consider promoting its use for appropriate queries"
                    )

        # Check for alert patterns
        if len(self.active_alerts) > 5:
            recommendations.append("Multiple active alerts detected - consider immediate system review")

        return recommendations

    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        # This would typically start async tasks for continuous monitoring
        # For now, we just log that monitoring is enabled
        self.logger.info("Performance monitoring started")

    def export_metrics(self, format: str = "json") -> str:
        """Export performance metrics in specified format."""
        report = self.get_performance_report()

        if format.lower() == "json":
            return json.dumps(report, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def reset_statistics(self, mode: str | None = None):
        """Reset statistics for a specific mode or all modes."""
        if mode:
            if mode in self.mode_stats:
                self.mode_stats[mode] = RetrievalModeStats(mode)
                self.logger.info(f"Reset statistics for {mode} mode")
            else:
                raise ValueError(f"Mode '{mode}' not found")
        else:
            for mode_name in self.mode_stats:
                self.mode_stats[mode_name] = RetrievalModeStats(mode_name)
            self.query_history.clear()
            self.active_alerts.clear()
            self.alert_history.clear()
            self.logger.info("Reset all statistics")


# Factory function
_performance_monitor_instance = None


def get_performance_monitor(max_history_size: int = 1000) -> RetrievalModePerformanceMonitor:
    """Get or create a RetrievalModePerformanceMonitor instance."""
    global _performance_monitor_instance

    if _performance_monitor_instance is None:
        _performance_monitor_instance = RetrievalModePerformanceMonitor(max_history_size)

    return _performance_monitor_instance
