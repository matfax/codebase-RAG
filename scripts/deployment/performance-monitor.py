#!/usr/bin/env python3
"""
Cache Performance Monitoring Script

Provides comprehensive performance monitoring for the cache system including:
- Real-time metrics collection
- Performance trend analysis
- Alert generation
- Dashboard data export
"""

import asyncio
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.cache_config import get_cache_config
from src.services.cache_service import get_cache_service


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: float
    cache_hit_rate: float
    l1_hit_rate: float
    l2_hit_rate: float
    memory_usage_mb: float
    memory_usage_percent: float
    redis_memory_mb: float
    total_operations: int
    operations_per_second: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    error_count: int
    error_rate: float
    cpu_usage_percent: float
    connection_count: int
    eviction_count: int


@dataclass
class Alert:
    """Performance alert."""

    timestamp: float
    level: str  # INFO, WARNING, CRITICAL
    category: str
    message: str
    value: float
    threshold: float


class PerformanceMonitor:
    """Cache performance monitoring system."""

    def __init__(self, retention_minutes: int = 60):
        self.cache_service = None
        self.config = None
        self.metrics_history: deque = deque(maxlen=retention_minutes * 60)  # 1 metric per second
        self.alerts: list[Alert] = []
        self.last_metrics: PerformanceMetrics | None = None
        self.start_time = time.time()

        # Alert thresholds
        self.thresholds = {
            "hit_rate_warning": 0.6,
            "hit_rate_critical": 0.4,
            "memory_usage_warning": 0.8,
            "memory_usage_critical": 0.9,
            "response_time_warning": 50,  # ms
            "response_time_critical": 100,  # ms
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.1,
            "cpu_usage_warning": 0.8,
            "cpu_usage_critical": 0.95,
        }

    async def initialize(self):
        """Initialize the performance monitor."""
        try:
            self.config = get_cache_config()
            self.cache_service = await get_cache_service()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize monitor: {e}")

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        timestamp = time.time()

        try:
            # Get cache statistics
            stats = self.cache_service.get_stats()
            tier_stats = self.cache_service.get_tier_stats()
            health_info = await self.cache_service.get_health()

            # Cache hit rates
            cache_hit_rate = getattr(stats, "hit_rate", 0)
            l1_hit_rate = 0
            l2_hit_rate = 0

            if tier_stats.get("l1_stats"):
                l1_hit_rate = getattr(tier_stats["l1_stats"], "hit_rate", 0)
            if tier_stats.get("l2_stats"):
                l2_hit_rate = getattr(tier_stats["l2_stats"], "hit_rate", 0)

            # Memory usage
            l1_info = tier_stats.get("l1_info", {})
            memory_usage_mb = l1_info.get("memory_usage_mb", 0)
            max_memory_mb = l1_info.get("max_memory_mb", 256)
            memory_usage_percent = (memory_usage_mb / max_memory_mb) if max_memory_mb > 0 else 0

            # Redis memory
            redis_memory_mb = 0
            if health_info and hasattr(health_info, "redis_memory_usage"):
                redis_memory_str = health_info.redis_memory_usage
                if isinstance(redis_memory_str, str) and "M" in redis_memory_str:
                    redis_memory_mb = float(redis_memory_str.replace("M", ""))

            # Operations
            total_operations = getattr(stats, "total_operations", 0)
            operations_per_second = 0
            if self.last_metrics:
                time_diff = timestamp - self.last_metrics.timestamp
                ops_diff = total_operations - self.last_metrics.total_operations
                if time_diff > 0:
                    operations_per_second = ops_diff / time_diff

            # Response times (mock values for now)
            avg_response_time_ms = getattr(health_info, "redis_ping_time", 0) if health_info else 0
            p95_response_time_ms = avg_response_time_ms * 1.5  # Estimate

            # Errors
            error_count = getattr(stats, "error_count", 0)
            error_rate = error_count / total_operations if total_operations > 0 else 0

            # System metrics
            cpu_usage_percent = psutil.cpu_percent()

            # Connection and eviction counts
            connection_count = l1_info.get("connection_count", 0)
            eviction_count = l1_info.get("eviction_count", 0)

            return PerformanceMetrics(
                timestamp=timestamp,
                cache_hit_rate=cache_hit_rate,
                l1_hit_rate=l1_hit_rate,
                l2_hit_rate=l2_hit_rate,
                memory_usage_mb=memory_usage_mb,
                memory_usage_percent=memory_usage_percent,
                redis_memory_mb=redis_memory_mb,
                total_operations=total_operations,
                operations_per_second=operations_per_second,
                avg_response_time_ms=avg_response_time_ms,
                p95_response_time_ms=p95_response_time_ms,
                error_count=error_count,
                error_rate=error_rate,
                cpu_usage_percent=cpu_usage_percent,
                connection_count=connection_count,
                eviction_count=eviction_count,
            )

        except Exception:
            # Return minimal metrics on error
            return PerformanceMetrics(
                timestamp=timestamp,
                cache_hit_rate=0,
                l1_hit_rate=0,
                l2_hit_rate=0,
                memory_usage_mb=0,
                memory_usage_percent=0,
                redis_memory_mb=0,
                total_operations=0,
                operations_per_second=0,
                avg_response_time_ms=0,
                p95_response_time_ms=0,
                error_count=1,
                error_rate=1,
                cpu_usage_percent=psutil.cpu_percent(),
                connection_count=0,
                eviction_count=0,
            )

    def check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions."""
        current_time = metrics.timestamp

        # Hit rate alerts
        if metrics.cache_hit_rate < self.thresholds["hit_rate_critical"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="CRITICAL",
                    category="Performance",
                    message=f"Cache hit rate critically low: {metrics.cache_hit_rate:.1%}",
                    value=metrics.cache_hit_rate,
                    threshold=self.thresholds["hit_rate_critical"],
                )
            )
        elif metrics.cache_hit_rate < self.thresholds["hit_rate_warning"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="WARNING",
                    category="Performance",
                    message=f"Cache hit rate low: {metrics.cache_hit_rate:.1%}",
                    value=metrics.cache_hit_rate,
                    threshold=self.thresholds["hit_rate_warning"],
                )
            )

        # Memory usage alerts
        if metrics.memory_usage_percent > self.thresholds["memory_usage_critical"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="CRITICAL",
                    category="Memory",
                    message=f"Memory usage critical: {metrics.memory_usage_percent:.1%}",
                    value=metrics.memory_usage_percent,
                    threshold=self.thresholds["memory_usage_critical"],
                )
            )
        elif metrics.memory_usage_percent > self.thresholds["memory_usage_warning"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="WARNING",
                    category="Memory",
                    message=f"Memory usage high: {metrics.memory_usage_percent:.1%}",
                    value=metrics.memory_usage_percent,
                    threshold=self.thresholds["memory_usage_warning"],
                )
            )

        # Response time alerts
        if metrics.avg_response_time_ms > self.thresholds["response_time_critical"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="CRITICAL",
                    category="Performance",
                    message=f"Response time critical: {metrics.avg_response_time_ms:.1f}ms",
                    value=metrics.avg_response_time_ms,
                    threshold=self.thresholds["response_time_critical"],
                )
            )
        elif metrics.avg_response_time_ms > self.thresholds["response_time_warning"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="WARNING",
                    category="Performance",
                    message=f"Response time high: {metrics.avg_response_time_ms:.1f}ms",
                    value=metrics.avg_response_time_ms,
                    threshold=self.thresholds["response_time_warning"],
                )
            )

        # Error rate alerts
        if metrics.error_rate > self.thresholds["error_rate_critical"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="CRITICAL",
                    category="Errors",
                    message=f"Error rate critical: {metrics.error_rate:.1%}",
                    value=metrics.error_rate,
                    threshold=self.thresholds["error_rate_critical"],
                )
            )
        elif metrics.error_rate > self.thresholds["error_rate_warning"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="WARNING",
                    category="Errors",
                    message=f"Error rate high: {metrics.error_rate:.1%}",
                    value=metrics.error_rate,
                    threshold=self.thresholds["error_rate_warning"],
                )
            )

        # CPU usage alerts
        if metrics.cpu_usage_percent > self.thresholds["cpu_usage_critical"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="CRITICAL",
                    category="System",
                    message=f"CPU usage critical: {metrics.cpu_usage_percent:.1f}%",
                    value=metrics.cpu_usage_percent,
                    threshold=self.thresholds["cpu_usage_critical"],
                )
            )
        elif metrics.cpu_usage_percent > self.thresholds["cpu_usage_warning"]:
            self.alerts.append(
                Alert(
                    timestamp=current_time,
                    level="WARNING",
                    category="System",
                    message=f"CPU usage high: {metrics.cpu_usage_percent:.1f}%",
                    value=metrics.cpu_usage_percent,
                    threshold=self.thresholds["cpu_usage_warning"],
                )
            )

        # Keep only recent alerts (last hour)
        cutoff_time = current_time - 3600
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

    def get_trend_analysis(self, minutes: int = 10) -> dict[str, Any]:
        """Analyze performance trends over the specified time period."""
        if not self.metrics_history:
            return {}

        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if len(recent_metrics) < 2:
            return {}

        # Calculate trends
        first_metric = recent_metrics[0]
        last_metric = recent_metrics[-1]

        trends = {
            "hit_rate_trend": last_metric.cache_hit_rate - first_metric.cache_hit_rate,
            "memory_trend": last_metric.memory_usage_percent - first_metric.memory_usage_percent,
            "response_time_trend": last_metric.avg_response_time_ms - first_metric.avg_response_time_ms,
            "operations_trend": last_metric.operations_per_second - first_metric.operations_per_second,
            "error_rate_trend": last_metric.error_rate - first_metric.error_rate,
        }

        # Calculate averages
        averages = {
            "avg_hit_rate": sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage_percent for m in recent_metrics) / len(recent_metrics),
            "avg_response_time": sum(m.avg_response_time_ms for m in recent_metrics) / len(recent_metrics),
            "avg_operations_per_sec": sum(m.operations_per_second for m in recent_metrics) / len(recent_metrics),
        }

        return {"period_minutes": minutes, "sample_count": len(recent_metrics), "trends": trends, "averages": averages}

    def generate_dashboard_data(self) -> dict[str, Any]:
        """Generate data for performance dashboard."""
        current_metrics = self.last_metrics
        if not current_metrics:
            return {}

        # Get recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp > time.time() - 3600]

        # Get trend analysis
        trend_analysis = self.get_trend_analysis(10)

        # Calculate uptime
        uptime_seconds = time.time() - self.start_time

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime_seconds,
            "current_metrics": asdict(current_metrics),
            "recent_alerts": [asdict(a) for a in recent_alerts],
            "trend_analysis": trend_analysis,
            "alert_summary": {
                "critical": len([a for a in recent_alerts if a.level == "CRITICAL"]),
                "warning": len([a for a in recent_alerts if a.level == "WARNING"]),
                "info": len([a for a in recent_alerts if a.level == "INFO"]),
            },
            "thresholds": self.thresholds,
        }

    def print_realtime_status(self, metrics: PerformanceMetrics):
        """Print real-time status information."""
        # Clear screen and print header
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print("=" * 80)
        print(f"Cache Performance Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Current metrics
        print("\n📊 Current Metrics:")
        print(f"   Hit Rate:         {metrics.cache_hit_rate:.1%} (L1: {metrics.l1_hit_rate:.1%}, L2: {metrics.l2_hit_rate:.1%})")
        print(f"   Memory Usage:     {metrics.memory_usage_mb:.1f}MB ({metrics.memory_usage_percent:.1%})")
        print(f"   Response Time:    {metrics.avg_response_time_ms:.1f}ms (P95: {metrics.p95_response_time_ms:.1f}ms)")
        print(f"   Operations/sec:   {metrics.operations_per_second:.1f}")
        print(f"   Error Rate:       {metrics.error_rate:.2%}")
        print(f"   CPU Usage:        {metrics.cpu_usage_percent:.1f}%")

        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp > time.time() - 300]  # Last 5 minutes
        if recent_alerts:
            print(f"\n🚨 Recent Alerts ({len(recent_alerts)}):")
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                age = time.time() - alert.timestamp
                print(f"   [{alert.level}] {alert.message} ({age:.0f}s ago)")
        else:
            print("\n✅ No recent alerts")

        # Trend indicators
        if len(self.metrics_history) > 10:
            trend = self.get_trend_analysis(5)
            if trend:
                print("\n📈 5-minute Trends:")
                trends = trend["trends"]
                for metric, change in trends.items():
                    indicator = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    print(f"   {metric:20}: {indicator} {change:+.3f}")

        print("\nPress Ctrl+C to stop monitoring...")

    async def run_monitoring(self, interval: int = 1, dashboard_export: str | None = None):
        """Run continuous performance monitoring."""
        print("Starting cache performance monitoring...")
        print(f"Collection interval: {interval} seconds")

        try:
            while True:
                # Collect metrics
                metrics = await self.collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Check for alerts
                self.check_alerts(metrics)

                # Update last metrics
                self.last_metrics = metrics

                # Print real-time status
                self.print_realtime_status(metrics)

                # Export dashboard data if requested
                if dashboard_export:
                    dashboard_data = self.generate_dashboard_data()
                    with open(dashboard_export, "w") as f:
                        json.dump(dashboard_data, f, indent=2, default=str)

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nMonitoring error: {e}")
            raise

    def export_historical_data(self, output_file: str, hours: int = 1):
        """Export historical performance data."""
        cutoff_time = time.time() - (hours * 3600)
        historical_data = [asdict(m) for m in self.metrics_history if m.timestamp > cutoff_time]

        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "sample_count": len(historical_data),
            "metrics": historical_data,
            "alerts": [asdict(a) for a in self.alerts if a.timestamp > cutoff_time],
            "summary": self.get_trend_analysis(hours * 60) if historical_data else {},
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Historical data exported to: {output_file}")
        print(f"Exported {len(historical_data)} metrics samples")


async def main():
    """Main monitoring function."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor cache performance")
    parser.add_argument("--interval", type=int, default=1, help="Collection interval in seconds")
    parser.add_argument("--dashboard", help="Export dashboard data to file")
    parser.add_argument("--export", help="Export historical data to file")
    parser.add_argument("--hours", type=int, default=1, help="Hours of historical data to export")
    parser.add_argument("--once", action="store_true", help="Collect metrics once and exit")
    args = parser.parse_args()

    # Create monitor
    monitor = PerformanceMonitor()

    try:
        # Initialize
        await monitor.initialize()

        if args.once:
            # Single metrics collection
            metrics = await monitor.collect_metrics()
            print("Current Performance Metrics:")
            print(json.dumps(asdict(metrics), indent=2, default=str))
        elif args.export:
            # Export mode (collect some data first)
            print("Collecting performance data...")
            for _ in range(60):  # Collect for 1 minute
                metrics = await monitor.collect_metrics()
                monitor.metrics_history.append(metrics)
                monitor.check_alerts(metrics)
                monitor.last_metrics = metrics
                await asyncio.sleep(1)

            monitor.export_historical_data(args.export, args.hours)
        else:
            # Continuous monitoring
            await monitor.run_monitoring(args.interval, args.dashboard)

    except Exception as e:
        print(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
