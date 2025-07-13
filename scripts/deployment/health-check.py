#!/usr/bin/env python3
"""
Cache Health Check Script

Performs comprehensive health checks on the cache system including:
- Redis connectivity and performance
- Memory usage and limits
- Cache operation functionality
- Performance metrics
"""

import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.cache_config import get_cache_config
from src.services.cache_service import get_cache_service


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass
class HealthCheckResult:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    details: dict = None
    duration_ms: float = 0


@dataclass
class HealthReport:
    """Complete health report."""

    overall_status: HealthStatus
    timestamp: str
    checks: list[HealthCheckResult]
    metrics: dict
    recommendations: list[str]


class CacheHealthChecker:
    """Comprehensive cache health checker."""

    def __init__(self):
        self.cache_service = None
        self.config = None
        self.results: list[HealthCheckResult] = []
        self.metrics: dict = {}

    async def initialize(self):
        """Initialize health checker."""
        try:
            self.config = get_cache_config()
            self.cache_service = await get_cache_service()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize: {e}")

    async def run_health_checks(self) -> HealthReport:
        """Run all health checks and generate report."""
        self.results = []
        self.metrics = {}

        # Run individual checks
        await self._check_redis_connectivity()
        await self._check_memory_usage()
        await self._check_cache_operations()
        await self._check_performance_metrics()
        await self._check_data_integrity()
        await self._check_cache_coherency()

        # Generate overall status
        overall_status = self._calculate_overall_status()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Create report
        return HealthReport(
            overall_status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            checks=self.results,
            metrics=self.metrics,
            recommendations=recommendations,
        )

    async def _check_redis_connectivity(self):
        """Check Redis connectivity and basic operations."""
        start_time = time.perf_counter()

        try:
            # Get health info from cache service
            health_info = await self.cache_service.get_health()

            if health_info.redis_connected:
                # Check ping time
                if health_info.redis_ping_time and health_info.redis_ping_time < 50:
                    status = HealthStatus.HEALTHY
                    message = f"Redis connected (ping: {health_info.redis_ping_time:.1f}ms)"
                elif health_info.redis_ping_time and health_info.redis_ping_time < 100:
                    status = HealthStatus.DEGRADED
                    message = f"Redis connected but slow (ping: {health_info.redis_ping_time:.1f}ms)"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Redis connected but very slow (ping: {health_info.redis_ping_time:.1f}ms)"

                details = {
                    "ping_time_ms": health_info.redis_ping_time,
                    "version": health_info.redis_version,
                    "memory_usage": health_info.redis_memory_usage,
                    "connected_clients": health_info.redis_connected_clients,
                }
            else:
                status = HealthStatus.UNHEALTHY
                message = "Redis not connected"
                details = {"error": health_info.error_message}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Redis connectivity check failed: {e}"
            details = {"error": str(e)}

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.results.append(
            HealthCheckResult(name="Redis Connectivity", status=status, message=message, details=details, duration_ms=duration_ms)
        )

    async def _check_memory_usage(self):
        """Check memory usage and limits."""
        start_time = time.perf_counter()

        try:
            # Get tier stats
            tier_stats = self.cache_service.get_tier_stats()

            # Check L1 memory usage
            l1_info = tier_stats.get("l1_info", {})
            memory_usage_mb = l1_info.get("memory_usage_mb", 0)
            max_memory_mb = l1_info.get("max_memory_mb", 256)
            usage_percent = (memory_usage_mb / max_memory_mb) * 100 if max_memory_mb > 0 else 0

            if usage_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal ({usage_percent:.1f}%)"
            elif usage_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high ({usage_percent:.1f}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical ({usage_percent:.1f}%)"

            details = {
                "l1_memory_usage_mb": memory_usage_mb,
                "l1_max_memory_mb": max_memory_mb,
                "l1_usage_percent": usage_percent,
                "l1_entry_count": l1_info.get("current_size", 0),
                "l1_eviction_count": l1_info.get("eviction_count", 0),
            }

            # Add Redis memory info if available
            if tier_stats.get("l2_info"):
                l2_info = tier_stats["l2_info"]
                details.update(
                    {
                        "redis_memory_usage": l2_info.get("memory_usage", "unknown"),
                        "redis_memory_peak": l2_info.get("memory_peak", "unknown"),
                    }
                )

            self.metrics["memory"] = details

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Memory check failed: {e}"
            details = {"error": str(e)}

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.results.append(
            HealthCheckResult(name="Memory Usage", status=status, message=message, details=details, duration_ms=duration_ms)
        )

    async def _check_cache_operations(self):
        """Test basic cache operations."""
        start_time = time.perf_counter()

        try:
            test_key = "health_check_test_key"
            test_value = {"test": "data", "timestamp": time.time()}

            # Test set operation
            set_start = time.perf_counter()
            await self.cache_service.set(test_key, test_value, ttl=60)
            set_duration = (time.perf_counter() - set_start) * 1000

            # Test get operation
            get_start = time.perf_counter()
            retrieved_value = await self.cache_service.get(test_key)
            get_duration = (time.perf_counter() - get_start) * 1000

            # Test delete operation
            delete_start = time.perf_counter()
            await self.cache_service.delete(test_key)
            delete_duration = (time.perf_counter() - delete_start) * 1000

            # Verify operations
            if retrieved_value == test_value:
                if max(set_duration, get_duration, delete_duration) < 10:
                    status = HealthStatus.HEALTHY
                    message = "Cache operations working normally"
                else:
                    status = HealthStatus.DEGRADED
                    message = "Cache operations slow"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Cache operations failed verification"

            details = {
                "set_duration_ms": set_duration,
                "get_duration_ms": get_duration,
                "delete_duration_ms": delete_duration,
                "verification": "passed" if retrieved_value == test_value else "failed",
            }

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Cache operations check failed: {e}"
            details = {"error": str(e)}

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.results.append(
            HealthCheckResult(name="Cache Operations", status=status, message=message, details=details, duration_ms=duration_ms)
        )

    async def _check_performance_metrics(self):
        """Check cache performance metrics."""
        start_time = time.perf_counter()

        try:
            # Get cache statistics
            stats = self.cache_service.get_stats()

            # Calculate hit rate
            hit_rate = stats.hit_rate if hasattr(stats, "hit_rate") else 0

            if hit_rate >= 0.8:
                status = HealthStatus.HEALTHY
                message = f"Cache hit rate excellent ({hit_rate:.1%})"
            elif hit_rate >= 0.6:
                status = HealthStatus.DEGRADED
                message = f"Cache hit rate acceptable ({hit_rate:.1%})"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Cache hit rate poor ({hit_rate:.1%})"

            details = {
                "hit_rate": hit_rate,
                "total_hits": stats.hits if hasattr(stats, "hits") else 0,
                "total_misses": stats.misses if hasattr(stats, "misses") else 0,
                "total_operations": stats.total_operations if hasattr(stats, "total_operations") else 0,
                "error_count": stats.error_count if hasattr(stats, "error_count") else 0,
            }

            # Get tier-specific stats
            tier_stats = self.cache_service.get_tier_stats()
            if tier_stats.get("l1_stats"):
                l1_stats = tier_stats["l1_stats"]
                details["l1_hit_rate"] = l1_stats.hit_rate if hasattr(l1_stats, "hit_rate") else 0
            if tier_stats.get("l2_stats"):
                l2_stats = tier_stats["l2_stats"]
                details["l2_hit_rate"] = l2_stats.hit_rate if hasattr(l2_stats, "hit_rate") else 0

            self.metrics["performance"] = details

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Performance metrics check failed: {e}"
            details = {"error": str(e)}

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.results.append(
            HealthCheckResult(name="Performance Metrics", status=status, message=message, details=details, duration_ms=duration_ms)
        )

    async def _check_data_integrity(self):
        """Check cache data integrity."""
        start_time = time.perf_counter()

        try:
            # Test data integrity with checksums
            test_data = {"integrity": "test", "data": "x" * 1000}
            test_key = "integrity_check_key"

            # Store with checksum
            await self.cache_service.set(test_key, test_data, ttl=60)

            # Retrieve and verify
            retrieved_data = await self.cache_service.get(test_key)

            if retrieved_data == test_data:
                status = HealthStatus.HEALTHY
                message = "Data integrity verified"
                details = {"integrity_check": "passed"}
            else:
                status = HealthStatus.UNHEALTHY
                message = "Data integrity check failed"
                details = {"integrity_check": "failed", "expected": str(test_data)[:100], "received": str(retrieved_data)[:100]}

            # Cleanup
            await self.cache_service.delete(test_key)

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Data integrity check failed: {e}"
            details = {"error": str(e)}

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.results.append(
            HealthCheckResult(name="Data Integrity", status=status, message=message, details=details, duration_ms=duration_ms)
        )

    async def _check_cache_coherency(self):
        """Check cache coherency between L1 and L2."""
        start_time = time.perf_counter()

        try:
            if hasattr(self.cache_service, "check_cache_coherency"):
                coherency_result = await self.cache_service.check_cache_coherency()

                if coherency_result.get("coherent", False):
                    status = HealthStatus.HEALTHY
                    message = "Cache coherency maintained"
                else:
                    mismatched = coherency_result.get("mismatched_keys", [])
                    if len(mismatched) < 5:
                        status = HealthStatus.DEGRADED
                        message = f"Minor coherency issues ({len(mismatched)} keys)"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = f"Major coherency issues ({len(mismatched)} keys)"

                details = coherency_result
            else:
                # Skip if coherency check not available
                status = HealthStatus.HEALTHY
                message = "Coherency check not applicable"
                details = {"skipped": True}

        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Coherency check failed: {e}"
            details = {"error": str(e)}

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.results.append(
            HealthCheckResult(name="Cache Coherency", status=status, message=message, details=details, duration_ms=duration_ms)
        )

    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall health status from individual checks."""
        if any(r.status == HealthStatus.UNHEALTHY for r in self.results):
            return HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in self.results):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on health check results."""
        recommendations = []

        for result in self.results:
            if result.status != HealthStatus.HEALTHY:
                if "Redis" in result.name and result.status == HealthStatus.UNHEALTHY:
                    recommendations.append("Check Redis service status and connectivity")
                elif "Memory" in result.name and result.status == HealthStatus.DEGRADED:
                    recommendations.append("Consider increasing cache memory limits or implementing more aggressive eviction")
                elif "Performance" in result.name and result.details and result.details.get("hit_rate", 1) < 0.6:
                    recommendations.append("Review cache TTL settings and warming strategies to improve hit rate")
                elif "Coherency" in result.name and result.status != HealthStatus.HEALTHY:
                    recommendations.append("Investigate cache coherency issues and consider cache invalidation")

        # Add performance recommendations based on metrics
        if self.metrics.get("performance", {}).get("error_count", 0) > 0:
            recommendations.append("Investigate cache operation errors in logs")

        return recommendations


def print_health_report(report: HealthReport):
    """Print health report in a formatted way."""
    # Status colors
    status_symbols = {HealthStatus.HEALTHY: "✅", HealthStatus.DEGRADED: "⚠️", HealthStatus.UNHEALTHY: "❌"}

    print("\n" + "=" * 60)
    print("Cache Health Check Report")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Overall Status: {status_symbols[report.overall_status]} {report.overall_status.value}")
    print("=" * 60 + "\n")

    # Individual checks
    print("Health Checks:")
    print("-" * 60)
    for check in report.checks:
        print(f"{status_symbols[check.status]} {check.name}")
        print(f"   Status: {check.status.value}")
        print(f"   Message: {check.message}")
        print(f"   Duration: {check.duration_ms:.1f}ms")
        if check.details and check.status != HealthStatus.HEALTHY:
            print(f"   Details: {json.dumps(check.details, indent=6)}")
        print()

    # Metrics summary
    if report.metrics:
        print("\nKey Metrics:")
        print("-" * 60)
        if "memory" in report.metrics:
            mem = report.metrics["memory"]
            print(f"Memory Usage: {mem.get('l1_usage_percent', 0):.1f}%")
            print(f"L1 Entries: {mem.get('l1_entry_count', 0)}")

        if "performance" in report.metrics:
            perf = report.metrics["performance"]
            print(f"Hit Rate: {perf.get('hit_rate', 0):.1%}")
            print(f"Total Operations: {perf.get('total_operations', 0)}")
            print(f"Error Count: {perf.get('error_count', 0)}")

    # Recommendations
    if report.recommendations:
        print("\nRecommendations:")
        print("-" * 60)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")


async def main():
    """Main health check function."""
    import argparse

    parser = argparse.ArgumentParser(description="Check cache system health")
    parser.add_argument("--export", help="Export results to JSON file")
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()

    # Create health checker
    checker = CacheHealthChecker()

    try:
        # Initialize
        await checker.initialize()

        if args.continuous:
            # Continuous monitoring
            print("Starting continuous health monitoring...")
            print(f"Check interval: {args.interval} seconds")
            print("Press Ctrl+C to stop\n")

            while True:
                report = await checker.run_health_checks()
                print_health_report(report)

                if args.export:
                    # Export with timestamp
                    export_file = f"{args.export}-{int(time.time())}.json"
                    with open(export_file, "w") as f:
                        json.dump(asdict(report), f, indent=2, default=str)

                await asyncio.sleep(args.interval)
        else:
            # Single check
            report = await checker.run_health_checks()
            print_health_report(report)

            if args.export:
                with open(args.export, "w") as f:
                    json.dump(asdict(report), f, indent=2, default=str)
                print(f"\nReport exported to: {args.export}")

            # Exit with appropriate code
            if report.overall_status == HealthStatus.HEALTHY:
                sys.exit(0)
            elif report.overall_status == HealthStatus.DEGRADED:
                sys.exit(1)
            else:
                sys.exit(2)

    except KeyboardInterrupt:
        print("\nHealth monitoring stopped")
    except Exception as e:
        print(f"\nHealth check failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
