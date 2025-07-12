# Cache Management and Maintenance Guide

## Overview

This guide provides comprehensive operational procedures for managing and maintaining the Query Caching Layer system in production environments. It covers daily operations, routine maintenance, emergency procedures, and long-term optimization strategies.

## Daily Operations

### Health Check Procedures

#### Morning Health Check Routine

```bash
#!/bin/bash
# scripts/daily-health-check.sh

echo "=== Daily Cache System Health Check ==="
echo "Date: $(date)"
echo

# 1. Check system resources
echo "1. System Resources:"
echo "   Memory usage: $(free -h | grep 'Mem:' | awk '{print $3"/"$2}')"
echo "   Disk usage: $(df -h / | tail -1 | awk '{print $5}')"
echo "   CPU load: $(uptime | awk -F'load average:' '{print $2}')"
echo

# 2. Check Redis health
echo "2. Redis Health:"
redis_ping=$(redis-cli ping 2>/dev/null || echo "FAILED")
echo "   Redis ping: $redis_ping"

if [ "$redis_ping" = "PONG" ]; then
    redis_memory=$(redis-cli info memory | grep used_memory_human | cut -d: -f2)
    redis_connections=$(redis-cli info clients | grep connected_clients | cut -d: -f2)
    echo "   Memory usage: $redis_memory"
    echo "   Active connections: $redis_connections"
else
    echo "   ⚠️  Redis is not responding!"
fi
echo

# 3. Check application health
echo "3. Application Health:"
app_health=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$app_health" = "200" ]; then
    echo "   ✓ Application is healthy"

    # Get cache metrics
    cache_metrics=$(curl -s http://localhost:8000/metrics/cache)
    hit_rate=$(echo $cache_metrics | jq -r '.hit_rate // "N/A"')
    total_ops=$(echo $cache_metrics | jq -r '.total_operations // "N/A"')
    echo "   Cache hit rate: $hit_rate"
    echo "   Total operations: $total_ops"
else
    echo "   ⚠️  Application health check failed (HTTP $app_health)"
fi
echo

# 4. Check Docker containers
echo "4. Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(redis|cache)"
echo

# 5. Summary
echo "=== Health Check Summary ==="
if [ "$redis_ping" = "PONG" ] && [ "$app_health" = "200" ]; then
    echo "✓ All systems operational"
    exit 0
else
    echo "⚠️  Issues detected - requires attention"
    exit 1
fi
```

#### Performance Metrics Collection

```python
#!/usr/bin/env python3
"""Daily performance metrics collection."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.services.cache_service import get_cache_service

class DailyMetricsCollector:
    def __init__(self, output_dir: str = "./metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def collect_daily_metrics(self):
        """Collect and store daily performance metrics."""
        cache_service = await get_cache_service()

        # Collect comprehensive metrics
        metrics = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "cache_stats": self._get_cache_stats(cache_service),
            "health_info": await self._get_health_info(cache_service),
            "tier_stats": self._get_tier_stats(cache_service),
            "performance_metrics": await self._get_performance_metrics(cache_service)
        }

        # Save to daily metrics file
        date_str = datetime.now().strftime("%Y-%m-%d")
        metrics_file = self.output_dir / f"cache_metrics_{date_str}.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _get_cache_stats(self, cache_service):
        """Get basic cache statistics."""
        stats = cache_service.get_stats()
        return {
            "hit_rate": stats.hit_rate,
            "miss_rate": stats.miss_rate,
            "total_operations": stats.total_operations,
            "hit_count": stats.hit_count,
            "miss_count": stats.miss_count,
            "set_count": stats.set_count,
            "delete_count": stats.delete_count,
            "error_count": stats.error_count
        }

    async def _get_health_info(self, cache_service):
        """Get cache health information."""
        health = await cache_service.get_health()
        return {
            "status": health.status.value,
            "redis_connected": health.redis_connected,
            "redis_ping_time": health.redis_ping_time,
            "memory_usage": health.memory_usage,
            "connection_pool_stats": health.connection_pool_stats
        }

    def _get_tier_stats(self, cache_service):
        """Get multi-tier cache statistics."""
        if hasattr(cache_service, 'get_tier_stats'):
            return cache_service.get_tier_stats()
        return None

    async def _get_performance_metrics(self, cache_service):
        """Get performance metrics."""
        # Measure operation latency
        latency_samples = []
        for _ in range(10):
            start_time = time.perf_counter()
            await cache_service.get("health_check_key")
            latency = (time.perf_counter() - start_time) * 1000
            latency_samples.append(latency)

        return {
            "avg_latency_ms": sum(latency_samples) / len(latency_samples),
            "max_latency_ms": max(latency_samples),
            "min_latency_ms": min(latency_samples)
        }

# Usage
if __name__ == "__main__":
    collector = DailyMetricsCollector()
    metrics = asyncio.run(collector.collect_daily_metrics())
    print(f"Daily metrics collected: {metrics['date']}")
```

### Operational Monitoring

#### Real-time Dashboard Commands

```bash
# Monitor cache hit rates
watch -n 5 'curl -s http://localhost:8000/metrics/cache | jq ".hit_rate"'

# Monitor Redis memory usage
watch -n 10 'redis-cli info memory | grep used_memory_human'

# Monitor active connections
watch -n 10 'redis-cli info clients | grep connected_clients'

# Monitor cache operations per second
watch -n 5 'curl -s http://localhost:8000/stats/cache | jq ".operations_per_second"'

# Monitor system resources
watch -n 5 'docker stats --no-stream | grep -E "(redis|cache)"'
```

#### Log Monitoring

```bash
# Monitor cache service logs
tail -f /var/log/cache-service.log | grep -E "(ERROR|WARN|Cache)"

# Monitor Redis logs
docker logs -f redis-cache | grep -E "(WARNING|ERROR)"

# Monitor performance issues
tail -f /var/log/cache-service.log | grep -E "(slow|timeout|latency)"

# Monitor security events
tail -f /var/log/cache-security.log | jq '.alert_type'
```

## Routine Maintenance

### Weekly Maintenance Tasks

```bash
#!/bin/bash
# scripts/weekly-maintenance.sh

echo "=== Weekly Cache System Maintenance ==="
echo "Date: $(date)"

# 1. Performance analysis
echo "1. Analyzing weekly performance trends..."
python3 scripts/analyze-weekly-performance.py

# 2. Cache optimization
echo "2. Running cache optimization..."
./scripts/optimize-cache-settings.sh

# 3. Log rotation and cleanup
echo "3. Rotating logs and cleaning up old files..."
docker exec redis-cache logrotate /etc/logrotate.d/redis
find /var/log/cache-* -name "*.log" -mtime +30 -delete

# 4. Security audit
echo "4. Running security audit..."
./scripts/security-audit.sh

# 5. Backup verification
echo "5. Verifying backup integrity..."
./scripts/verify-backups.sh

# 6. Update system packages
echo "6. Updating system packages..."
sudo apt update && sudo apt list --upgradable

# 7. Memory defragmentation (if needed)
echo "7. Checking memory fragmentation..."
fragmentation=$(redis-cli memory doctor | grep fragmentation || echo "OK")
echo "   Memory fragmentation: $fragmentation"

echo "Weekly maintenance completed!"
```

#### Cache Optimization Script

```python
#!/usr/bin/env python3
"""Weekly cache optimization analysis and recommendations."""

import asyncio
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

from src.services.cache_service import get_cache_service
from src.config.cache_config import get_cache_config

class CacheOptimizer:
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)

    async def analyze_and_optimize(self):
        """Analyze cache performance and provide optimization recommendations."""

        # Load weekly metrics
        weekly_metrics = self._load_weekly_metrics()

        # Analyze performance trends
        analysis = {
            "hit_rate_analysis": self._analyze_hit_rates(weekly_metrics),
            "latency_analysis": self._analyze_latency(weekly_metrics),
            "memory_analysis": self._analyze_memory_usage(weekly_metrics),
            "operation_analysis": self._analyze_operations(weekly_metrics),
            "recommendations": []
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        analysis["recommendations"] = recommendations

        # Save analysis report
        report_file = self.metrics_dir / f"optimization_report_{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(report_file, "w") as f:
            json.dump(analysis, f, indent=2)

        # Print summary
        self._print_optimization_summary(analysis)

        return analysis

    def _load_weekly_metrics(self):
        """Load metrics from the past week."""
        weekly_metrics = []
        today = datetime.now().date()

        for i in range(7):
            date = today - timedelta(days=i)
            metrics_file = self.metrics_dir / f"cache_metrics_{date}.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    weekly_metrics.append(json.load(f))

        return weekly_metrics

    def _analyze_hit_rates(self, metrics):
        """Analyze cache hit rate trends."""
        hit_rates = [m["cache_stats"]["hit_rate"] for m in metrics if "cache_stats" in m]

        if not hit_rates:
            return {"status": "no_data"}

        return {
            "average_hit_rate": statistics.mean(hit_rates),
            "min_hit_rate": min(hit_rates),
            "max_hit_rate": max(hit_rates),
            "trend": "improving" if hit_rates[-1] > hit_rates[0] else "declining",
            "volatility": statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0
        }

    def _analyze_latency(self, metrics):
        """Analyze response latency trends."""
        latencies = []
        for m in metrics:
            if "performance_metrics" in m and "avg_latency_ms" in m["performance_metrics"]:
                latencies.append(m["performance_metrics"]["avg_latency_ms"])

        if not latencies:
            return {"status": "no_data"}

        return {
            "average_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "trend": "improving" if latencies[-1] < latencies[0] else "declining",
            "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        }

    def _generate_recommendations(self, analysis):
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # Hit rate recommendations
        hit_rate_analysis = analysis.get("hit_rate_analysis", {})
        if hit_rate_analysis.get("average_hit_rate", 0) < 0.7:
            recommendations.append({
                "category": "hit_rate",
                "priority": "high",
                "issue": "Low cache hit rate",
                "recommendation": "Consider increasing TTL values or cache size",
                "current_value": hit_rate_analysis.get("average_hit_rate"),
                "target_value": 0.8
            })

        # Latency recommendations
        latency_analysis = analysis.get("latency_analysis", {})
        if latency_analysis.get("average_latency", 0) > 50:
            recommendations.append({
                "category": "latency",
                "priority": "medium",
                "issue": "High response latency",
                "recommendation": "Optimize Redis configuration or increase connection pool size",
                "current_value": latency_analysis.get("average_latency"),
                "target_value": 25
            })

        # Memory recommendations
        memory_analysis = analysis.get("memory_analysis", {})
        if memory_analysis.get("memory_pressure", False):
            recommendations.append({
                "category": "memory",
                "priority": "high",
                "issue": "Memory pressure detected",
                "recommendation": "Increase memory allocation or optimize eviction policy",
                "action": "Scale up memory resources"
            })

        return recommendations

    def _print_optimization_summary(self, analysis):
        """Print optimization summary to console."""
        print("\n=== Cache Optimization Summary ===")

        # Hit rate summary
        hit_rate = analysis.get("hit_rate_analysis", {})
        if "average_hit_rate" in hit_rate:
            print(f"Hit Rate: {hit_rate['average_hit_rate']:.1%} (trend: {hit_rate.get('trend', 'unknown')})")

        # Latency summary
        latency = analysis.get("latency_analysis", {})
        if "average_latency" in latency:
            print(f"Latency: {latency['average_latency']:.1f}ms (trend: {latency.get('trend', 'unknown')})")

        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. [{rec['priority'].upper()}] {rec['issue']}")
                print(f"     → {rec['recommendation']}")
        else:
            print("\n✓ No optimization recommendations at this time")

# Usage
if __name__ == "__main__":
    optimizer = CacheOptimizer()
    asyncio.run(optimizer.analyze_and_optimize())
```

### Monthly Maintenance Tasks

#### Comprehensive System Review

```bash
#!/bin/bash
# scripts/monthly-maintenance.sh

echo "=== Monthly Cache System Maintenance ==="
echo "Date: $(date)"

# 1. Comprehensive performance review
echo "1. Generating monthly performance report..."
python3 scripts/generate-monthly-report.py

# 2. Security compliance audit
echo "2. Running comprehensive security audit..."
./scripts/comprehensive-security-audit.sh

# 3. Capacity planning analysis
echo "3. Analyzing capacity requirements..."
./scripts/capacity-planning-analysis.sh

# 4. Configuration review and optimization
echo "4. Reviewing configuration settings..."
./scripts/review-configuration.sh

# 5. Disaster recovery test
echo "5. Testing disaster recovery procedures..."
./scripts/test-disaster-recovery.sh

# 6. Update dependencies and patches
echo "6. Checking for updates..."
./scripts/update-dependencies.sh

# 7. Performance benchmarking
echo "7. Running performance benchmarks..."
./scripts/run-performance-benchmarks.sh

# 8. Data integrity verification
echo "8. Verifying data integrity..."
./scripts/verify-data-integrity.sh

echo "Monthly maintenance completed!"
```

#### Capacity Planning Analysis

```python
#!/usr/bin/env python3
"""Monthly capacity planning analysis."""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path

class CapacityPlanner:
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)

    def analyze_capacity_trends(self):
        """Analyze capacity trends over the past month."""

        # Load monthly metrics
        monthly_metrics = self._load_monthly_metrics()

        analysis = {
            "memory_trends": self._analyze_memory_trends(monthly_metrics),
            "operation_trends": self._analyze_operation_trends(monthly_metrics),
            "growth_projections": self._calculate_growth_projections(monthly_metrics),
            "capacity_recommendations": []
        }

        # Generate capacity recommendations
        recommendations = self._generate_capacity_recommendations(analysis)
        analysis["capacity_recommendations"] = recommendations

        # Save capacity report
        report_file = self.metrics_dir / f"capacity_report_{datetime.now().strftime('%Y-%m')}.json"
        with open(report_file, "w") as f:
            json.dump(analysis, f, indent=2)

        self._print_capacity_summary(analysis)
        return analysis

    def _load_monthly_metrics(self):
        """Load metrics from the past month."""
        monthly_metrics = []
        today = datetime.now().date()

        for i in range(30):
            date = today - timedelta(days=i)
            metrics_file = self.metrics_dir / f"cache_metrics_{date}.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    monthly_metrics.append(json.load(f))

        return monthly_metrics

    def _analyze_memory_trends(self, metrics):
        """Analyze memory usage trends."""
        memory_usage = []
        for m in metrics:
            if "health_info" in m and m["health_info"].get("memory_usage"):
                used_memory = m["health_info"]["memory_usage"].get("used_memory", 0)
                memory_usage.append(used_memory)

        if not memory_usage:
            return {"status": "no_data"}

        # Calculate growth rate
        if len(memory_usage) > 1:
            growth_rate = (memory_usage[-1] - memory_usage[0]) / memory_usage[0] * 100
        else:
            growth_rate = 0

        return {
            "average_usage": statistics.mean(memory_usage),
            "peak_usage": max(memory_usage),
            "growth_rate_percent": growth_rate,
            "trend": "increasing" if growth_rate > 5 else "stable" if growth_rate > -5 else "decreasing"
        }

    def _analyze_operation_trends(self, metrics):
        """Analyze operation volume trends."""
        daily_operations = []
        for m in metrics:
            if "cache_stats" in m:
                total_ops = m["cache_stats"].get("total_operations", 0)
                daily_operations.append(total_ops)

        if not daily_operations:
            return {"status": "no_data"}

        # Calculate operations per day growth
        if len(daily_operations) > 1:
            growth_rate = (daily_operations[-1] - daily_operations[0]) / daily_operations[0] * 100
        else:
            growth_rate = 0

        return {
            "average_daily_operations": statistics.mean(daily_operations),
            "peak_daily_operations": max(daily_operations),
            "growth_rate_percent": growth_rate,
            "operations_per_second_peak": max(daily_operations) / 86400  # Convert to ops/sec
        }

    def _calculate_growth_projections(self, metrics):
        """Calculate growth projections for the next 6 months."""
        memory_trends = self._analyze_memory_trends(metrics)
        operation_trends = self._analyze_operation_trends(metrics)

        # Project memory growth
        current_memory = memory_trends.get("peak_usage", 0)
        memory_growth_rate = memory_trends.get("growth_rate_percent", 0) / 100
        projected_memory_6m = current_memory * (1 + memory_growth_rate * 6)

        # Project operation growth
        current_ops = operation_trends.get("peak_daily_operations", 0)
        ops_growth_rate = operation_trends.get("growth_rate_percent", 0) / 100
        projected_ops_6m = current_ops * (1 + ops_growth_rate * 6)

        return {
            "6_month_memory_projection": projected_memory_6m,
            "6_month_operations_projection": projected_ops_6m,
            "memory_growth_monthly": memory_growth_rate,
            "operations_growth_monthly": ops_growth_rate
        }

    def _generate_capacity_recommendations(self, analysis):
        """Generate capacity planning recommendations."""
        recommendations = []

        # Memory recommendations
        memory_trends = analysis.get("memory_trends", {})
        projections = analysis.get("growth_projections", {})

        if memory_trends.get("growth_rate_percent", 0) > 10:
            recommendations.append({
                "category": "memory",
                "priority": "high",
                "issue": "Rapid memory growth detected",
                "recommendation": "Plan for memory upgrade within 3 months",
                "projected_requirement": f"{projections.get('6_month_memory_projection', 0) / (1024**3):.1f} GB"
            })

        # Operations recommendations
        operation_trends = analysis.get("operation_trends", {})

        if operation_trends.get("growth_rate_percent", 0) > 15:
            recommendations.append({
                "category": "performance",
                "priority": "medium",
                "issue": "Rapid operation volume growth",
                "recommendation": "Consider horizontal scaling or performance optimization",
                "projected_ops_per_second": f"{projections.get('6_month_operations_projection', 0) / 86400:.0f}"
            })

        return recommendations

    def _print_capacity_summary(self, analysis):
        """Print capacity planning summary."""
        print("\n=== Capacity Planning Summary ===")

        # Memory trends
        memory_trends = analysis.get("memory_trends", {})
        if "average_usage" in memory_trends:
            print(f"Memory Usage: {memory_trends['average_usage'] / (1024**3):.1f} GB average")
            print(f"Memory Growth: {memory_trends.get('growth_rate_percent', 0):.1f}% monthly")

        # Operation trends
        operation_trends = analysis.get("operation_trends", {})
        if "average_daily_operations" in operation_trends:
            print(f"Daily Operations: {operation_trends['average_daily_operations']:,.0f} average")
            print(f"Operations Growth: {operation_trends.get('growth_rate_percent', 0):.1f}% monthly")

        # Projections
        projections = analysis.get("growth_projections", {})
        print(f"\n6-Month Projections:")
        print(f"  Memory: {projections.get('6_month_memory_projection', 0) / (1024**3):.1f} GB")
        print(f"  Operations: {projections.get('6_month_operations_projection', 0):,.0f} daily")

        # Recommendations
        recommendations = analysis.get("capacity_recommendations", [])
        if recommendations:
            print(f"\nCapacity Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. [{rec['priority'].upper()}] {rec['issue']}")
                print(f"     → {rec['recommendation']}")

# Usage
if __name__ == "__main__":
    planner = CapacityPlanner()
    planner.analyze_capacity_trends()
```

## Cache Management Operations

### Cache Warming Strategies

#### Predictive Cache Warming

```python
#!/usr/bin/env python3
"""Intelligent cache warming based on usage patterns."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.services.cache_service import get_cache_service
from src.services.cache_warmup_service import CacheWarmupService

class IntelligentCacheWarmer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.usage_patterns_file = Path("./cache_usage_patterns.json")

    async def warm_cache_intelligently(self):
        """Warm cache based on historical usage patterns."""

        # Load usage patterns
        patterns = self._load_usage_patterns()

        # Get cache service
        cache_service = await get_cache_service()
        warmup_service = CacheWarmupService(cache_service)

        # Warm based on time of day patterns
        current_hour = datetime.now().hour
        hourly_patterns = patterns.get("hourly_patterns", {})

        if str(current_hour) in hourly_patterns:
            popular_keys = hourly_patterns[str(current_hour)].get("popular_keys", [])

            self.logger.info(f"Warming {len(popular_keys)} popular keys for hour {current_hour}")

            for key_pattern in popular_keys:
                try:
                    await self._warm_key_pattern(warmup_service, key_pattern)
                except Exception as e:
                    self.logger.error(f"Failed to warm key pattern {key_pattern}: {e}")

        # Warm based on project usage
        project_patterns = patterns.get("project_patterns", {})
        for project, project_data in project_patterns.items():
            if project_data.get("active", False):
                await warmup_service.warm_project_cache(project)

        return True

    def _load_usage_patterns(self):
        """Load historical usage patterns."""
        if self.usage_patterns_file.exists():
            with open(self.usage_patterns_file) as f:
                return json.load(f)
        return {}

    async def _warm_key_pattern(self, warmup_service, key_pattern):
        """Warm cache for a specific key pattern."""
        pattern_type = key_pattern.get("type")

        if pattern_type == "embedding":
            await self._warm_embedding_cache(warmup_service, key_pattern)
        elif pattern_type == "search":
            await self._warm_search_cache(warmup_service, key_pattern)
        elif pattern_type == "project":
            await self._warm_project_cache(warmup_service, key_pattern)

    async def _warm_embedding_cache(self, warmup_service, pattern):
        """Warm embedding cache with popular queries."""
        popular_queries = pattern.get("queries", [])

        for query in popular_queries[:10]:  # Limit to top 10
            try:
                # Generate embedding for popular query
                await warmup_service.warm_embedding_cache(query)
            except Exception as e:
                self.logger.warning(f"Failed to warm embedding for query '{query}': {e}")

# Usage
if __name__ == "__main__":
    warmer = IntelligentCacheWarmer()
    asyncio.run(warmer.warm_cache_intelligently())
```

### Cache Cleanup and Maintenance

#### Automated Cache Cleanup

```python
#!/usr/bin/env python3
"""Automated cache cleanup and maintenance."""

import asyncio
import logging
import time
from datetime import datetime, timedelta

from src.services.cache_service import get_cache_service

class CacheCleanupManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run_cleanup_cycle(self):
        """Run comprehensive cache cleanup cycle."""

        cache_service = await get_cache_service()

        cleanup_tasks = [
            self._cleanup_expired_entries(cache_service),
            self._cleanup_orphaned_keys(cache_service),
            self._optimize_memory_usage(cache_service),
            self._cleanup_old_metrics(cache_service),
            self._validate_cache_integrity(cache_service)
        ]

        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Log cleanup results
        for i, result in enumerate(results):
            task_name = [
                "expired_entries", "orphaned_keys", "memory_optimization",
                "old_metrics", "cache_integrity"
            ][i]

            if isinstance(result, Exception):
                self.logger.error(f"Cleanup task {task_name} failed: {result}")
            else:
                self.logger.info(f"Cleanup task {task_name} completed: {result}")

        return results

    async def _cleanup_expired_entries(self, cache_service):
        """Clean up expired cache entries."""
        cleaned_count = 0

        # Clean L1 cache
        if hasattr(cache_service, 'l1_cache'):
            cleaned_count += cache_service.l1_cache.cleanup_expired()

        # Clean L2 cache (Redis handles this automatically with TTL)
        # But we can scan for entries that should have expired
        if hasattr(cache_service, 'get_redis_client'):
            async with cache_service.get_redis_client() as redis:
                # Scan for potentially stale entries
                stale_keys = []
                async for key in redis.scan_iter(match="codebase_rag:*"):
                    ttl = await redis.ttl(key)
                    if ttl == -1:  # No TTL set
                        # Check if this key type should have TTL
                        key_str = key.decode()
                        if self._should_have_ttl(key_str):
                            stale_keys.append(key)

                # Set TTL for keys that should have it
                for key in stale_keys:
                    default_ttl = self._get_default_ttl_for_key(key.decode())
                    await redis.expire(key, default_ttl)
                    cleaned_count += 1

        return f"Cleaned {cleaned_count} expired entries"

    async def _cleanup_orphaned_keys(self, cache_service):
        """Clean up orphaned cache keys."""
        orphaned_count = 0

        if hasattr(cache_service, 'get_redis_client'):
            async with cache_service.get_redis_client() as redis:
                # Find keys that don't match current patterns
                all_keys = []
                async for key in redis.scan_iter(match="codebase_rag:*"):
                    all_keys.append(key.decode())

                # Check each key for validity
                orphaned_keys = []
                for key in all_keys:
                    if self._is_orphaned_key(key):
                        orphaned_keys.append(key)

                # Remove orphaned keys
                if orphaned_keys:
                    await redis.delete(*orphaned_keys)
                    orphaned_count = len(orphaned_keys)

        return f"Cleaned {orphaned_count} orphaned keys"

    def _should_have_ttl(self, key: str) -> bool:
        """Check if a key should have TTL set."""
        # Keys that should always have TTL
        ttl_required_patterns = [
            ":search:", ":embedding:", ":file:", ":temp:"
        ]

        return any(pattern in key for pattern in ttl_required_patterns)

    def _get_default_ttl_for_key(self, key: str) -> int:
        """Get default TTL for a key based on its type."""
        if ":embedding:" in key:
            return 7200  # 2 hours
        elif ":search:" in key:
            return 1800  # 30 minutes
        elif ":project:" in key:
            return 3600  # 1 hour
        elif ":file:" in key:
            return 1800  # 30 minutes
        else:
            return 3600  # 1 hour default

    def _is_orphaned_key(self, key: str) -> bool:
        """Check if a key is orphaned (invalid pattern)."""
        # Define valid key patterns
        valid_patterns = [
            "codebase_rag:embedding:",
            "codebase_rag:search:",
            "codebase_rag:project:",
            "codebase_rag:file:",
            "codebase_rag:session:",
            "codebase_rag:temp:"
        ]

        return not any(key.startswith(pattern) for pattern in valid_patterns)

# Usage
if __name__ == "__main__":
    cleanup_manager = CacheCleanupManager()
    asyncio.run(cleanup_manager.run_cleanup_cycle())
```

## Backup and Recovery

### Automated Backup System

```bash
#!/bin/bash
# scripts/automated-backup.sh

set -e

BACKUP_DIR="/var/backups/cache-system"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

echo "Starting automated cache backup - $DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Redis data
echo "Creating Redis backup..."
docker exec redis-cache redis-cli BGSAVE
sleep 10  # Wait for background save to complete

# Copy Redis dump
docker cp redis-cache:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Backup configuration files
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
    .env* \
    docker-compose*.yml \
    configs/ \
    certs/ \
    2>/dev/null || true

# Backup metrics and logs
echo "Backing up metrics and logs..."
tar -czf $BACKUP_DIR/metrics_$DATE.tar.gz \
    metrics/ \
    logs/ \
    2>/dev/null || true

# Create backup manifest
cat > $BACKUP_DIR/manifest_$DATE.json << EOF
{
    "backup_date": "$DATE",
    "backup_type": "automated",
    "files": {
        "redis_data": "redis_$DATE.rdb",
        "configuration": "config_$DATE.tar.gz",
        "metrics_logs": "metrics_$DATE.tar.gz"
    },
    "system_info": {
        "hostname": "$(hostname)",
        "redis_version": "$(docker exec redis-cache redis-cli INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')",
        "backup_size": "$(du -sh $BACKUP_DIR | cut -f1)"
    }
}
EOF

# Compress Redis backup
gzip $BACKUP_DIR/redis_$DATE.rdb

# Clean old backups
echo "Cleaning old backups (older than $RETENTION_DAYS days)..."
find $BACKUP_DIR -name "*_*.rdb.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*_*.tar.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "manifest_*.json" -mtime +$RETENTION_DAYS -delete

# Verify backup integrity
echo "Verifying backup integrity..."
if [ -f "$BACKUP_DIR/redis_${DATE}.rdb.gz" ]; then
    gunzip -t $BACKUP_DIR/redis_${DATE}.rdb.gz
    echo "✓ Redis backup verification passed"
else
    echo "✗ Redis backup verification failed"
    exit 1
fi

echo "Backup completed successfully: $BACKUP_DIR"
```

### Disaster Recovery Procedures

```bash
#!/bin/bash
# scripts/disaster-recovery.sh

set -e

BACKUP_DIR="/var/backups/cache-system"
RECOVERY_MODE=${1:-"latest"}  # latest, date, or specific backup

echo "=== Cache System Disaster Recovery ==="
echo "Recovery mode: $RECOVERY_MODE"

# Stop services
echo "Stopping cache services..."
docker-compose down

# Determine which backup to use
if [ "$RECOVERY_MODE" = "latest" ]; then
    BACKUP_FILE=$(ls -t $BACKUP_DIR/redis_*.rdb.gz | head -1)
    CONFIG_FILE=$(ls -t $BACKUP_DIR/config_*.tar.gz | head -1)
elif [ "$RECOVERY_MODE" = "date" ]; then
    echo "Available backups:"
    ls -la $BACKUP_DIR/redis_*.rdb.gz
    read -p "Enter backup date (YYYYMMDD_HHMMSS): " BACKUP_DATE
    BACKUP_FILE="$BACKUP_DIR/redis_${BACKUP_DATE}.rdb.gz"
    CONFIG_FILE="$BACKUP_DIR/config_${BACKUP_DATE}.tar.gz"
else
    BACKUP_FILE="$BACKUP_DIR/$RECOVERY_MODE"
    CONFIG_FILE=$(echo $BACKUP_FILE | sed 's/redis_/config_/' | sed 's/.rdb.gz/.tar.gz/')
fi

# Verify backup files exist
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Configuration backup not found: $CONFIG_FILE"
fi

echo "Using backup: $BACKUP_FILE"

# Restore Redis data
echo "Restoring Redis data..."
gunzip -c $BACKUP_FILE > ./redis_restore.rdb

# Create temporary Redis container for verification
echo "Verifying backup integrity..."
docker run -d --name redis-verify \
    -v $(pwd)/redis_restore.rdb:/data/dump.rdb \
    redis:7-alpine redis-server --save ""

sleep 5

# Test Redis data
if docker exec redis-verify redis-cli ping > /dev/null 2>&1; then
    echo "✓ Backup verification successful"
    docker stop redis-verify
    docker rm redis-verify
else
    echo "✗ Backup verification failed"
    docker stop redis-verify || true
    docker rm redis-verify || true
    rm -f ./redis_restore.rdb
    exit 1
fi

# Restore configuration if available
if [ -f "$CONFIG_FILE" ]; then
    echo "Restoring configuration..."
    tar -xzf $CONFIG_FILE
fi

# Start services with restored data
echo "Starting services with restored data..."
docker volume create redis_data_restored
docker run -d --name redis-temp \
    -v redis_data_restored:/data \
    redis:7-alpine redis-server --save ""

docker cp ./redis_restore.rdb redis-temp:/data/dump.rdb
docker exec redis-temp chown redis:redis /data/dump.rdb
docker stop redis-temp
docker rm redis-temp

# Update docker-compose to use restored volume
sed -i.bak 's/redis_data:/redis_data_restored:/' docker-compose.yml

# Start services
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Verify recovery
echo "Verifying recovery..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Cache system recovery successful"

    # Create recovery report
    cat > recovery_report_$(date +%Y%m%d_%H%M%S).txt << EOF
Cache System Recovery Report
Date: $(date)
Backup Used: $BACKUP_FILE
Recovery Mode: $RECOVERY_MODE
Status: SUCCESS

Services Status:
$(docker-compose ps)

Cache Health:
$(curl -s http://localhost:8000/health | jq .)
EOF

else
    echo "✗ Cache system recovery failed"
    exit 1
fi

# Cleanup
rm -f ./redis_restore.rdb

echo "Disaster recovery completed successfully!"
```

## Performance Monitoring and Alerts

### Automated Performance Monitoring

```python
#!/usr/bin/env python3
"""Automated performance monitoring and alerting."""

import asyncio
import json
import logging
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from typing import Dict, List

from src.services.cache_service import get_cache_service

class PerformanceMonitor:
    def __init__(self, config_file: str = "monitoring_config.json"):
        self.config = self._load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.alert_history = {}

    def _load_config(self, config_file: str) -> dict:
        """Load monitoring configuration."""
        try:
            with open(config_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default monitoring configuration."""
        return {
            "thresholds": {
                "hit_rate_min": 0.7,
                "latency_max_ms": 50,
                "memory_usage_max_percent": 85,
                "error_rate_max_percent": 5,
                "connection_usage_max_percent": 90
            },
            "alert_cooldown_minutes": 30,
            "monitoring_interval_seconds": 60,
            "email": {
                "enabled": False,
                "smtp_server": "localhost",
                "smtp_port": 587,
                "from_address": "cache-monitor@yourorg.com",
                "to_addresses": ["ops-team@yourorg.com"]
            }
        }

    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        self.logger.info("Starting performance monitoring...")

        while True:
            try:
                await self._check_performance_metrics()
                await asyncio.sleep(self.config["monitoring_interval_seconds"])
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _check_performance_metrics(self):
        """Check current performance metrics against thresholds."""
        cache_service = await get_cache_service()

        # Collect metrics
        metrics = {
            "timestamp": time.time(),
            "cache_stats": cache_service.get_stats(),
            "health_info": await cache_service.get_health(),
            "tier_stats": getattr(cache_service, 'get_tier_stats', lambda: {})()
        }

        # Check thresholds and generate alerts
        alerts = self._check_thresholds(metrics)

        # Process alerts
        for alert in alerts:
            await self._process_alert(alert, metrics)

    def _check_thresholds(self, metrics: dict) -> List[dict]:
        """Check metrics against configured thresholds."""
        alerts = []
        thresholds = self.config["thresholds"]

        # Check hit rate
        hit_rate = metrics["cache_stats"].hit_rate
        if hit_rate < thresholds["hit_rate_min"]:
            alerts.append({
                "type": "low_hit_rate",
                "severity": "warning",
                "message": f"Cache hit rate {hit_rate:.1%} below threshold {thresholds['hit_rate_min']:.1%}",
                "current_value": hit_rate,
                "threshold": thresholds["hit_rate_min"]
            })

        # Check latency
        health_info = metrics["health_info"]
        if health_info.redis_ping_time and health_info.redis_ping_time > thresholds["latency_max_ms"]:
            alerts.append({
                "type": "high_latency",
                "severity": "critical",
                "message": f"Redis latency {health_info.redis_ping_time:.1f}ms above threshold {thresholds['latency_max_ms']}ms",
                "current_value": health_info.redis_ping_time,
                "threshold": thresholds["latency_max_ms"]
            })

        # Check memory usage
        if health_info.memory_usage:
            used_memory = health_info.memory_usage.get("used_memory", 0)
            # Assuming we know max memory from config (this would be Redis maxmemory)
            memory_usage_percent = (used_memory / (8 * 1024**3)) * 100  # Assume 8GB max

            if memory_usage_percent > thresholds["memory_usage_max_percent"]:
                alerts.append({
                    "type": "high_memory_usage",
                    "severity": "warning",
                    "message": f"Memory usage {memory_usage_percent:.1f}% above threshold {thresholds['memory_usage_max_percent']}%",
                    "current_value": memory_usage_percent,
                    "threshold": thresholds["memory_usage_max_percent"]
                })

        return alerts

    async def _process_alert(self, alert: dict, metrics: dict):
        """Process an alert (logging, notifications, etc.)."""
        alert_key = f"{alert['type']}_{alert['severity']}"
        current_time = time.time()

        # Check cooldown period
        if alert_key in self.alert_history:
            last_alert_time = self.alert_history[alert_key]
            cooldown_seconds = self.config["alert_cooldown_minutes"] * 60

            if current_time - last_alert_time < cooldown_seconds:
                return  # Skip alert due to cooldown

        # Log alert
        self.logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")

        # Send notification
        if self.config["email"]["enabled"]:
            await self._send_email_alert(alert, metrics)

        # Update alert history
        self.alert_history[alert_key] = current_time

    async def _send_email_alert(self, alert: dict, metrics: dict):
        """Send email alert notification."""
        try:
            subject = f"Cache System Alert: {alert['type'].replace('_', ' ').title()}"

            body = f"""
Cache System Alert

Alert Type: {alert['type']}
Severity: {alert['severity'].upper()}
Message: {alert['message']}
Timestamp: {datetime.fromtimestamp(metrics['timestamp'])}

Current Metrics:
- Hit Rate: {metrics['cache_stats'].hit_rate:.1%}
- Total Operations: {metrics['cache_stats'].total_operations:,}
- Redis Connected: {metrics['health_info'].redis_connected}
- Redis Ping Time: {metrics['health_info'].redis_ping_time:.1f}ms

Please investigate and take appropriate action.
            """

            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.config["email"]["from_address"]
            msg['To'] = ', '.join(self.config["email"]["to_addresses"])

            with smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"]) as server:
                server.send_message(msg)

            self.logger.info(f"Alert email sent for {alert['type']}")

        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")

# Usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    asyncio.run(monitor.start_monitoring())
```

This comprehensive management and maintenance guide provides all the operational procedures needed to successfully manage a production cache system. The scripts and procedures can be automated and integrated into your existing operational workflows.
