# Cache Monitoring and Metrics Guide

## Overview

This comprehensive guide covers monitoring, metrics collection, and dashboard setup for the Query Caching Layer system. It provides detailed instructions for implementing observability, creating dashboards, and setting up alerting systems to ensure optimal cache performance.

## Monitoring Architecture

### Monitoring Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Cache App     │ │   MCP Tools     │ │   Custom        │   │
│  │   Metrics       │ │   Metrics       │ │   Applications  │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Metrics Collection Layer                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Prometheus    │ │   OpenTelemetry │ │   Custom        │   │
│  │   Metrics       │ │   Tracing       │ │   Collectors    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Storage and Processing                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │   Prometheus    │ │   Jaeger        │ │   InfluxDB      │   │
│  │   TSDB          │ │   Traces        │ │   (Optional)    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               Visualization and Alerting                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │    Grafana      │ │   AlertManager  │ │   PagerDuty     │   │
│  │   Dashboards    │ │   Rules         │ │   Slack, etc.   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics Collection

### Application Metrics

#### Cache Performance Metrics

```python
# src/utils/cache_metrics.py
"""Cache metrics collection and reporting."""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

@dataclass
class CacheMetrics:
    """Cache metrics data structure."""

    # Performance metrics
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    total_operations: int = 0
    operations_per_second: float = 0.0

    # Latency metrics (in milliseconds)
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    # Memory metrics
    l1_memory_usage_mb: float = 0.0
    l1_memory_usage_percent: float = 0.0
    l2_memory_usage_mb: float = 0.0
    l2_memory_usage_percent: float = 0.0

    # Connection metrics
    active_connections: int = 0
    connection_pool_usage_percent: float = 0.0

    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0

    # Cache-specific metrics
    cache_size_entries: int = 0
    eviction_count: int = 0
    expiration_count: int = 0

class PrometheusMetricsCollector:
    """Prometheus metrics collector for cache system."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._init_metrics()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""

        # Cache operation counters
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total number of cache operations',
            ['operation_type', 'cache_tier', 'cache_name'],
            registry=self.registry
        )

        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['cache_tier', 'cache_name'],
            registry=self.registry
        )

        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total number of cache misses',
            ['cache_tier', 'cache_name'],
            registry=self.registry
        )

        # Cache latency histograms
        self.cache_operation_duration = Histogram(
            'cache_operation_duration_seconds',
            'Duration of cache operations',
            ['operation_type', 'cache_tier'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )

        # Cache size and memory gauges
        self.cache_size_entries = Gauge(
            'cache_size_entries',
            'Number of entries in cache',
            ['cache_tier', 'cache_name'],
            registry=self.registry
        )

        self.cache_memory_usage_bytes = Gauge(
            'cache_memory_usage_bytes',
            'Memory usage of cache in bytes',
            ['cache_tier', 'cache_name'],
            registry=self.registry
        )

        self.cache_memory_usage_percent = Gauge(
            'cache_memory_usage_percent',
            'Memory usage percentage of cache',
            ['cache_tier', 'cache_name'],
            registry=self.registry
        )

        # Connection metrics
        self.cache_connections_active = Gauge(
            'cache_connections_active',
            'Number of active cache connections',
            ['cache_name'],
            registry=self.registry
        )

        # Hit rate gauge
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate as percentage',
            ['cache_tier', 'cache_name'],
            registry=self.registry
        )

        # Error metrics
        self.cache_errors_total = Counter(
            'cache_errors_total',
            'Total number of cache errors',
            ['error_type', 'cache_tier'],
            registry=self.registry
        )

    def record_cache_operation(
        self,
        operation: str,
        cache_tier: str,
        cache_name: str = "default",
        duration: Optional[float] = None,
        hit: Optional[bool] = None,
        error: bool = False
    ):
        """Record a cache operation."""

        # Record operation
        self.cache_operations_total.labels(
            operation_type=operation,
            cache_tier=cache_tier,
            cache_name=cache_name
        ).inc()

        # Record hit/miss
        if hit is not None:
            if hit:
                self.cache_hits_total.labels(
                    cache_tier=cache_tier,
                    cache_name=cache_name
                ).inc()
            else:
                self.cache_misses_total.labels(
                    cache_tier=cache_tier,
                    cache_name=cache_name
                ).inc()

        # Record duration
        if duration is not None:
            self.cache_operation_duration.labels(
                operation_type=operation,
                cache_tier=cache_tier
            ).observe(duration)

        # Record error
        if error:
            self.cache_errors_total.labels(
                error_type="operation_error",
                cache_tier=cache_tier
            ).inc()

    def update_cache_size(self, cache_tier: str, cache_name: str, size: int):
        """Update cache size metric."""
        self.cache_size_entries.labels(
            cache_tier=cache_tier,
            cache_name=cache_name
        ).set(size)

    def update_memory_usage(
        self,
        cache_tier: str,
        cache_name: str,
        usage_bytes: int,
        usage_percent: float
    ):
        """Update memory usage metrics."""
        self.cache_memory_usage_bytes.labels(
            cache_tier=cache_tier,
            cache_name=cache_name
        ).set(usage_bytes)

        self.cache_memory_usage_percent.labels(
            cache_tier=cache_tier,
            cache_name=cache_name
        ).set(usage_percent)

    def update_hit_rate(self, cache_tier: str, cache_name: str, hit_rate: float):
        """Update hit rate metric."""
        self.cache_hit_rate.labels(
            cache_tier=cache_tier,
            cache_name=cache_name
        ).set(hit_rate * 100)  # Convert to percentage
```

#### Metrics Integration with Cache Service

```python
# Integration with cache service
class InstrumentedCacheService:
    """Cache service with metrics instrumentation."""

    def __init__(self, cache_service, metrics_collector):
        self.cache_service = cache_service
        self.metrics = metrics_collector

    async def get(self, key: str):
        """Instrumented get operation."""
        start_time = time.perf_counter()

        try:
            result = await self.cache_service.get(key)
            duration = time.perf_counter() - start_time

            # Record metrics
            self.metrics.record_cache_operation(
                operation="get",
                cache_tier="multi",
                duration=duration,
                hit=result is not None
            )

            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.metrics.record_cache_operation(
                operation="get",
                cache_tier="multi",
                duration=duration,
                error=True
            )
            raise

    async def set(self, key: str, value, ttl: Optional[int] = None):
        """Instrumented set operation."""
        start_time = time.perf_counter()

        try:
            result = await self.cache_service.set(key, value, ttl)
            duration = time.perf_counter() - start_time

            self.metrics.record_cache_operation(
                operation="set",
                cache_tier="multi",
                duration=duration
            )

            return result

        except Exception as e:
            duration = time.perf_counter() - start_time
            self.metrics.record_cache_operation(
                operation="set",
                cache_tier="multi",
                duration=duration,
                error=True
            )
            raise
```

### System Metrics

#### Redis Metrics Collection

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis-cache:6379
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    depends_on:
      - redis-cache
    networks:
      - cache_network

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - cache_network

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-cache
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - cache_network

volumes:
  prometheus_data:

networks:
  cache_network:
    external: true
```

#### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'cache-system'
    environment: 'production'

rule_files:
  - "rules/*.yml"

scrape_configs:
  # Cache application metrics
  - job_name: 'cache-app'
    static_configs:
      - targets: ['cache-app-1:8000', 'cache-app-2:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s
    params:
      format: ['prometheus']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s
    scrape_timeout: 10s

  # System metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s
    scrape_timeout: 10s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Remote write (optional - for long-term storage)
remote_write:
  - url: "http://cortex:9009/api/prom/push"
    queue_config:
      max_samples_per_send: 1000
      max_shards: 200
      capacity: 2500
```

## Dashboard Setup

### Grafana Configuration

#### Docker Compose for Grafana

```yaml
# Add to docker-compose.monitoring.yml
  grafana:
    image: grafana/grafana:latest
    container_name: grafana-cache
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    networks:
      - cache_network

volumes:
  grafana_data:
```

#### Grafana Provisioning

```yaml
# monitoring/grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus-cache:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "15s"
      queryTimeout: "60s"
      httpMethod: "POST"
```

```yaml
# monitoring/grafana/provisioning/dashboards/dashboard.yml
apiVersion: 1

providers:
  - name: 'Cache System Dashboards'
    orgId: 1
    folder: 'Cache System'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

### Cache System Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Cache System Overview",
    "tags": ["cache", "performance", "redis"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Cache Hit Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "cache_hit_rate{cache_tier=\"multi\"}",
            "legendFormat": "Hit Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 60},
                {"color": "green", "value": 80}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Operations per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(cache_operations_total[5m])",
            "legendFormat": "Ops/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ops",
            "decimals": 1
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "Cache Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cache_memory_usage_bytes{cache_tier=\"l1\"} / 1024 / 1024",
            "legendFormat": "L1 Memory (MB)"
          },
          {
            "expr": "redis_memory_used_bytes / 1024 / 1024",
            "legendFormat": "L2 Memory (MB)"
          }
        ],
        "yAxes": [
          {
            "label": "Memory (MB)",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Cache Operation Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, cache_operation_duration_seconds_bucket)",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, cache_operation_duration_seconds_bucket)",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, cache_operation_duration_seconds_bucket)",
            "legendFormat": "p99"
          }
        ],
        "yAxes": [
          {
            "label": "Latency (seconds)",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 5,
        "title": "Redis Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_connected_clients",
            "legendFormat": "Connected Clients"
          },
          {
            "expr": "redis_config_maxclients",
            "legendFormat": "Max Clients"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "id": 6,
        "title": "Cache Errors",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cache_errors_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "yAxes": [
          {
            "label": "Errors/sec",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### Performance Dashboard

```json
{
  "dashboard": {
    "title": "Cache Performance Deep Dive",
    "panels": [
      {
        "id": 10,
        "title": "Hit Rate by Cache Type",
        "type": "graph",
        "targets": [
          {
            "expr": "cache_hit_rate by (cache_name)",
            "legendFormat": "{{cache_name}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 11,
        "title": "Operations Breakdown",
        "type": "piechart",
        "targets": [
          {
            "expr": "increase(cache_operations_total[1h]) by (operation_type)",
            "legendFormat": "{{operation_type}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 12,
        "title": "Cache Size Trends",
        "type": "graph",
        "targets": [
          {
            "expr": "cache_size_entries by (cache_tier)",
            "legendFormat": "{{cache_tier}} entries"
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      },
      {
        "id": 13,
        "title": "Memory Pressure",
        "type": "graph",
        "targets": [
          {
            "expr": "cache_memory_usage_percent",
            "legendFormat": "Memory Usage %"
          }
        ],
        "alert": {
          "conditions": [
            {
              "query": {"params": ["A", "5m", "now"]},
              "reducer": {"type": "last"},
              "evaluator": {
                "params": [85],
                "type": "gt"
              }
            }
          ],
          "executionErrorState": "alerting",
          "noDataState": "no_data",
          "frequency": "10s",
          "handler": 1,
          "name": "High Memory Usage",
          "message": "Cache memory usage is above 85%"
        },
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
      }
    ]
  }
}
```

## Alerting Rules

### Prometheus Alerting Rules

```yaml
# monitoring/rules/cache_alerts.yml
groups:
  - name: cache_performance
    rules:
      - alert: LowCacheHitRate
        expr: cache_hit_rate < 60
        for: 5m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Low cache hit rate detected"
          description: "Cache hit rate is {{ $value }}% which is below the 60% threshold"
          runbook_url: "https://docs.yourorg.com/runbooks/cache-hit-rate"

      - alert: HighCacheLatency
        expr: histogram_quantile(0.95, cache_operation_duration_seconds_bucket) > 0.05
        for: 2m
        labels:
          severity: critical
          component: cache
        annotations:
          summary: "High cache operation latency"
          description: "95th percentile latency is {{ $value }}s which exceeds 50ms threshold"

      - alert: CacheMemoryPressure
        expr: cache_memory_usage_percent > 85
        for: 3m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Cache memory pressure detected"
          description: "Cache memory usage is {{ $value }}% which exceeds 85% threshold"

      - alert: RedisDown
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
          component: redis
        annotations:
          summary: "Redis is down"
          description: "Redis instance has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(cache_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "High cache error rate"
          description: "Cache error rate is {{ $value }} errors/sec"

  - name: redis_alerts
    rules:
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 5m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis memory usage high"
          description: "Redis memory usage is {{ $value | humanizePercentage }}"

      - alert: RedisConnectionsHigh
        expr: redis_connected_clients / redis_config_maxclients > 0.8
        for: 3m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis connections high"
          description: "Redis has {{ $value }} connections, approaching limit"

      - alert: RedisSlowQueries
        expr: increase(redis_slowlog_length[5m]) > 10
        for: 1m
        labels:
          severity: warning
          component: redis
        annotations:
          summary: "Redis slow queries detected"
          description: "{{ $value }} slow queries detected in the last 5 minutes"
```

### AlertManager Configuration

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourorg.com'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://webhook-service:5000/alerts'

  - name: 'critical-alerts'
    email_configs:
      - to: 'ops-team@yourorg.com'
        subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#ops-alerts'
        title: 'CRITICAL Cache Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'warning-alerts'
    email_configs:
      - to: 'dev-team@yourorg.com'
        subject: 'WARNING: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

## Custom Monitoring Scripts

### Health Check Monitor

```python
#!/usr/bin/env python3
"""Comprehensive health monitoring script."""

import asyncio
import json
import logging
import requests
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

class CacheHealthMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run_health_checks(self) -> Dict:
        """Run comprehensive health checks."""

        health_results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {}
        }

        # Run individual health checks
        checks = [
            ("cache_service", self._check_cache_service),
            ("redis_connectivity", self._check_redis_connectivity),
            ("memory_usage", self._check_memory_usage),
            ("performance", self._check_performance),
            ("error_rates", self._check_error_rates)
        ]

        passed_checks = 0
        total_checks = len(checks)

        for check_name, check_func in checks:
            try:
                result = await check_func()
                health_results["checks"][check_name] = result

                if result.get("status") == "healthy":
                    passed_checks += 1

            except Exception as e:
                health_results["checks"][check_name] = {
                    "status": "error",
                    "error": str(e)
                }

        # Determine overall status
        if passed_checks == total_checks:
            health_results["overall_status"] = "healthy"
        elif passed_checks > total_checks * 0.5:
            health_results["overall_status"] = "degraded"
        else:
            health_results["overall_status"] = "unhealthy"

        return health_results

    async def _check_cache_service(self) -> Dict:
        """Check cache service health."""
        try:
            response = requests.get(
                f"{self.config['cache_service_url']}/health",
                timeout=10
            )

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "http_status": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _check_redis_connectivity(self) -> Dict:
        """Check Redis connectivity and performance."""
        try:
            import redis

            r = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                password=self.config['redis_password'],
                socket_timeout=5
            )

            start_time = time.perf_counter()
            r.ping()
            ping_time = (time.perf_counter() - start_time) * 1000

            # Get Redis info
            info = r.info()

            return {
                "status": "healthy",
                "ping_time_ms": ping_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _check_performance(self) -> Dict:
        """Check cache performance metrics."""
        try:
            response = requests.get(
                f"{self.config['cache_service_url']}/metrics/cache",
                timeout=10
            )

            if response.status_code == 200:
                metrics = response.json()
                hit_rate = metrics.get("hit_rate", 0)

                if hit_rate >= 0.7:
                    status = "healthy"
                elif hit_rate >= 0.5:
                    status = "degraded"
                else:
                    status = "unhealthy"

                return {
                    "status": status,
                    "hit_rate": hit_rate,
                    "total_operations": metrics.get("total_operations", 0),
                    "avg_latency_ms": metrics.get("avg_latency_ms", 0)
                }
            else:
                return {
                    "status": "error",
                    "http_status": response.status_code
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# Usage example
if __name__ == "__main__":
    config = {
        "cache_service_url": "http://localhost:8000",
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_password": None
    }

    monitor = CacheHealthMonitor(config)
    results = asyncio.run(monitor.run_health_checks())

    print(json.dumps(results, indent=2))
```

### Performance Metrics Collector

```python
#!/usr/bin/env python3
"""Performance metrics collection script."""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

class PerformanceMetricsCollector:
    def __init__(self, output_dir: str = "./metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    async def collect_metrics(self) -> Dict:
        """Collect comprehensive performance metrics."""

        metrics = {
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "cache_metrics": await self._collect_cache_metrics(),
            "redis_metrics": await self._collect_redis_metrics(),
            "system_metrics": await self._collect_system_metrics()
        }

        # Save metrics to file
        metrics_file = self.output_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    async def _collect_cache_metrics(self) -> Dict:
        """Collect cache-specific metrics."""
        try:
            import requests
            response = requests.get("http://localhost:8000/metrics/cache", timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    async def _collect_redis_metrics(self) -> Dict:
        """Collect Redis metrics."""
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379)
            info = r.info()

            return {
                "memory": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "N/A"),
                    "used_memory_peak": info.get("used_memory_peak", 0),
                    "used_memory_peak_human": info.get("used_memory_peak_human", "N/A")
                },
                "clients": {
                    "connected_clients": info.get("connected_clients", 0),
                    "client_recent_max_input_buffer": info.get("client_recent_max_input_buffer", 0),
                    "client_recent_max_output_buffer": info.get("client_recent_max_output_buffer", 0)
                },
                "stats": {
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                    "total_net_input_bytes": info.get("total_net_input_bytes", 0),
                    "total_net_output_bytes": info.get("total_net_output_bytes", 0)
                },
                "persistence": {
                    "loading": info.get("loading", 0),
                    "rdb_changes_since_last_save": info.get("rdb_changes_since_last_save", 0),
                    "rdb_bgsave_in_progress": info.get("rdb_bgsave_in_progress", 0),
                    "rdb_last_save_time": info.get("rdb_last_save_time", 0)
                }
            }

        except Exception as e:
            return {"error": str(e)}

    async def _collect_system_metrics(self) -> Dict:
        """Collect system-level metrics."""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()

            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            }

        except Exception as e:
            return {"error": str(e)}

# Usage
if __name__ == "__main__":
    collector = PerformanceMetricsCollector()
    metrics = asyncio.run(collector.collect_metrics())
    print(f"Metrics collected: {metrics['date']}")
```

## Deployment Scripts

### Complete Monitoring Stack Deployment

```bash
#!/bin/bash
# scripts/deploy-monitoring.sh

set -e

ENVIRONMENT=${1:-staging}
GRAFANA_PASSWORD=${2:-admin}

echo "Deploying monitoring stack for $ENVIRONMENT environment..."

# Create monitoring directories
mkdir -p monitoring/{prometheus,grafana,alertmanager}
mkdir -p monitoring/grafana/{dashboards,provisioning}
mkdir -p monitoring/prometheus/rules

# Generate Grafana password
export GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Import Grafana dashboards
echo "Importing Grafana dashboards..."
./scripts/import-dashboards.sh

# Configure alerting rules
echo "Setting up alerting rules..."
curl -X POST http://localhost:9090/-/reload

# Verify deployment
echo "Verifying monitoring deployment..."
curl -f http://localhost:9090/api/v1/status/config > /dev/null
curl -f http://localhost:3000/api/health > /dev/null

echo "Monitoring stack deployed successfully!"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/$GRAFANA_PASSWORD)"
```

This comprehensive monitoring and metrics guide provides everything needed to implement robust observability for the cache system, including real-time dashboards, alerting, and performance tracking.
