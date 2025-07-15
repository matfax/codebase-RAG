# Cache System Deployment Guide

This guide provides comprehensive instructions for deploying the advanced cache system in production environments.

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Redis**: 7.0 or higher
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **CPU**: Multi-core processor recommended for optimal performance
- **Storage**: SSD recommended for Redis persistence

### Dependencies

```bash
# Core dependencies
redis>=4.5.0
redis-py>=4.5.0
asyncio
aioredis>=2.0.0

# Optimization dependencies
orjson>=3.8.0
msgpack>=1.0.0
lz4>=4.0.0
brotli>=1.0.0

# Monitoring dependencies
prometheus-client>=0.16.0
structlog>=22.0.0
```

## Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd query-caching-layer-wave

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Redis Setup

#### Using Docker (Recommended)

```bash
# Start Redis with optimized configuration
docker-compose -f docker-compose.cache.yml up -d
```

#### Manual Installation

```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis                 # macOS

# Configure Redis (edit /etc/redis/redis.conf)
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### 3. Configuration

Create a production configuration file:

```python
# config/production.py
from src.config.cache_config import CacheConfig, CacheLevel, CacheWriteStrategy

CACHE_CONFIG = CacheConfig(
    enabled=True,
    cache_level=CacheLevel.BOTH,
    write_strategy=CacheWriteStrategy.WRITE_THROUGH,

    # Redis configuration
    redis_host="localhost",
    redis_port=6333,
    redis_password="your_redis_password",
    redis_ssl_enabled=True,

    # Performance tuning
    default_ttl=3600,
    max_connections=50,
    connection_timeout=30,

    # Memory management
    memory_max_size=10000,
    memory_max_memory_mb=1024,
    memory_cleanup_interval=300,

    # Security
    encryption_enabled=True,
    encryption_key="your_32_byte_key_here",

    # Monitoring
    metrics_enabled=True,
    health_check_interval=60
)
```

## Deployment Options

### Option 1: Standalone Deployment

```bash
# Start the cache service
python -m src.main --config config/production.py
```

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000
CMD ["python", "-m", "src.main", "--config", "config/production.py"]
```

```bash
# Build and run
docker build -t cache-system .
docker run -d --name cache-system -p 8000:8000 cache-system
```

### Option 3: Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cache-system
  template:
    metadata:
      labels:
        app: cache-system
    spec:
      containers:
      - name: cache-system
        image: cache-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: cache-system-service
spec:
  selector:
    app: cache-system
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

## Performance Optimization

### Memory Configuration

```python
# Optimize for your workload
CACHE_CONFIG.memory.max_size = 50000        # Increase for larger datasets
CACHE_CONFIG.memory.max_memory_mb = 4096    # 4GB memory limit
CACHE_CONFIG.memory.cleanup_interval = 180  # More frequent cleanup
```

### Redis Optimization

```bash
# Redis configuration optimizations
maxmemory 8gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 300

# Persistence optimization
save 300 10
save 60 1000
```

### Connection Pool Tuning

```python
CACHE_CONFIG.redis.max_connections = 100
CACHE_CONFIG.redis.connection_timeout = 10
CACHE_CONFIG.redis.socket_timeout = 5
CACHE_CONFIG.redis.retry_on_timeout = True
```

## Security Configuration

### SSL/TLS Setup

```python
CACHE_CONFIG.redis.ssl_enabled = True
CACHE_CONFIG.redis.ssl_cert_path = "/path/to/cert.pem"
CACHE_CONFIG.redis.ssl_key_path = "/path/to/key.pem"
CACHE_CONFIG.redis.ssl_ca_cert_path = "/path/to/ca.pem"
```

### Encryption Configuration

```python
# Generate encryption key
from cryptography.fernet import Fernet
encryption_key = Fernet.generate_key()

CACHE_CONFIG.encryption.enabled = True
CACHE_CONFIG.encryption.key = encryption_key
CACHE_CONFIG.encryption.algorithm = "AES-256-GCM"
```

### Access Control

```python
# Redis AUTH
CACHE_CONFIG.redis.password = "strong_password_here"

# Network security
CACHE_CONFIG.redis.host = "internal-redis.company.com"
CACHE_CONFIG.allowed_hosts = ["10.0.0.0/8", "192.168.0.0/16"]
```

## Monitoring and Observability

### Metrics Collection

```python
# Enable comprehensive metrics
CACHE_CONFIG.metrics.enabled = True
CACHE_CONFIG.metrics.export_interval = 30
CACHE_CONFIG.metrics.include_detailed_stats = True
```

### Prometheus Integration

```python
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Start metrics server
start_http_server(9090)

# Custom metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_latency = Histogram('cache_operation_duration_seconds', 'Cache operation latency')
cache_memory = Gauge('cache_memory_usage_bytes', 'Cache memory usage')
```

### Health Checks

```python
async def health_check():
    """Comprehensive health check endpoint."""
    cache_service = await get_cache_service()

    health_info = await cache_service.get_health()
    return {
        "status": "healthy" if health_info.status == CacheHealthStatus.HEALTHY else "unhealthy",
        "redis_connected": health_info.redis_connected,
        "memory_usage": health_info.memory_usage,
        "timestamp": time.time()
    }
```

### Logging Configuration

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

## Backup and Recovery

### Redis Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_backup_$DATE.rdb

# Cleanup old backups (keep 7 days)
find $BACKUP_DIR -name "redis_backup_*.rdb" -mtime +7 -delete
```

### Cache Warm-up

```python
async def warm_cache_on_startup():
    """Warm cache with critical data on startup."""
    cache_service = await get_cache_service()

    # Load critical project data
    for project in critical_projects:
        await cache_service.trigger_cache_warmup('aggressive', project_keys[project])

    # Pre-load frequent embeddings
    await embedding_service.warm_frequent_embeddings()

    # Pre-compute common search results
    await search_service.warm_popular_searches()
```

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check Redis memory usage
redis-cli INFO memory

# Optimize memory settings
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET maxmemory 2gb
```

#### Connection Pool Exhaustion

```python
# Increase connection pool size
CACHE_CONFIG.redis.max_connections = 200

# Implement connection monitoring
async def monitor_connections():
    pool_stats = await cache_service.get_connection_pool_stats()
    if pool_stats['available_connections'] < 10:
        logger.warning("Low connection pool availability")
```

#### Cache Inconsistency

```python
# Run cache coherency check
coherency_result = await cache_service.check_cache_coherency()
if not coherency_result['coherent']:
    logger.error(f"Cache inconsistency detected: {coherency_result}")
    await cache_service.clear()  # Emergency cache clear
```

### Performance Debugging

```python
# Enable debug mode
CACHE_CONFIG.debug_mode = True

# Performance profiling
import cProfile
profiler = cProfile.Profile()
profiler.enable()

# Your cache operations here

profiler.disable()
profiler.dump_stats('cache_performance.prof')
```

### Log Analysis

```bash
# Common log patterns to monitor
grep "ERROR" cache.log | tail -20
grep "high_latency" cache.log | wc -l
grep "cache_miss" cache.log | head -10
```

## Scaling Considerations

### Horizontal Scaling

```python
# Multiple cache instances with consistent hashing
CACHE_CONFIG.cluster.enabled = True
CACHE_CONFIG.cluster.nodes = [
    "cache-1.internal:6379",
    "cache-2.internal:6379",
    "cache-3.internal:6379"
]
```

### Vertical Scaling

```python
# Optimize for larger machines
CACHE_CONFIG.memory.max_memory_mb = 16384  # 16GB
CACHE_CONFIG.redis.max_connections = 500
CACHE_CONFIG.concurrency.max_workers = 20
```

### Database Sharding

```python
# Shard by project or content type
def get_shard_key(cache_key: str) -> str:
    project = cache_key.split(':')[0]
    return f"shard_{hash(project) % 4}"
```

## Migration Guide

### From Basic Redis

```python
# Migration script
async def migrate_from_basic_redis():
    # Read existing Redis data
    old_keys = await old_redis.keys("*")

    # Migrate to new cache system
    for key in old_keys:
        value = await old_redis.get(key)
        await new_cache_service.set(key, value)

    logger.info(f"Migrated {len(old_keys)} keys")
```

### Version Upgrades

```bash
# Backup before upgrade
./scripts/backup_cache.sh

# Upgrade with rolling deployment
kubectl set image deployment/cache-system cache-system=cache-system:v2.0.0
kubectl rollout status deployment/cache-system

# Validate upgrade
./scripts/validate_cache_system.py --config config/production.py
```

## Maintenance

### Regular Tasks

```bash
# Daily tasks
0 2 * * * /app/scripts/backup_cache.sh
0 3 * * * /app/scripts/cleanup_expired_keys.sh

# Weekly tasks
0 1 * * 0 /app/scripts/optimize_cache.sh
0 2 * * 0 /app/scripts/validate_cache_system.py

# Monthly tasks
0 0 1 * * /app/scripts/performance_report.sh
```

### Cache Optimization

```python
# Automated optimization
async def daily_optimization():
    optimizer = AdaptivePerformanceOptimizer()

    # Collect performance profile
    profile = await collect_performance_metrics()

    # Run optimization
    result = await optimizer.optimize_performance(profile)

    if result['recommendations_applied'] > 0:
        logger.info(f"Applied {result['recommendations_applied']} optimizations")
```

## Support and Documentation

### Additional Resources

- [API Documentation](./API_REFERENCE.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [Security Best Practices](./SECURITY.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

### Getting Help

- GitHub Issues: [Create an issue](https://github.com/your-org/cache-system/issues)
- Documentation: [Full documentation](https://docs.your-org.com/cache-system)
- Support: support@your-org.com

---

**Note**: This deployment guide assumes a production environment. For development setups, see the [Development Guide](./DEVELOPMENT.md).
