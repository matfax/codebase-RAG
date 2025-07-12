# Cache Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, diagnostic procedures, and solutions for the Query Caching Layer system. Use this guide to quickly identify and resolve cache-related problems.

## Quick Diagnostic Commands

### Health Check Commands

```bash
# Basic cache health check
curl -X GET http://localhost:8000/health/cache

# Detailed cache statistics
curl -X GET http://localhost:8000/stats/cache

# Redis connectivity test
redis-cli -h localhost -p 6379 ping

# Check Docker containers
docker-compose -f docker-compose.cache.yml ps

# View cache service logs
docker-compose -f docker-compose.cache.yml logs redis-cache

# Check memory usage
docker stats --no-stream
```

### Python Diagnostic Tools

```python
# Quick cache service test
from src.services.cache_service import get_cache_service

async def test_cache_health():
    cache_service = await get_cache_service()

    # Test basic operations
    await cache_service.set("health_check", "test_value")
    result = await cache_service.get("health_check")
    await cache_service.delete("health_check")

    print(f"Cache test result: {result}")

    # Get health info
    health = await cache_service.get_health()
    print(f"Cache health: {health}")

# Run diagnostics
import asyncio
asyncio.run(test_cache_health())
```

## Common Issues and Solutions

### 1. Connection Issues

#### Issue: Redis Connection Failed

**Symptoms:**
- `ConnectionError: Error connecting to Redis`
- Cache operations timing out
- MCP tools failing with cache errors

**Diagnostic Steps:**
```bash
# 1. Check if Redis is running
docker-compose -f docker-compose.cache.yml ps

# 2. Test Redis connectivity
redis-cli -h localhost -p 6379 ping
# Expected output: PONG

# 3. Check Redis logs
docker-compose -f docker-compose.cache.yml logs redis-cache

# 4. Verify network connectivity
telnet localhost 6379

# 5. Check firewall settings
sudo ufw status
```

**Solutions:**

1. **Start Redis Service:**
```bash
docker-compose -f docker-compose.cache.yml up -d redis-cache
```

2. **Fix Configuration Issues:**
```bash
# Check Redis configuration
docker exec -it redis-cache redis-cli config get "*"

# Verify bind address
docker exec -it redis-cache redis-cli config get bind
```

3. **Reset Redis Data:**
```bash
# Warning: This will delete all cached data
docker-compose -f docker-compose.cache.yml down -v
docker-compose -f docker-compose.cache.yml up -d
```

4. **Check Environment Variables:**
```bash
# Verify cache configuration
python -c "
from src.config.cache_config import get_cache_config
config = get_cache_config()
print(f'Redis host: {config.redis.host}')
print(f'Redis port: {config.redis.port}')
print(f'Redis password: {\"***\" if config.redis.password else \"None\"}')
"
```

#### Issue: SSL/TLS Connection Problems

**Symptoms:**
- SSL handshake failures
- Certificate verification errors
- Encrypted connection timeouts

**Diagnostic Steps:**
```bash
# 1. Check SSL configuration
openssl s_client -connect localhost:6380 -tls1_2

# 2. Verify certificates
openssl x509 -in /path/to/redis.crt -text -noout

# 3. Check certificate chain
openssl verify -CAfile /path/to/ca.crt /path/to/redis.crt
```

**Solutions:**

1. **Fix Certificate Issues:**
```bash
# Generate new certificates if needed
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout redis.key -out redis.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

2. **Update SSL Configuration:**
```python
# In cache configuration
REDIS_SSL_ENABLED=true
REDIS_SSL_CERT_PATH=/path/to/redis.crt
REDIS_SSL_KEY_PATH=/path/to/redis.key
REDIS_SSL_CA_CERT_PATH=/path/to/ca.crt
```

### 2. Performance Issues

#### Issue: High Cache Latency

**Symptoms:**
- Cache operations taking > 50ms
- Slow MCP tool responses
- High response time percentiles

**Diagnostic Steps:**
```python
# Performance diagnostic script
import time
import asyncio
from src.services.cache_service import get_cache_service

async def diagnose_performance():
    cache_service = await get_cache_service()

    # Test operation latency
    latencies = []
    for i in range(100):
        start_time = time.perf_counter()
        await cache_service.set(f"perf_test_{i}", f"value_{i}")
        latency = (time.perf_counter() - start_time) * 1000
        latencies.append(latency)

    import statistics
    print(f"Average latency: {statistics.mean(latencies):.2f}ms")
    print(f"P95 latency: {sorted(latencies)[95]:.2f}ms")
    print(f"Max latency: {max(latencies):.2f}ms")

asyncio.run(diagnose_performance())
```

**Solutions:**

1. **Optimize Redis Configuration:**
```conf
# redis.conf optimizations
tcp-keepalive 300
timeout 0
tcp-nodelay yes
save 900 1
save 300 10
save 60 10000
```

2. **Increase Connection Pool:**
```python
# In cache configuration
REDIS_MAX_CONNECTIONS=20
CACHE_CONNECTION_POOL_SIZE=15
```

3. **Enable Compression:**
```python
# For large cache values
CACHE_COMPRESSION_ENABLED=true
CACHE_COMPRESSION_THRESHOLD=1024
```

4. **Optimize Memory Settings:**
```conf
# Redis memory optimization
maxmemory 2gb
maxmemory-policy allkeys-lru
```

#### Issue: Low Cache Hit Rate

**Symptoms:**
- Cache hit rate < 60%
- Frequent cache misses
- Poor performance improvement

**Diagnostic Steps:**
```python
# Cache hit rate analysis
async def analyze_hit_rate():
    cache_service = await get_cache_service()
    stats = cache_service.get_stats()

    print(f"Hit rate: {stats.hit_rate:.2%}")
    print(f"Miss rate: {stats.miss_rate:.2%}")
    print(f"Total operations: {stats.total_operations}")

    # Analyze per cache type
    tier_stats = cache_service.get_tier_stats()
    print(f"L1 hit rate: {tier_stats['l1_stats'].hit_rate:.2%}")
    print(f"L2 hit rate: {tier_stats['l2_stats'].hit_rate:.2%}")

asyncio.run(analyze_hit_rate())
```

**Solutions:**

1. **Adjust TTL Settings:**
```python
# Increase TTL for stable data
EMBEDDING_CACHE_TTL=7200  # 2 hours
PROJECT_CACHE_TTL=3600    # 1 hour
SEARCH_CACHE_TTL=1800     # 30 minutes
```

2. **Increase Cache Size:**
```python
# Expand memory allocation
MEMORY_CACHE_MAX_SIZE=2000
MEMORY_CACHE_MAX_MEMORY_MB=512
```

3. **Implement Cache Warming:**
```python
# Add cache warming strategies
from src.services.cache_warmup_service import CacheWarmupService

async def warm_cache():
    warmup_service = CacheWarmupService(cache_service)
    await warmup_service.warm_project_cache("your_project")
```

### 3. Memory Issues

#### Issue: Memory Pressure and Eviction

**Symptoms:**
- High memory usage warnings
- Frequent cache evictions
- Out of memory errors

**Diagnostic Steps:**
```bash
# Check system memory
free -h

# Check Redis memory usage
redis-cli info memory

# Monitor Docker container memory
docker stats --no-stream

# Check cache memory metrics
python -c "
from src.services.cache_service import get_cache_service
import asyncio

async def check_memory():
    cache_service = await get_cache_service()
    tier_stats = cache_service.get_tier_stats()
    print(f'L1 memory usage: {tier_stats[\"l1_info\"][\"memory_usage_mb\"]:.1f} MB')
    print(f'L1 max memory: {tier_stats[\"l1_info\"][\"max_memory_mb\"]} MB')

asyncio.run(check_memory())
"
```

**Solutions:**

1. **Increase Memory Limits:**
```python
# Expand memory allocation
MEMORY_CACHE_MAX_MEMORY_MB=1024
REDIS_MAXMEMORY=4gb
```

2. **Optimize Eviction Policy:**
```python
# Better eviction strategy
MEMORY_CACHE_EVICTION_POLICY=LRU
REDIS_MAXMEMORY_POLICY=allkeys-lru
```

3. **Enable Memory Monitoring:**
```python
# Add memory pressure handling
CACHE_MEMORY_PRESSURE_ENABLED=true
CACHE_MEMORY_WARNING_THRESHOLD=0.8
CACHE_MEMORY_CRITICAL_THRESHOLD=0.9
```

4. **Implement Data Compression:**
```python
# Reduce memory footprint
CACHE_COMPRESSION_ENABLED=true
CACHE_COMPRESSION_ALGORITHM=lz4
```

#### Issue: Memory Leaks

**Symptoms:**
- Continuously increasing memory usage
- Cache size growing without bounds
- System becoming unresponsive

**Diagnostic Steps:**
```python
# Memory leak detection
import psutil
import time

def monitor_memory_usage(duration=300):
    """Monitor memory usage for potential leaks."""
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Baseline memory: {baseline_memory:.1f} MB")

    for i in range(duration // 10):
        time.sleep(10)
        current_memory = process.memory_info().rss / 1024 / 1024
        growth = current_memory - baseline_memory
        print(f"Memory after {(i+1)*10}s: {current_memory:.1f} MB (+{growth:.1f} MB)")

monitor_memory_usage()
```

**Solutions:**

1. **Enable Automatic Cleanup:**
```python
# Add cleanup intervals
MEMORY_CACHE_CLEANUP_INTERVAL=300  # 5 minutes
CACHE_EXPIRED_CLEANUP_ENABLED=true
```

2. **Implement Memory Monitoring:**
```python
# Add memory leak detection
from src.services.cache_memory_leak_detector import CacheMemoryLeakDetector

leak_detector = CacheMemoryLeakDetector()
await leak_detector.start_monitoring()
```

3. **Fix Code Issues:**
```python
# Ensure proper resource cleanup
async def proper_cache_usage():
    cache_service = await get_cache_service()
    try:
        # Cache operations
        pass
    finally:
        # Cleanup if needed
        await cache_service.cleanup_expired()
```

### 4. Data Consistency Issues

#### Issue: Cache Coherency Problems

**Symptoms:**
- Different values in L1 and L2 cache
- Stale data being returned
- Inconsistent search results

**Diagnostic Steps:**
```python
# Cache coherency check
async def check_cache_coherency():
    cache_service = await get_cache_service()

    if hasattr(cache_service, 'check_cache_coherency'):
        coherency_result = await cache_service.check_cache_coherency()
        print(f"Cache coherent: {coherency_result['coherent']}")
        print(f"L1-L2 consistent: {coherency_result['l1_l2_consistent']}")
        print(f"Stale entries: {coherency_result['stale_entries']}")
        print(f"Mismatched keys: {coherency_result['mismatched_keys']}")

asyncio.run(check_cache_coherency())
```

**Solutions:**

1. **Fix Write Strategy:**
```python
# Use write-through for consistency
CACHE_WRITE_STRATEGY=WRITE_THROUGH
```

2. **Manual Cache Sync:**
```python
# Force cache synchronization
async def sync_cache_tiers():
    cache_service = await get_cache_service()

    # Clear L1 cache to force L2 reload
    cache_service.l1_cache.clear()

    # Or flush dirty keys
    if hasattr(cache_service, '_flush_dirty_keys'):
        await cache_service._flush_dirty_keys()
```

3. **Enable Coherency Monitoring:**
```python
# Add coherency checks
CACHE_COHERENCY_CHECK_ENABLED=true
CACHE_COHERENCY_CHECK_INTERVAL=300
```

#### Issue: Cache Invalidation Problems

**Symptoms:**
- Stale data persisting after updates
- File changes not reflected in cache
- Inconsistent invalidation behavior

**Diagnostic Steps:**
```python
# Test invalidation
async def test_invalidation():
    from src.services.cache_invalidation_service import CacheInvalidationService

    invalidation_service = CacheInvalidationService()

    # Test file invalidation
    test_file = "/path/to/test/file.py"
    await invalidation_service.invalidate_file_cache(test_file)

    # Check if invalidation worked
    cache_service = await get_cache_service()
    file_cache_key = f"file:{test_file}"
    result = await cache_service.get(file_cache_key)
    print(f"File cache after invalidation: {result}")

asyncio.run(test_invalidation())
```

**Solutions:**

1. **Fix File Monitoring:**
```python
# Ensure file monitoring is active
from src.services.file_monitoring_service import FileMonitoringService

monitor = FileMonitoringService()
await monitor.start_monitoring(["/path/to/project"])
```

2. **Manual Invalidation:**
```python
# Force invalidation
async def force_invalidation():
    from src.services.cache_invalidation_service import CacheInvalidationService

    invalidation_service = CacheInvalidationService()

    # Invalidate specific project
    await invalidation_service.invalidate_project_cache("project_name")

    # Or invalidate all caches
    await invalidation_service.invalidate_all_caches()

asyncio.run(force_invalidation())
```

### 5. Authentication and Security Issues

#### Issue: Encryption/Decryption Errors

**Symptoms:**
- `EncryptionOperationError` exceptions
- Corrupted cache data
- Authentication failures

**Diagnostic Steps:**
```python
# Test encryption
async def test_encryption():
    from src.utils.encryption_utils import AESEncryption, KeyManager

    key_manager = KeyManager()
    encryption = AESEncryption(key_manager)

    # Test basic encryption/decryption
    test_data = "test encryption data"
    encrypted = await encryption.encrypt(test_data)
    decrypted = await encryption.decrypt(encrypted)

    print(f"Original: {test_data}")
    print(f"Decrypted: {decrypted.decode()}")
    print(f"Success: {test_data == decrypted.decode()}")

asyncio.run(test_encryption())
```

**Solutions:**

1. **Fix Key Management:**
```bash
# Check encryption key
echo $CACHE_ENCRYPTION_KEY

# Generate new key if needed
python -c "
import secrets
import base64
key = secrets.token_bytes(32)
print(base64.b64encode(key).decode())
"
```

2. **Reset Encryption:**
```python
# Reset encryption keys
from src.utils.encryption_utils import KeyManager

key_manager = KeyManager()
new_key = key_manager.generate_key()
print(f"New key generated: {new_key.key_id}")
```

3. **Disable Encryption Temporarily:**
```bash
# For debugging only
export CACHE_ENCRYPTION_ENABLED=false
```

#### Issue: Access Control Failures

**Symptoms:**
- Permission denied errors
- Users accessing wrong project data
- Session validation failures

**Diagnostic Steps:**
```python
# Test access controls
async def test_access_control():
    from src.services.project_cache_service import ProjectIsolationManager

    isolation_manager = ProjectIsolationManager(cache_service)

    # Test project key generation
    key1 = isolation_manager.generate_project_key("project1", "test")
    key2 = isolation_manager.generate_project_key("project2", "test")

    print(f"Project 1 key: {key1}")
    print(f"Project 2 key: {key2}")
    print(f"Keys different: {key1 != key2}")

asyncio.run(test_access_control())
```

**Solutions:**

1. **Fix Project Isolation:**
```python
# Enable project isolation
CACHE_PROJECT_ISOLATION=true

# Reset project permissions if needed
from src.services.project_cache_service import ProjectIsolationManager
isolation_manager = ProjectIsolationManager(cache_service)
# Clear and reconfigure permissions
```

2. **Session Management:**
```python
# Clear invalid sessions
from src.services.cache_service import SessionSecurityManager
session_manager = SessionSecurityManager(cache_service, encryption)
session_manager.active_sessions.clear()
```

## Advanced Troubleshooting

### 1. Performance Profiling

#### CPU Profiling

```python
# Profile cache operations
import cProfile
import asyncio
from src.services.cache_service import get_cache_service

async def profile_cache_operations():
    cache_service = await get_cache_service()

    # Perform operations to profile
    for i in range(1000):
        await cache_service.set(f"profile_key_{i}", f"value_{i}")
        await cache_service.get(f"profile_key_{i}")

def run_profiling():
    cProfile.run('asyncio.run(profile_cache_operations())', 'cache_profile.prof')

# Analyze profile
import pstats
stats = pstats.Stats('cache_profile.prof')
stats.sort_stats('cumulative').print_stats(20)
```

#### Memory Profiling

```python
# Memory profiling with memory_profiler
from memory_profiler import profile

@profile
async def memory_intensive_cache_operations():
    cache_service = await get_cache_service()

    # Store large amounts of data
    large_data = "x" * 10000  # 10KB
    for i in range(1000):
        await cache_service.set(f"large_key_{i}", large_data)

# Run with: python -m memory_profiler script.py
```

### 2. Network Analysis

#### Redis Network Monitoring

```bash
# Monitor Redis network traffic
sudo tcpdump -i lo -n port 6379

# Monitor network latency
ping -c 10 localhost

# Check network connections
netstat -an | grep 6379

# Monitor bandwidth usage
iftop -i lo
```

#### Connection Pool Analysis

```python
# Monitor connection pool
async def analyze_connection_pool():
    cache_service = await get_cache_service()

    if hasattr(cache_service, 'redis_manager'):
        health_info = await cache_service.redis_manager.get_health_info()
        pool_stats = health_info.connection_pool_stats

        print(f"Created connections: {pool_stats['created_connections']}")
        print(f"Available connections: {pool_stats['available_connections']}")
        print(f"In-use connections: {pool_stats['in_use_connections']}")
        print(f"Max connections: {pool_stats['max_connections']}")

asyncio.run(analyze_connection_pool())
```

### 3. Data Analysis

#### Cache Data Inspection

```python
# Inspect cache data patterns
async def inspect_cache_data():
    cache_service = await get_cache_service()

    # Get all keys with pattern
    async with cache_service.get_redis_client() as redis:
        keys = []
        async for key in redis.scan_iter(match="codebase_rag:*"):
            keys.append(key.decode())

    # Analyze key patterns
    key_patterns = {}
    for key in keys:
        parts = key.split(":")
        pattern = ":".join(parts[:3]) if len(parts) >= 3 else key
        key_patterns[pattern] = key_patterns.get(pattern, 0) + 1

    print("Cache key patterns:")
    for pattern, count in sorted(key_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")

asyncio.run(inspect_cache_data())
```

#### Cache Size Analysis

```python
# Analyze cache memory usage by type
async def analyze_cache_sizes():
    cache_service = await get_cache_service()

    async with cache_service.get_redis_client() as redis:
        total_memory = 0
        type_memory = {}

        async for key in redis.scan_iter(match="codebase_rag:*"):
            key_str = key.decode()
            memory_usage = await redis.memory_usage(key)
            total_memory += memory_usage

            # Categorize by cache type
            if ":embedding:" in key_str:
                cache_type = "embedding"
            elif ":search:" in key_str:
                cache_type = "search"
            elif ":project:" in key_str:
                cache_type = "project"
            elif ":file:" in key_str:
                cache_type = "file"
            else:
                cache_type = "other"

            type_memory[cache_type] = type_memory.get(cache_type, 0) + memory_usage

    print(f"Total cache memory: {total_memory / 1024 / 1024:.1f} MB")
    print("\nMemory usage by type:")
    for cache_type, memory in sorted(type_memory.items(), key=lambda x: x[1], reverse=True):
        percentage = (memory / total_memory) * 100 if total_memory > 0 else 0
        print(f"  {cache_type}: {memory / 1024 / 1024:.1f} MB ({percentage:.1f}%)")

asyncio.run(analyze_cache_sizes())
```

## Monitoring and Alerting Setup

### 1. Health Check Monitoring

```python
# Automated health monitoring
import asyncio
import logging
from datetime import datetime

class CacheHealthMonitor:
    def __init__(self, cache_service, alert_thresholds):
        self.cache_service = cache_service
        self.alert_thresholds = alert_thresholds
        self.logger = logging.getLogger(__name__)

    async def monitor_health(self, interval=60):
        """Continuous health monitoring."""
        while True:
            try:
                # Collect health metrics
                health_info = await self.cache_service.get_health()
                stats = self.cache_service.get_stats()

                # Check thresholds
                alerts = []

                if stats.hit_rate < self.alert_thresholds['min_hit_rate']:
                    alerts.append(f"Low hit rate: {stats.hit_rate:.1%}")

                if health_info.redis_ping_time and health_info.redis_ping_time > self.alert_thresholds['max_ping_time']:
                    alerts.append(f"High Redis latency: {health_info.redis_ping_time:.1f}ms")

                # Log alerts
                if alerts:
                    self.logger.warning(f"Cache health alerts: {', '.join(alerts)}")
                else:
                    self.logger.info("Cache health check passed")

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)

# Usage
alert_thresholds = {
    'min_hit_rate': 0.6,
    'max_ping_time': 50,  # ms
    'max_memory_usage': 0.8
}

monitor = CacheHealthMonitor(cache_service, alert_thresholds)
asyncio.create_task(monitor.monitor_health())
```

### 2. Performance Metrics Dashboard

```python
# Simple metrics dashboard
from flask import Flask, jsonify
import asyncio

app = Flask(__name__)

@app.route('/metrics/cache')
async def cache_metrics():
    cache_service = await get_cache_service()

    stats = cache_service.get_stats()
    health = await cache_service.get_health()
    tier_stats = cache_service.get_tier_stats()

    return jsonify({
        'timestamp': time.time(),
        'hit_rate': stats.hit_rate,
        'miss_rate': stats.miss_rate,
        'total_operations': stats.total_operations,
        'redis_connected': health.redis_connected,
        'redis_ping_time': health.redis_ping_time,
        'l1_memory_usage': tier_stats['l1_info']['memory_usage_mb'],
        'l1_hit_rate': tier_stats['l1_stats'].hit_rate,
        'l2_hit_rate': tier_stats['l2_stats'].hit_rate if tier_stats['l2_stats'] else 0
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Emergency Procedures

### 1. Cache System Recovery

#### Complete Cache Reset

```bash
#!/bin/bash
# emergency_cache_reset.sh

echo "WARNING: This will delete all cached data!"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo "Stopping cache services..."
    docker-compose -f docker-compose.cache.yml down

    echo "Removing cache data..."
    docker volume rm $(docker volume ls -q | grep cache)

    echo "Starting fresh cache services..."
    docker-compose -f docker-compose.cache.yml up -d

    echo "Cache system reset complete"
else
    echo "Reset cancelled"
fi
```

#### Partial Cache Cleanup

```python
# Selective cache cleanup
async def emergency_cache_cleanup():
    cache_service = await get_cache_service()

    # Clear only specific cache types
    cache_types = ["embedding", "search", "project", "file"]

    for cache_type in cache_types:
        try:
            pattern = f"codebase_rag:{cache_type}:*"

            async with cache_service.get_redis_client() as redis:
                keys_to_delete = []
                async for key in redis.scan_iter(match=pattern):
                    keys_to_delete.append(key)

                if keys_to_delete:
                    await redis.delete(*keys_to_delete)
                    print(f"Deleted {len(keys_to_delete)} {cache_type} cache entries")

        except Exception as e:
            print(f"Error cleaning {cache_type} cache: {e}")

asyncio.run(emergency_cache_cleanup())
```

### 2. Rollback Procedures

#### Configuration Rollback

```bash
# Rollback to previous configuration
cp .env.backup .env
cp docker-compose.cache.yml.backup docker-compose.cache.yml

# Restart with previous config
docker-compose -f docker-compose.cache.yml down
docker-compose -f docker-compose.cache.yml up -d
```

#### Data Recovery

```python
# Recover from cache backup
async def recover_from_backup(backup_file):
    import json

    cache_service = await get_cache_service()

    with open(backup_file, 'r') as f:
        backup_data = json.load(f)

    for key, value in backup_data.items():
        try:
            await cache_service.set(key, value)
        except Exception as e:
            print(f"Error recovering key {key}: {e}")

    print(f"Recovered {len(backup_data)} cache entries")
```

## Support and Documentation

### Getting Help

1. **Check Logs First:**
   - Cache service logs: `docker-compose logs redis-cache`
   - Application logs: Check for cache-related errors
   - System logs: `/var/log/syslog` for system-level issues

2. **Gather Diagnostic Information:**
   ```bash
   # Create diagnostic report
   python -c "
   import asyncio
   from src.services.cache_service import get_cache_service

   async def create_diagnostic_report():
       cache_service = await get_cache_service()
       health = await cache_service.get_health()
       stats = cache_service.get_stats()

       print('=== Cache Diagnostic Report ===')
       print(f'Health Status: {health.status}')
       print(f'Redis Connected: {health.redis_connected}')
       print(f'Hit Rate: {stats.hit_rate:.1%}')
       print(f'Total Operations: {stats.total_operations}')
       print(f'Error Count: {stats.error_count}')

   asyncio.run(create_diagnostic_report())
   "
   ```

3. **Common Resolution Steps:**
   - Restart cache services
   - Check configuration files
   - Verify network connectivity
   - Review recent changes
   - Check system resources

4. **Escalation Path:**
   - Level 1: Basic troubleshooting (this guide)
   - Level 2: Advanced diagnostics and profiling
   - Level 3: Code-level debugging and patches

### Prevention Strategies

1. **Regular Maintenance:**
   - Weekly cache statistics review
   - Monthly performance optimization
   - Quarterly security audits
   - Semi-annual disaster recovery testing

2. **Monitoring Setup:**
   - Real-time health monitoring
   - Performance trend analysis
   - Alert threshold configuration
   - Automated issue detection

3. **Best Practices:**
   - Regular backups
   - Configuration version control
   - Change management procedures
   - Documentation updates
