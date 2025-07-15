# Cache Performance Optimization Guide

## Overview

This guide provides comprehensive strategies for optimizing cache performance in the Query Caching Layer system. It covers benchmarking methodologies, performance tuning techniques, and monitoring strategies to achieve optimal cache efficiency.

## Performance Metrics and KPIs

### Primary Performance Indicators

#### Cache Hit Rate
```python
# Target: > 80% for embedding cache, > 60% for search cache
hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100

# By cache type targets:
embedding_cache_hit_rate = 85%      # High stability expected
search_cache_hit_rate = 65%         # Variable query patterns
project_cache_hit_rate = 90%        # Stable metadata
file_cache_hit_rate = 75%           # Depends on file change frequency
```

#### Response Time Metrics
```python
# L1 Cache (Memory) targets:
l1_avg_response_time = "< 1ms"
l1_p95_response_time = "< 2ms"
l1_p99_response_time = "< 5ms"

# L2 Cache (Redis) targets:
l2_avg_response_time = "< 10ms"
l2_p95_response_time = "< 25ms"
l2_p99_response_time = "< 50ms"

# Network-dependent thresholds:
local_redis_response = "< 5ms"
remote_redis_response = "< 15ms"
```

#### Memory Utilization
```python
# Memory usage targets:
l1_memory_utilization = "< 80%"     # Leave room for spikes
l1_eviction_rate = "< 5%"           # Low eviction indicates good sizing
memory_fragmentation = "< 15%"      # Healthy fragmentation level

# Redis memory targets:
redis_memory_utilization = "< 75%"  # Room for persistence operations
redis_eviction_rate = "< 2%"        # Very low for cache persistence
```

#### Throughput Metrics
```python
# Operations per second targets:
l1_ops_per_second = "> 10,000"      # Memory operations
l2_ops_per_second = "> 1,000"       # Redis operations
batch_operation_efficiency = "> 5x" # Batch vs individual operations

# Concurrent operation handling:
max_concurrent_operations = 100
avg_queue_depth = "< 10"
max_queue_wait_time = "< 100ms"
```

## Benchmarking Strategies

### 1. Comprehensive Cache Benchmarking

```python
class CachePerformanceBenchmark:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.metrics_collector = MetricsCollector()

    async def run_comprehensive_benchmark(self) -> dict:
        """Run comprehensive cache performance benchmark."""

        results = {
            "operation_latency": await self._benchmark_operation_latency(),
            "throughput": await self._benchmark_throughput(),
            "concurrency": await self._benchmark_concurrency(),
            "memory_efficiency": await self._benchmark_memory_efficiency(),
            "cache_effectiveness": await self._benchmark_cache_effectiveness(),
            "scalability": await self._benchmark_scalability()
        }

        return results

    async def _benchmark_operation_latency(self) -> dict:
        """Benchmark individual operation latency."""

        # Generate test data
        test_keys = [f"benchmark:key:{i}" for i in range(1000)]
        test_values = [f"value_{i}" * 100 for i in range(1000)]  # ~600 bytes each

        latency_metrics = {
            "set_operations": [],
            "get_operations": [],
            "delete_operations": [],
            "batch_operations": []
        }

        # Benchmark SET operations
        for key, value in zip(test_keys, test_values):
            start_time = time.perf_counter()
            await self.cache_service.set(key, value)
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
            latency_metrics["set_operations"].append(latency)

        # Benchmark GET operations
        for key in test_keys:
            start_time = time.perf_counter()
            await self.cache_service.get(key)
            latency = (time.perf_counter() - start_time) * 1000
            latency_metrics["get_operations"].append(latency)

        # Benchmark batch operations
        batch_size = 50
        for i in range(0, len(test_keys), batch_size):
            batch_keys = test_keys[i:i + batch_size]

            start_time = time.perf_counter()
            await self.cache_service.get_batch(batch_keys)
            latency = (time.perf_counter() - start_time) * 1000
            latency_metrics["batch_operations"].append(latency / len(batch_keys))

        # Calculate statistics
        return {
            operation: {
                "avg": statistics.mean(latencies),
                "p50": statistics.median(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "min": min(latencies),
                "max": max(latencies)
            }
            for operation, latencies in latency_metrics.items()
        }

    async def _benchmark_throughput(self) -> dict:
        """Benchmark cache throughput under load."""

        duration = 60  # 60 second test
        concurrent_clients = [1, 5, 10, 25, 50, 100]

        throughput_results = {}

        for client_count in concurrent_clients:
            async def client_workload():
                """Individual client workload."""
                operations = 0
                start_time = time.time()

                while time.time() - start_time < duration:
                    # Mix of operations
                    key = f"throughput:key:{random.randint(1, 10000)}"

                    if random.random() < 0.7:  # 70% reads
                        await self.cache_service.get(key)
                    elif random.random() < 0.9:  # 20% writes
                        await self.cache_service.set(key, f"value_{operations}")
                    else:  # 10% deletes
                        await self.cache_service.delete(key)

                    operations += 1

                return operations

            # Run concurrent clients
            tasks = [client_workload() for _ in range(client_count)]
            operation_counts = await asyncio.gather(*tasks)

            total_ops = sum(operation_counts)
            ops_per_second = total_ops / duration

            throughput_results[client_count] = {
                "total_operations": total_ops,
                "ops_per_second": ops_per_second,
                "ops_per_client": total_ops / client_count
            }

        return throughput_results

    async def _benchmark_memory_efficiency(self) -> dict:
        """Benchmark memory usage and efficiency."""

        # Test data with varying sizes
        test_sizes = [100, 1000, 10000, 100000]  # bytes
        entries_per_size = 1000

        memory_metrics = {}

        for size in test_sizes:
            test_value = "x" * size
            initial_memory = await self._get_memory_usage()

            # Store entries
            for i in range(entries_per_size):
                await self.cache_service.set(f"mem_test:{size}:{i}", test_value)

            final_memory = await self._get_memory_usage()

            memory_used = final_memory - initial_memory
            overhead_ratio = memory_used / (size * entries_per_size)

            memory_metrics[size] = {
                "entries": entries_per_size,
                "theoretical_size": size * entries_per_size,
                "actual_memory_used": memory_used,
                "overhead_ratio": overhead_ratio,
                "memory_efficiency": 1 / overhead_ratio
            }

            # Cleanup
            for i in range(entries_per_size):
                await self.cache_service.delete(f"mem_test:{size}:{i}")

        return memory_metrics
```

### 2. Real-World Usage Patterns

```python
class RealWorldBenchmark:
    """Benchmark cache performance with realistic usage patterns."""

    async def simulate_embedding_workload(self, duration: int = 300) -> dict:
        """Simulate realistic embedding cache workload."""

        # Realistic embedding patterns
        patterns = {
            "repeat_queries": 0.4,      # 40% repeated queries
            "similar_queries": 0.3,     # 30% similar queries
            "new_queries": 0.3          # 30% completely new queries
        }

        query_templates = [
            "function implementation of {concept}",
            "how to {action} in {language}",
            "error handling for {operation}",
            "best practices for {topic}",
            "{pattern} design pattern example"
        ]

        concepts = ["authentication", "caching", "database", "networking", "algorithms"]
        actions = ["optimize", "implement", "debug", "test", "deploy"]
        languages = ["python", "javascript", "java", "go", "rust"]

        metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time": 0,
            "embedding_generation_time": 0
        }

        start_time = time.time()
        response_times = []

        while time.time() - start_time < duration:
            # Generate query based on patterns
            rand = random.random()

            if rand < patterns["repeat_queries"]:
                # Use existing query
                query = random.choice(self.previous_queries[-100:]) if hasattr(self, 'previous_queries') else "default query"
            elif rand < patterns["repeat_queries"] + patterns["similar_queries"]:
                # Generate similar query
                template = random.choice(query_templates)
                query = template.format(
                    concept=random.choice(concepts),
                    action=random.choice(actions),
                    language=random.choice(languages),
                    operation=random.choice(concepts),
                    topic=random.choice(concepts),
                    pattern=random.choice(concepts)
                )
            else:
                # Generate new query
                query = f"unique query {random.randint(1000000, 9999999)}"

            # Track queries
            if not hasattr(self, 'previous_queries'):
                self.previous_queries = []
            self.previous_queries.append(query)

            # Simulate embedding request
            start_req = time.perf_counter()
            cache_hit = await self._check_embedding_cache(query)
            request_time = (time.perf_counter() - start_req) * 1000

            metrics["total_requests"] += 1
            response_times.append(request_time)

            if cache_hit:
                metrics["cache_hits"] += 1
            else:
                metrics["cache_misses"] += 1
                # Simulate embedding generation time
                await asyncio.sleep(0.1)  # 100ms embedding generation
                metrics["embedding_generation_time"] += 100

            # Random delay between requests
            await asyncio.sleep(random.uniform(0.01, 0.1))

        metrics["hit_rate"] = metrics["cache_hits"] / metrics["total_requests"]
        metrics["avg_response_time"] = statistics.mean(response_times)
        metrics["p95_response_time"] = np.percentile(response_times, 95)

        return metrics
```

### 3. Load Testing and Stress Testing

```python
class CacheLoadTester:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service

    async def stress_test_cache_limits(self) -> dict:
        """Test cache behavior under extreme load conditions."""

        stress_results = {
            "memory_pressure": await self._test_memory_pressure(),
            "connection_exhaustion": await self._test_connection_limits(),
            "high_concurrency": await self._test_high_concurrency(),
            "data_volume": await self._test_large_data_volume()
        }

        return stress_results

    async def _test_memory_pressure(self) -> dict:
        """Test cache behavior under memory pressure."""

        # Fill cache to capacity
        large_value = "x" * 10000  # 10KB values
        stored_keys = []

        try:
            # Store until memory pressure
            for i in range(100000):  # Try to store 1GB
                key = f"stress:memory:{i}"
                success = await self.cache_service.set(key, large_value)

                if success:
                    stored_keys.append(key)
                else:
                    break

                # Check memory usage every 1000 entries
                if i % 1000 == 0:
                    memory_info = await self._get_memory_info()
                    if memory_info["pressure_level"] == "critical":
                        break

            # Test operations under pressure
            pressure_metrics = await self._measure_operations_under_pressure()

            return {
                "max_entries_stored": len(stored_keys),
                "estimated_memory_used": len(stored_keys) * 10000,
                "pressure_performance": pressure_metrics
            }

        finally:
            # Cleanup
            for key in stored_keys:
                await self.cache_service.delete(key)

    async def _test_high_concurrency(self) -> dict:
        """Test cache performance with high concurrent load."""

        concurrent_levels = [50, 100, 200, 500, 1000]
        concurrency_results = {}

        for concurrency in concurrent_levels:
            async def concurrent_worker(worker_id: int):
                """Worker function for concurrent testing."""
                operations = 0
                errors = 0
                latencies = []

                for i in range(100):  # 100 operations per worker
                    try:
                        start_time = time.perf_counter()

                        # Random operation
                        key = f"concurrent:{worker_id}:{i}"
                        operation = random.choice(["get", "set", "delete"])

                        if operation == "set":
                            await self.cache_service.set(key, f"value_{i}")
                        elif operation == "get":
                            await self.cache_service.get(key)
                        else:
                            await self.cache_service.delete(key)

                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        operations += 1

                    except Exception:
                        errors += 1

                return {
                    "operations": operations,
                    "errors": errors,
                    "avg_latency": statistics.mean(latencies) if latencies else 0,
                    "max_latency": max(latencies) if latencies else 0
                }

            # Run concurrent workers
            start_time = time.time()
            tasks = [concurrent_worker(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time

            # Aggregate results
            total_ops = sum(r["operations"] for r in results if isinstance(r, dict))
            total_errors = sum(r["errors"] for r in results if isinstance(r, dict))
            avg_latencies = [r["avg_latency"] for r in results if isinstance(r, dict)]

            concurrency_results[concurrency] = {
                "total_operations": total_ops,
                "total_errors": total_errors,
                "error_rate": total_errors / (total_ops + total_errors) if total_ops + total_errors > 0 else 0,
                "ops_per_second": total_ops / duration,
                "avg_latency": statistics.mean(avg_latencies) if avg_latencies else 0,
                "duration": duration
            }

        return concurrency_results
```

## Performance Optimization Techniques

### 1. Cache Configuration Optimization

```python
class CacheConfigurationOptimizer:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service

    async def optimize_ttl_settings(self, usage_patterns: dict) -> dict:
        """Optimize TTL settings based on usage patterns."""

        # Analyze access patterns for different cache types
        ttl_recommendations = {}

        for cache_type, pattern_data in usage_patterns.items():
            access_frequency = pattern_data["access_frequency"]
            data_volatility = pattern_data["data_volatility"]
            cost_of_regeneration = pattern_data["regeneration_cost"]

            # Calculate optimal TTL based on multiple factors
            base_ttl = self._calculate_base_ttl(cache_type)

            # Adjust based on access frequency
            frequency_multiplier = min(access_frequency / 10.0, 2.0)

            # Adjust based on data volatility (lower volatility = longer TTL)
            volatility_multiplier = max(1.0 - data_volatility, 0.1)

            # Adjust based on regeneration cost (higher cost = longer TTL)
            cost_multiplier = min(cost_of_regeneration / 100.0, 3.0)

            optimal_ttl = int(base_ttl * frequency_multiplier * volatility_multiplier * cost_multiplier)

            ttl_recommendations[cache_type] = {
                "current_ttl": pattern_data["current_ttl"],
                "recommended_ttl": optimal_ttl,
                "improvement_potential": self._calculate_improvement_potential(
                    pattern_data["current_ttl"], optimal_ttl, pattern_data
                )
            }

        return ttl_recommendations

    async def optimize_memory_allocation(self, memory_usage_data: dict) -> dict:
        """Optimize memory allocation across cache tiers."""

        total_memory = memory_usage_data["total_available_memory"]
        current_allocation = memory_usage_data["current_allocation"]
        access_patterns = memory_usage_data["access_patterns"]

        # Calculate optimal allocation
        l1_optimal = self._calculate_optimal_l1_size(access_patterns, total_memory)
        l2_optimal = total_memory - l1_optimal

        return {
            "current_l1_size": current_allocation["l1_memory"],
            "current_l2_size": current_allocation["l2_memory"],
            "recommended_l1_size": l1_optimal,
            "recommended_l2_size": l2_optimal,
            "expected_improvement": self._estimate_allocation_improvement(
                current_allocation, {"l1": l1_optimal, "l2": l2_optimal}, access_patterns
            )
        }

    def _calculate_optimal_l1_size(self, access_patterns: dict, total_memory: int) -> int:
        """Calculate optimal L1 cache size based on access patterns."""

        # Working set size analysis
        working_set_size = access_patterns["working_set_size"]
        access_skew = access_patterns["access_skew"]  # How skewed the access pattern is

        # Base L1 size on working set + buffer for access skew
        base_l1_size = working_set_size * 1.2  # 20% buffer

        # Adjust for access skew (higher skew benefits from larger L1)
        skew_multiplier = 1.0 + (access_skew * 0.5)
        optimal_l1_size = base_l1_size * skew_multiplier

        # Cap at reasonable percentage of total memory
        max_l1_size = total_memory * 0.3  # Max 30% for L1

        return min(int(optimal_l1_size), max_l1_size)
```

### 2. Cache Warming Strategies

```python
class IntelligentCacheWarming:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.access_predictor = AccessPatternPredictor()

    async def warm_cache_intelligently(self, context: dict) -> dict:
        """Intelligent cache warming based on predictive analytics."""

        warming_strategies = {
            "predictive_warming": await self._predictive_warming(context),
            "pattern_based_warming": await self._pattern_based_warming(context),
            "dependency_warming": await self._dependency_based_warming(context),
            "time_based_warming": await self._time_based_warming(context)
        }

        return warming_strategies

    async def _predictive_warming(self, context: dict) -> dict:
        """Warm cache based on access predictions."""

        # Predict likely cache misses
        predicted_accesses = await self.access_predictor.predict_next_accesses(
            context["recent_activity"],
            context["user_patterns"],
            context["project_context"]
        )

        warmed_keys = []
        warming_time = 0

        for prediction in predicted_accesses:
            if prediction["probability"] > 0.7:  # High confidence threshold
                start_time = time.perf_counter()

                success = await self._warm_cache_key(
                    prediction["key"],
                    prediction["data_generator"]
                )

                warming_time += (time.perf_counter() - start_time) * 1000

                if success:
                    warmed_keys.append(prediction["key"])

        return {
            "keys_warmed": len(warmed_keys),
            "total_warming_time": warming_time,
            "warming_efficiency": len(warmed_keys) / warming_time if warming_time > 0 else 0
        }

    async def _dependency_based_warming(self, context: dict) -> dict:
        """Warm cache based on data dependencies."""

        # Identify dependency chains
        dependency_chains = self._analyze_dependency_chains(context)

        warmed_dependencies = 0

        for chain in dependency_chains:
            # Warm dependencies in order
            for dependency in chain["dependencies"]:
                if await self._should_warm_dependency(dependency, context):
                    await self._warm_cache_key(
                        dependency["key"],
                        dependency["generator"]
                    )
                    warmed_dependencies += 1

        return {
            "dependency_chains_processed": len(dependency_chains),
            "dependencies_warmed": warmed_dependencies
        }
```

### 3. Cache Eviction Optimization

```python
class AdaptiveEvictionManager:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.access_tracker = AccessPatternTracker()

    async def optimize_eviction_policy(self, cache_analytics: dict) -> dict:
        """Optimize cache eviction policy based on access patterns."""

        current_policy = cache_analytics["current_eviction_policy"]
        access_patterns = cache_analytics["access_patterns"]

        # Analyze different eviction policies
        policy_analysis = {
            "LRU": self._analyze_lru_effectiveness(access_patterns),
            "LFU": self._analyze_lfu_effectiveness(access_patterns),
            "ARC": self._analyze_arc_effectiveness(access_patterns),
            "CLOCK": self._analyze_clock_effectiveness(access_patterns)
        }

        # Find optimal policy
        best_policy = max(policy_analysis.items(), key=lambda x: x[1]["effectiveness_score"])

        return {
            "current_policy": current_policy,
            "recommended_policy": best_policy[0],
            "policy_analysis": policy_analysis,
            "expected_improvement": best_policy[1]["effectiveness_score"] - policy_analysis[current_policy]["effectiveness_score"]
        }

    async def implement_adaptive_eviction(self) -> None:
        """Implement adaptive eviction based on real-time patterns."""

        while True:
            # Analyze recent access patterns
            recent_patterns = await self.access_tracker.get_recent_patterns(
                time_window=300  # 5 minutes
            )

            # Determine optimal eviction strategy
            if recent_patterns["access_skew"] > 0.8:
                # High skew - LRU works well
                await self._set_eviction_policy("LRU")
            elif recent_patterns["frequency_importance"] > 0.7:
                # Frequency matters - use LFU
                await self._set_eviction_policy("LFU")
            else:
                # Balanced - use ARC (Adaptive Replacement Cache)
                await self._set_eviction_policy("ARC")

            # Sleep before next evaluation
            await asyncio.sleep(60)  # Check every minute
```

### 4. Network and Serialization Optimization

```python
class NetworkOptimizationManager:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service

    async def optimize_serialization(self, data_samples: list) -> dict:
        """Optimize serialization format based on data characteristics."""

        serialization_formats = ["json", "pickle", "msgpack", "protobuf", "avro"]
        optimization_results = {}

        for format_name in serialization_formats:
            format_metrics = await self._benchmark_serialization_format(
                format_name, data_samples
            )
            optimization_results[format_name] = format_metrics

        # Find optimal format
        optimal_format = min(
            optimization_results.items(),
            key=lambda x: x[1]["total_cost"]  # Combine size + time cost
        )

        return {
            "current_format": "json",  # Default
            "recommended_format": optimal_format[0],
            "format_analysis": optimization_results,
            "expected_savings": self._calculate_format_savings(
                optimization_results["json"], optimal_format[1]
            )
        }

    async def optimize_compression(self, data_samples: list) -> dict:
        """Optimize compression settings for cache data."""

        compression_algorithms = ["none", "gzip", "lz4", "zstd", "brotli"]
        compression_levels = [1, 3, 6, 9]  # Different compression levels

        optimization_results = {}

        for algorithm in compression_algorithms:
            if algorithm == "none":
                metrics = await self._benchmark_no_compression(data_samples)
                optimization_results[algorithm] = metrics
            else:
                for level in compression_levels:
                    config_name = f"{algorithm}_level_{level}"
                    metrics = await self._benchmark_compression(
                        algorithm, level, data_samples
                    )
                    optimization_results[config_name] = metrics

        # Find optimal compression
        optimal_compression = min(
            optimization_results.items(),
            key=lambda x: self._calculate_compression_score(x[1])
        )

        return {
            "compression_analysis": optimization_results,
            "recommended_compression": optimal_compression[0],
            "expected_improvement": optimal_compression[1]
        }

    async def optimize_batch_operations(self, operation_patterns: dict) -> dict:
        """Optimize batch operation sizes and strategies."""

        batch_sizes = [10, 25, 50, 100, 250, 500]
        operation_types = ["get", "set", "delete"]

        batch_optimization = {}

        for op_type in operation_types:
            type_optimization = {}

            for batch_size in batch_sizes:
                metrics = await self._benchmark_batch_size(op_type, batch_size)
                type_optimization[batch_size] = metrics

            # Find optimal batch size for this operation type
            optimal_size = max(
                type_optimization.items(),
                key=lambda x: x[1]["efficiency_score"]
            )

            batch_optimization[op_type] = {
                "optimal_batch_size": optimal_size[0],
                "size_analysis": type_optimization
            }

        return batch_optimization
```

## Monitoring and Alerting

### 1. Performance Monitoring Dashboard

```python
class CachePerformanceMonitor:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.metrics_collector = MetricsCollector()

    async def collect_real_time_metrics(self) -> dict:
        """Collect real-time cache performance metrics."""

        metrics = {
            "timestamp": time.time(),
            "cache_stats": await self._collect_cache_stats(),
            "performance_metrics": await self._collect_performance_metrics(),
            "health_indicators": await self._collect_health_indicators(),
            "resource_utilization": await self._collect_resource_metrics()
        }

        return metrics

    async def _collect_cache_stats(self) -> dict:
        """Collect cache statistics."""

        l1_stats = self.cache_service.l1_cache.get_stats()
        l2_stats = await self.cache_service.l2_cache.get_stats() if self.cache_service.l2_cache else None

        return {
            "l1_cache": {
                "hit_rate": l1_stats.hit_rate,
                "miss_rate": l1_stats.miss_rate,
                "total_operations": l1_stats.total_operations,
                "memory_usage": l1_stats.memory_usage_mb,
                "entry_count": l1_stats.entry_count
            },
            "l2_cache": {
                "hit_rate": l2_stats.hit_rate if l2_stats else 0,
                "miss_rate": l2_stats.miss_rate if l2_stats else 0,
                "total_operations": l2_stats.total_operations if l2_stats else 0,
                "memory_usage": l2_stats.memory_usage_mb if l2_stats else 0,
                "connection_count": l2_stats.connection_count if l2_stats else 0
            },
            "overall": {
                "combined_hit_rate": self._calculate_combined_hit_rate(l1_stats, l2_stats),
                "tier_distribution": self._calculate_tier_distribution(l1_stats, l2_stats)
            }
        }

    async def setup_performance_alerts(self, alert_config: dict) -> None:
        """Setup performance alerting system."""

        alert_rules = [
            {
                "name": "Low Cache Hit Rate",
                "condition": lambda metrics: metrics["cache_stats"]["overall"]["combined_hit_rate"] < 0.6,
                "severity": "warning",
                "description": "Cache hit rate below 60%"
            },
            {
                "name": "High Response Latency",
                "condition": lambda metrics: metrics["performance_metrics"]["avg_response_time"] > 50,
                "severity": "critical",
                "description": "Average response time above 50ms"
            },
            {
                "name": "Memory Pressure",
                "condition": lambda metrics: metrics["resource_utilization"]["memory_usage_percent"] > 85,
                "severity": "warning",
                "description": "Memory usage above 85%"
            },
            {
                "name": "Connection Pool Exhaustion",
                "condition": lambda metrics: metrics["resource_utilization"]["connection_usage_percent"] > 90,
                "severity": "critical",
                "description": "Redis connection pool above 90% capacity"
            }
        ]

        # Setup monitoring loop
        asyncio.create_task(self._alert_monitoring_loop(alert_rules))

    async def _alert_monitoring_loop(self, alert_rules: list) -> None:
        """Monitor metrics and trigger alerts."""

        while True:
            try:
                current_metrics = await self.collect_real_time_metrics()

                for rule in alert_rules:
                    if rule["condition"](current_metrics):
                        await self._trigger_alert(rule, current_metrics)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logging.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(60)  # Longer delay on error
```

### 2. Performance Trend Analysis

```python
class PerformanceTrendAnalyzer:
    def __init__(self, metrics_storage: MetricsStorage):
        self.metrics_storage = metrics_storage

    async def analyze_performance_trends(self, time_range: dict) -> dict:
        """Analyze performance trends over time."""

        metrics_history = await self.metrics_storage.get_metrics_range(
            start_time=time_range["start"],
            end_time=time_range["end"]
        )

        trend_analysis = {
            "hit_rate_trend": self._analyze_hit_rate_trend(metrics_history),
            "latency_trend": self._analyze_latency_trend(metrics_history),
            "memory_usage_trend": self._analyze_memory_trend(metrics_history),
            "error_rate_trend": self._analyze_error_trend(metrics_history),
            "recommendations": self._generate_trend_recommendations(metrics_history)
        }

        return trend_analysis

    def _analyze_hit_rate_trend(self, metrics_history: list) -> dict:
        """Analyze cache hit rate trends."""

        hit_rates = [m["cache_stats"]["overall"]["combined_hit_rate"] for m in metrics_history]
        timestamps = [m["timestamp"] for m in metrics_history]

        # Calculate trend direction and slope
        slope, intercept, r_value = self._calculate_linear_trend(timestamps, hit_rates)

        return {
            "current_hit_rate": hit_rates[-1] if hit_rates else 0,
            "average_hit_rate": statistics.mean(hit_rates) if hit_rates else 0,
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "trend_strength": abs(r_value),
            "slope": slope,
            "volatility": statistics.stdev(hit_rates) if len(hit_rates) > 1 else 0
        }

    def _generate_trend_recommendations(self, metrics_history: list) -> list:
        """Generate recommendations based on trend analysis."""

        recommendations = []

        # Analyze hit rate trends
        hit_rate_trend = self._analyze_hit_rate_trend(metrics_history)
        if hit_rate_trend["trend_direction"] == "decreasing" and hit_rate_trend["current_hit_rate"] < 0.7:
            recommendations.append({
                "type": "hit_rate_improvement",
                "priority": "high",
                "description": "Cache hit rate declining - consider adjusting TTL settings or cache size",
                "actions": [
                    "Increase cache memory allocation",
                    "Optimize TTL settings for frequently accessed data",
                    "Review cache invalidation patterns"
                ]
            })

        # Analyze latency trends
        latency_trend = self._analyze_latency_trend(metrics_history)
        if latency_trend["trend_direction"] == "increasing":
            recommendations.append({
                "type": "latency_optimization",
                "priority": "medium",
                "description": "Response latency increasing - investigate performance bottlenecks",
                "actions": [
                    "Review Redis connection pool settings",
                    "Optimize serialization format",
                    "Consider cache warming strategies"
                ]
            })

        return recommendations
```

## Performance Tuning Checklist

### Infrastructure Level
- [ ] Redis memory allocation optimized
- [ ] Connection pool size tuned for workload
- [ ] Network latency minimized (local Redis preferred)
- [ ] Persistence settings optimized for use case
- [ ] Memory overcommit settings configured

### Cache Configuration Level
- [ ] TTL values optimized per cache type
- [ ] Eviction policies configured based on access patterns
- [ ] Compression enabled for large objects
- [ ] Batch operation sizes optimized
- [ ] Key namespacing implemented efficiently

### Application Level
- [ ] Cache-aside pattern implemented correctly
- [ ] Batch operations used where applicable
- [ ] Async operations utilized throughout
- [ ] Error handling and fallbacks implemented
- [ ] Cache warming strategies in place

### Monitoring Level
- [ ] Performance metrics collection active
- [ ] Alerting thresholds configured
- [ ] Trend analysis automated
- [ ] Regular performance reviews scheduled
- [ ] Capacity planning process established

## Common Performance Anti-patterns

### 1. Cache Stampede
**Problem**: Multiple concurrent requests for same missing data
**Solution**: Implement single-flight pattern or cache locking

### 2. Hot Key Problems
**Problem**: Uneven access distribution causing bottlenecks
**Solution**: Data sharding, local caching, or read replicas

### 3. Memory Thrashing
**Problem**: Constant eviction due to undersized cache
**Solution**: Proper capacity planning and adaptive sizing

### 4. Serialization Overhead
**Problem**: Inefficient serialization formats causing latency
**Solution**: Optimize serialization format and compression

### 5. Network Chatty Operations
**Problem**: Too many individual requests instead of batching
**Solution**: Implement intelligent batching strategies
