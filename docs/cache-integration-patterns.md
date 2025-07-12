# Cache Layer Integration Patterns and Best Practices

## Overview

This document provides comprehensive guidance on integrating the Query Caching Layer with existing services, implementing best practices, and following established patterns for optimal performance and maintainability.

## Integration Architecture Patterns

### 1. Service-Level Cache Integration

#### Pattern: Cache-Aside (Lazy Loading)
```python
# Example: EmbeddingService Integration
class EmbeddingService:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.ollama_client = OllamaClient()

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with cache-aside pattern."""
        results = []
        cache_misses = []
        cache_keys = []

        # 1. Check cache first (Cache-Aside Read)
        for text in texts:
            cache_key = self._generate_cache_key(text)
            cache_keys.append(cache_key)

            cached_embedding = await self.cache_service.get(cache_key)
            if cached_embedding:
                results.append(cached_embedding)
            else:
                cache_misses.append((text, cache_key))
                results.append(None)

        # 2. Generate missing embeddings
        if cache_misses:
            texts_to_generate = [text for text, _ in cache_misses]
            new_embeddings = await self._generate_from_ollama(texts_to_generate)

            # 3. Update cache (Cache-Aside Write)
            cache_updates = {}
            for (text, cache_key), embedding in zip(cache_misses, new_embeddings):
                cache_updates[cache_key] = embedding
                # Update results
                index = cache_keys.index(cache_key)
                results[index] = embedding

            # Batch update cache
            await self.cache_service.set_batch(cache_updates, ttl=7200)

        return results
```

#### Pattern: Write-Through Caching
```python
# Example: Search Results Caching
class SearchCacheService:
    async def cache_search_results(self, query: str, results: list, params: dict):
        """Write-through pattern for search results."""
        cache_key = self._generate_search_key(query, params)

        # Write to both L1 and L2 simultaneously
        cache_entry = {
            "results": results,
            "query": query,
            "params": params,
            "timestamp": time.time(),
            "result_count": len(results)
        }

        # Write-through ensures data consistency
        success = await self.cache_service.set(
            cache_key,
            cache_entry,
            ttl=1800  # 30 minutes for search results
        )

        if success:
            self.metrics.record_cache_write(cache_key, "search")

        return success
```

#### Pattern: Cache-Through with Background Refresh
```python
# Example: Project Metadata Caching
class ProjectCacheService:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.background_tasks = set()

    async def get_project_info(self, project_name: str) -> dict:
        """Cache-through pattern with background refresh."""
        cache_key = f"project:{project_name}:info"

        # Try cache first
        cached_info = await self.cache_service.get(cache_key)

        if cached_info:
            # Check if refresh is needed (TTL < 25% remaining)
            if self._needs_background_refresh(cached_info):
                # Schedule background refresh
                task = asyncio.create_task(
                    self._refresh_project_info(project_name, cache_key)
                )
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

            return cached_info["data"]

        # Cache miss - fetch and cache
        project_info = await self._fetch_project_info(project_name)
        await self._cache_project_info(cache_key, project_info)

        return project_info
```

### 2. Multi-Tier Cache Patterns

#### Pattern: Cache Promotion/Demotion
```python
class MultiTierCacheManager:
    async def promote_to_l1(self, key: str, access_count_threshold: int = 3):
        """Promote frequently accessed items to L1 cache."""
        # Check access pattern
        access_count = await self._get_access_count(key)

        if access_count >= access_count_threshold:
            # Get from L2 and promote to L1
            value = await self.l2_cache.get(key)
            if value:
                await self.l1_cache.set(key, value, ttl=self.l1_ttl)
                self.metrics.record_cache_promotion(key)

    async def demote_from_l1(self, key: str):
        """Demote items from L1 during memory pressure."""
        value = await self.l1_cache.get(key)
        if value:
            # Ensure it's in L2 before removing from L1
            await self.l2_cache.set(key, value)
            await self.l1_cache.delete(key)
            self.metrics.record_cache_demotion(key)
```

#### Pattern: Cache Coherency Management
```python
class CacheCoherencyManager:
    async def invalidate_coherent_data(self, file_path: str):
        """Maintain cache coherency across tiers."""
        # Generate all related cache keys
        related_keys = await self._get_related_cache_keys(file_path)

        # Invalidate from both tiers
        for key in related_keys:
            await self.l1_cache.delete(key)
            await self.l2_cache.delete(key)

        # Mark dependent caches for cascade invalidation
        await self._trigger_cascade_invalidation(file_path, related_keys)
```

### 3. Cache Invalidation Patterns

#### Pattern: Event-Driven Invalidation
```python
class FileSystemEventHandler:
    def __init__(self, cache_invalidation_service):
        self.invalidation_service = cache_invalidation_service
        self.event_queue = asyncio.Queue()

    async def handle_file_change(self, event: FileSystemEvent):
        """Event-driven cache invalidation."""
        if event.event_type in ["modified", "deleted"]:
            # Queue invalidation event
            await self.event_queue.put({
                "type": "file_change",
                "file_path": event.src_path,
                "event_type": event.event_type,
                "timestamp": time.time()
            })

    async def process_invalidation_events(self):
        """Process invalidation events asynchronously."""
        while True:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
                await self.invalidation_service.invalidate_file_cache(
                    event["file_path"]
                )
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
```

#### Pattern: Cascade Invalidation
```python
class CascadeInvalidationService:
    def __init__(self):
        self.dependency_graph = DependencyGraph()

    async def register_dependency(self, parent_key: str, child_key: str):
        """Register cache dependencies."""
        self.dependency_graph.add_edge(parent_key, child_key)

    async def cascade_invalidate(self, root_key: str):
        """Perform cascade invalidation following dependency graph."""
        # Get all dependent keys
        dependent_keys = self.dependency_graph.get_descendants(root_key)

        # Invalidate in dependency order
        for key in self.dependency_graph.topological_sort(dependent_keys):
            await self._invalidate_cache_key(key)
            self.metrics.record_cascade_invalidation(root_key, key)
```

## Service Integration Best Practices

### 1. EmbeddingService Integration

```python
class OptimizedEmbeddingService:
    def __init__(self, cache_service: EmbeddingCacheService):
        self.cache_service = cache_service
        self.batch_size = 10

    async def generate_embeddings_with_batching(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Optimized batch embedding generation with caching."""

        # 1. Batch cache lookup
        cache_results = await self._batch_cache_lookup(texts)

        # 2. Identify cache misses
        misses = [
            (i, text) for i, (text, cached) in enumerate(
                zip(texts, cache_results)
            ) if cached is None
        ]

        if not misses:
            return cache_results

        # 3. Batch generate missing embeddings
        missing_texts = [text for _, text in misses]
        new_embeddings = await self._batch_generate_embeddings(missing_texts)

        # 4. Batch cache update
        cache_updates = {}
        for (index, text), embedding in zip(misses, new_embeddings):
            cache_key = self._generate_cache_key(text)
            cache_updates[cache_key] = embedding
            cache_results[index] = embedding

        await self.cache_service.set_batch(cache_updates, ttl=7200)

        return cache_results

    async def _batch_cache_lookup(self, texts: list[str]) -> list[list[float] | None]:
        """Perform batch cache lookup."""
        cache_keys = [self._generate_cache_key(text) for text in texts]
        cache_results = await self.cache_service.get_batch(cache_keys)

        return [
            cache_results.get(key) for key in cache_keys
        ]
```

### 2. SearchService Integration

```python
class CachedSearchService:
    def __init__(self, search_cache_service: SearchCacheService):
        self.cache_service = search_cache_service

    async def search_with_cache(
        self,
        query: str,
        params: dict,
        cache_strategy: str = "aggressive"
    ) -> dict:
        """Cached search with configurable strategies."""

        # Generate cache key based on query and parameters
        cache_key = self._generate_search_cache_key(query, params)

        # Try cache first
        if cache_strategy in ["aggressive", "normal"]:
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                # Validate cache freshness
                if self._is_cache_fresh(cached_result, cache_strategy):
                    self.metrics.record_cache_hit("search", cache_key)
                    return cached_result["results"]

        # Perform actual search
        search_results = await self._perform_vector_search(query, params)

        # Cache results based on strategy
        if cache_strategy != "no_cache":
            ttl = self._get_cache_ttl(cache_strategy, search_results)
            await self.cache_service.set(
                cache_key,
                {
                    "results": search_results,
                    "query": query,
                    "params": params,
                    "timestamp": time.time(),
                    "strategy": cache_strategy
                },
                ttl=ttl
            )

        return search_results

    def _get_cache_ttl(self, strategy: str, results: dict) -> int:
        """Dynamic TTL based on strategy and result quality."""
        base_ttl = 1800  # 30 minutes

        if strategy == "conservative":
            return base_ttl // 2
        elif strategy == "aggressive":
            return base_ttl * 2
        elif len(results.get("results", [])) > 10:
            # More results = higher confidence = longer cache
            return base_ttl * 1.5

        return base_ttl
```

### 3. Project Analysis Integration

```python
class CachedProjectAnalysisService:
    def __init__(self, project_cache_service: ProjectCacheService):
        self.cache_service = project_cache_service

    async def analyze_project_with_cache(self, project_path: str) -> dict:
        """Project analysis with intelligent caching."""

        # Check for cached analysis
        cache_key = f"project_analysis:{self._hash_path(project_path)}"

        cached_analysis = await self.cache_service.get(cache_key)
        if cached_analysis:
            # Verify cache is still valid
            if await self._is_project_unchanged(
                project_path, cached_analysis["metadata"]
            ):
                return cached_analysis["analysis"]

        # Perform analysis
        analysis_result = await self._perform_project_analysis(project_path)

        # Cache with project metadata for validation
        cache_entry = {
            "analysis": analysis_result,
            "metadata": {
                "project_path": project_path,
                "last_modified": await self._get_project_last_modified(project_path),
                "file_count": analysis_result.get("file_count", 0),
                "analysis_timestamp": time.time()
            }
        }

        await self.cache_service.set(cache_key, cache_entry, ttl=3600)

        return analysis_result
```

## Performance Optimization Patterns

### 1. Batch Operations Pattern

```python
class BatchCacheOperations:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.batch_size = 100

    async def batch_process_with_cache(self, items: list[Any]) -> list[Any]:
        """Process items in batches with optimized caching."""

        results = []

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]

            # Batch cache lookup
            cache_keys = [self._generate_key(item) for item in batch]
            cached_results = await self.cache_service.get_batch(cache_keys)

            # Identify items needing processing
            to_process = []
            batch_results = []

            for item, key in zip(batch, cache_keys):
                if key in cached_results:
                    batch_results.append(cached_results[key])
                else:
                    to_process.append((item, key))
                    batch_results.append(None)

            # Process missing items
            if to_process:
                processed = await self._batch_process_items(
                    [item for item, _ in to_process]
                )

                # Update cache and results
                cache_updates = {}
                processed_index = 0

                for i, result in enumerate(batch_results):
                    if result is None:
                        item, key = to_process[processed_index]
                        processed_result = processed[processed_index]
                        cache_updates[key] = processed_result
                        batch_results[i] = processed_result
                        processed_index += 1

                # Batch cache update
                await self.cache_service.set_batch(cache_updates)

            results.extend(batch_results)

        return results
```

### 2. Memory-Aware Caching Pattern

```python
class MemoryAwareCacheManager:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.memory_monitor = MemoryMonitor()

    async def adaptive_cache_operation(self, key: str, value: Any) -> bool:
        """Adaptive caching based on memory pressure."""

        # Check memory pressure
        memory_info = await self.memory_monitor.get_memory_info()

        if memory_info["pressure_level"] == "high":
            # Under memory pressure - use more aggressive eviction
            if self._is_large_object(value):
                # Skip caching large objects
                return False

            # Force L1 cleanup
            await self.cache_service.l1_cache.cleanup_expired()

            # Use shorter TTL
            ttl = self._calculate_pressure_ttl(memory_info)
            return await self.cache_service.set(key, value, ttl=ttl)

        elif memory_info["pressure_level"] == "low":
            # Low pressure - more aggressive caching
            extended_ttl = self._calculate_extended_ttl(value)
            return await self.cache_service.set(key, value, ttl=extended_ttl)

        # Normal operation
        return await self.cache_service.set(key, value)
```

### 3. Cache Warming Pattern

```python
class CacheWarmingService:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service
        self.warming_strategies = {}

    async def warm_cache_predictively(self, context: dict):
        """Predictive cache warming based on usage patterns."""

        # Analyze recent access patterns
        access_patterns = await self._analyze_access_patterns()

        # Predict likely cache misses
        predicted_keys = self._predict_likely_accesses(
            context, access_patterns
        )

        # Warm cache with predicted data
        for key, probability in predicted_keys:
            if probability > 0.7:  # High confidence threshold
                await self._warm_cache_key(key)

    async def warm_project_cache(self, project_name: str):
        """Warm cache for a specific project."""

        warming_tasks = [
            self._warm_project_metadata(project_name),
            self._warm_frequent_searches(project_name),
            self._warm_file_parsing_cache(project_name),
        ]

        await asyncio.gather(*warming_tasks, return_exceptions=True)

    async def _warm_cache_key(self, key: str):
        """Warm a specific cache key."""
        # Check if already cached
        if await self.cache_service.exists(key):
            return

        # Generate data for cache key
        try:
            data = await self._generate_cache_data(key)
            await self.cache_service.set(key, data)
            self.metrics.record_cache_warming(key)
        except Exception as e:
            self.logger.warning(f"Failed to warm cache key {key}: {e}")
```

## Error Handling and Resilience Patterns

### 1. Circuit Breaker Pattern

```python
class CacheCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call_with_circuit_breaker(self, cache_operation):
        """Execute cache operation with circuit breaker protection."""

        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                # Circuit is open - fail fast
                raise CacheUnavailableError("Cache circuit breaker is OPEN")

        try:
            result = await cache_operation()

            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

            raise e
```

### 2. Graceful Degradation Pattern

```python
class GracefulCacheService:
    def __init__(self, primary_cache: BaseCacheService):
        self.primary_cache = primary_cache
        self.fallback_cache = LocalMemoryCache()
        self.circuit_breaker = CacheCircuitBreaker()

    async def get_with_fallback(self, key: str) -> Any:
        """Get operation with graceful degradation."""

        try:
            # Try primary cache with circuit breaker
            return await self.circuit_breaker.call_with_circuit_breaker(
                lambda: self.primary_cache.get(key)
            )
        except CacheUnavailableError:
            # Primary cache unavailable - try fallback
            try:
                return await self.fallback_cache.get(key)
            except Exception:
                # Both caches failed - return None gracefully
                self.logger.warning(f"All cache layers failed for key: {key}")
                return None

    async def set_with_fallback(self, key: str, value: Any) -> bool:
        """Set operation with graceful degradation."""

        primary_success = False
        fallback_success = False

        # Try primary cache
        try:
            primary_success = await self.circuit_breaker.call_with_circuit_breaker(
                lambda: self.primary_cache.set(key, value)
            )
        except Exception as e:
            self.logger.warning(f"Primary cache set failed: {e}")

        # Always try fallback cache
        try:
            fallback_success = await self.fallback_cache.set(key, value)
        except Exception as e:
            self.logger.warning(f"Fallback cache set failed: {e}")

        return primary_success or fallback_success
```

## Testing and Validation Patterns

### 1. Cache Testing Pattern

```python
class CacheTestingUtilities:
    @staticmethod
    async def verify_cache_consistency(
        cache_service: BaseCacheService,
        test_data: dict
    ) -> bool:
        """Verify cache consistency across operations."""

        # Set test data
        for key, value in test_data.items():
            await cache_service.set(key, value)

        # Verify all data is retrievable
        for key, expected_value in test_data.items():
            cached_value = await cache_service.get(key)
            if cached_value != expected_value:
                return False

        # Test batch operations
        batch_results = await cache_service.get_batch(list(test_data.keys()))
        for key, expected_value in test_data.items():
            if batch_results.get(key) != expected_value:
                return False

        return True

    @staticmethod
    async def test_cache_invalidation(
        cache_service: BaseCacheService,
        invalidation_service: CacheInvalidationService
    ):
        """Test cache invalidation scenarios."""

        # Setup test data
        test_key = "test:invalidation:key"
        test_value = {"data": "test_value"}

        await cache_service.set(test_key, test_value)

        # Verify data is cached
        assert await cache_service.get(test_key) == test_value

        # Trigger invalidation
        await invalidation_service.invalidate_key(test_key)

        # Verify data is invalidated
        assert await cache_service.get(test_key) is None
```

### 2. Performance Testing Pattern

```python
class CachePerformanceTester:
    def __init__(self, cache_service: BaseCacheService):
        self.cache_service = cache_service

    async def benchmark_cache_operations(
        self,
        operation_count: int = 1000
    ) -> dict:
        """Benchmark cache operation performance."""

        metrics = {
            "set_operations": [],
            "get_operations": [],
            "batch_operations": [],
            "hit_rates": []
        }

        # Generate test data
        test_data = {
            f"test:key:{i}": f"test_value_{i}"
            for i in range(operation_count)
        }

        # Benchmark SET operations
        start_time = time.time()
        for key, value in test_data.items():
            await self.cache_service.set(key, value)
        set_duration = time.time() - start_time
        metrics["set_operations"].append(set_duration / operation_count)

        # Benchmark GET operations
        start_time = time.time()
        for key in test_data.keys():
            await self.cache_service.get(key)
        get_duration = time.time() - start_time
        metrics["get_operations"].append(get_duration / operation_count)

        # Benchmark BATCH operations
        keys = list(test_data.keys())
        batch_size = 100
        start_time = time.time()

        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            await self.cache_service.get_batch(batch_keys)

        batch_duration = time.time() - start_time
        metrics["batch_operations"].append(batch_duration / (operation_count / batch_size))

        return metrics
```

## Integration Checklist

### Pre-Integration
- [ ] Verify cache service configuration
- [ ] Test Redis connectivity and persistence
- [ ] Validate memory cache configuration
- [ ] Review security and encryption settings
- [ ] Plan cache key namespace strategy

### During Integration
- [ ] Implement cache-aside pattern for reads
- [ ] Add appropriate write strategy (write-through/write-back)
- [ ] Implement error handling and fallback logic
- [ ] Add cache invalidation triggers
- [ ] Include performance monitoring

### Post-Integration
- [ ] Validate cache hit/miss rates
- [ ] Monitor memory usage and performance
- [ ] Test cache invalidation scenarios
- [ ] Verify data consistency across cache tiers
- [ ] Review and optimize cache TTL settings

### Monitoring and Maintenance
- [ ] Set up cache performance dashboards
- [ ] Configure cache health alerts
- [ ] Implement regular cache maintenance procedures
- [ ] Monitor cache growth and eviction patterns
- [ ] Review and adjust cache configuration periodically

## Common Integration Pitfalls

### 1. Cache Key Collision
**Problem**: Different data types using similar keys
**Solution**: Use hierarchical namespacing with type prefixes

### 2. Cache Stampede
**Problem**: Multiple concurrent requests for same missing data
**Solution**: Implement cache locking or single-flight pattern

### 3. Stale Data Issues
**Problem**: Cached data becomes outdated
**Solution**: Proper TTL settings and active invalidation

### 4. Memory Leaks
**Problem**: Unbounded cache growth
**Solution**: Proper eviction policies and memory monitoring

### 5. Cache Consistency Issues
**Problem**: Data inconsistency between cache tiers
**Solution**: Proper write strategies and consistency checks
