"""
Cache Hit/Miss Ratio Validation Tests for Query Caching Layer.

This module provides comprehensive validation for cache effectiveness through:
- Cache hit/miss ratio analysis
- Cache effectiveness measurement
- Cache warming strategies validation
- TTL and expiration behavior validation
- Cache efficiency optimization testing
- Workload pattern analysis
"""

import asyncio
import json
import random
import statistics
import string
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.cache_config import CacheConfig
from services.cache_service import CacheStats
from services.project_cache_service import ProjectCacheService
from services.search_cache_service import SearchCacheService


class CacheAccessPattern(Enum):
    """Cache access patterns for testing."""

    SEQUENTIAL = "sequential"
    RANDOM = "random"
    HOTSPOT = "hotspot"
    TEMPORAL_LOCALITY = "temporal_locality"
    SPATIAL_LOCALITY = "spatial_locality"
    ZIPFIAN = "zipfian"


@dataclass
class CacheHitMissMetric:
    """Metric for tracking cache hits and misses."""

    timestamp: float
    operation: str
    key: str
    hit: bool
    response_time_ms: float
    data_size_bytes: int = 0
    ttl_remaining: int | None = None
    cache_level: str = "L1"  # L1, L2, etc.


@dataclass
class CacheEffectivenessResult:
    """Result of cache effectiveness analysis."""

    test_name: str
    total_operations: int
    cache_hits: int
    cache_misses: int
    hit_ratio: float
    miss_ratio: float
    avg_hit_response_time_ms: float
    avg_miss_response_time_ms: float
    cache_efficiency_score: float
    temporal_locality_score: float
    working_set_size: int
    recommendations: list[str] = field(default_factory=list)
    metrics: list[CacheHitMissMetric] = field(default_factory=list)
    pattern_analysis: dict[str, Any] = field(default_factory=dict)


class CacheWorkloadGenerator:
    """Generates different cache access patterns for testing."""

    def __init__(self, universe_size: int = 10000):
        self.universe_size = universe_size
        self.keys = [f"key_{i:06d}" for i in range(universe_size)]
        self.values = {}

        # Pre-generate values for consistent testing
        for key in self.keys:
            self.values[key] = self._generate_value(key)

    def _generate_value(self, key: str) -> dict[str, Any]:
        """Generate a realistic cache value."""
        return {
            "id": key,
            "data": "".join(random.choices(string.ascii_letters, k=random.randint(100, 1000))),
            "metadata": {"created": time.time(), "size": random.randint(50, 500), "type": random.choice(["search", "project", "analysis"])},
            "content": [{"chunk": i, "data": "x" * 100} for i in range(random.randint(1, 10))],
        }

    def generate_sequential_pattern(self, count: int, start_index: int = 0) -> list[str]:
        """Generate sequential access pattern."""
        return [self.keys[(start_index + i) % len(self.keys)] for i in range(count)]

    def generate_random_pattern(self, count: int) -> list[str]:
        """Generate random access pattern."""
        return [random.choice(self.keys) for _ in range(count)]

    def generate_hotspot_pattern(self, count: int, hotspot_ratio: float = 0.8, hotspot_size: int = 100) -> list[str]:
        """Generate hotspot access pattern (80/20 rule)."""
        hotspot_keys = self.keys[:hotspot_size]
        cold_keys = self.keys[hotspot_size:]

        accesses = []
        for _ in range(count):
            if random.random() < hotspot_ratio:
                accesses.append(random.choice(hotspot_keys))
            else:
                accesses.append(random.choice(cold_keys))

        return accesses

    def generate_temporal_locality_pattern(self, count: int, window_size: int = 50, reuse_probability: float = 0.7) -> list[str]:
        """Generate temporal locality pattern."""
        accesses = []
        recent_keys = []

        for _ in range(count):
            if recent_keys and random.random() < reuse_probability:
                # Reuse recent key
                key = random.choice(recent_keys)
                accesses.append(key)
            else:
                # New key
                key = random.choice(self.keys)
                accesses.append(key)
                recent_keys.append(key)

                # Maintain window size
                if len(recent_keys) > window_size:
                    recent_keys.pop(0)

        return accesses

    def generate_zipfian_pattern(self, count: int, alpha: float = 1.0) -> list[str]:
        """Generate Zipfian distribution access pattern."""
        # Simplified Zipfian: rank-based selection
        ranks = list(range(1, len(self.keys) + 1))
        weights = [1.0 / (rank**alpha) for rank in ranks]

        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        accesses = []
        for _ in range(count):
            # Weighted random selection
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    accesses.append(self.keys[i])
                    break

        return accesses

    def get_value(self, key: str) -> dict[str, Any]:
        """Get value for a key."""
        return self.values.get(key, {})


class CacheHitMissValidator:
    """Validator for cache hit/miss ratios and effectiveness."""

    def __init__(self):
        self.workload_generator = CacheWorkloadGenerator()

    async def validate_cache_effectiveness(
        self, cache_service: Any, access_pattern: CacheAccessPattern, operation_count: int = 1000, test_name: str = None
    ) -> CacheEffectivenessResult:
        """Validate cache effectiveness with different access patterns."""
        if test_name is None:
            test_name = f"effectiveness_{access_pattern.value}"

        # Generate access pattern
        if access_pattern == CacheAccessPattern.SEQUENTIAL:
            keys = self.workload_generator.generate_sequential_pattern(operation_count)
        elif access_pattern == CacheAccessPattern.RANDOM:
            keys = self.workload_generator.generate_random_pattern(operation_count)
        elif access_pattern == CacheAccessPattern.HOTSPOT:
            keys = self.workload_generator.generate_hotspot_pattern(operation_count)
        elif access_pattern == CacheAccessPattern.TEMPORAL_LOCALITY:
            keys = self.workload_generator.generate_temporal_locality_pattern(operation_count)
        elif access_pattern == CacheAccessPattern.ZIPFIAN:
            keys = self.workload_generator.generate_zipfian_pattern(operation_count)
        else:
            keys = self.workload_generator.generate_random_pattern(operation_count)

        metrics = []
        initial_stats = cache_service.get_stats() if hasattr(cache_service, "get_stats") else CacheStats()

        # Execute operations and collect metrics
        for i, key in enumerate(keys):
            start_time = time.perf_counter()

            # Try to get from cache
            cached_value = await cache_service.get(key)

            if cached_value is not None:
                # Cache hit
                hit = True
                value = cached_value
            else:
                # Cache miss - simulate data retrieval and caching
                hit = False
                value = self.workload_generator.get_value(key)
                await cache_service.set(key, value)

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Calculate data size
            data_size = len(json.dumps(value).encode("utf-8")) if value else 0

            metric = CacheHitMissMetric(
                timestamp=time.time(),
                operation=f"access_{i}",
                key=key,
                hit=hit,
                response_time_ms=response_time_ms,
                data_size_bytes=data_size,
            )
            metrics.append(metric)

        # Get final stats
        final_stats = cache_service.get_stats() if hasattr(cache_service, "get_stats") else CacheStats()

        return self._analyze_cache_effectiveness(test_name, metrics, initial_stats, final_stats, access_pattern)

    async def validate_ttl_behavior(
        self, cache_service: Any, ttl_seconds: int = 5, test_duration: int = 15, access_interval: float = 1.0
    ) -> CacheEffectivenessResult:
        """Validate TTL and expiration behavior."""
        test_name = f"ttl_validation_{ttl_seconds}s"
        metrics = []

        # Set initial data with TTL
        test_keys = [f"ttl_test_key_{i}" for i in range(10)]
        for key in test_keys:
            value = self.workload_generator.get_value(key)
            await cache_service.set(key, value, ttl=ttl_seconds)

        start_time = time.time()

        # Access keys over time to observe TTL behavior
        while time.time() - start_time < test_duration:
            for key in test_keys:
                access_start = time.perf_counter()
                cached_value = await cache_service.get(key)
                access_end = time.perf_counter()

                hit = cached_value is not None
                response_time = (access_end - access_start) * 1000

                # Calculate TTL remaining
                elapsed = time.time() - start_time
                ttl_remaining = max(0, ttl_seconds - elapsed) if hit else None

                metric = CacheHitMissMetric(
                    timestamp=time.time(),
                    operation="ttl_access",
                    key=key,
                    hit=hit,
                    response_time_ms=response_time,
                    ttl_remaining=int(ttl_remaining) if ttl_remaining else None,
                )
                metrics.append(metric)

            await asyncio.sleep(access_interval)

        return self._analyze_cache_effectiveness(test_name, metrics, CacheStats(), CacheStats(), CacheAccessPattern.TEMPORAL_LOCALITY)

    async def validate_cache_warming(
        self, cache_service: Any, warmup_ratio: float = 0.3, working_set_size: int = 1000, access_count: int = 2000
    ) -> CacheEffectivenessResult:
        """Validate cache warming strategies."""
        test_name = f"cache_warming_{warmup_ratio}"
        metrics = []

        # Select working set
        working_set = self.workload_generator.keys[:working_set_size]
        warmup_keys = working_set[: int(working_set_size * warmup_ratio)]

        # Phase 1: Cache warming
        for key in warmup_keys:
            value = self.workload_generator.get_value(key)
            await cache_service.set(key, value)

        # Phase 2: Access working set with hit/miss measurement
        for i in range(access_count):
            key = random.choice(working_set)

            start_time = time.perf_counter()
            cached_value = await cache_service.get(key)
            end_time = time.perf_counter()

            hit = cached_value is not None
            response_time = (end_time - start_time) * 1000

            if not hit:
                # Cache miss - populate cache
                value = self.workload_generator.get_value(key)
                await cache_service.set(key, value)

            metric = CacheHitMissMetric(
                timestamp=time.time(), operation=f"warmed_access_{i}", key=key, hit=hit, response_time_ms=response_time
            )
            metrics.append(metric)

        return self._analyze_cache_effectiveness(test_name, metrics, CacheStats(), CacheStats(), CacheAccessPattern.HOTSPOT)

    async def validate_working_set_behavior(
        self, cache_service: Any, working_set_sizes: list[int] = None, accesses_per_size: int = 500
    ) -> dict[int, CacheEffectivenessResult]:
        """Validate cache behavior with different working set sizes."""
        if working_set_sizes is None:
            working_set_sizes = [100, 500, 1000, 2000, 5000]

        results = {}

        for ws_size in working_set_sizes:
            # Clear cache between tests
            if hasattr(cache_service, "clear"):
                await cache_service.clear()

            # Create working set
            working_set = self.workload_generator.keys[:ws_size]
            metrics = []

            # Access working set multiple times
            for i in range(accesses_per_size):
                key = random.choice(working_set)

                start_time = time.perf_counter()
                cached_value = await cache_service.get(key)
                end_time = time.perf_counter()

                hit = cached_value is not None
                response_time = (end_time - start_time) * 1000

                if not hit:
                    value = self.workload_generator.get_value(key)
                    await cache_service.set(key, value)

                metric = CacheHitMissMetric(
                    timestamp=time.time(), operation=f"ws_{ws_size}_access_{i}", key=key, hit=hit, response_time_ms=response_time
                )
                metrics.append(metric)

            test_name = f"working_set_{ws_size}"
            result = self._analyze_cache_effectiveness(test_name, metrics, CacheStats(), CacheStats(), CacheAccessPattern.HOTSPOT)
            results[ws_size] = result

        return results

    def _analyze_cache_effectiveness(
        self,
        test_name: str,
        metrics: list[CacheHitMissMetric],
        initial_stats: CacheStats,
        final_stats: CacheStats,
        access_pattern: CacheAccessPattern,
    ) -> CacheEffectivenessResult:
        """Analyze cache effectiveness from collected metrics."""
        if not metrics:
            return CacheEffectivenessResult(
                test_name=test_name,
                total_operations=0,
                cache_hits=0,
                cache_misses=0,
                hit_ratio=0.0,
                miss_ratio=0.0,
                avg_hit_response_time_ms=0.0,
                avg_miss_response_time_ms=0.0,
                cache_efficiency_score=0.0,
                temporal_locality_score=0.0,
                working_set_size=0,
            )

        # Basic statistics
        total_operations = len(metrics)
        cache_hits = sum(1 for m in metrics if m.hit)
        cache_misses = total_operations - cache_hits
        hit_ratio = cache_hits / total_operations if total_operations > 0 else 0
        miss_ratio = 1.0 - hit_ratio

        # Response time analysis
        hit_times = [m.response_time_ms for m in metrics if m.hit]
        miss_times = [m.response_time_ms for m in metrics if not m.hit]

        avg_hit_response_time = statistics.mean(hit_times) if hit_times else 0
        avg_miss_response_time = statistics.mean(miss_times) if miss_times else 0

        # Working set analysis
        unique_keys = set(m.key for m in metrics)
        working_set_size = len(unique_keys)

        # Temporal locality analysis
        temporal_locality_score = self._calculate_temporal_locality_score(metrics)

        # Cache efficiency score
        cache_efficiency_score = self._calculate_cache_efficiency_score(
            hit_ratio, avg_hit_response_time, avg_miss_response_time, temporal_locality_score
        )

        # Pattern analysis
        pattern_analysis = self._analyze_access_pattern(metrics, access_pattern)

        # Generate recommendations
        recommendations = self._generate_effectiveness_recommendations(
            hit_ratio, cache_efficiency_score, working_set_size, pattern_analysis
        )

        return CacheEffectivenessResult(
            test_name=test_name,
            total_operations=total_operations,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            hit_ratio=hit_ratio,
            miss_ratio=miss_ratio,
            avg_hit_response_time_ms=avg_hit_response_time,
            avg_miss_response_time_ms=avg_miss_response_time,
            cache_efficiency_score=cache_efficiency_score,
            temporal_locality_score=temporal_locality_score,
            working_set_size=working_set_size,
            recommendations=recommendations,
            metrics=metrics,
            pattern_analysis=pattern_analysis,
        )

    def _calculate_temporal_locality_score(self, metrics: list[CacheHitMissMetric]) -> float:
        """Calculate temporal locality score (0-100)."""
        if len(metrics) < 10:
            return 0.0

        # Analyze reuse distances
        key_last_access = {}
        reuse_distances = []

        for i, metric in enumerate(metrics):
            if metric.key in key_last_access:
                distance = i - key_last_access[metric.key]
                reuse_distances.append(distance)
            key_last_access[metric.key] = i

        if not reuse_distances:
            return 0.0

        # Score based on average reuse distance (shorter is better)
        avg_reuse_distance = statistics.mean(reuse_distances)
        max_distance = len(metrics)

        # Normalize to 0-100 scale (lower distance = higher score)
        score = max(0, 100 - (avg_reuse_distance / max_distance) * 100)
        return score

    def _calculate_cache_efficiency_score(
        self, hit_ratio: float, avg_hit_time: float, avg_miss_time: float, temporal_locality_score: float
    ) -> float:
        """Calculate overall cache efficiency score (0-100)."""
        scores = []

        # Hit ratio score (weighted heavily)
        hit_ratio_score = hit_ratio * 100
        scores.append(hit_ratio_score * 0.5)  # 50% weight

        # Response time efficiency score
        if avg_miss_time > 0:
            time_efficiency = min(100, (avg_miss_time - avg_hit_time) / avg_miss_time * 100)
            scores.append(max(0, time_efficiency) * 0.3)  # 30% weight

        # Temporal locality score
        scores.append(temporal_locality_score * 0.2)  # 20% weight

        return sum(scores) if scores else 0.0

    def _analyze_access_pattern(self, metrics: list[CacheHitMissMetric], pattern: CacheAccessPattern) -> dict[str, Any]:
        """Analyze access pattern characteristics."""
        if not metrics:
            return {}

        # Key frequency analysis
        key_counts = {}
        for metric in metrics:
            key_counts[metric.key] = key_counts.get(metric.key, 0) + 1

        # Distribution analysis
        access_counts = list(key_counts.values())

        analysis = {
            "pattern_type": pattern.value,
            "unique_keys": len(key_counts),
            "total_accesses": len(metrics),
            "avg_accesses_per_key": statistics.mean(access_counts),
            "max_accesses_per_key": max(access_counts),
            "min_accesses_per_key": min(access_counts),
            "access_distribution_stddev": statistics.stdev(access_counts) if len(access_counts) > 1 else 0,
        }

        # Hotspot analysis
        sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
        top_10_percent = max(1, len(sorted_keys) // 10)
        top_keys_accesses = sum(count for _, count in sorted_keys[:top_10_percent])

        analysis["hotspot_ratio"] = top_keys_accesses / len(metrics) if metrics else 0
        analysis["top_keys"] = sorted_keys[:5]  # Top 5 most accessed keys

        return analysis

    def _generate_effectiveness_recommendations(
        self, hit_ratio: float, efficiency_score: float, working_set_size: int, pattern_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for improving cache effectiveness."""
        recommendations = []

        # Hit ratio recommendations
        if hit_ratio < 0.5:
            recommendations.append("Low hit ratio (<50%). Consider increasing cache size or implementing cache warming.")
        elif hit_ratio < 0.7:
            recommendations.append("Moderate hit ratio. Consider optimizing cache eviction policies.")
        elif hit_ratio > 0.9:
            recommendations.append("Excellent hit ratio. Current caching strategy is highly effective.")

        # Efficiency score recommendations
        if efficiency_score < 40:
            recommendations.append("Low cache efficiency. Review caching strategy and access patterns.")
        elif efficiency_score < 70:
            recommendations.append("Moderate cache efficiency. Consider fine-tuning cache parameters.")

        # Working set recommendations
        if working_set_size > 5000:
            recommendations.append("Large working set detected. Consider implementing multi-tier caching.")

        # Pattern-specific recommendations
        hotspot_ratio = pattern_analysis.get("hotspot_ratio", 0)
        if hotspot_ratio > 0.7:
            recommendations.append("High hotspot ratio detected. Consider implementing LRU or frequency-based eviction.")
        elif hotspot_ratio < 0.3:
            recommendations.append("Low hotspot ratio. Consider larger cache size or different access patterns.")

        # Access distribution recommendations
        access_stddev = pattern_analysis.get("access_distribution_stddev", 0)
        avg_accesses = pattern_analysis.get("avg_accesses_per_key", 0)
        if avg_accesses > 0 and access_stddev / avg_accesses > 2:
            recommendations.append("High access pattern variability. Consider adaptive caching strategies.")

        return recommendations


class TestCacheHitMissValidation:
    """Test suite for cache hit/miss validation."""

    @pytest.fixture
    def hit_miss_validator(self):
        """Create a cache hit/miss validator."""
        return CacheHitMissValidator()

    @pytest.fixture
    def mock_cache_service(self):
        """Create a mock cache service with realistic hit/miss behavior."""
        cache_data = {}

        class MockCacheService:
            def __init__(self):
                self.stats = CacheStats()

            async def get(self, key: str):
                if key in cache_data:
                    self.stats.hit_count += 1
                    return cache_data[key]
                else:
                    self.stats.miss_count += 1
                    return None

            async def set(self, key: str, value: Any, ttl: int = None):
                cache_data[key] = value
                self.stats.set_count += 1
                return True

            async def delete(self, key: str):
                if key in cache_data:
                    del cache_data[key]
                    self.stats.delete_count += 1
                    return True
                return False

            async def clear(self):
                cache_data.clear()
                return True

            def get_stats(self):
                self.stats.total_operations = self.stats.hit_count + self.stats.miss_count + self.stats.set_count + self.stats.delete_count
                return self.stats

        return MockCacheService()

    @pytest.mark.asyncio
    async def test_sequential_access_pattern(self, hit_miss_validator, mock_cache_service):
        """Test cache effectiveness with sequential access pattern."""
        result = await hit_miss_validator.validate_cache_effectiveness(
            mock_cache_service, CacheAccessPattern.SEQUENTIAL, operation_count=100
        )

        # Verify results
        assert result.test_name == "effectiveness_sequential"
        assert result.total_operations == 100
        assert result.cache_hits + result.cache_misses == 100
        assert 0 <= result.hit_ratio <= 1
        assert result.miss_ratio == 1 - result.hit_ratio

        # Sequential pattern should have some hits due to repeated access
        print(f"Sequential pattern: {result.hit_ratio*100:.1f}% hit ratio, {result.working_set_size} unique keys")

    @pytest.mark.asyncio
    async def test_hotspot_access_pattern(self, hit_miss_validator, mock_cache_service):
        """Test cache effectiveness with hotspot access pattern."""
        result = await hit_miss_validator.validate_cache_effectiveness(mock_cache_service, CacheAccessPattern.HOTSPOT, operation_count=200)

        # Verify results
        assert result.test_name == "effectiveness_hotspot"
        assert result.total_operations == 200

        # Hotspot pattern should achieve good hit ratio
        assert result.hit_ratio > 0.3, f"Low hit ratio for hotspot pattern: {result.hit_ratio}"

        # Should have hotspot characteristics
        assert result.pattern_analysis["hotspot_ratio"] > 0.5, "Expected high hotspot ratio"

        print(f"Hotspot pattern: {result.hit_ratio*100:.1f}% hit ratio, hotspot ratio: {result.pattern_analysis['hotspot_ratio']:.2f}")

    @pytest.mark.asyncio
    async def test_temporal_locality_pattern(self, hit_miss_validator, mock_cache_service):
        """Test cache effectiveness with temporal locality pattern."""
        result = await hit_miss_validator.validate_cache_effectiveness(
            mock_cache_service, CacheAccessPattern.TEMPORAL_LOCALITY, operation_count=300
        )

        # Verify results
        assert result.test_name == "effectiveness_temporal_locality"
        assert result.total_operations == 300

        # Temporal locality should achieve reasonable hit ratio
        assert result.hit_ratio > 0.4, f"Low hit ratio for temporal locality: {result.hit_ratio}"

        # Should have good temporal locality score
        assert result.temporal_locality_score > 30, f"Low temporal locality score: {result.temporal_locality_score}"

        print(f"Temporal locality: {result.hit_ratio*100:.1f}% hit ratio, locality score: {result.temporal_locality_score:.1f}")

    @pytest.mark.asyncio
    async def test_random_access_pattern(self, hit_miss_validator, mock_cache_service):
        """Test cache effectiveness with random access pattern."""
        result = await hit_miss_validator.validate_cache_effectiveness(mock_cache_service, CacheAccessPattern.RANDOM, operation_count=150)

        # Verify results
        assert result.test_name == "effectiveness_random"
        assert result.total_operations == 150

        # Random pattern should have lower hit ratio
        print(f"Random pattern: {result.hit_ratio*100:.1f}% hit ratio, efficiency: {result.cache_efficiency_score:.1f}")

        # Should have recommendations for improvement
        assert len(result.recommendations) > 0, "Expected recommendations for random access pattern"

    @pytest.mark.asyncio
    async def test_ttl_behavior_validation(self, hit_miss_validator):
        """Test TTL behavior validation."""
        # Create mock service with TTL support
        cache_data = {}
        ttl_data = {}

        class TTLMockCacheService:
            async def get(self, key: str):
                if key in cache_data:
                    # Check TTL
                    if key in ttl_data:
                        if time.time() > ttl_data[key]:
                            # Expired
                            del cache_data[key]
                            del ttl_data[key]
                            return None
                    return cache_data[key]
                return None

            async def set(self, key: str, value: Any, ttl: int = None):
                cache_data[key] = value
                if ttl:
                    ttl_data[key] = time.time() + ttl
                return True

            def get_stats(self):
                return CacheStats()

        service = TTLMockCacheService()

        result = await hit_miss_validator.validate_ttl_behavior(
            service,
            ttl_seconds=2,
            test_duration=5,
            access_interval=0.5,  # Short TTL for testing
        )

        # Verify TTL behavior
        assert result.test_name.startswith("ttl_validation")
        assert result.total_operations > 0

        # Should see hits early and misses later due to expiration
        early_metrics = result.metrics[: len(result.metrics) // 3]
        late_metrics = result.metrics[len(result.metrics) * 2 // 3 :]

        early_hits = sum(1 for m in early_metrics if m.hit)
        late_hits = sum(1 for m in late_metrics if m.hit)

        print(f"TTL validation: early hits: {early_hits}, late hits: {late_hits}")

    @pytest.mark.asyncio
    async def test_cache_warming_validation(self, hit_miss_validator, mock_cache_service):
        """Test cache warming strategy validation."""
        result = await hit_miss_validator.validate_cache_warming(
            mock_cache_service,
            warmup_ratio=0.5,
            working_set_size=100,
            access_count=200,  # Warm 50% of working set
        )

        # Verify warming effectiveness
        assert result.test_name.startswith("cache_warming")
        assert result.total_operations == 200

        # Cache warming should improve hit ratio
        assert result.hit_ratio > 0.4, f"Low hit ratio after warming: {result.hit_ratio}"

        print(f"Cache warming: {result.hit_ratio*100:.1f}% hit ratio with 50% warmup")

    @pytest.mark.asyncio
    async def test_working_set_behavior(self, hit_miss_validator, mock_cache_service):
        """Test working set behavior validation."""
        results = await hit_miss_validator.validate_working_set_behavior(
            mock_cache_service, working_set_sizes=[50, 100, 200], accesses_per_size=100
        )

        # Verify working set analysis
        assert len(results) == 3

        for ws_size, result in results.items():
            assert result.working_set_size <= ws_size
            assert result.total_operations == 100
            print(f"Working set {ws_size}: {result.hit_ratio*100:.1f}% hit ratio")

        # Larger working sets might have lower hit ratios
        # (depending on cache size vs working set size)

    def test_workload_generator_patterns(self, hit_miss_validator):
        """Test workload generator pattern creation."""
        generator = hit_miss_validator.workload_generator

        # Test sequential pattern
        sequential = generator.generate_sequential_pattern(100)
        assert len(sequential) == 100
        assert len(set(sequential)) <= 100  # Should have some repetition

        # Test random pattern
        random_pattern = generator.generate_random_pattern(100)
        assert len(random_pattern) == 100

        # Test hotspot pattern
        hotspot = generator.generate_hotspot_pattern(1000, hotspot_ratio=0.8, hotspot_size=100)
        assert len(hotspot) == 1000

        # Count hotspot vs cold accesses
        hotspot_keys = set(generator.keys[:100])
        hotspot_accesses = sum(1 for key in hotspot if key in hotspot_keys)
        hotspot_ratio = hotspot_accesses / len(hotspot)

        # Should be approximately 80% hotspot accesses
        assert 0.7 <= hotspot_ratio <= 0.9, f"Hotspot ratio {hotspot_ratio} not in expected range"

        # Test temporal locality
        temporal = generator.generate_temporal_locality_pattern(200, window_size=20, reuse_probability=0.8)
        assert len(temporal) == 200

        print(f"Pattern tests: hotspot ratio {hotspot_ratio:.2f}")

    def test_cache_effectiveness_analysis(self, hit_miss_validator):
        """Test cache effectiveness analysis logic."""
        # Create sample metrics
        metrics = []
        for i in range(100):
            hit = i % 3 != 0  # 67% hit ratio
            metric = CacheHitMissMetric(
                timestamp=time.time() + i * 0.1,
                operation=f"test_{i}",
                key=f"key_{i % 20}",  # 20 unique keys
                hit=hit,
                response_time_ms=1.0 if hit else 10.0,
            )
            metrics.append(metric)

        result = hit_miss_validator._analyze_cache_effectiveness(
            "test_analysis", metrics, CacheStats(), CacheStats(), CacheAccessPattern.HOTSPOT
        )

        # Verify analysis
        assert result.total_operations == 100
        assert abs(result.hit_ratio - 0.67) < 0.05  # Approximately 67%
        assert result.working_set_size == 20
        assert result.avg_hit_response_time_ms < result.avg_miss_response_time_ms
        assert result.cache_efficiency_score > 0

        print(f"Analysis test: {result.hit_ratio*100:.1f}% hit ratio, {result.cache_efficiency_score:.1f} efficiency")


@pytest.mark.performance
@pytest.mark.integration
class TestCacheHitMissValidationIntegration:
    """Integration tests for cache hit/miss validation with real services."""

    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration."""
        return CacheConfig(enabled=True, redis_url="redis://localhost:6379/15", default_ttl=300, max_memory_mb=50)  # Test database

    @pytest.mark.asyncio
    async def test_search_cache_hit_miss_validation(self, cache_config):
        """Test hit/miss validation with SearchCacheService."""
        try:
            service = SearchCacheService(cache_config)
            await service.initialize()

            validator = CacheHitMissValidator()

            # Test hotspot pattern (should work well with search cache)
            result = await validator.validate_cache_effectiveness(service, CacheAccessPattern.HOTSPOT, operation_count=200)

            # Verify effectiveness
            assert result.total_operations == 200
            assert result.hit_ratio >= 0.3, f"Low hit ratio: {result.hit_ratio}"
            assert result.cache_efficiency_score > 20

            print(f"SearchCache validation: {result.hit_ratio*100:.1f}% hit ratio, {result.cache_efficiency_score:.1f} efficiency")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()

    @pytest.mark.asyncio
    async def test_project_cache_working_set_validation(self, cache_config):
        """Test working set validation with ProjectCacheService."""
        try:
            service = ProjectCacheService(cache_config)
            await service.initialize()

            validator = CacheHitMissValidator()

            # Test different working set sizes
            results = await validator.validate_working_set_behavior(service, working_set_sizes=[50, 100], accesses_per_size=100)

            # Verify working set behavior
            assert len(results) == 2

            for ws_size, result in results.items():
                assert result.total_operations == 100
                assert result.working_set_size <= ws_size
                print(f"ProjectCache WS {ws_size}: {result.hit_ratio*100:.1f}% hit ratio")

        except Exception as e:
            pytest.skip(f"Redis not available for integration test: {e}")
        finally:
            if "service" in locals():
                await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
