"""
Unit tests for the cache eviction service.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cache_eviction_service import (
    AdaptiveEvictionPolicy,
    CacheEntry,
    CacheEvictionService,
    EvictionConfig,
    EvictionStrategy,
    EvictionTrigger,
    LFUEvictionPolicy,
    LRUEvictionPolicy,
    MemoryPressureEvictionPolicy,
    TTLEvictionPolicy,
    get_eviction_service,
    shutdown_eviction_service,
)

from ..utils.memory_utils import MemoryPressureLevel


class TestCacheEntry:
    """Test cache entry functionality."""

    def test_cache_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(key="test_key", value="test_value", timestamp=time.time(), last_access=time.time(), size_bytes=1024, ttl=300)

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 1024
        assert entry.ttl == 300
        assert entry.access_count == 0

    def test_cache_entry_touch(self):
        """Test cache entry touch functionality."""
        entry = CacheEntry(key="test_key", value="test_value", timestamp=time.time(), last_access=time.time(), size_bytes=1024)

        initial_access_count = entry.access_count
        initial_last_access = entry.last_access

        time.sleep(0.1)  # Small delay
        entry.touch()

        assert entry.access_count == initial_access_count + 1
        assert entry.last_access > initial_last_access

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Create entry with short TTL
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time() - 10,  # 10 seconds ago
            last_access=time.time() - 10,
            size_bytes=1024,
            ttl=5,  # 5 second TTL
        )

        assert entry.is_expired is True

        # Create entry with long TTL
        entry2 = CacheEntry(
            key="test_key2",
            value="test_value2",
            timestamp=time.time(),
            last_access=time.time(),
            size_bytes=1024,
            ttl=300,  # 5 minute TTL
        )

        assert entry2.is_expired is False

    def test_cache_entry_scores(self):
        """Test cache entry scoring methods."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time() - 60,  # 1 minute ago
            last_access=time.time() - 30,  # 30 seconds ago
            size_bytes=1024,
            access_count=10,
        )

        # Test frequency score
        frequency_score = entry.frequency_score
        assert frequency_score > 0

        # Test recency score
        recency_score = entry.recency_score
        assert recency_score > 0

        # Test age and idle time
        assert entry.age > 0
        assert entry.idle_time > 0


class TestEvictionPolicies:
    """Test different eviction policies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EvictionConfig(batch_size=5)
        self.current_time = time.time()

        # Create test entries
        self.entries = []
        for i in range(10):
            entry = CacheEntry(
                key=f"key_{i}",
                value=f"value_{i}",
                timestamp=self.current_time - (i * 10),  # Different ages
                last_access=self.current_time - (i * 5),  # Different access times
                size_bytes=1024 * (i + 1),  # Different sizes
                access_count=i * 2,  # Different access counts
                ttl=300 if i < 5 else None,  # Some with TTL, some without
            )
            self.entries.append(entry)

    @pytest.mark.asyncio
    async def test_lru_eviction_policy(self):
        """Test LRU eviction policy."""
        policy = LRUEvictionPolicy(self.config)

        candidates = await policy.select_candidates(self.entries, 3)

        assert len(candidates) == 3
        # Should select oldest accessed entries
        assert candidates[0].key == "key_9"  # Oldest access
        assert candidates[1].key == "key_8"
        assert candidates[2].key == "key_7"

    @pytest.mark.asyncio
    async def test_lfu_eviction_policy(self):
        """Test LFU eviction policy."""
        policy = LFUEvictionPolicy(self.config)

        candidates = await policy.select_candidates(self.entries, 3)

        assert len(candidates) == 3
        # Should select least frequently used entries
        assert candidates[0].key == "key_0"  # Lowest frequency
        assert candidates[1].key == "key_1"
        assert candidates[2].key == "key_2"

    @pytest.mark.asyncio
    async def test_ttl_eviction_policy(self):
        """Test TTL eviction policy."""
        policy = TTLEvictionPolicy(self.config)

        # Create some expired entries
        expired_entries = []
        for i in range(3):
            entry = CacheEntry(
                key=f"expired_{i}",
                value=f"value_{i}",
                timestamp=self.current_time - 400,  # 400 seconds ago
                last_access=self.current_time - 400,
                size_bytes=1024,
                ttl=300,  # 5 minute TTL (expired)
            )
            expired_entries.append(entry)

        all_entries = expired_entries + self.entries
        candidates = await policy.select_candidates(all_entries, 3)

        assert len(candidates) == 3
        # Should prioritize expired entries
        assert all(candidate.is_expired for candidate in candidates)

    @pytest.mark.asyncio
    async def test_memory_pressure_eviction_policy(self):
        """Test memory pressure eviction policy."""
        policy = MemoryPressureEvictionPolicy(self.config)

        with patch("cache_eviction_service.get_system_memory_pressure") as mock_pressure:
            mock_pressure.return_value.level = MemoryPressureLevel.CRITICAL

            candidates = await policy.select_candidates(self.entries, 3)

            assert len(candidates) == 3
            # Should select largest entries first during critical pressure
            assert candidates[0].size_bytes >= candidates[1].size_bytes
            assert candidates[1].size_bytes >= candidates[2].size_bytes

    @pytest.mark.asyncio
    async def test_adaptive_eviction_policy(self):
        """Test adaptive eviction policy."""
        policy = AdaptiveEvictionPolicy(self.config)

        # Test adaptation to memory pressure
        with patch("cache_eviction_service.get_system_memory_pressure") as mock_pressure:
            mock_pressure.return_value.level = MemoryPressureLevel.HIGH

            candidates = await policy.select_candidates(self.entries, 3)

            assert len(candidates) == 3
            assert policy.current_strategy == EvictionStrategy.MEMORY_PRESSURE


class TestCacheEvictionService:
    """Test cache eviction service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EvictionConfig(batch_size=5)
        self.service = CacheEvictionService(self.config)
        self.mock_cache = MagicMock()
        self.mock_cache.delete = AsyncMock()

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service initialization."""
        await self.service.initialize()

        assert self.service.ttl_cleanup_task is not None
        assert self.service.memory_monitor_task is not None

        await self.service.shutdown()

    def test_cache_registration(self):
        """Test cache registration."""
        self.service.register_cache("test_cache", self.mock_cache)

        assert "test_cache" in self.service.cache_registry
        assert self.service.cache_registry["test_cache"]() is self.mock_cache

    def test_cache_entry_tracking(self):
        """Test cache entry tracking."""
        cache_name = "test_cache"

        # Track an entry
        self.service.track_cache_entry(cache_name, "test_key", "test_value", 1024, ttl=300)

        assert cache_name in self.service.cache_entries
        assert "test_key" in self.service.cache_entries[cache_name]

        entry = self.service.cache_entries[cache_name]["test_key"]
        assert entry.value == "test_value"
        assert entry.size_bytes == 1024
        assert entry.ttl == 300

    def test_cache_entry_access_update(self):
        """Test cache entry access update."""
        cache_name = "test_cache"

        # Track an entry
        self.service.track_cache_entry(cache_name, "test_key", "test_value", 1024)

        initial_access_count = self.service.cache_entries[cache_name]["test_key"].access_count

        # Update access
        self.service.update_cache_entry_access(cache_name, "test_key")

        updated_access_count = self.service.cache_entries[cache_name]["test_key"].access_count
        assert updated_access_count == initial_access_count + 1

    def test_cache_entry_removal(self):
        """Test cache entry removal."""
        cache_name = "test_cache"

        # Track an entry
        self.service.track_cache_entry(cache_name, "test_key", "test_value", 1024)

        assert "test_key" in self.service.cache_entries[cache_name]

        # Remove entry
        self.service.remove_cache_entry(cache_name, "test_key")

        assert "test_key" not in self.service.cache_entries[cache_name]

    @pytest.mark.asyncio
    async def test_entry_eviction(self):
        """Test entry eviction."""
        cache_name = "test_cache"

        # Register cache
        self.service.register_cache(cache_name, self.mock_cache)

        # Track some entries
        for i in range(5):
            self.service.track_cache_entry(cache_name, f"key_{i}", f"value_{i}", 1024, ttl=300)

        # Evict entries
        evicted_keys = await self.service.evict_entries(cache_name, 3, EvictionStrategy.LRU)

        assert len(evicted_keys) == 3
        assert len(self.service.cache_entries[cache_name]) == 2

        # Verify cache.delete was called
        assert self.mock_cache.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test memory pressure handling."""
        cache_name = "test_cache"

        # Register cache
        self.service.register_cache(cache_name, self.mock_cache)

        # Track some entries
        for i in range(10):
            self.service.track_cache_entry(cache_name, f"key_{i}", f"value_{i}", 1024 * (i + 1))  # Different sizes

        # Handle critical memory pressure
        evicted_count = await self.service.handle_memory_pressure(cache_name, MemoryPressureLevel.CRITICAL)

        assert evicted_count > 0
        assert evicted_count <= self.config.aggressive_batch_size

    @pytest.mark.asyncio
    async def test_expired_entries_cleanup(self):
        """Test expired entries cleanup."""
        cache_name = "test_cache"

        # Register cache
        self.service.register_cache(cache_name, self.mock_cache)

        # Track expired entries
        current_time = time.time()
        for i in range(3):
            entry = CacheEntry(
                key=f"expired_{i}",
                value=f"value_{i}",
                timestamp=current_time - 400,  # 400 seconds ago
                last_access=current_time - 400,
                size_bytes=1024,
                ttl=300,  # 5 minute TTL (expired)
            )
            self.service.cache_entries[cache_name][f"expired_{i}"] = entry

        # Track non-expired entries
        for i in range(2):
            entry = CacheEntry(
                key=f"active_{i}",
                value=f"value_{i}",
                timestamp=current_time,
                last_access=current_time,
                size_bytes=1024,
                ttl=600,  # 10 minute TTL (not expired)
            )
            self.service.cache_entries[cache_name][f"active_{i}"] = entry

        # Cleanup expired entries
        results = await self.service.cleanup_expired_entries(cache_name)

        assert results[cache_name] == 3  # 3 expired entries cleaned
        assert len(self.service.cache_entries[cache_name]) == 2  # 2 active entries remain

    def test_cache_info(self):
        """Test cache info retrieval."""
        cache_name = "test_cache"

        # Track some entries
        for i in range(5):
            self.service.track_cache_entry(cache_name, f"key_{i}", f"value_{i}", 1024 * (i + 1))

        info = self.service.get_cache_info(cache_name)

        assert info["cache_name"] == cache_name
        assert info["entry_count"] == 5
        assert info["total_size_bytes"] == 1024 * (1 + 2 + 3 + 4 + 5)
        assert info["total_size_mb"] > 0

    def test_eviction_stats(self):
        """Test eviction statistics."""
        stats = self.service.get_eviction_stats()

        assert stats.total_evictions == 0
        assert stats.total_size_evicted == 0
        assert len(stats.evictions_by_strategy) == 0
        assert len(stats.evictions_by_trigger) == 0

    def test_performance_metrics(self):
        """Test performance metrics."""
        metrics = self.service.get_performance_metrics()

        assert "avg_eviction_time" in metrics
        assert "max_eviction_time" in metrics
        assert "min_eviction_time" in metrics
        assert "total_evictions" in metrics

    def test_stats_reset(self):
        """Test statistics reset."""
        # Add some fake eviction times
        self.service.eviction_times = [0.1, 0.2, 0.3]

        self.service.reset_stats()

        assert len(self.service.eviction_times) == 0
        assert self.service.stats.total_evictions == 0


class TestGlobalService:
    """Test global service functions."""

    @pytest.mark.asyncio
    async def test_global_service_lifecycle(self):
        """Test global service lifecycle."""
        # Get service instance
        service = await get_eviction_service()

        assert service is not None
        assert isinstance(service, CacheEvictionService)

        # Get same instance
        service2 = await get_eviction_service()
        assert service is service2

        # Shutdown service
        await shutdown_eviction_service()

        # Should create new instance
        service3 = await get_eviction_service()
        assert service3 is not service

        await shutdown_eviction_service()


class TestEvictionConfig:
    """Test eviction configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvictionConfig()

        assert config.primary_strategy == EvictionStrategy.LRU
        assert EvictionStrategy.TTL in config.fallback_strategies
        assert EvictionStrategy.RANDOM in config.fallback_strategies
        assert config.batch_size == 100
        assert config.aggressive_batch_size == 500
        assert config.memory_pressure_threshold == 0.8
        assert config.critical_memory_threshold == 0.9

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvictionConfig(primary_strategy=EvictionStrategy.LFU, batch_size=50, memory_pressure_threshold=0.7)

        assert config.primary_strategy == EvictionStrategy.LFU
        assert config.batch_size == 50
        assert config.memory_pressure_threshold == 0.7


if __name__ == "__main__":
    pytest.main([__file__])
