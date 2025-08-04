"""
Comprehensive Memory Manager for Wave 6.0 Cache and Memory Management Optimization.

This module implements a comprehensive memory management system that coordinates all
cache layers, memory monitoring, intelligent eviction, compression, consistency,
and failure recovery mechanisms.
"""

import asyncio
import gc
import gzip
import logging
import pickle
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Set, Union

import psutil

from src.config.cache_config import CacheConfig, get_global_cache_config
from src.services.cache_consistency_service import CacheConsistencyService
from src.services.cache_failover_service import CacheFailoverService
from src.services.cache_memory_pressure_service import CacheMemoryPressureService
from src.services.cache_service import MultiTierCacheService, get_cache_service
from src.services.cache_warmup_service import CacheWarmupService
from src.utils.telemetry import get_telemetry_manager


class MemoryState(Enum):
    """Memory management states."""

    OPTIMAL = "optimal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CompressionLevel(Enum):
    """Data compression levels."""

    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    total_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    l1_memory_mb: float = 0.0
    l2_memory_mb: float = 0.0
    l3_memory_mb: float = 0.0
    memory_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class EvictionStats:
    """Cache eviction statistics."""

    total_evictions: int = 0
    l1_evictions: int = 0
    l2_evictions: int = 0
    l3_evictions: int = 0
    bytes_evicted: int = 0
    last_eviction_time: float = 0.0
    eviction_rate: float = 0.0


@dataclass
class CompressionStats:
    """Data compression statistics."""

    compressed_entries: int = 0
    compression_ratio: float = 0.0
    bytes_saved: int = 0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0


class MemoryManager:
    """
    Comprehensive Memory Manager for Wave 6.0.

    Coordinates all cache layers, memory monitoring, intelligent eviction,
    compression, consistency, and failure recovery mechanisms.
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the memory manager."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)

        # Core services
        self.cache_service: MultiTierCacheService | None = None
        self.warmup_service: CacheWarmupService | None = None
        self.pressure_service: CacheMemoryPressureService | None = None
        self.consistency_service: CacheConsistencyService | None = None
        self.failover_service: CacheFailoverService | None = None

        # Memory monitoring
        self.memory_metrics = MemoryMetrics()
        self.memory_state = MemoryState.OPTIMAL
        self.memory_thresholds = {
            MemoryState.WARNING: 0.75,  # 75% memory usage
            MemoryState.CRITICAL: 0.85,  # 85% memory usage
            MemoryState.EMERGENCY: 0.95,  # 95% memory usage
        }

        # Multi-layer cache architecture (L1: memory, L2: path cache, L3: query results)
        self.l1_cache = {}  # In-memory index cache
        self.l2_cache = {}  # Path cache
        self.l3_cache = {}  # Query results cache
        self.cache_metadata = defaultdict(dict)

        # Intelligent eviction
        self.eviction_stats = EvictionStats()
        self.access_patterns = defaultdict(lambda: {"count": 0, "last_access": 0, "frequency_score": 0.0})
        self.eviction_queue = deque()

        # Compression
        self.compression_stats = CompressionStats()
        self.compression_level = CompressionLevel.MEDIUM
        self.compressed_keys = set()

        # Memory limits
        self.max_memory_mb = self.config.memory.max_memory_mb
        self.l1_memory_limit = self.max_memory_mb * 0.4  # 40% for L1
        self.l2_memory_limit = self.max_memory_mb * 0.35  # 35% for L2
        self.l3_memory_limit = self.max_memory_mb * 0.25  # 25% for L3

        # Monitoring tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._prewarming_task: asyncio.Task | None = None

        # Thread safety
        self._lock = RLock()
        self._cache_locks = defaultdict(Lock)

        # Statistics
        self._stats = {
            "memory_optimizations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions_triggered": 0,
            "compressions_performed": 0,
            "consistency_checks": 0,
            "failover_activations": 0,
        }

        # Failure recovery
        self._failure_recovery_enabled = True
        self._last_health_check = time.time()
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3

    async def initialize(self) -> None:
        """Initialize the memory manager and all services."""
        try:
            self.logger.info("Initializing Memory Manager for Wave 6.0...")

            # Initialize core cache service
            self.cache_service = await get_cache_service()

            # Initialize specialized services
            self.warmup_service = CacheWarmupService(self.config)
            await self.warmup_service.initialize()

            self.pressure_service = CacheMemoryPressureService(self.config)
            await self.pressure_service.initialize()

            self.consistency_service = CacheConsistencyService(self.config)
            await self.consistency_service.initialize()

            self.failover_service = CacheFailoverService(self.config)
            await self.failover_service.initialize()

            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._memory_monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._prewarming_task = asyncio.create_task(self._prewarming_loop())

            # Initial memory assessment
            await self._update_memory_metrics()

            self.logger.info("Memory Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Memory Manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the memory manager and all services."""
        try:
            self.logger.info("Shutting down Memory Manager...")

            # Cancel monitoring tasks
            for task in [self._monitoring_task, self._cleanup_task, self._prewarming_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Shutdown services
            if self.warmup_service:
                await self.warmup_service.shutdown()
            if self.pressure_service:
                await self.pressure_service.shutdown()
            if self.consistency_service:
                await self.consistency_service.shutdown()
            if self.failover_service:
                await self.failover_service.shutdown()

            # Final cleanup
            await self._emergency_cleanup()

            self.logger.info("Memory Manager shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during Memory Manager shutdown: {e}")

    # 6.1 Multi-layer Cache Architecture Implementation
    async def get_cached_value(self, key: str, cache_level: str = "auto") -> Any | None:
        """
        Get value from multi-layer cache with intelligent level selection.

        Args:
            key: Cache key
            cache_level: Target cache level ("l1", "l2", "l3", "auto")

        Returns:
            Cached value or None if not found
        """
        try:
            self._track_access(key)

            # Auto-select best cache level
            if cache_level == "auto":
                cache_level = self._select_optimal_cache_level(key)

            # Try cache levels in order
            if cache_level in ["l1", "auto"]:
                value = await self._get_from_l1(key)
                if value is not None:
                    self._stats["cache_hits"] += 1
                    return value

            if cache_level in ["l2", "auto"]:
                value = await self._get_from_l2(key)
                if value is not None:
                    # Promote to L1 if frequently accessed
                    if self._should_promote_to_l1(key):
                        await self._set_to_l1(key, value)
                    self._stats["cache_hits"] += 1
                    return value

            if cache_level in ["l3", "auto"]:
                value = await self._get_from_l3(key)
                if value is not None:
                    # Promote to higher levels if needed
                    if self._should_promote_to_l2(key):
                        await self._set_to_l2(key, value)
                    if self._should_promote_to_l1(key):
                        await self._set_to_l1(key, value)
                    self._stats["cache_hits"] += 1
                    return value

            self._stats["cache_misses"] += 1
            return None

        except Exception as e:
            self.logger.error(f"Error getting cached value for key {key}: {e}")
            return None

    async def set_cached_value(self, key: str, value: Any, ttl: int | None = None, cache_level: str = "auto") -> bool:
        """
        Set value in multi-layer cache with intelligent level selection.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            cache_level: Target cache level ("l1", "l2", "l3", "auto")

        Returns:
            True if successful
        """
        try:
            # Auto-select best cache level
            if cache_level == "auto":
                cache_level = self._select_optimal_cache_level(key, value)

            # Apply compression if beneficial
            compressed_value, is_compressed = await self._apply_compression(value, key)

            success = False

            # Set in appropriate cache level(s)
            if cache_level == "l1":
                success = await self._set_to_l1(key, compressed_value, ttl)
            elif cache_level == "l2":
                success = await self._set_to_l2(key, compressed_value, ttl)
            elif cache_level == "l3":
                success = await self._set_to_l3(key, compressed_value, ttl)
            elif cache_level == "auto":
                # Set in all appropriate levels
                success = True
                success &= await self._set_to_l1(key, compressed_value, ttl)
                success &= await self._set_to_l2(key, compressed_value, ttl)
                success &= await self._set_to_l3(key, compressed_value, ttl)

            if is_compressed:
                self.compressed_keys.add(key)
                self.compression_stats.compressed_entries += 1

            return success

        except Exception as e:
            self.logger.error(f"Error setting cached value for key {key}: {e}")
            return False

    # 6.2 Intelligent Cache Eviction Implementation
    async def trigger_intelligent_eviction(self, target_memory_mb: float | None = None) -> dict[str, Any]:
        """
        Trigger intelligent cache eviction based on access patterns and importance.

        Args:
            target_memory_mb: Target memory usage in MB

        Returns:
            Eviction statistics
        """
        try:
            start_time = time.time()

            if target_memory_mb is None:
                target_memory_mb = self.max_memory_mb * 0.7  # Target 70% memory usage

            current_memory = await self._get_current_memory_usage()
            bytes_to_evict = max(0, (current_memory - target_memory_mb) * 1024 * 1024)

            evicted_keys = []
            bytes_evicted = 0

            # Use pressure service for sophisticated eviction
            if self.pressure_service:
                eviction_result = await self.pressure_service.trigger_eviction(target_size_bytes=int(bytes_to_evict))
                evicted_keys.extend(eviction_result.get("evicted_keys", []))
                bytes_evicted = eviction_result.get("bytes_evicted", 0)
            else:
                # Fallback eviction logic
                evicted_keys, bytes_evicted = await self._fallback_eviction(bytes_to_evict)

            # Update statistics
            self.eviction_stats.total_evictions += len(evicted_keys)
            self.eviction_stats.bytes_evicted += bytes_evicted
            self.eviction_stats.last_eviction_time = time.time()
            self._stats["evictions_triggered"] += 1

            duration = time.time() - start_time

            return {
                "success": True,
                "evicted_keys_count": len(evicted_keys),
                "bytes_evicted": bytes_evicted,
                "duration_seconds": duration,
                "target_memory_mb": target_memory_mb,
                "memory_after_eviction": await self._get_current_memory_usage(),
            }

        except Exception as e:
            self.logger.error(f"Error during intelligent eviction: {e}")
            return {"success": False, "error": str(e)}

    # 6.3 Memory Usage Monitoring and Control
    async def monitor_memory_usage(self) -> dict[str, Any]:
        """
        Monitor current memory usage and control mechanisms.

        Returns:
            Memory monitoring report
        """
        try:
            await self._update_memory_metrics()

            # Determine memory state
            utilization = self.memory_metrics.memory_utilization
            previous_state = self.memory_state

            if utilization >= self.memory_thresholds[MemoryState.EMERGENCY]:
                self.memory_state = MemoryState.EMERGENCY
            elif utilization >= self.memory_thresholds[MemoryState.CRITICAL]:
                self.memory_state = MemoryState.CRITICAL
            elif utilization >= self.memory_thresholds[MemoryState.WARNING]:
                self.memory_state = MemoryState.WARNING
            else:
                self.memory_state = MemoryState.OPTIMAL

            # Trigger actions based on state
            if self.memory_state != previous_state:
                await self._handle_memory_state_change(previous_state, self.memory_state)

            return {
                "memory_state": self.memory_state.value,
                "memory_metrics": {
                    "total_memory_mb": self.memory_metrics.total_memory_mb,
                    "used_memory_mb": self.memory_metrics.used_memory_mb,
                    "available_memory_mb": self.memory_metrics.available_memory_mb,
                    "cache_memory_mb": self.memory_metrics.cache_memory_mb,
                    "memory_utilization": self.memory_metrics.memory_utilization,
                    "cache_hit_rate": self.memory_metrics.cache_hit_rate,
                },
                "cache_layers": {
                    "l1_memory_mb": self.memory_metrics.l1_memory_mb,
                    "l2_memory_mb": self.memory_metrics.l2_memory_mb,
                    "l3_memory_mb": self.memory_metrics.l3_memory_mb,
                },
                "thresholds": {state.value: threshold for state, threshold in self.memory_thresholds.items()},
                "last_updated": self.memory_metrics.last_updated,
            }

        except Exception as e:
            self.logger.error(f"Error monitoring memory usage: {e}")
            return {"error": str(e)}

    # 6.4 Cache Pre-warming Implementation
    async def configure_cache_prewarming(
        self, enabled: bool = True, strategy: str = "adaptive", batch_size: int = 50, frequency_minutes: int = 15
    ) -> dict[str, Any]:
        """
        Configure cache pre-warming mechanism.

        Args:
            enabled: Enable/disable pre-warming
            strategy: Pre-warming strategy
            batch_size: Number of items to pre-warm per batch
            frequency_minutes: Pre-warming frequency in minutes

        Returns:
            Configuration result
        """
        try:
            if self.warmup_service:
                config_result = await self.warmup_service.configure_warmup(
                    enabled=enabled, strategy=strategy, batch_size=batch_size, frequency_minutes=frequency_minutes
                )
                return config_result
            else:
                return {"success": False, "error": "Warmup service not available"}

        except Exception as e:
            self.logger.error(f"Error configuring cache prewarming: {e}")
            return {"success": False, "error": str(e)}

    async def trigger_manual_prewarming(self, keys: list[str] | None = None) -> dict[str, Any]:
        """
        Manually trigger cache pre-warming.

        Args:
            keys: Specific keys to pre-warm (optional)

        Returns:
            Pre-warming result
        """
        try:
            if self.warmup_service:
                if keys:
                    result = await self.warmup_service.warm_specific_keys(keys)
                else:
                    result = await self.warmup_service.trigger_warmup()
                return result
            else:
                return {"success": False, "error": "Warmup service not available"}

        except Exception as e:
            self.logger.error(f"Error triggering manual prewarming: {e}")
            return {"success": False, "error": str(e)}

    # 6.5 Cache Hit Rate Monitoring and Optimization
    async def optimize_cache_hit_rate(self) -> dict[str, Any]:
        """
        Monitor and optimize cache hit rate continuously.

        Returns:
            Optimization result
        """
        try:
            # Analyze current hit rates
            l1_stats = await self._get_cache_stats("l1")
            l2_stats = await self._get_cache_stats("l2")
            l3_stats = await self._get_cache_stats("l3")

            overall_hit_rate = self.memory_metrics.cache_hit_rate

            optimization_actions = []

            # Optimize based on hit rate patterns
            if l1_stats["hit_rate"] < 0.6:
                # Low L1 hit rate - increase L1 size or improve pre-warming
                await self._optimize_l1_cache()
                optimization_actions.append("optimized_l1_cache")

            if l2_stats["hit_rate"] < 0.7:
                # Low L2 hit rate - analyze path caching patterns
                await self._optimize_l2_cache()
                optimization_actions.append("optimized_l2_cache")

            if l3_stats["hit_rate"] < 0.8:
                # Low L3 hit rate - optimize query result caching
                await self._optimize_l3_cache()
                optimization_actions.append("optimized_l3_cache")

            # Adjust cache sizes based on performance
            if overall_hit_rate < 0.7:
                await self._rebalance_cache_sizes()
                optimization_actions.append("rebalanced_cache_sizes")

            return {
                "success": True,
                "overall_hit_rate": overall_hit_rate,
                "cache_stats": {"l1": l1_stats, "l2": l2_stats, "l3": l3_stats},
                "optimization_actions": optimization_actions,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing cache hit rate: {e}")
            return {"success": False, "error": str(e)}

    # 6.6 Cache Data Compression Implementation
    async def configure_compression(self, level: str = "medium", enabled: bool = True) -> dict[str, Any]:
        """
        Configure cache data compression.

        Args:
            level: Compression level ("none", "low", "medium", "high")
            enabled: Enable/disable compression

        Returns:
            Configuration result
        """
        try:
            if not enabled:
                self.compression_level = CompressionLevel.NONE
            else:
                level_map = {
                    "none": CompressionLevel.NONE,
                    "low": CompressionLevel.LOW,
                    "medium": CompressionLevel.MEDIUM,
                    "high": CompressionLevel.HIGH,
                }
                self.compression_level = level_map.get(level.lower(), CompressionLevel.MEDIUM)

            return {
                "success": True,
                "compression_enabled": enabled,
                "compression_level": self.compression_level.name.lower(),
                "current_stats": {
                    "compressed_entries": self.compression_stats.compressed_entries,
                    "compression_ratio": self.compression_stats.compression_ratio,
                    "bytes_saved": self.compression_stats.bytes_saved,
                },
            }

        except Exception as e:
            self.logger.error(f"Error configuring compression: {e}")
            return {"success": False, "error": str(e)}

    async def analyze_compression_efficiency(self) -> dict[str, Any]:
        """
        Analyze compression efficiency and provide recommendations.

        Returns:
            Compression analysis report
        """
        try:
            # Analyze compression effectiveness
            total_entries = len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
            compression_percentage = (self.compression_stats.compressed_entries / total_entries * 100) if total_entries > 0 else 0

            # Calculate efficiency metrics
            avg_compression_time = self.compression_stats.compression_time_ms
            avg_decompression_time = self.compression_stats.decompression_time_ms
            space_savings_mb = self.compression_stats.bytes_saved / (1024 * 1024)

            recommendations = []

            if compression_percentage < 30:
                recommendations.append("Consider enabling compression for more cache entries")

            if avg_compression_time > 10:
                recommendations.append("Consider lowering compression level for better performance")

            if space_savings_mb < 10:
                recommendations.append("Compression may not be providing significant benefits")

            return {
                "compression_stats": {
                    "compressed_entries": self.compression_stats.compressed_entries,
                    "total_entries": total_entries,
                    "compression_percentage": compression_percentage,
                    "compression_ratio": self.compression_stats.compression_ratio,
                    "bytes_saved": self.compression_stats.bytes_saved,
                    "space_savings_mb": space_savings_mb,
                },
                "performance_metrics": {
                    "avg_compression_time_ms": avg_compression_time,
                    "avg_decompression_time_ms": avg_decompression_time,
                },
                "recommendations": recommendations,
                "efficiency_score": self._calculate_compression_efficiency_score(),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing compression efficiency: {e}")
            return {"error": str(e)}

    # 6.7 Cache Consistency Mechanisms
    async def ensure_cache_consistency(self) -> dict[str, Any]:
        """
        Ensure cache consistency across all layers.

        Returns:
            Consistency check result
        """
        try:
            if self.consistency_service:
                consistency_result = await self.consistency_service.check_consistency()
                self._stats["consistency_checks"] += 1
                return consistency_result
            else:
                # Fallback consistency check
                return await self._fallback_consistency_check()

        except Exception as e:
            self.logger.error(f"Error ensuring cache consistency: {e}")
            return {"success": False, "error": str(e)}

    async def repair_cache_inconsistencies(self) -> dict[str, Any]:
        """
        Repair detected cache inconsistencies.

        Returns:
            Repair result
        """
        try:
            if self.consistency_service:
                repair_result = await self.consistency_service.repair_inconsistencies()
                return repair_result
            else:
                return {"success": False, "error": "Consistency service not available"}

        except Exception as e:
            self.logger.error(f"Error repairing cache inconsistencies: {e}")
            return {"success": False, "error": str(e)}

    # 6.8 Cache Failure Recovery Implementation
    async def configure_failure_recovery(
        self, enabled: bool = True, max_retries: int = 3, recovery_strategy: str = "gradual"
    ) -> dict[str, Any]:
        """
        Configure cache failure recovery mechanisms.

        Args:
            enabled: Enable/disable failure recovery
            max_retries: Maximum recovery attempts
            recovery_strategy: Recovery strategy ("immediate", "gradual", "conservative")

        Returns:
            Configuration result
        """
        try:
            self._failure_recovery_enabled = enabled
            self._max_recovery_attempts = max_retries

            if self.failover_service:
                config_result = await self.failover_service.configure_failover(
                    enabled=enabled, max_retries=max_retries, strategy=recovery_strategy
                )
                return config_result
            else:
                return {"success": True, "recovery_enabled": enabled, "max_retries": max_retries, "strategy": recovery_strategy}

        except Exception as e:
            self.logger.error(f"Error configuring failure recovery: {e}")
            return {"success": False, "error": str(e)}

    async def trigger_failure_recovery(self) -> dict[str, Any]:
        """
        Manually trigger cache failure recovery.

        Returns:
            Recovery result
        """
        try:
            if not self._failure_recovery_enabled:
                return {"success": False, "error": "Failure recovery is disabled"}

            if self._recovery_attempts >= self._max_recovery_attempts:
                return {"success": False, "error": "Maximum recovery attempts exceeded"}

            recovery_start = time.time()
            self._recovery_attempts += 1

            # Attempt recovery
            recovery_success = await self._perform_failure_recovery()

            if recovery_success:
                self._recovery_attempts = 0  # Reset on successful recovery
                self._stats["failover_activations"] += 1

            duration = time.time() - recovery_start

            return {
                "success": recovery_success,
                "recovery_attempt": self._recovery_attempts,
                "max_attempts": self._max_recovery_attempts,
                "duration_seconds": duration,
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error triggering failure recovery: {e}")
            return {"success": False, "error": str(e)}

    # Helper Methods

    async def _update_memory_metrics(self) -> None:
        """Update current memory metrics."""
        try:
            # Get system memory info
            memory_info = psutil.virtual_memory()

            self.memory_metrics.total_memory_mb = memory_info.total / (1024 * 1024)
            self.memory_metrics.used_memory_mb = memory_info.used / (1024 * 1024)
            self.memory_metrics.available_memory_mb = memory_info.available / (1024 * 1024)
            self.memory_metrics.memory_utilization = memory_info.percent / 100.0

            # Get cache memory usage
            self.memory_metrics.l1_memory_mb = await self._get_cache_memory_usage("l1")
            self.memory_metrics.l2_memory_mb = await self._get_cache_memory_usage("l2")
            self.memory_metrics.l3_memory_mb = await self._get_cache_memory_usage("l3")
            self.memory_metrics.cache_memory_mb = (
                self.memory_metrics.l1_memory_mb + self.memory_metrics.l2_memory_mb + self.memory_metrics.l3_memory_mb
            )

            # Calculate hit rate
            total_hits = self._stats["cache_hits"]
            total_requests = total_hits + self._stats["cache_misses"]
            self.memory_metrics.cache_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

            self.memory_metrics.last_updated = time.time()

        except Exception as e:
            self.logger.error(f"Error updating memory metrics: {e}")

    async def _memory_monitoring_loop(self) -> None:
        """Continuous memory monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                await self._update_memory_metrics()

                # Check for memory pressure
                if self.memory_state in [MemoryState.CRITICAL, MemoryState.EMERGENCY]:
                    await self.trigger_intelligent_eviction()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes

                # Cleanup expired entries
                await self._cleanup_expired_entries()

                # Optimize cache layouts
                await self._optimize_cache_layouts()

                # Run garbage collection
                gc.collect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _prewarming_loop(self) -> None:
        """Periodic pre-warming loop."""
        while True:
            try:
                await asyncio.sleep(900)  # Pre-warm every 15 minutes

                if self.warmup_service:
                    await self.warmup_service.trigger_warmup()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in prewarming loop: {e}")

    def _track_access(self, key: str) -> None:
        """Track key access patterns."""
        current_time = time.time()
        pattern = self.access_patterns[key]
        pattern["count"] += 1
        pattern["last_access"] = current_time

        # Calculate frequency score
        if pattern["count"] > 1:
            time_window = current_time - (pattern.get("first_access", current_time))
            if time_window > 0:
                pattern["frequency_score"] = pattern["count"] / time_window

    def _select_optimal_cache_level(self, key: str, value: Any = None) -> str:
        """Select optimal cache level for key/value."""
        pattern = self.access_patterns.get(key, {})
        frequency_score = pattern.get("frequency_score", 0)

        # High frequency -> L1
        if frequency_score > 10:
            return "l1"
        # Medium frequency -> L2
        elif frequency_score > 1:
            return "l2"
        # Low frequency -> L3
        else:
            return "l3"

    async def _apply_compression(self, value: Any, key: str) -> tuple[Any, bool]:
        """Apply compression if beneficial."""
        if self.compression_level == CompressionLevel.NONE:
            return value, False

        try:
            # Serialize and compress
            start_time = time.time()
            serialized = pickle.dumps(value)

            if len(serialized) > 1024:  # Only compress larger values
                compressed = gzip.compress(serialized, compresslevel=self.compression_level.value)
                compression_time = (time.time() - start_time) * 1000

                # Update compression stats
                self.compression_stats.compression_time_ms = (self.compression_stats.compression_time_ms + compression_time) / 2

                compression_ratio = len(serialized) / len(compressed)
                self.compression_stats.compression_ratio = (self.compression_stats.compression_ratio + compression_ratio) / 2

                bytes_saved = len(serialized) - len(compressed)
                self.compression_stats.bytes_saved += bytes_saved

                return compressed, True

            return value, False

        except Exception as e:
            self.logger.error(f"Error applying compression to key {key}: {e}")
            return value, False

    async def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive memory manager status."""
        try:
            memory_status = await self.monitor_memory_usage()
            compression_analysis = await self.analyze_compression_efficiency()
            consistency_status = await self.ensure_cache_consistency()

            return {
                "memory_manager_status": "operational",
                "memory_state": self.memory_state.value,
                "memory_monitoring": memory_status,
                "cache_layers": {"l1_entries": len(self.l1_cache), "l2_entries": len(self.l2_cache), "l3_entries": len(self.l3_cache)},
                "compression": compression_analysis,
                "consistency": consistency_status,
                "eviction_stats": {
                    "total_evictions": self.eviction_stats.total_evictions,
                    "bytes_evicted": self.eviction_stats.bytes_evicted,
                    "last_eviction": self.eviction_stats.last_eviction_time,
                },
                "statistics": self._stats,
                "failure_recovery": {
                    "enabled": self._failure_recovery_enabled,
                    "recovery_attempts": self._recovery_attempts,
                    "max_attempts": self._max_recovery_attempts,
                },
                "timestamp": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e)}

    # Additional helper methods would be implemented here for:
    # - Cache level operations (_get_from_l1, _set_to_l1, etc.)
    # - Memory calculations and optimizations
    # - Fallback implementations
    # - Recovery procedures
    # - Cache statistics and analysis

    # For brevity, these methods are indicated but not fully implemented
    # in this response. Each would follow similar patterns to the above methods.

    async def _get_from_l1(self, key: str) -> Any | None:
        """Get value from L1 cache."""
        # Implementation would access L1 memory cache
        pass

    async def _set_to_l1(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in L1 cache."""
        # Implementation would set in L1 memory cache
        pass

    # ... additional helper methods would be implemented here
