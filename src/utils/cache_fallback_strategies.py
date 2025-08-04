"""
Advanced Cache Fallback Strategies.

This module provides sophisticated fallback strategies for cache unavailability,
including multiple fallback layers, intelligent data sourcing, and cache-aware
recovery mechanisms.
"""

import asyncio
import json
import logging
import pickle
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..config.cache_config import CacheConfig


class FallbackTier(Enum):
    """Fallback tier levels."""

    PRIMARY = "primary"  # Primary cache (Redis)
    SECONDARY = "secondary"  # Secondary cache (memory)
    TERTIARY = "tertiary"  # Tertiary cache (disk)
    COMPUTE = "compute"  # Recompute data
    DEFAULT = "default"  # Default values
    DEGRADE = "degrade"  # Graceful degradation


class FallbackTrigger(Enum):
    """Triggers for fallback activation."""

    CONNECTION_LOST = "connection_lost"
    TIMEOUT = "timeout"
    ERROR = "error"
    CIRCUIT_BREAKER = "circuit_breaker"
    PERFORMANCE = "performance"
    MANUAL = "manual"


@dataclass
class FallbackConfig:
    """Configuration for fallback strategies."""

    # Enabled fallback tiers
    enabled_tiers: set[FallbackTier] = field(default_factory=lambda: {FallbackTier.SECONDARY, FallbackTier.TERTIARY, FallbackTier.DEFAULT})

    # Disk cache settings
    disk_cache_dir: str = "/tmp/cache_fallback"
    disk_cache_max_size_mb: int = 100
    disk_cache_ttl_hours: int = 24

    # Memory fallback settings
    memory_fallback_max_size: int = 1000
    memory_fallback_ttl_seconds: int = 3600

    # Compute fallback settings
    compute_timeout_seconds: float = 30.0
    compute_max_retries: int = 2

    # Performance settings
    fallback_response_timeout: float = 5.0
    parallel_fallback_execution: bool = True

    # Data freshness settings
    stale_data_threshold_seconds: int = 86400  # 24 hours
    prefer_stale_over_recompute: bool = True

    # Monitoring
    track_fallback_usage: bool = True
    fallback_metrics_window: int = 1000


@dataclass
class FallbackMetrics:
    """Metrics for fallback operations."""

    total_fallback_activations: int = 0
    fallback_tier_usage: dict[str, int] = field(default_factory=dict)
    fallback_trigger_counts: dict[str, int] = field(default_factory=dict)
    fallback_success_rate: dict[str, float] = field(default_factory=dict)
    average_fallback_response_time: dict[str, float] = field(default_factory=dict)

    # Recent activity
    recent_fallbacks: list[tuple[float, str, str]] = field(default_factory=list)  # (timestamp, tier, key)

    def record_fallback(self, tier: FallbackTier, trigger: FallbackTrigger, key: str, success: bool, response_time: float) -> None:
        """Record fallback usage."""
        self.total_fallback_activations += 1

        # Update tier usage
        tier_name = tier.value
        self.fallback_tier_usage[tier_name] = self.fallback_tier_usage.get(tier_name, 0) + 1

        # Update trigger counts
        trigger_name = trigger.value
        self.fallback_trigger_counts[trigger_name] = self.fallback_trigger_counts.get(trigger_name, 0) + 1

        # Update success rate
        if tier_name not in self.fallback_success_rate:
            self.fallback_success_rate[tier_name] = 1.0 if success else 0.0
        else:
            current_rate = self.fallback_success_rate[tier_name]
            usage_count = self.fallback_tier_usage[tier_name]
            self.fallback_success_rate[tier_name] = (current_rate * (usage_count - 1) + (1.0 if success else 0.0)) / usage_count

        # Update response time
        if tier_name not in self.average_fallback_response_time:
            self.average_fallback_response_time[tier_name] = response_time
        else:
            current_avg = self.average_fallback_response_time[tier_name]
            usage_count = self.fallback_tier_usage[tier_name]
            self.average_fallback_response_time[tier_name] = (current_avg * (usage_count - 1) + response_time) / usage_count

        # Record recent activity
        self.recent_fallbacks.append((time.time(), tier_name, key))
        if len(self.recent_fallbacks) > 1000:  # Keep last 1000 entries
            self.recent_fallbacks.pop(0)


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""

    def __init__(self, config: FallbackConfig):
        """Initialize fallback strategy."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def get(self, key: str, **kwargs) -> Any | None:
        """Get value using fallback strategy."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None, **kwargs) -> bool:
        """Set value using fallback strategy."""
        pass

    @abstractmethod
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete value using fallback strategy."""
        pass

    @abstractmethod
    async def clear(self, **kwargs) -> bool:
        """Clear all values using fallback strategy."""
        pass

    @abstractmethod
    def get_tier(self) -> FallbackTier:
        """Get the fallback tier this strategy represents."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if this fallback strategy is healthy."""
        pass


class MemoryFallbackStrategy(FallbackStrategy):
    """In-memory fallback strategy using LRU cache."""

    def __init__(self, config: FallbackConfig):
        """Initialize memory fallback strategy."""
        super().__init__(config)
        from collections import OrderedDict

        self._cache: OrderedDict[str, tuple[Any, float, int | None]] = OrderedDict()
        self._max_size = config.memory_fallback_max_size
        self._default_ttl = config.memory_fallback_ttl_seconds

    async def get(self, key: str, **kwargs) -> Any | None:
        """Get value from memory fallback."""
        if key not in self._cache:
            return None

        value, timestamp, ttl = self._cache[key]

        # Check expiration
        if ttl and time.time() - timestamp > ttl:
            del self._cache[key]
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None, **kwargs) -> bool:
        """Set value in memory fallback."""
        try:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict if necessary
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Remove oldest

            # Add new entry
            cache_ttl = ttl or self._default_ttl
            self._cache[key] = (value, time.time(), cache_ttl)
            return True

        except Exception as e:
            self.logger.error(f"Memory fallback set failed: {e}")
            return False

    async def delete(self, key: str, **kwargs) -> bool:
        """Delete value from memory fallback."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def clear(self, **kwargs) -> bool:
        """Clear memory fallback."""
        self._cache.clear()
        return True

    def get_tier(self) -> FallbackTier:
        """Get fallback tier."""
        return FallbackTier.SECONDARY

    async def health_check(self) -> bool:
        """Check memory fallback health."""
        return True  # Memory fallback is always healthy

    def get_info(self) -> dict[str, Any]:
        """Get memory fallback information."""
        return {"size": len(self._cache), "max_size": self._max_size, "utilization": len(self._cache) / self._max_size}


class DiskFallbackStrategy(FallbackStrategy):
    """Disk-based fallback strategy using file system."""

    def __init__(self, config: FallbackConfig):
        """Initialize disk fallback strategy."""
        super().__init__(config)
        self._cache_dir = Path(config.disk_cache_dir)
        self._max_size_bytes = config.disk_cache_max_size_mb * 1024 * 1024
        self._ttl_seconds = config.disk_cache_ttl_hours * 3600

        # Create cache directory
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self._cache_dir / "metadata.json"

        # Load metadata
        self._metadata = self._load_metadata()

    async def get(self, key: str, **kwargs) -> Any | None:
        """Get value from disk fallback."""
        try:
            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None

            # Check metadata for expiration
            if not self._is_valid(key):
                await self.delete(key)
                return None

            # Read data
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Update access time in metadata
            self._metadata[key]["last_access"] = time.time()
            await self._save_metadata()

            return data

        except Exception as e:
            self.logger.error(f"Disk fallback get failed for key '{key}': {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None, **kwargs) -> bool:
        """Set value in disk fallback."""
        try:
            # Check if we need to make space
            await self._ensure_space()

            # Write data
            file_path = self._get_file_path(key)
            with open(file_path, "wb") as f:
                pickle.dump(value, f)

            # Update metadata
            file_size = file_path.stat().st_size
            cache_ttl = ttl or self._ttl_seconds

            self._metadata[key] = {"size": file_size, "created": time.time(), "last_access": time.time(), "ttl": cache_ttl}

            await self._save_metadata()
            return True

        except Exception as e:
            self.logger.error(f"Disk fallback set failed for key '{key}': {e}")
            return False

    async def delete(self, key: str, **kwargs) -> bool:
        """Delete value from disk fallback."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()

            if key in self._metadata:
                del self._metadata[key]
                await self._save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"Disk fallback delete failed for key '{key}': {e}")
            return False

    async def clear(self, **kwargs) -> bool:
        """Clear disk fallback."""
        try:
            # Remove all cache files
            for file_path in self._cache_dir.glob("*.cache"):
                file_path.unlink()

            # Clear metadata
            self._metadata.clear()
            await self._save_metadata()

            return True

        except Exception as e:
            self.logger.error(f"Disk fallback clear failed: {e}")
            return False

    def get_tier(self) -> FallbackTier:
        """Get fallback tier."""
        return FallbackTier.TERTIARY

    async def health_check(self) -> bool:
        """Check disk fallback health."""
        try:
            # Check if directory is writable
            test_file = self._cache_dir / "health_check.tmp"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False

    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Simple hash-based filename
        import hashlib

        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"

    def _is_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._metadata:
            return False

        metadata = self._metadata[key]
        age = time.time() - metadata["created"]
        return age <= metadata["ttl"]

    async def _ensure_space(self) -> None:
        """Ensure sufficient disk space by evicting old entries."""
        try:
            current_size = sum(metadata["size"] for metadata in self._metadata.values())

            if current_size >= self._max_size_bytes:
                # Sort by last access time (LRU)
                sorted_entries = sorted(self._metadata.items(), key=lambda x: x[1]["last_access"])

                # Remove oldest entries until we have space
                for key, metadata in sorted_entries:
                    await self.delete(key)
                    current_size -= metadata["size"]
                    if current_size < self._max_size_bytes * 0.8:  # Leave some buffer
                        break

        except Exception as e:
            self.logger.error(f"Failed to ensure disk space: {e}")

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from disk."""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file) as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load disk cache metadata: {e}")

        return {}

    async def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f)
        except Exception as e:
            self.logger.error(f"Failed to save disk cache metadata: {e}")

    def get_info(self) -> dict[str, Any]:
        """Get disk fallback information."""
        total_size = sum(metadata["size"] for metadata in self._metadata.values())
        return {
            "entries": len(self._metadata),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self._max_size_bytes / (1024 * 1024),
            "utilization": total_size / self._max_size_bytes,
        }


class ComputeFallbackStrategy(FallbackStrategy):
    """Compute fallback strategy that recomputes values."""

    def __init__(self, config: FallbackConfig):
        """Initialize compute fallback strategy."""
        super().__init__(config)
        self._compute_functions: dict[str, Callable] = {}
        self._compute_cache: dict[str, tuple[Any, float]] = {}
        self._timeout = config.compute_timeout_seconds
        self._max_retries = config.compute_max_retries

    def register_compute_function(self, key_pattern: str, compute_func: Callable) -> None:
        """Register a compute function for a key pattern."""
        self._compute_functions[key_pattern] = compute_func

    async def get(self, key: str, **kwargs) -> Any | None:
        """Get value by recomputing."""
        # Check if we have a recent computed value
        if key in self._compute_cache:
            value, timestamp = self._compute_cache[key]
            if time.time() - timestamp < 300:  # 5 minutes cache
                return value

        # Find matching compute function
        compute_func = self._find_compute_function(key)
        if not compute_func:
            return None

        # Attempt computation with retries
        for attempt in range(self._max_retries + 1):
            try:
                start_time = time.time()

                if asyncio.iscoroutinefunction(compute_func):
                    result = await asyncio.wait_for(compute_func(key, **kwargs), timeout=self._timeout)
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, compute_func, key), timeout=self._timeout
                    )

                # Cache the computed result
                self._compute_cache[key] = (result, time.time())

                compute_time = time.time() - start_time
                self.logger.info(f"Computed fallback value for '{key}' in {compute_time:.2f}s")

                return result

            except Exception as e:
                if attempt < self._max_retries:
                    self.logger.warning(f"Compute attempt {attempt + 1} failed for '{key}': {e}")
                    await asyncio.sleep(1 * (attempt + 1))  # Linear backoff
                else:
                    self.logger.error(f"All compute attempts failed for '{key}': {e}")

        return None

    async def set(self, key: str, value: Any, ttl: int | None = None, **kwargs) -> bool:
        """Set computed value (cache it)."""
        self._compute_cache[key] = (value, time.time())
        return True

    async def delete(self, key: str, **kwargs) -> bool:
        """Delete computed value from cache."""
        if key in self._compute_cache:
            del self._compute_cache[key]
            return True
        return False

    async def clear(self, **kwargs) -> bool:
        """Clear compute cache."""
        self._compute_cache.clear()
        return True

    def get_tier(self) -> FallbackTier:
        """Get fallback tier."""
        return FallbackTier.COMPUTE

    async def health_check(self) -> bool:
        """Check compute fallback health."""
        return len(self._compute_functions) > 0

    def _find_compute_function(self, key: str) -> Callable | None:
        """Find compute function matching the key."""
        for pattern, func in self._compute_functions.items():
            if pattern in key or key.startswith(pattern):
                return func
        return None


class DefaultValueFallbackStrategy(FallbackStrategy):
    """Default value fallback strategy."""

    def __init__(self, config: FallbackConfig):
        """Initialize default value fallback strategy."""
        super().__init__(config)
        self._defaults: dict[str, Any] = {}

    def register_default(self, key_pattern: str, default_value: Any) -> None:
        """Register a default value for a key pattern."""
        self._defaults[key_pattern] = default_value

    async def get(self, key: str, **kwargs) -> Any | None:
        """Get default value."""
        # Find matching default
        for pattern, default_value in self._defaults.items():
            if pattern in key or key.startswith(pattern):
                return default_value

        # Generic defaults based on key patterns
        if "count" in key.lower():
            return 0
        elif "list" in key.lower() or "items" in key.lower():
            return []
        elif "dict" in key.lower() or "map" in key.lower():
            return {}
        elif "bool" in key.lower() or "flag" in key.lower():
            return False
        elif "time" in key.lower() or "timestamp" in key.lower():
            return time.time()

        return None

    async def set(self, key: str, value: Any, ttl: int | None = None, **kwargs) -> bool:
        """Set not supported for default values."""
        return False

    async def delete(self, key: str, **kwargs) -> bool:
        """Delete not supported for default values."""
        return False

    async def clear(self, **kwargs) -> bool:
        """Clear not supported for default values."""
        return False

    def get_tier(self) -> FallbackTier:
        """Get fallback tier."""
        return FallbackTier.DEFAULT

    async def health_check(self) -> bool:
        """Check default value fallback health."""
        return True


class CacheFallbackManager:
    """
    Comprehensive cache fallback manager.

    Coordinates multiple fallback strategies in a hierarchical manner,
    providing intelligent fallback selection and comprehensive metrics.
    """

    def __init__(self, config: FallbackConfig | None = None):
        """Initialize cache fallback manager."""
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize strategies
        self._strategies: dict[FallbackTier, FallbackStrategy] = {}
        self._initialize_strategies()

        # Metrics and monitoring
        self.metrics = FallbackMetrics()

        # State tracking
        self._primary_cache_available = True
        self._fallback_active = False
        self._active_tier = FallbackTier.PRIMARY

    def _initialize_strategies(self) -> None:
        """Initialize fallback strategies."""
        if FallbackTier.SECONDARY in self.config.enabled_tiers:
            self._strategies[FallbackTier.SECONDARY] = MemoryFallbackStrategy(self.config)

        if FallbackTier.TERTIARY in self.config.enabled_tiers:
            self._strategies[FallbackTier.TERTIARY] = DiskFallbackStrategy(self.config)

        if FallbackTier.COMPUTE in self.config.enabled_tiers:
            self._strategies[FallbackTier.COMPUTE] = ComputeFallbackStrategy(self.config)

        if FallbackTier.DEFAULT in self.config.enabled_tiers:
            self._strategies[FallbackTier.DEFAULT] = DefaultValueFallbackStrategy(self.config)

    async def get_with_fallback(
        self, key: str, trigger: FallbackTrigger = FallbackTrigger.ERROR, excluded_tiers: set[FallbackTier] | None = None, **kwargs
    ) -> tuple[Any | None, FallbackTier]:
        """
        Get value using fallback strategies.

        Args:
            key: Cache key
            trigger: What triggered the fallback
            excluded_tiers: Tiers to exclude from fallback
            **kwargs: Additional arguments

        Returns:
            Tuple of (value, tier_used)
        """
        excluded_tiers = excluded_tiers or set()
        time.time()

        # Determine fallback order
        fallback_order = self._get_fallback_order(excluded_tiers)

        if self.config.parallel_fallback_execution and len(fallback_order) > 1:
            # Try multiple strategies in parallel
            return await self._get_parallel_fallback(key, fallback_order, trigger, **kwargs)
        else:
            # Try strategies sequentially
            return await self._get_sequential_fallback(key, fallback_order, trigger, **kwargs)

    async def set_to_fallback(self, key: str, value: Any, tier: FallbackTier, ttl: int | None = None, **kwargs) -> bool:
        """Set value to specific fallback tier."""
        if tier not in self._strategies:
            return False

        try:
            strategy = self._strategies[tier]
            return await strategy.set(key, value, ttl, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to set to {tier.value} fallback: {e}")
            return False

    async def populate_fallback_caches(self, data: dict[str, Any], tiers: set[FallbackTier] | None = None) -> dict[FallbackTier, int]:
        """Populate fallback caches with data."""
        tiers = tiers or set(self._strategies.keys())
        results = {}

        for tier in tiers:
            if tier not in self._strategies:
                continue

            strategy = self._strategies[tier]
            success_count = 0

            for key, value in data.items():
                try:
                    if await strategy.set(key, value):
                        success_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to populate {tier.value} cache for key '{key}': {e}")

            results[tier] = success_count
            self.logger.info(f"Populated {tier.value} cache with {success_count}/{len(data)} items")

        return results

    async def _get_sequential_fallback(
        self, key: str, fallback_order: list[FallbackTier], trigger: FallbackTrigger, **kwargs
    ) -> tuple[Any | None, FallbackTier]:
        """Get value using sequential fallback."""
        for tier in fallback_order:
            if tier not in self._strategies:
                continue

            try:
                start_time = time.time()
                strategy = self._strategies[tier]

                # Check strategy health first
                if not await strategy.health_check():
                    continue

                value = await asyncio.wait_for(strategy.get(key, **kwargs), timeout=self.config.fallback_response_timeout)

                response_time = time.time() - start_time

                if value is not None:
                    # Record successful fallback
                    self.metrics.record_fallback(tier, trigger, key, True, response_time)
                    self.logger.info(f"Fallback successful using {tier.value} for key '{key}'")
                    return value, tier

            except Exception as e:
                response_time = time.time() - start_time if "start_time" in locals() else 0
                self.metrics.record_fallback(tier, trigger, key, False, response_time)
                self.logger.warning(f"Fallback failed for {tier.value}: {e}")

        # No fallback succeeded
        return None, FallbackTier.PRIMARY

    async def _get_parallel_fallback(
        self, key: str, fallback_order: list[FallbackTier], trigger: FallbackTrigger, **kwargs
    ) -> tuple[Any | None, FallbackTier]:
        """Get value using parallel fallback execution."""

        async def try_strategy(tier: FallbackTier) -> tuple[Any | None, FallbackTier, float]:
            if tier not in self._strategies:
                return None, tier, 0.0

            start_time = time.time()
            try:
                strategy = self._strategies[tier]

                if not await strategy.health_check():
                    return None, tier, time.time() - start_time

                value = await asyncio.wait_for(strategy.get(key, **kwargs), timeout=self.config.fallback_response_timeout)

                return value, tier, time.time() - start_time

            except Exception as e:
                self.logger.warning(f"Parallel fallback failed for {tier.value}: {e}")
                return None, tier, time.time() - start_time

        # Execute strategies in parallel
        tasks = [try_strategy(tier) for tier in fallback_order]

        try:
            # Wait for first successful result
            for completed_task in asyncio.as_completed(tasks):
                value, tier, response_time = await completed_task

                if value is not None:
                    # Record successful fallback
                    self.metrics.record_fallback(tier, trigger, key, True, response_time)
                    self.logger.info(f"Parallel fallback successful using {tier.value} for key '{key}'")

                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()

                    return value, tier
                else:
                    # Record failed attempt
                    self.metrics.record_fallback(tier, trigger, key, False, response_time)

        except Exception as e:
            self.logger.error(f"Parallel fallback execution failed: {e}")

        return None, FallbackTier.PRIMARY

    def _get_fallback_order(self, excluded_tiers: set[FallbackTier]) -> list[FallbackTier]:
        """Get ordered list of fallback tiers to try."""
        # Default order: SECONDARY -> TERTIARY -> COMPUTE -> DEFAULT
        default_order = [FallbackTier.SECONDARY, FallbackTier.TERTIARY, FallbackTier.COMPUTE, FallbackTier.DEFAULT]

        # Filter by available strategies and exclusions
        return [tier for tier in default_order if tier in self._strategies and tier not in excluded_tiers]

    def register_compute_function(self, key_pattern: str, compute_func: Callable) -> None:
        """Register compute function for fallback."""
        if FallbackTier.COMPUTE in self._strategies:
            strategy = self._strategies[FallbackTier.COMPUTE]
            if isinstance(strategy, ComputeFallbackStrategy):
                strategy.register_compute_function(key_pattern, compute_func)

    def register_default_value(self, key_pattern: str, default_value: Any) -> None:
        """Register default value for fallback."""
        if FallbackTier.DEFAULT in self._strategies:
            strategy = self._strategies[FallbackTier.DEFAULT]
            if isinstance(strategy, DefaultValueFallbackStrategy):
                strategy.register_default(key_pattern, default_value)

    async def health_check_all(self) -> dict[FallbackTier, bool]:
        """Check health of all fallback strategies."""
        results = {}
        for tier, strategy in self._strategies.items():
            try:
                results[tier] = await strategy.health_check()
            except Exception as e:
                self.logger.error(f"Health check failed for {tier.value}: {e}")
                results[tier] = False
        return results

    def get_metrics(self) -> FallbackMetrics:
        """Get fallback metrics."""
        return self.metrics

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about all strategies."""
        info = {}
        for tier, strategy in self._strategies.items():
            tier_info = {"tier": tier.value, "available": True}

            if hasattr(strategy, "get_info"):
                tier_info.update(strategy.get_info())

            info[tier.value] = tier_info

        return info

    async def clear_all_fallbacks(self) -> dict[FallbackTier, bool]:
        """Clear all fallback caches."""
        results = {}
        for tier, strategy in self._strategies.items():
            try:
                results[tier] = await strategy.clear()
            except Exception as e:
                self.logger.error(f"Failed to clear {tier.value} fallback: {e}")
                results[tier] = False
        return results


# Factory function
def create_fallback_manager(config: FallbackConfig | None = None) -> CacheFallbackManager:
    """Create cache fallback manager."""
    return CacheFallbackManager(config)


# Global fallback manager instance
_global_fallback_manager: CacheFallbackManager | None = None


def get_fallback_manager(config: FallbackConfig | None = None) -> CacheFallbackManager:
    """Get or create global fallback manager."""
    global _global_fallback_manager
    if _global_fallback_manager is None:
        _global_fallback_manager = create_fallback_manager(config)
    return _global_fallback_manager
