"""
Advanced cache eviction policies for optimized memory management.

This module provides sophisticated eviction strategies beyond basic LRU,
including adaptive policies, machine learning-based prediction, and
multi-criteria decision making for optimal cache performance.
"""

import heapq
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class EvictionPolicy(Enum):
    """Cache eviction policy types."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ARC = "arc"  # Adaptive Replacement Cache
    SLRU = "slru"  # Segmented LRU
    W_TINY_LFU = "w_tiny_lfu"  # Window TinyLFU
    ADAPTIVE = "adaptive"  # Adaptive policy selection


@dataclass
class CacheEntry:
    """Enhanced cache entry with comprehensive metadata."""

    key: str
    value: Any
    size: int
    timestamp: float
    last_access: float
    access_count: int
    frequency_decay: float = 0.95
    cost_score: float = 1.0
    priority: int = 0

    def __post_init__(self):
        """Initialize computed fields."""
        self.age = 0.0
        self.frequency_score = 0.0
        self.recency_score = 1.0

    def update_access(self) -> None:
        """Update access statistics."""
        current_time = time.time()
        self.last_access = current_time
        self.access_count += 1

        # Update derived scores
        self.age = current_time - self.timestamp
        self.recency_score = 1.0 / (1.0 + self.age / 3600)  # Decay over hours

        # Update frequency with decay
        time_delta = current_time - self.last_access
        decay_factor = self.frequency_decay ** (time_delta / 3600)
        self.frequency_score = self.frequency_score * decay_factor + 1.0


class BaseEvictionPolicy(ABC):
    """Base class for cache eviction policies."""

    def __init__(self, capacity: int, logger: logging.Logger | None = None):
        """Initialize eviction policy."""
        self.capacity = capacity
        self.logger = logger or logging.getLogger(__name__)
        self.entries: dict[str, CacheEntry] = {}
        self.eviction_count = 0
        self.hit_count = 0
        self.miss_count = 0

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value and update access patterns."""
        pass

    @abstractmethod
    def put(self, key: str, value: Any, size: int = 1) -> bool:
        """Add/update entry and handle eviction."""
        pass

    @abstractmethod
    def evict(self) -> str | None:
        """Select and evict an entry."""
        pass

    def remove(self, key: str) -> bool:
        """Remove specific entry."""
        if key in self.entries:
            del self.entries[key]
            return True
        return False

    def size(self) -> int:
        """Get current cache size."""
        return len(self.entries)

    def get_stats(self) -> dict[str, Any]:
        """Get eviction policy statistics."""
        total_operations = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_operations if total_operations > 0 else 0.0

        return {
            "policy": self.__class__.__name__,
            "capacity": self.capacity,
            "current_size": self.size(),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
        }


class LRUEvictionPolicy(BaseEvictionPolicy):
    """Least Recently Used eviction policy."""

    def __init__(self, capacity: int, logger: logging.Logger | None = None):
        """Initialize LRU policy."""
        super().__init__(capacity, logger)
        self.access_order: list[str] = []

    def get(self, key: str) -> Any | None:
        """Get value and update LRU order."""
        if key in self.entries:
            entry = self.entries[key]
            entry.update_access()

            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)

            self.hit_count += 1
            return entry.value

        self.miss_count += 1
        return None

    def put(self, key: str, value: Any, size: int = 1) -> bool:
        """Add/update entry with LRU eviction."""
        current_time = time.time()

        if key in self.entries:
            # Update existing entry
            entry = self.entries[key]
            entry.value = value
            entry.size = size
            entry.update_access()

            # Move to end
            self.access_order.remove(key)
            self.access_order.append(key)
            return True

        # Check capacity
        while len(self.entries) >= self.capacity:
            if not self.evict():
                return False

        # Add new entry
        entry = CacheEntry(key=key, value=value, size=size, timestamp=current_time, last_access=current_time, access_count=1)

        self.entries[key] = entry
        self.access_order.append(key)
        return True

    def evict(self) -> str | None:
        """Evict least recently used entry."""
        if not self.access_order:
            return None

        # Remove oldest (first in list)
        evicted_key = self.access_order.pop(0)
        if evicted_key in self.entries:
            del self.entries[evicted_key]
            self.eviction_count += 1
            return evicted_key

        return None


class LFUEvictionPolicy(BaseEvictionPolicy):
    """Least Frequently Used eviction policy."""

    def __init__(self, capacity: int, logger: logging.Logger | None = None):
        """Initialize LFU policy."""
        super().__init__(capacity, logger)
        self.frequency_heap: list[tuple[float, str]] = []

    def get(self, key: str) -> Any | None:
        """Get value and update frequency."""
        if key in self.entries:
            entry = self.entries[key]
            entry.update_access()

            # Update frequency heap
            heapq.heappush(self.frequency_heap, (entry.frequency_score, key))

            self.hit_count += 1
            return entry.value

        self.miss_count += 1
        return None

    def put(self, key: str, value: Any, size: int = 1) -> bool:
        """Add/update entry with LFU eviction."""
        current_time = time.time()

        if key in self.entries:
            # Update existing entry
            entry = self.entries[key]
            entry.value = value
            entry.size = size
            entry.update_access()

            # Update heap
            heapq.heappush(self.frequency_heap, (entry.frequency_score, key))
            return True

        # Check capacity
        while len(self.entries) >= self.capacity:
            if not self.evict():
                return False

        # Add new entry
        entry = CacheEntry(key=key, value=value, size=size, timestamp=current_time, last_access=current_time, access_count=1)
        entry.update_access()

        self.entries[key] = entry
        heapq.heappush(self.frequency_heap, (entry.frequency_score, key))
        return True

    def evict(self) -> str | None:
        """Evict least frequently used entry."""
        while self.frequency_heap:
            frequency, key = heapq.heappop(self.frequency_heap)

            # Check if entry still exists and frequency is current
            if key in self.entries and abs(self.entries[key].frequency_score - frequency) < 0.01:
                del self.entries[key]
                self.eviction_count += 1
                return key

        return None


class ARCEvictionPolicy(BaseEvictionPolicy):
    """Adaptive Replacement Cache (ARC) eviction policy."""

    def __init__(self, capacity: int, logger: logging.Logger | None = None):
        """Initialize ARC policy."""
        super().__init__(capacity, logger)
        self.c = capacity  # Total cache size
        self.p = 0  # Target size for T1

        # ARC lists
        self.t1: list[str] = []  # Recent cache entries
        self.t2: list[str] = []  # Frequent cache entries
        self.b1: list[str] = []  # Ghost entries for T1
        self.b2: list[str] = []  # Ghost entries for T2

        # Mapping for quick lookup
        self.location: dict[str, str] = {}  # key -> list location

    def get(self, key: str) -> Any | None:
        """Get value with ARC policy."""
        if key in self.entries:
            entry = self.entries[key]
            entry.update_access()

            # Move from T1 to T2 or update position in T2
            if key in self.t1:
                self.t1.remove(key)
                self.t2.append(key)
                self.location[key] = "t2"
            elif key in self.t2:
                self.t2.remove(key)
                self.t2.append(key)

            self.hit_count += 1
            return entry.value

        # Check ghost lists for potential promotion
        if key in self.b1:
            self.p = min(self.c, self.p + max(1, len(self.b2) // len(self.b1)))
            self.b1.remove(key)
        elif key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
            self.b2.remove(key)

        self.miss_count += 1
        return None

    def put(self, key: str, value: Any, size: int = 1) -> bool:
        """Add/update entry with ARC eviction."""
        current_time = time.time()

        if key in self.entries:
            # Update existing entry
            entry = self.entries[key]
            entry.value = value
            entry.size = size
            entry.update_access()
            return True

        # Check capacity and evict if necessary
        while len(self.entries) >= self.capacity:
            if not self.evict():
                return False

        # Add new entry to T1
        entry = CacheEntry(key=key, value=value, size=size, timestamp=current_time, last_access=current_time, access_count=1)

        self.entries[key] = entry
        self.t1.append(key)
        self.location[key] = "t1"

        return True

    def evict(self) -> str | None:
        """Evict using ARC replacement strategy."""
        # Replace based on ARC algorithm
        if len(self.t1) > 0 and (len(self.t1) > self.p or (len(self.t1) == self.p and len(self.b2) > 0)):
            # Evict from T1
            evicted_key = self.t1.pop(0)
            self.b1.append(evicted_key)

            # Maintain B1 size
            if len(self.b1) > self.c:
                self.b1.pop(0)

        else:
            # Evict from T2
            if self.t2:
                evicted_key = self.t2.pop(0)
                self.b2.append(evicted_key)

                # Maintain B2 size
                if len(self.b2) > self.c:
                    self.b2.pop(0)
            else:
                return None

        # Remove from entries
        if evicted_key in self.entries:
            del self.entries[evicted_key]
            del self.location[evicted_key]
            self.eviction_count += 1
            return evicted_key

        return None


class AdaptiveEvictionPolicy(BaseEvictionPolicy):
    """Adaptive eviction policy that switches between strategies."""

    def __init__(self, capacity: int, logger: logging.Logger | None = None):
        """Initialize adaptive policy."""
        super().__init__(capacity, logger)

        # Initialize sub-policies
        self.policies = {
            "lru": LRUEvictionPolicy(capacity, logger),
            "lfu": LFUEvictionPolicy(capacity, logger),
            "arc": ARCEvictionPolicy(capacity, logger),
        }

        self.current_policy = "lru"
        self.evaluation_interval = 1000  # Operations between evaluations
        self.operation_count = 0
        self.policy_performance: dict[str, list[float]] = {name: [] for name in self.policies}

    def get(self, key: str) -> Any | None:
        """Get value using current policy."""
        self.operation_count += 1
        result = self.policies[self.current_policy].get(key)

        if result is not None:
            self.hit_count += 1
        else:
            self.miss_count += 1

        # Evaluate and potentially switch policies
        if self.operation_count % self.evaluation_interval == 0:
            self._evaluate_policies()

        return result

    def put(self, key: str, value: Any, size: int = 1) -> bool:
        """Add/update entry using current policy."""
        return self.policies[self.current_policy].put(key, value, size)

    def evict(self) -> str | None:
        """Evict using current policy."""
        return self.policies[self.current_policy].evict()

    def _evaluate_policies(self) -> None:
        """Evaluate policy performance and switch if beneficial."""
        current_hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0.0

        # Record current policy performance
        self.policy_performance[self.current_policy].append(current_hit_rate)

        # Keep only recent performance data
        for policy_name in self.policy_performance:
            self.policy_performance[policy_name] = self.policy_performance[policy_name][-10:]

        # Find best performing policy
        best_policy = self.current_policy
        best_performance = current_hit_rate

        for policy_name, performances in self.policy_performance.items():
            if performances:
                avg_performance = sum(performances) / len(performances)
                if avg_performance > best_performance:
                    best_policy = policy_name
                    best_performance = avg_performance

        # Switch policy if significant improvement
        if best_policy != self.current_policy and best_performance > current_hit_rate * 1.05:
            self.logger.info(f"Switching eviction policy from {self.current_policy} to {best_policy}")
            self.current_policy = best_policy

            # Migrate current entries to new policy
            self._migrate_to_policy(best_policy)

    def _migrate_to_policy(self, new_policy: str) -> None:
        """Migrate current cache state to new policy."""
        # Get current entries
        current_entries = dict(self.policies[self.current_policy].entries)

        # Clear all policies
        for policy in self.policies.values():
            policy.entries.clear()
            policy.eviction_count = 0

        # Repopulate new policy
        target_policy = self.policies[new_policy]
        for key, entry in current_entries.items():
            target_policy.put(key, entry.value, entry.size)

        # Update global entries reference
        self.entries = target_policy.entries


class EvictionPolicyFactory:
    """Factory for creating eviction policy instances."""

    @staticmethod
    def create_policy(policy_type: EvictionPolicy, capacity: int, logger: logging.Logger | None = None) -> BaseEvictionPolicy:
        """Create eviction policy instance."""
        policy_map = {
            EvictionPolicy.LRU: LRUEvictionPolicy,
            EvictionPolicy.LFU: LFUEvictionPolicy,
            EvictionPolicy.ARC: ARCEvictionPolicy,
            EvictionPolicy.ADAPTIVE: AdaptiveEvictionPolicy,
        }

        if policy_type not in policy_map:
            raise ValueError(f"Unsupported eviction policy: {policy_type}")

        return policy_map[policy_type](capacity, logger)

    @staticmethod
    def get_available_policies() -> list[str]:
        """Get list of available eviction policies."""
        return [policy.value for policy in EvictionPolicy]


class EvictionOptimizer:
    """Optimizer for cache eviction policies and parameters."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize eviction optimizer."""
        self.logger = logger or logging.getLogger(__name__)
        self.performance_history: dict[str, list[dict[str, Any]]] = {}

    def optimize_policy_parameters(self, policy: BaseEvictionPolicy, workload_stats: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize eviction policy parameters based on workload characteristics.

        Args:
            policy: Current eviction policy
            workload_stats: Workload statistics

        Returns:
            Recommended parameter adjustments
        """
        recommendations = {"policy_type": policy.__class__.__name__, "current_performance": policy.get_stats(), "recommendations": []}

        # Analyze hit rate patterns
        hit_rate = policy.get_stats()["hit_rate"]

        if hit_rate < 0.5:
            recommendations["recommendations"].append(
                {"type": "capacity", "message": "Consider increasing cache capacity - low hit rate detected", "suggested_increase": "25%"}
            )

        # Analyze access patterns
        if "temporal_locality" in workload_stats:
            temporal_locality = workload_stats["temporal_locality"]

            if temporal_locality > 0.8 and not isinstance(policy, LRUEvictionPolicy):
                recommendations["recommendations"].append(
                    {"type": "policy", "message": "High temporal locality detected - LRU policy recommended", "suggested_policy": "LRU"}
                )
            elif temporal_locality < 0.3 and not isinstance(policy, LFUEvictionPolicy):
                recommendations["recommendations"].append(
                    {"type": "policy", "message": "Low temporal locality detected - LFU policy recommended", "suggested_policy": "LFU"}
                )

        # Analyze frequency patterns
        if "access_frequency_distribution" in workload_stats:
            freq_dist = workload_stats["access_frequency_distribution"]

            if freq_dist.get("highly_skewed", False):
                recommendations["recommendations"].append(
                    {
                        "type": "policy",
                        "message": "Highly skewed access pattern - consider frequency-based eviction",
                        "suggested_policy": "LFU",
                    }
                )

        return recommendations

    def analyze_workload_characteristics(self, access_log: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze workload characteristics from access log.

        Args:
            access_log: List of access events

        Returns:
            Workload analysis results
        """
        if not access_log:
            return {}

        # Calculate temporal locality
        recent_accesses = {}
        temporal_hits = 0

        for i, access in enumerate(access_log):
            key = access.get("key")
            if key in recent_accesses and i - recent_accesses[key] <= 10:
                temporal_hits += 1
            recent_accesses[key] = i

        temporal_locality = temporal_hits / len(access_log) if access_log else 0.0

        # Calculate access frequency distribution
        access_counts = {}
        for access in access_log:
            key = access.get("key")
            access_counts[key] = access_counts.get(key, 0) + 1

        frequencies = list(access_counts.values())
        if frequencies:
            freq_std = (sum((f - sum(frequencies) / len(frequencies)) ** 2 for f in frequencies) / len(frequencies)) ** 0.5
            freq_mean = sum(frequencies) / len(frequencies)
            freq_cv = freq_std / freq_mean if freq_mean > 0 else 0
        else:
            freq_cv = 0

        return {
            "total_accesses": len(access_log),
            "unique_keys": len(access_counts),
            "temporal_locality": temporal_locality,
            "access_frequency_distribution": {
                "mean": freq_mean if frequencies else 0,
                "std_dev": freq_std if frequencies else 0,
                "coefficient_of_variation": freq_cv,
                "highly_skewed": freq_cv > 1.0,
            },
            "working_set_size": len(access_counts),
        }
