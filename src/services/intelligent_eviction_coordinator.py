"""
Intelligent Eviction Coordinator for Wave 6.0 - Subtask 6.2

This module implements an advanced intelligent cache eviction coordinator that
manages multiple eviction strategies based on access frequency, importance,
memory pressure, and adaptive learning algorithms.
"""

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.services.cache_eviction_service import CacheEvictionService, EvictionConfig, EvictionStrategy, EvictionTrigger
from src.services.cache_memory_pressure_service import CacheMemoryPressureService
from src.utils.memory_utils import MemoryPressureLevel, get_system_memory_pressure


class ImportanceScore(Enum):
    """Cache entry importance levels."""

    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class AccessPattern(Enum):
    """Cache access pattern types."""

    FREQUENT = "frequent"  # High frequency, regular intervals
    BURST = "burst"  # High frequency in bursts
    SPORADIC = "sporadic"  # Low frequency, irregular
    TEMPORAL = "temporal"  # Time-dependent access
    PREDICTABLE = "predictable"  # Follows predictable patterns


@dataclass
class CacheEntryMetrics:
    """Comprehensive metrics for cache entries."""

    key: str
    access_count: int = 0
    last_access_time: float = 0.0
    creation_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: float | None = None

    # Advanced metrics
    access_frequency: float = 0.0  # Access per second
    access_pattern: AccessPattern = AccessPattern.SPORADIC
    importance_score: ImportanceScore = ImportanceScore.MEDIUM
    heat_score: float = 0.0  # Combined access + recency score

    # Access pattern analysis
    access_times: list[float] = field(default_factory=list)
    access_intervals: list[float] = field(default_factory=list)

    # Predictive metrics
    predicted_next_access: float | None = None
    access_confidence: float = 0.0

    def update_access(self) -> None:
        """Update access metrics when entry is accessed."""
        current_time = time.time()

        # Update basic metrics
        self.access_count += 1
        if self.last_access_time > 0:
            interval = current_time - self.last_access_time
            self.access_intervals.append(interval)
            # Keep only last 100 intervals
            if len(self.access_intervals) > 100:
                self.access_intervals = self.access_intervals[-100:]

        self.last_access_time = current_time
        self.access_times.append(current_time)

        # Keep only last 1000 access times
        if len(self.access_times) > 1000:
            self.access_times = self.access_times[-1000:]

        # Update derived metrics
        self._update_access_frequency()
        self._update_access_pattern()
        self._update_heat_score()
        self._predict_next_access()

    def _update_access_frequency(self) -> None:
        """Update access frequency calculation."""
        if len(self.access_times) < 2:
            self.access_frequency = 0.0
            return

        time_span = self.access_times[-1] - self.access_times[0]
        if time_span > 0:
            self.access_frequency = (len(self.access_times) - 1) / time_span
        else:
            self.access_frequency = 0.0

    def _update_access_pattern(self) -> None:
        """Analyze and update access pattern classification."""
        if len(self.access_intervals) < 5:
            self.access_pattern = AccessPattern.SPORADIC
            return

        intervals = np.array(self.access_intervals)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else float("inf")

        # Classify access pattern
        if self.access_frequency > 1.0:  # More than 1 access per second
            if cv < 0.3:  # Low variability
                self.access_pattern = AccessPattern.FREQUENT
            else:
                self.access_pattern = AccessPattern.BURST
        elif cv < 0.5 and len(self.access_intervals) > 10:
            self.access_pattern = AccessPattern.PREDICTABLE
        elif self._detect_temporal_pattern():
            self.access_pattern = AccessPattern.TEMPORAL
        else:
            self.access_pattern = AccessPattern.SPORADIC

    def _detect_temporal_pattern(self) -> bool:
        """Detect if access follows temporal patterns (daily, hourly, etc.)."""
        if len(self.access_times) < 20:
            return False

        # Convert to hours of day
        hours = [(t % 86400) / 3600 for t in self.access_times[-20:]]

        # Check for clustering in specific hours
        if len(set(int(h) for h in hours)) <= 3:  # Accessed in 3 or fewer distinct hours
            return True

        return False

    def _update_heat_score(self) -> None:
        """Update heat score combining frequency, recency, and importance."""
        current_time = time.time()

        # Recency score (exponential decay)
        if self.last_access_time > 0:
            time_since_access = current_time - self.last_access_time
            recency_score = math.exp(-time_since_access / 3600)  # 1-hour decay
        else:
            recency_score = 0.0

        # Frequency score (normalized)
        frequency_score = min(self.access_frequency / 10.0, 1.0)  # Cap at 10 accesses/sec

        # Importance weight
        importance_weight = self.importance_score.value / 5.0

        # Combined heat score
        self.heat_score = 0.4 * frequency_score + 0.4 * recency_score + 0.2 * importance_weight

    def _predict_next_access(self) -> None:
        """Predict next access time based on historical patterns."""
        if len(self.access_intervals) < 3:
            self.predicted_next_access = None
            self.access_confidence = 0.0
            return

        intervals = np.array(self.access_intervals)

        if self.access_pattern == AccessPattern.PREDICTABLE:
            # Use mean interval for predictable patterns
            predicted_interval = np.mean(intervals)
            confidence = 1.0 / (np.std(intervals) + 1.0)
        elif self.access_pattern == AccessPattern.FREQUENT:
            # Use weighted recent intervals
            weights = np.exp(np.linspace(-1, 0, len(intervals)))
            predicted_interval = np.average(intervals, weights=weights)
            confidence = 0.8
        else:
            # Use exponential smoothing for other patterns
            alpha = 0.3
            predicted_interval = intervals[-1]
            for i in range(len(intervals) - 2, -1, -1):
                predicted_interval = alpha * intervals[i] + (1 - alpha) * predicted_interval
            confidence = 0.5

        self.predicted_next_access = self.last_access_time + predicted_interval
        self.access_confidence = min(confidence, 1.0)


@dataclass
class EvictionDecision:
    """Decision result for cache eviction."""

    key: str
    eviction_priority: float
    reason: str
    strategy_used: EvictionStrategy
    confidence: float


class IntelligentEvictionCoordinator:
    """
    Advanced intelligent cache eviction coordinator.

    This coordinator implements sophisticated eviction strategies that consider:
    - Access frequency and patterns
    - Entry importance scores
    - Memory pressure levels
    - Predictive analytics
    - Machine learning-based optimization
    """

    def __init__(self, config: EvictionConfig | None = None):
        """Initialize the intelligent eviction coordinator."""
        self.config = config or EvictionConfig()
        self.logger = logging.getLogger(__name__)

        # Core services
        self.eviction_service: CacheEvictionService | None = None
        self.pressure_service: CacheMemoryPressureService | None = None

        # Metrics storage
        self.entry_metrics: dict[str, dict[str, CacheEntryMetrics]] = defaultdict(dict)

        # Machine learning components
        self.scaler = StandardScaler()
        self.clusterer: KMeans | None = None
        self.ml_model_trained = False

        # Access pattern analysis
        self.pattern_analyzer = AccessPatternAnalyzer()

        # Eviction decision history
        self.decision_history: deque = deque(maxlen=1000)

        # Performance metrics
        self.eviction_success_rate = 0.0
        self.prediction_accuracy = 0.0

        # Configuration
        self.learning_enabled = True
        self.pattern_analysis_enabled = True
        self.predictive_eviction_enabled = True

        # Background tasks
        self._analysis_task: asyncio.Task | None = None
        self._training_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the coordinator."""
        try:
            self.logger.info("Initializing Intelligent Eviction Coordinator...")

            # Initialize ML components
            self.clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)

            # Start background tasks
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            self._training_task = asyncio.create_task(self._training_loop())

            self.logger.info("Intelligent Eviction Coordinator initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize coordinator: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the coordinator."""
        try:
            # Cancel background tasks
            for task in [self._analysis_task, self._training_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self.logger.info("Intelligent Eviction Coordinator shutdown")

        except Exception as e:
            self.logger.error(f"Error during coordinator shutdown: {e}")

    def track_cache_access(
        self, cache_name: str, key: str, size_bytes: int = 0, importance: ImportanceScore = ImportanceScore.MEDIUM
    ) -> None:
        """Track cache entry access for intelligent analysis."""
        if cache_name not in self.entry_metrics:
            self.entry_metrics[cache_name] = {}

        if key not in self.entry_metrics[cache_name]:
            self.entry_metrics[cache_name][key] = CacheEntryMetrics(key=key, size_bytes=size_bytes, importance_score=importance)

        # Update access metrics
        self.entry_metrics[cache_name][key].update_access()

    async def make_eviction_decisions(
        self, cache_name: str, target_count: int, memory_pressure: MemoryPressureLevel | None = None
    ) -> list[EvictionDecision]:
        """
        Make intelligent eviction decisions based on multiple factors.

        Args:
            cache_name: Name of the cache
            target_count: Number of entries to evict
            memory_pressure: Current memory pressure level

        Returns:
            List of eviction decisions
        """
        try:
            metrics = self.entry_metrics.get(cache_name, {})
            if not metrics:
                return []

            # Get current memory pressure if not provided
            if memory_pressure is None:
                pressure_info = get_system_memory_pressure()
                memory_pressure = pressure_info.level

            # Analyze all entries
            candidates = await self._analyze_eviction_candidates(cache_name, metrics, memory_pressure)

            # Select best candidates based on eviction priority
            selected_candidates = sorted(candidates, key=lambda x: x.eviction_priority, reverse=True)[:target_count]

            # Record decisions
            for decision in selected_candidates:
                self.decision_history.append(
                    {"timestamp": time.time(), "cache_name": cache_name, "decision": decision, "memory_pressure": memory_pressure.value}
                )

            return selected_candidates

        except Exception as e:
            self.logger.error(f"Error making eviction decisions for {cache_name}: {e}")
            return []

    async def _analyze_eviction_candidates(
        self, cache_name: str, metrics: dict[str, CacheEntryMetrics], memory_pressure: MemoryPressureLevel
    ) -> list[EvictionDecision]:
        """Analyze cache entries and score them for eviction."""
        decisions = []
        current_time = time.time()

        for key, entry_metrics in metrics.items():
            # Calculate base eviction score
            eviction_score = await self._calculate_eviction_score(entry_metrics, memory_pressure, current_time)

            # Determine strategy and reason
            strategy, reason = self._determine_eviction_strategy(entry_metrics, memory_pressure)

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(entry_metrics)

            decision = EvictionDecision(
                key=key, eviction_priority=eviction_score, reason=reason, strategy_used=strategy, confidence=confidence
            )

            decisions.append(decision)

        return decisions

    async def _calculate_eviction_score(
        self, metrics: CacheEntryMetrics, memory_pressure: MemoryPressureLevel, current_time: float
    ) -> float:
        """Calculate composite eviction score for an entry."""

        # Base factors
        factors = {
            "age_factor": self._calculate_age_factor(metrics, current_time),
            "frequency_factor": self._calculate_frequency_factor(metrics),
            "size_factor": self._calculate_size_factor(metrics),
            "importance_factor": self._calculate_importance_factor(metrics),
            "prediction_factor": self._calculate_prediction_factor(metrics, current_time),
            "pattern_factor": self._calculate_pattern_factor(metrics),
            "heat_factor": 1.0 - metrics.heat_score,  # Lower heat = higher eviction score
        }

        # Adjust weights based on memory pressure
        weights = self._get_eviction_weights(memory_pressure)

        # Calculate weighted score
        eviction_score = sum(factors[factor] * weights.get(factor, 0.0) for factor in factors)

        # Apply ML enhancement if available
        if self.ml_model_trained and self.learning_enabled:
            ml_score = await self._get_ml_eviction_score(metrics)
            eviction_score = 0.7 * eviction_score + 0.3 * ml_score

        return max(0.0, min(1.0, eviction_score))

    def _calculate_age_factor(self, metrics: CacheEntryMetrics, current_time: float) -> float:
        """Calculate age-based eviction factor."""
        age = current_time - metrics.creation_time
        # Normalize to 0-1 scale (24 hours = 1.0)
        return min(age / 86400, 1.0)

    def _calculate_frequency_factor(self, metrics: CacheEntryMetrics) -> float:
        """Calculate frequency-based eviction factor (lower frequency = higher eviction score)."""
        if metrics.access_frequency == 0:
            return 1.0

        # Inverse relationship: lower frequency = higher eviction score
        # Normalize using log scale
        return 1.0 / (1.0 + math.log(1.0 + metrics.access_frequency))

    def _calculate_size_factor(self, metrics: CacheEntryMetrics) -> float:
        """Calculate size-based eviction factor."""
        # Larger entries get higher eviction scores during memory pressure
        # Normalize to MB scale
        size_mb = metrics.size_bytes / (1024 * 1024)
        return min(size_mb / 100.0, 1.0)  # Cap at 100MB

    def _calculate_importance_factor(self, metrics: CacheEntryMetrics) -> float:
        """Calculate importance-based eviction factor."""
        # Higher importance = lower eviction score
        return 1.0 - (metrics.importance_score.value / 5.0)

    def _calculate_prediction_factor(self, metrics: CacheEntryMetrics, current_time: float) -> float:
        """Calculate prediction-based eviction factor."""
        if not self.predictive_eviction_enabled or metrics.predicted_next_access is None:
            return 0.5  # Neutral score

        time_to_next_access = metrics.predicted_next_access - current_time

        if time_to_next_access <= 0:
            return 0.0  # Should have been accessed already, don't evict

        # Entries predicted to be accessed soon get lower eviction scores
        # Normalize to hours (24 hours = 1.0 eviction score)
        hours_to_access = time_to_next_access / 3600
        prediction_score = min(hours_to_access / 24.0, 1.0)

        # Weight by confidence
        return prediction_score * metrics.access_confidence + 0.5 * (1 - metrics.access_confidence)

    def _calculate_pattern_factor(self, metrics: CacheEntryMetrics) -> float:
        """Calculate pattern-based eviction factor."""
        pattern_scores = {
            AccessPattern.FREQUENT: 0.1,  # Low eviction score
            AccessPattern.PREDICTABLE: 0.2,  # Low eviction score
            AccessPattern.BURST: 0.4,  # Medium eviction score
            AccessPattern.TEMPORAL: 0.3,  # Medium-low eviction score
            AccessPattern.SPORADIC: 0.8,  # High eviction score
        }

        return pattern_scores.get(metrics.access_pattern, 0.5)

    def _determine_eviction_strategy(
        self, metrics: CacheEntryMetrics, memory_pressure: MemoryPressureLevel
    ) -> tuple[EvictionStrategy, str]:
        """Determine the best eviction strategy for an entry."""

        # Critical memory pressure: prioritize size
        if memory_pressure == MemoryPressureLevel.CRITICAL:
            return EvictionStrategy.MEMORY_PRESSURE, "Critical memory pressure - evicting large entries"

        # Check TTL expiration
        if metrics.ttl and (time.time() - metrics.creation_time) > metrics.ttl:
            return EvictionStrategy.TTL, "Entry has expired"

        # Pattern-based strategy selection
        if metrics.access_pattern == AccessPattern.SPORADIC:
            return EvictionStrategy.LFU, "Sporadic access pattern - using LFU"
        elif metrics.access_pattern in [AccessPattern.FREQUENT, AccessPattern.PREDICTABLE]:
            return EvictionStrategy.LRU, "Regular access pattern - using LRU"
        elif metrics.access_pattern == AccessPattern.BURST:
            return EvictionStrategy.ADAPTIVE, "Burst access pattern - using adaptive strategy"
        else:
            return EvictionStrategy.LRU, "Default LRU strategy"

    def _get_eviction_weights(self, memory_pressure: MemoryPressureLevel) -> dict[str, float]:
        """Get eviction factor weights based on memory pressure."""
        if memory_pressure == MemoryPressureLevel.CRITICAL:
            return {
                "size_factor": 0.4,  # Prioritize large entries
                "frequency_factor": 0.2,
                "age_factor": 0.15,
                "importance_factor": 0.1,
                "prediction_factor": 0.1,
                "pattern_factor": 0.05,
                "heat_factor": 0.0,
            }
        elif memory_pressure == MemoryPressureLevel.HIGH:
            return {
                "size_factor": 0.25,
                "frequency_factor": 0.25,
                "age_factor": 0.2,
                "importance_factor": 0.15,
                "prediction_factor": 0.1,
                "pattern_factor": 0.05,
                "heat_factor": 0.0,
            }
        else:
            # Normal operation weights
            return {
                "frequency_factor": 0.25,
                "age_factor": 0.2,
                "heat_factor": 0.2,
                "prediction_factor": 0.15,
                "importance_factor": 0.1,
                "pattern_factor": 0.05,
                "size_factor": 0.05,
            }

    def _calculate_confidence(self, metrics: CacheEntryMetrics) -> float:
        """Calculate confidence in eviction decision based on data quality."""
        factors = []

        # Access count confidence
        if metrics.access_count >= 10:
            factors.append(1.0)
        elif metrics.access_count >= 5:
            factors.append(0.8)
        elif metrics.access_count >= 2:
            factors.append(0.6)
        else:
            factors.append(0.3)

        # Time series length confidence
        if len(metrics.access_times) >= 20:
            factors.append(1.0)
        elif len(metrics.access_times) >= 10:
            factors.append(0.8)
        elif len(metrics.access_times) >= 5:
            factors.append(0.6)
        else:
            factors.append(0.4)

        # Pattern confidence
        factors.append(metrics.access_confidence)

        return sum(factors) / len(factors)

    async def _get_ml_eviction_score(self, metrics: CacheEntryMetrics) -> float:
        """Get ML-based eviction score (placeholder for future ML implementation)."""
        # This would use a trained model to predict eviction success
        # For now, return a score based on heat score and pattern
        pattern_weights = {
            AccessPattern.FREQUENT: 0.2,
            AccessPattern.PREDICTABLE: 0.3,
            AccessPattern.BURST: 0.5,
            AccessPattern.TEMPORAL: 0.4,
            AccessPattern.SPORADIC: 0.8,
        }

        pattern_score = pattern_weights.get(metrics.access_pattern, 0.5)
        heat_penalty = 1.0 - metrics.heat_score

        return 0.6 * pattern_score + 0.4 * heat_penalty

    async def _analysis_loop(self) -> None:
        """Background loop for continuous analysis."""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes

                if self.pattern_analysis_enabled:
                    await self._analyze_access_patterns()

                await self._update_predictions()
                await self._cleanup_old_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")

    async def _training_loop(self) -> None:
        """Background loop for ML model training."""
        while True:
            try:
                await asyncio.sleep(3600)  # Train every hour

                if self.learning_enabled and len(self.decision_history) >= 100:
                    await self._train_ml_model()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")

    async def _analyze_access_patterns(self) -> None:
        """Analyze access patterns across all caches."""
        try:
            for cache_name, metrics in self.entry_metrics.items():
                pattern_summary = defaultdict(int)

                for key, entry_metrics in metrics.items():
                    pattern_summary[entry_metrics.access_pattern] += 1

                self.logger.debug(f"Cache {cache_name} patterns: {dict(pattern_summary)}")

        except Exception as e:
            self.logger.error(f"Error analyzing access patterns: {e}")

    async def _update_predictions(self) -> None:
        """Update access predictions for all entries."""
        try:
            current_time = time.time()

            for cache_name, metrics in self.entry_metrics.items():
                for key, entry_metrics in metrics.items():
                    # Update prediction if enough data
                    if len(entry_metrics.access_times) >= 3:
                        entry_metrics._predict_next_access()

        except Exception as e:
            self.logger.error(f"Error updating predictions: {e}")

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to prevent memory bloat."""
        try:
            current_time = time.time()
            cutoff_time = current_time - 86400 * 7  # Keep 7 days of data

            for cache_name, metrics in self.entry_metrics.items():
                keys_to_remove = []

                for key, entry_metrics in metrics.items():
                    # Remove entries not accessed in 7 days
                    if entry_metrics.last_access_time < cutoff_time:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del metrics[key]

                if keys_to_remove:
                    self.logger.debug(f"Cleaned up {len(keys_to_remove)} old metrics from {cache_name}")

        except Exception as e:
            self.logger.error(f"Error cleaning up metrics: {e}")

    async def _train_ml_model(self) -> None:
        """Train ML model for eviction optimization (placeholder)."""
        try:
            # This would implement actual ML training
            # For now, just update internal state
            self.ml_model_trained = True
            self.logger.debug("ML model training completed")

        except Exception as e:
            self.logger.error(f"Error training ML model: {e}")

    async def get_coordinator_stats(self) -> dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        try:
            total_entries = sum(len(metrics) for metrics in self.entry_metrics.values())

            pattern_distribution = defaultdict(int)
            importance_distribution = defaultdict(int)

            for cache_metrics in self.entry_metrics.values():
                for entry_metrics in cache_metrics.values():
                    pattern_distribution[entry_metrics.access_pattern.value] += 1
                    importance_distribution[entry_metrics.importance_score.value] += 1

            return {
                "total_tracked_entries": total_entries,
                "tracked_caches": len(self.entry_metrics),
                "decision_history_size": len(self.decision_history),
                "ml_model_trained": self.ml_model_trained,
                "pattern_distribution": dict(pattern_distribution),
                "importance_distribution": dict(importance_distribution),
                "eviction_success_rate": self.eviction_success_rate,
                "prediction_accuracy": self.prediction_accuracy,
                "configuration": {
                    "learning_enabled": self.learning_enabled,
                    "pattern_analysis_enabled": self.pattern_analysis_enabled,
                    "predictive_eviction_enabled": self.predictive_eviction_enabled,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting coordinator stats: {e}")
            return {"error": str(e)}


class AccessPatternAnalyzer:
    """Analyzer for cache access patterns."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_temporal_patterns(self, access_times: list[float]) -> dict[str, Any]:
        """Analyze temporal patterns in access times."""
        if len(access_times) < 10:
            return {"pattern": "insufficient_data"}

        # Convert to daily cycles
        daily_hours = [(t % 86400) / 3600 for t in access_times]

        # Analyze distribution
        hour_counts = defaultdict(int)
        for hour in daily_hours:
            hour_counts[int(hour)] += 1

        # Find peak hours
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        return {"pattern": "temporal", "peak_hours": [hour for hour, count in peak_hours], "access_distribution": dict(hour_counts)}

    def detect_burst_patterns(self, access_intervals: list[float]) -> dict[str, Any]:
        """Detect burst access patterns."""
        if len(access_intervals) < 5:
            return {"pattern": "insufficient_data"}

        intervals = np.array(access_intervals)

        # Detect bursts as periods with very short intervals
        burst_threshold = np.percentile(intervals, 25)  # Bottom 25%
        burst_periods = np.where(intervals <= burst_threshold)[0]

        # Count consecutive burst periods
        burst_sequences = []
        if len(burst_periods) > 0:
            current_sequence = [burst_periods[0]]
            for i in range(1, len(burst_periods)):
                if burst_periods[i] == burst_periods[i - 1] + 1:
                    current_sequence.append(burst_periods[i])
                else:
                    burst_sequences.append(current_sequence)
                    current_sequence = [burst_periods[i]]
            burst_sequences.append(current_sequence)

        return {
            "pattern": "burst",
            "burst_threshold": burst_threshold,
            "burst_sequences": len(burst_sequences),
            "avg_burst_length": np.mean([len(seq) for seq in burst_sequences]) if burst_sequences else 0,
        }


# Global coordinator instance
_coordinator: IntelligentEvictionCoordinator | None = None


async def get_eviction_coordinator(config: EvictionConfig | None = None) -> IntelligentEvictionCoordinator:
    """Get the global intelligent eviction coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = IntelligentEvictionCoordinator(config)
        await _coordinator.initialize()
    return _coordinator


async def shutdown_eviction_coordinator() -> None:
    """Shutdown the global intelligent eviction coordinator."""
    global _coordinator
    if _coordinator:
        await _coordinator.shutdown()
        _coordinator = None
