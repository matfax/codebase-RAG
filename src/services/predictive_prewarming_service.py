"""
Predictive Pre-warming Service for Wave 6.0 - Subtask 6.4

This module implements an advanced predictive cache pre-warming mechanism that
uses machine learning, access pattern analysis, and intelligent scheduling
to proactively load frequently used data and improve response times.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.config.cache_config import CacheConfig, get_global_cache_config
from src.services.cache_warmup_service import CacheWarmupService, WarmupStrategy
from src.services.intelligent_eviction_coordinator import AccessPattern, IntelligentEvictionCoordinator
from src.utils.memory_utils import MemoryPressureLevel, get_system_memory_pressure


class PredictionModel(Enum):
    """Types of prediction models."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PATTERN_MATCHING = "pattern_matching"


class PrewarmingTrigger(Enum):
    """Triggers for cache pre-warming."""

    SCHEDULED = "scheduled"
    PREDICTED_ACCESS = "predicted_access"
    PATTERN_BASED = "pattern_based"
    DEPENDENCY_DRIVEN = "dependency_driven"
    MANUAL = "manual"
    MEMORY_AVAILABLE = "memory_available"


@dataclass
class AccessPrediction:
    """Prediction for future cache access."""

    key: str
    cache_name: str
    predicted_access_time: float
    confidence: float
    priority: int
    estimated_size_mb: float
    access_pattern: AccessPattern
    dependencies: set[str] = field(default_factory=set)

    @property
    def time_to_access(self) -> float:
        """Time until predicted access in seconds."""
        return max(0, self.predicted_access_time - time.time())

    @property
    def should_prewarm_now(self) -> bool:
        """Check if item should be pre-warmed now."""
        # Pre-warm items predicted to be accessed within next hour
        return self.time_to_access <= 3600 and self.confidence >= 0.6


@dataclass
class PrewarmingJob:
    """Pre-warming job with scheduling information."""

    prediction: AccessPrediction
    scheduled_time: float
    trigger: PrewarmingTrigger
    priority_score: float
    status: str = "pending"  # pending, running, completed, failed

    @property
    def is_ready(self) -> bool:
        """Check if job is ready to execute."""
        return time.time() >= self.scheduled_time and self.status == "pending"


@dataclass
class PrewarmingStats:
    """Statistics for pre-warming operations."""

    total_predictions: int = 0
    successful_predictions: int = 0
    false_positives: int = 0
    cache_hits_after_prewarm: int = 0
    total_prewarmed_mb: float = 0.0
    prediction_accuracy: float = 0.0
    response_time_improvement: float = 0.0
    last_updated: float = field(default_factory=time.time)


class AccessPatternAnalyzer:
    """Analyzes historical access patterns for prediction."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.access_history: dict[str, list[float]] = defaultdict(list)
        self.pattern_cache: dict[str, dict] = {}

    def record_access(self, cache_name: str, key: str, timestamp: float = None) -> None:
        """Record cache access for pattern analysis."""
        if timestamp is None:
            timestamp = time.time()

        access_key = f"{cache_name}:{key}"
        self.access_history[access_key].append(timestamp)

        # Keep only recent history (last 1000 accesses)
        if len(self.access_history[access_key]) > 1000:
            self.access_history[access_key] = self.access_history[access_key][-1000:]

        # Invalidate pattern cache for this key
        self.pattern_cache.pop(access_key, None)

    def analyze_pattern(self, cache_name: str, key: str) -> dict[str, Any]:
        """Analyze access pattern for a specific cache key."""
        access_key = f"{cache_name}:{key}"

        # Check cache first
        if access_key in self.pattern_cache:
            return self.pattern_cache[access_key]

        accesses = self.access_history.get(access_key, [])
        if len(accesses) < 3:
            return {"pattern_type": "insufficient_data", "confidence": 0.0}

        # Calculate intervals between accesses
        intervals = np.diff(accesses)

        # Analyze pattern characteristics
        pattern_analysis = {
            "access_count": len(accesses),
            "mean_interval": np.mean(intervals),
            "std_interval": np.std(intervals),
            "coefficient_of_variation": np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float("inf"),
            "min_interval": np.min(intervals),
            "max_interval": np.max(intervals),
            "last_access": accesses[-1],
            "pattern_type": self._classify_pattern(intervals),
            "confidence": self._calculate_pattern_confidence(intervals),
        }

        # Cache the analysis
        self.pattern_cache[access_key] = pattern_analysis

        return pattern_analysis

    def _classify_pattern(self, intervals: np.ndarray) -> str:
        """Classify the access pattern type."""
        if len(intervals) < 3:
            return "insufficient_data"

        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float("inf")
        mean_interval = np.mean(intervals)

        # Regular pattern (low coefficient of variation)
        if cv < 0.3:
            if mean_interval < 3600:  # Less than 1 hour
                return "regular_frequent"
            elif mean_interval < 86400:  # Less than 1 day
                return "regular_daily"
            else:
                return "regular_periodic"

        # Burst pattern (short intervals followed by long gaps)
        elif cv > 2.0:
            short_intervals = intervals[intervals < np.percentile(intervals, 25)]
            if len(short_intervals) > len(intervals) * 0.3:
                return "burst"

        # Check for time-based patterns (hourly, daily)
        if self._detect_temporal_pattern(intervals):
            return "temporal"

        return "irregular"

    def _calculate_pattern_confidence(self, intervals: np.ndarray) -> float:
        """Calculate confidence in the pattern analysis."""
        if len(intervals) < 5:
            return 0.3

        # More data points = higher confidence
        data_confidence = min(len(intervals) / 50.0, 1.0)

        # Lower coefficient of variation = higher confidence
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float("inf")
        pattern_confidence = 1.0 / (1.0 + cv)

        return (data_confidence + pattern_confidence) / 2.0

    def _detect_temporal_pattern(self, intervals: np.ndarray) -> bool:
        """Detect if intervals follow temporal patterns (hourly, daily)."""
        # Check for common temporal intervals
        common_periods = [3600, 86400, 604800]  # 1 hour, 1 day, 1 week

        for period in common_periods:
            # Check if intervals cluster around multiples of the period
            normalized = intervals % period
            if np.std(normalized) < period * 0.1:  # Within 10% of period
                return True

        return False


class PredictionEngine:
    """Machine learning engine for access predictions."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: dict[PredictionModel, Any] = {}
        self.scalers: dict[PredictionModel, StandardScaler] = {}
        self.training_data: list[dict] = []
        self.model_performance: dict[PredictionModel, float] = {}

        # Initialize models
        self._initialize_models()

        # Training parameters
        self.min_training_samples = 100
        self.retrain_interval = 3600  # 1 hour
        self.last_training_time = 0.0

    def _initialize_models(self) -> None:
        """Initialize prediction models."""
        self.models[PredictionModel.LINEAR_REGRESSION] = LinearRegression()
        self.models[PredictionModel.RANDOM_FOREST] = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

        # Initialize scalers
        for model_type in [PredictionModel.LINEAR_REGRESSION, PredictionModel.RANDOM_FOREST]:
            self.scalers[model_type] = StandardScaler()

    def add_training_sample(self, features: dict[str, float], actual_next_access: float) -> None:
        """Add a training sample to the dataset."""
        sample = {"features": features, "target": actual_next_access, "timestamp": time.time()}

        self.training_data.append(sample)

        # Keep only recent training data (last 10000 samples)
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]

    async def train_models(self) -> dict[str, Any]:
        """Train all prediction models."""
        if len(self.training_data) < self.min_training_samples:
            return {"error": "Insufficient training data"}

        try:
            # Prepare training data
            X, y = self._prepare_training_data()

            if len(X) == 0:
                return {"error": "No valid training data"}

            training_results = {}

            # Train each model
            for model_type, model in self.models.items():
                if model_type in [PredictionModel.LINEAR_REGRESSION, PredictionModel.RANDOM_FOREST]:
                    try:
                        # Scale features
                        scaler = self.scalers[model_type]
                        X_scaled = scaler.fit_transform(X)

                        # Train model
                        model.fit(X_scaled, y)

                        # Calculate performance (R² score)
                        score = model.score(X_scaled, y)
                        self.model_performance[model_type] = score

                        training_results[model_type.value] = {"success": True, "r2_score": score, "samples_used": len(X)}

                    except Exception as e:
                        self.logger.error(f"Failed to train {model_type.value}: {e}")
                        training_results[model_type.value] = {"success": False, "error": str(e)}

            self.last_training_time = time.time()

            return {
                "success": True,
                "models_trained": len([r for r in training_results.values() if r.get("success")]),
                "training_results": training_results,
                "training_samples": len(X),
            }

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}

    def predict_next_access(self, features: dict[str, float], model_type: PredictionModel = None) -> tuple[float, float]:
        """
        Predict next access time for given features.

        Returns:
            Tuple of (predicted_time, confidence)
        """
        if model_type is None:
            # Use best performing model
            model_type = self._get_best_model()

        if model_type not in self.models:
            return 0.0, 0.0

        try:
            # Prepare features
            feature_vector = self._prepare_feature_vector(features)

            if model_type in [PredictionModel.LINEAR_REGRESSION, PredictionModel.RANDOM_FOREST]:
                model = self.models[model_type]
                scaler = self.scalers[model_type]

                # Scale features
                feature_scaled = scaler.transform([feature_vector])

                # Predict
                prediction = model.predict(feature_scaled)[0]
                confidence = self.model_performance.get(model_type, 0.0)

                return prediction, confidence

            elif model_type == PredictionModel.MOVING_AVERAGE:
                return self._moving_average_prediction(features)

            elif model_type == PredictionModel.EXPONENTIAL_SMOOTHING:
                return self._exponential_smoothing_prediction(features)

            else:
                return 0.0, 0.0

        except Exception as e:
            self.logger.error(f"Prediction failed with {model_type.value}: {e}")
            return 0.0, 0.0

    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training."""
        X, y = [], []

        for sample in self.training_data:
            features = sample["features"]
            target = sample["target"]

            feature_vector = self._prepare_feature_vector(features)

            X.append(feature_vector)
            y.append(target)

        return np.array(X), np.array(y)

    def _prepare_feature_vector(self, features: dict[str, float]) -> list[float]:
        """Prepare feature vector for model input."""
        # Standard feature order
        feature_names = [
            "mean_interval",
            "std_interval",
            "cv",
            "last_access_age",
            "access_count",
            "hour_of_day",
            "day_of_week",
            "pattern_confidence",
        ]

        return [features.get(name, 0.0) for name in feature_names]

    def _get_best_model(self) -> PredictionModel:
        """Get the best performing model."""
        if not self.model_performance:
            return PredictionModel.MOVING_AVERAGE

        best_model = max(self.model_performance.items(), key=lambda x: x[1])
        return best_model[0]

    def _moving_average_prediction(self, features: dict[str, float]) -> tuple[float, float]:
        """Simple moving average prediction."""
        mean_interval = features.get("mean_interval", 3600)
        last_access_age = features.get("last_access_age", 0)

        # Predict next access based on mean interval
        predicted_time = time.time() + mean_interval - last_access_age
        confidence = 0.5  # Moderate confidence for simple method

        return predicted_time, confidence

    def _exponential_smoothing_prediction(self, features: dict[str, float]) -> tuple[float, float]:
        """Exponential smoothing prediction."""
        mean_interval = features.get("mean_interval", 3600)
        cv = features.get("cv", 1.0)
        last_access_age = features.get("last_access_age", 0)

        # Adjust prediction based on variability
        alpha = 0.3  # Smoothing factor
        adjusted_interval = mean_interval * (1 + alpha * (1 - 1 / cv if cv > 0 else 0))

        predicted_time = time.time() + adjusted_interval - last_access_age
        confidence = 1.0 / (1.0 + cv)  # Lower confidence for high variability

        return predicted_time, confidence


class PredictivePrewarmingService:
    """
    Advanced predictive cache pre-warming service.

    This service uses machine learning and pattern analysis to predict
    future cache accesses and proactively warm up the cache to improve
    response times.
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the predictive pre-warming service."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.pattern_analyzer = AccessPatternAnalyzer()
        self.prediction_engine = PredictionEngine()
        self.warmup_service: CacheWarmupService | None = None

        # Scheduling and job management
        self.pending_jobs: list[PrewarmingJob] = []
        self.completed_jobs: deque = deque(maxlen=1000)
        self.job_scheduler_task: asyncio.Task | None = None
        self.prediction_task: asyncio.Task | None = None

        # Statistics and monitoring
        self.stats = PrewarmingStats()
        self.access_tracking: dict[str, dict] = defaultdict(dict)

        # Configuration
        self.prediction_window_hours = 4  # Predict 4 hours ahead
        self.prewarm_threshold_minutes = 30  # Pre-warm 30 minutes before predicted access
        self.max_concurrent_jobs = 5
        self.prediction_interval = 600  # Run predictions every 10 minutes

        # Performance tracking
        self.response_times_before: deque = deque(maxlen=1000)
        self.response_times_after: deque = deque(maxlen=1000)

    async def initialize(self) -> None:
        """Initialize the predictive pre-warming service."""
        try:
            self.logger.info("Initializing Predictive Pre-warming Service...")

            # Initialize warmup service
            from src.services.cache_warmup_service import get_cache_warmup_service

            self.warmup_service = await get_cache_warmup_service()

            # Start background tasks
            self.job_scheduler_task = asyncio.create_task(self._job_scheduler_loop())
            self.prediction_task = asyncio.create_task(self._prediction_loop())

            self.logger.info("Predictive Pre-warming Service initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize Predictive Pre-warming Service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the predictive pre-warming service."""
        try:
            # Cancel background tasks
            for task in [self.job_scheduler_task, self.prediction_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self.logger.info("Predictive Pre-warming Service shutdown")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def record_cache_access(
        self, cache_name: str, key: str, response_time: float = None, size_mb: float = None, timestamp: float = None
    ) -> None:
        """
        Record cache access for pattern learning.

        Args:
            cache_name: Name of the cache
            key: Cache key that was accessed
            response_time: Response time for the access
            size_mb: Size of the cached item
            timestamp: Access timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        # Record in pattern analyzer
        self.pattern_analyzer.record_access(cache_name, key, timestamp)

        # Track for predictions
        access_key = f"{cache_name}:{key}"
        if access_key not in self.access_tracking:
            self.access_tracking[access_key] = {
                "first_seen": timestamp,
                "access_count": 0,
                "total_response_time": 0.0,
                "size_mb": size_mb or 0.0,
            }

        tracking = self.access_tracking[access_key]
        tracking["access_count"] += 1
        tracking["last_access"] = timestamp

        if response_time is not None:
            tracking["total_response_time"] += response_time
            tracking["avg_response_time"] = tracking["total_response_time"] / tracking["access_count"]

            # Track response times for performance analysis
            if self._was_prewarmed(cache_name, key):
                self.response_times_after.append(response_time)
            else:
                self.response_times_before.append(response_time)

        if size_mb is not None:
            tracking["size_mb"] = size_mb

    async def generate_predictions(self, cache_name: str | None = None) -> list[AccessPrediction]:
        """
        Generate predictions for future cache accesses.

        Args:
            cache_name: Specific cache to predict for (None for all caches)

        Returns:
            List of access predictions
        """
        try:
            predictions = []
            current_time = time.time()

            # Filter keys to analyze
            keys_to_analyze = []
            for access_key in self.access_tracking:
                key_cache_name, key = access_key.split(":", 1)
                if cache_name is None or key_cache_name == cache_name:
                    keys_to_analyze.append((key_cache_name, key))

            # Generate predictions for each key
            for key_cache_name, key in keys_to_analyze:
                try:
                    prediction = await self._predict_single_key(key_cache_name, key)
                    if prediction and prediction.should_prewarm_now:
                        predictions.append(prediction)
                except Exception as e:
                    self.logger.error(f"Failed to predict for {key_cache_name}:{key}: {e}")

            # Sort by priority and confidence
            predictions.sort(key=lambda p: (p.priority * p.confidence), reverse=True)

            self.stats.total_predictions += len(predictions)

            return predictions

        except Exception as e:
            self.logger.error(f"Failed to generate predictions: {e}")
            return []

    async def schedule_prewarming(self, predictions: list[AccessPrediction]) -> dict[str, Any]:
        """
        Schedule pre-warming jobs based on predictions.

        Args:
            predictions: List of access predictions

        Returns:
            Scheduling result
        """
        try:
            jobs_created = 0
            memory_budget = await self._calculate_memory_budget()
            total_memory_needed = 0.0

            for prediction in predictions:
                # Check memory budget
                if total_memory_needed + prediction.estimated_size_mb > memory_budget:
                    break

                # Calculate optimal scheduling time
                prewarm_time = prediction.predicted_access_time - (self.prewarm_threshold_minutes * 60)
                prewarm_time = max(prewarm_time, time.time() + 60)  # At least 1 minute from now

                # Calculate priority score
                priority_score = self._calculate_priority_score(prediction)

                # Create job
                job = PrewarmingJob(
                    prediction=prediction,
                    scheduled_time=prewarm_time,
                    trigger=PrewarmingTrigger.PREDICTED_ACCESS,
                    priority_score=priority_score,
                )

                self.pending_jobs.append(job)
                total_memory_needed += prediction.estimated_size_mb
                jobs_created += 1

            # Sort jobs by priority
            self.pending_jobs.sort(key=lambda j: j.priority_score, reverse=True)

            return {
                "success": True,
                "jobs_created": jobs_created,
                "memory_budget_mb": memory_budget,
                "memory_allocated_mb": total_memory_needed,
                "next_job_time": min((j.scheduled_time for j in self.pending_jobs), default=0),
            }

        except Exception as e:
            self.logger.error(f"Failed to schedule pre-warming: {e}")
            return {"success": False, "error": str(e)}

    async def trigger_immediate_prewarming(self, cache_name: str, keys: list[str] = None) -> dict[str, Any]:
        """
        Trigger immediate pre-warming for specific cache/keys.

        Args:
            cache_name: Cache to pre-warm
            keys: Specific keys to pre-warm (None for auto-selection)

        Returns:
            Pre-warming result
        """
        try:
            if keys is None:
                # Generate predictions and select top keys
                predictions = await self.generate_predictions(cache_name)
                keys = [p.key for p in predictions[:20]]  # Top 20 predictions

            if not keys:
                return {"success": False, "error": "No keys to pre-warm"}

            # Execute immediate pre-warming
            warmup_results = []
            memory_used = 0.0

            for key in keys:
                try:
                    # Check memory pressure
                    pressure = get_system_memory_pressure()
                    if pressure.level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                        break

                    # Perform pre-warming
                    result = await self._execute_prewarming_job(cache_name, key)
                    warmup_results.append(result)
                    memory_used += result.get("memory_used_mb", 0)

                except Exception as e:
                    self.logger.error(f"Failed to pre-warm {cache_name}:{key}: {e}")

            return {
                "success": True,
                "cache_name": cache_name,
                "keys_processed": len(warmup_results),
                "successful_prewarms": len([r for r in warmup_results if r.get("success")]),
                "memory_used_mb": memory_used,
                "results": warmup_results,
            }

        except Exception as e:
            self.logger.error(f"Immediate pre-warming failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_prewarming_status(self) -> dict[str, Any]:
        """Get comprehensive pre-warming status."""
        try:
            # Calculate performance metrics
            await self._update_performance_metrics()

            return {
                "service_status": "active",
                "pending_jobs": len(self.pending_jobs),
                "completed_jobs": len(self.completed_jobs),
                "statistics": {
                    "total_predictions": self.stats.total_predictions,
                    "successful_predictions": self.stats.successful_predictions,
                    "prediction_accuracy": self.stats.prediction_accuracy,
                    "cache_hits_after_prewarm": self.stats.cache_hits_after_prewarm,
                    "total_prewarmed_mb": self.stats.total_prewarmed_mb,
                    "response_time_improvement": self.stats.response_time_improvement,
                },
                "configuration": {
                    "prediction_window_hours": self.prediction_window_hours,
                    "prewarm_threshold_minutes": self.prewarm_threshold_minutes,
                    "max_concurrent_jobs": self.max_concurrent_jobs,
                    "prediction_interval": self.prediction_interval,
                },
                "model_performance": self.prediction_engine.model_performance,
                "next_prediction_run": time.time() + self.prediction_interval,
                "memory_status": await self._get_memory_status(),
            }

        except Exception as e:
            self.logger.error(f"Error getting pre-warming status: {e}")
            return {"error": str(e)}

    # Helper Methods

    async def _predict_single_key(self, cache_name: str, key: str) -> AccessPrediction | None:
        """Generate prediction for a single cache key."""
        try:
            # Analyze pattern
            pattern_analysis = self.pattern_analyzer.analyze_pattern(cache_name, key)

            if pattern_analysis["pattern_type"] == "insufficient_data":
                return None

            # Prepare features for prediction
            current_time = time.time()
            last_access = pattern_analysis["last_access"]

            features = {
                "mean_interval": pattern_analysis["mean_interval"],
                "std_interval": pattern_analysis["std_interval"],
                "cv": pattern_analysis["coefficient_of_variation"],
                "last_access_age": current_time - last_access,
                "access_count": pattern_analysis["access_count"],
                "hour_of_day": time.localtime(current_time).tm_hour,
                "day_of_week": time.localtime(current_time).tm_wday,
                "pattern_confidence": pattern_analysis["confidence"],
            }

            # Get prediction
            predicted_time, confidence = self.prediction_engine.predict_next_access(features)

            # Only create prediction if confidence is sufficient
            if confidence < 0.3:
                return None

            # Get additional metadata
            access_key = f"{cache_name}:{key}"
            tracking = self.access_tracking.get(access_key, {})

            # Determine access pattern
            pattern_type = pattern_analysis["pattern_type"]
            if pattern_type in ["regular_frequent", "regular_daily", "regular_periodic"]:
                access_pattern = AccessPattern.FREQUENT
            elif pattern_type == "burst":
                access_pattern = AccessPattern.BURST
            elif pattern_type == "temporal":
                access_pattern = AccessPattern.TEMPORAL
            else:
                access_pattern = AccessPattern.SPORADIC

            # Calculate priority based on access frequency and pattern
            priority = min(
                10,
                max(
                    1,
                    int(
                        pattern_analysis["access_count"] / 10
                        + confidence * 5
                        + (1.0 / max(pattern_analysis["mean_interval"], 1) * 86400) * 3
                    ),
                ),
            )

            return AccessPrediction(
                key=key,
                cache_name=cache_name,
                predicted_access_time=predicted_time,
                confidence=confidence,
                priority=priority,
                estimated_size_mb=tracking.get("size_mb", 1.0),
                access_pattern=access_pattern,
            )

        except Exception as e:
            self.logger.error(f"Failed to predict for {cache_name}:{key}: {e}")
            return None

    async def _job_scheduler_loop(self) -> None:
        """Background job scheduler loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_time = time.time()
                ready_jobs = [job for job in self.pending_jobs if job.is_ready]

                # Sort by priority
                ready_jobs.sort(key=lambda j: j.priority_score, reverse=True)

                # Execute jobs (respecting concurrency limit)
                executing_jobs = [job for job in self.pending_jobs if job.status == "running"]
                available_slots = self.max_concurrent_jobs - len(executing_jobs)

                for job in ready_jobs[:available_slots]:
                    asyncio.create_task(self._execute_job(job))

                # Clean up old jobs
                self.pending_jobs = [
                    job
                    for job in self.pending_jobs
                    if job.status in ["pending", "running"] and current_time - job.scheduled_time < 3600  # Remove jobs older than 1 hour
                ]

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in job scheduler loop: {e}")

    async def _prediction_loop(self) -> None:
        """Background prediction generation loop."""
        while True:
            try:
                await asyncio.sleep(self.prediction_interval)

                # Generate new predictions
                predictions = await self.generate_predictions()

                # Schedule pre-warming jobs
                if predictions:
                    await self.schedule_prewarming(predictions)

                # Train models if needed
                if time.time() - self.prediction_engine.last_training_time > self.prediction_engine.retrain_interval:
                    await self.prediction_engine.train_models()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")

    async def _execute_job(self, job: PrewarmingJob) -> None:
        """Execute a pre-warming job."""
        try:
            job.status = "running"

            result = await self._execute_prewarming_job(job.prediction.cache_name, job.prediction.key)

            if result.get("success"):
                job.status = "completed"
                self.stats.total_prewarmed_mb += result.get("memory_used_mb", 0)
            else:
                job.status = "failed"

            # Move to completed jobs
            self.completed_jobs.append({"job": job.__dict__, "result": result, "completed_at": time.time()})

            # Remove from pending
            self.pending_jobs.remove(job)

        except Exception as e:
            self.logger.error(f"Job execution failed: {e}")
            job.status = "failed"

    async def _execute_prewarming_job(self, cache_name: str, key: str) -> dict[str, Any]:
        """Execute actual pre-warming for a cache key."""
        try:
            # This would integrate with specific cache implementations
            # For now, return a mock result

            # Simulate pre-warming operation
            await asyncio.sleep(0.1)  # Simulate work

            return {"success": True, "cache_name": cache_name, "key": key, "memory_used_mb": 1.0, "execution_time": 0.1}  # Estimated

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _was_prewarmed(self, cache_name: str, key: str) -> bool:
        """Check if a key was pre-warmed recently."""
        # Check if key was in recent completed jobs
        recent_cutoff = time.time() - 3600  # Last hour

        for job_record in self.completed_jobs:
            if (
                job_record["completed_at"] >= recent_cutoff
                and job_record["job"]["prediction"]["cache_name"] == cache_name
                and job_record["job"]["prediction"]["key"] == key
            ):
                return True

        return False

    async def _calculate_memory_budget(self) -> float:
        """Calculate available memory budget for pre-warming."""
        pressure = get_system_memory_pressure()

        if pressure.level == MemoryPressureLevel.CRITICAL:
            return 0.0
        elif pressure.level == MemoryPressureLevel.HIGH:
            return min(50.0, pressure.available_mb * 0.1)  # 10% of available, max 50MB
        elif pressure.level == MemoryPressureLevel.MODERATE:
            return min(200.0, pressure.available_mb * 0.2)  # 20% of available, max 200MB
        else:
            return min(500.0, pressure.available_mb * 0.3)  # 30% of available, max 500MB

    def _calculate_priority_score(self, prediction: AccessPrediction) -> float:
        """Calculate priority score for a prediction."""
        # Combine multiple factors
        time_factor = 1.0 / max(1.0, prediction.time_to_access / 3600)  # Higher for sooner access
        confidence_factor = prediction.confidence
        priority_factor = prediction.priority / 10.0

        return time_factor * confidence_factor * priority_factor

    async def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Calculate prediction accuracy
            if self.stats.total_predictions > 0:
                self.stats.prediction_accuracy = self.stats.successful_predictions / self.stats.total_predictions

            # Calculate response time improvement
            if self.response_times_before and self.response_times_after:
                avg_before = sum(self.response_times_before) / len(self.response_times_before)
                avg_after = sum(self.response_times_after) / len(self.response_times_after)

                if avg_before > 0:
                    self.stats.response_time_improvement = (avg_before - avg_after) / avg_before

            self.stats.last_updated = time.time()

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def _get_memory_status(self) -> dict[str, Any]:
        """Get current memory status for pre-warming."""
        pressure = get_system_memory_pressure()
        budget = await self._calculate_memory_budget()

        return {
            "pressure_level": pressure.level.value,
            "available_mb": pressure.available_mb,
            "budget_mb": budget,
            "budget_utilization": (self.stats.total_prewarmed_mb / budget * 100) if budget > 0 else 0,
        }


# Global service instance
_prewarming_service: PredictivePrewarmingService | None = None


async def get_prewarming_service(config: CacheConfig | None = None) -> PredictivePrewarmingService:
    """Get the global predictive pre-warming service instance."""
    global _prewarming_service
    if _prewarming_service is None:
        _prewarming_service = PredictivePrewarmingService(config)
        await _prewarming_service.initialize()
    return _prewarming_service


async def shutdown_prewarming_service() -> None:
    """Shutdown the global predictive pre-warming service."""
    global _prewarming_service
    if _prewarming_service:
        await _prewarming_service.shutdown()
        _prewarming_service = None
