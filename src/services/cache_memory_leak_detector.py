"""
Cache memory leak detection service.

This module provides comprehensive memory leak detection capabilities for cache
systems including leak pattern recognition, growth trend analysis, orphaned
reference detection, and automated leak alerts.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, median, stdev
from threading import Lock
from typing import Any, Optional, Union

from ..utils.memory_utils import (
    CacheMemoryEvent,
    get_memory_stats,
    get_system_memory_pressure,
    get_total_cache_memory_usage,
    track_cache_memory_event,
)


class LeakSeverity(Enum):
    """Memory leak severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LeakType(Enum):
    """Types of memory leaks."""

    GRADUAL_GROWTH = "gradual_growth"  # Slow but steady memory growth
    SUDDEN_SPIKE = "sudden_spike"  # Sudden memory increase
    ORPHANED_REFERENCES = "orphaned_references"  # Unreferenced cached objects
    FRAGMENTATION = "fragmentation"  # Memory fragmentation
    RETENTION_LEAK = "retention_leak"  # Objects not being released
    CIRCULAR_REFERENCE = "circular_reference"  # Circular references preventing GC


@dataclass
class LeakEvidence:
    """Evidence of a memory leak."""

    timestamp: float
    leak_type: LeakType
    severity: LeakSeverity
    cache_name: str
    description: str
    metrics: dict[str, Any] = field(default_factory=dict)
    suggested_actions: list[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0.0 to 1.0
    related_keys: list[str] = field(default_factory=list)
    stack_trace: list[str] = field(default_factory=list)

    @property
    def age(self) -> float:
        """Age of the leak evidence in seconds."""
        return time.time() - self.timestamp


@dataclass
class LeakDetectionConfig:
    """Configuration for memory leak detection."""

    # Growth detection thresholds
    growth_threshold_mb: float = 50.0  # MB growth to trigger detection
    growth_time_window_minutes: int = 30  # Time window for growth detection
    growth_rate_threshold_mb_per_min: float = 2.0  # MB/minute growth rate threshold

    # Retention detection thresholds
    retention_threshold_minutes: int = 60  # Minutes before considering object retained
    retention_ratio_threshold: float = 0.8  # Ratio of retained objects to trigger alert

    # Fragmentation detection
    fragmentation_threshold: float = 0.3  # Fragmentation ratio threshold
    fragmentation_check_interval_minutes: int = 15  # Check interval for fragmentation

    # Orphaned reference detection
    orphaned_check_interval_minutes: int = 30  # Check interval for orphaned references
    orphaned_threshold_count: int = 100  # Number of orphaned objects to trigger alert

    # Leak confirmation settings
    confirmation_samples: int = 3  # Number of samples to confirm leak
    confirmation_interval_minutes: int = 5  # Interval between confirmation samples

    # Alert settings
    alert_cooldown_minutes: int = 30  # Cooldown period between alerts
    max_alerts_per_hour: int = 5  # Maximum alerts per hour


@dataclass
class MemoryGrowthPattern:
    """Pattern of memory growth over time."""

    cache_name: str
    start_time: float
    end_time: float
    start_memory_mb: float
    end_memory_mb: float
    growth_rate_mb_per_min: float
    confidence: float
    data_points: int
    trend_slope: float
    correlation_coefficient: float

    @property
    def total_growth_mb(self) -> float:
        """Total memory growth in MB."""
        return self.end_memory_mb - self.start_memory_mb

    @property
    def duration_minutes(self) -> float:
        """Duration of the growth pattern in minutes."""
        return (self.end_time - self.start_time) / 60.0


@dataclass
class RetentionAnalysis:
    """Analysis of object retention patterns."""

    cache_name: str
    total_objects: int
    retained_objects: int
    retention_ratio: float
    avg_retention_time_minutes: float
    max_retention_time_minutes: float
    retention_threshold_minutes: float
    old_objects_count: int
    very_old_objects_count: int

    @property
    def is_leak_suspected(self) -> bool:
        """Check if retention suggests a leak."""
        return self.retention_ratio > 0.8 and self.avg_retention_time_minutes > 60


class CacheMemoryLeakDetector:
    """
    Comprehensive cache memory leak detector.

    This service analyzes cache memory usage patterns to detect various types
    of memory leaks including gradual growth, sudden spikes, orphaned references,
    fragmentation, and retention leaks.
    """

    def __init__(self, config: LeakDetectionConfig | None = None):
        self.config = config or LeakDetectionConfig()
        self.logger = logging.getLogger(__name__)

        # Detection state
        self.is_monitoring = False
        self.monitoring_lock = Lock()

        # Memory tracking
        self.memory_history: deque[tuple[float, dict[str, float]]] = deque(maxlen=1000)
        self.growth_patterns: dict[str, deque[MemoryGrowthPattern]] = defaultdict(lambda: deque(maxlen=100))

        # Leak evidence
        self.leak_evidence: deque[LeakEvidence] = deque(maxlen=500)
        self.evidence_lock = Lock()

        # Object retention tracking
        self.object_lifecycles: dict[str, dict[str, float]] = defaultdict(dict)  # cache_name -> {key: creation_time}
        self.retention_analyses: dict[str, RetentionAnalysis] = {}

        # Alert management
        self.recent_alerts: deque[tuple[float, str]] = deque(maxlen=100)
        self.alert_counts: dict[str, int] = defaultdict(int)

        # Background tasks
        self.monitoring_task: asyncio.Task | None = None
        self.analysis_task: asyncio.Task | None = None

        # Statistics
        self.detection_stats = {
            "total_leaks_detected": 0,
            "leaks_by_type": defaultdict(int),
            "leaks_by_severity": defaultdict(int),
            "false_positives": 0,
            "confirmed_leaks": 0,
        }

    async def initialize(self) -> None:
        """Initialize the leak detector."""
        self.is_monitoring = True

        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.analysis_task = asyncio.create_task(self._analysis_loop())

        self.logger.info("Cache memory leak detector initialized")

    async def shutdown(self) -> None:
        """Shutdown the leak detector."""
        self.is_monitoring = False

        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Cache memory leak detector shutdown")

    def track_object_lifecycle(self, cache_name: str, key: str, operation: str) -> None:
        """Track object lifecycle for retention analysis."""
        current_time = time.time()

        if operation == "create":
            self.object_lifecycles[cache_name][key] = current_time
        elif operation == "delete":
            self.object_lifecycles[cache_name].pop(key, None)
        elif operation == "access":
            # Update last access time (if tracking granular access)
            if key in self.object_lifecycles[cache_name]:
                self.object_lifecycles[cache_name][key] = current_time

    def record_memory_snapshot(self, cache_breakdown: dict[str, float]) -> None:
        """Record a memory snapshot for analysis."""
        current_time = time.time()
        self.memory_history.append((current_time, cache_breakdown.copy()))

        # Trigger growth pattern analysis
        self._analyze_growth_patterns(cache_breakdown)

    def _analyze_growth_patterns(self, current_memory: dict[str, float]) -> None:
        """Analyze memory growth patterns."""
        if len(self.memory_history) < 2:
            return

        current_time = time.time()
        window_seconds = self.config.growth_time_window_minutes * 60
        cutoff_time = current_time - window_seconds

        # Filter recent history
        recent_history = [(t, m) for t, m in self.memory_history if t >= cutoff_time]

        if len(recent_history) < 2:
            return

        # Analyze growth for each cache
        for cache_name in current_memory:
            cache_history = [(t, m.get(cache_name, 0)) for t, m in recent_history]
            self._detect_growth_leak(cache_name, cache_history)

    def _detect_growth_leak(self, cache_name: str, memory_history: list[tuple[float, float]]) -> None:
        """Detect gradual growth leaks."""
        if len(memory_history) < 3:
            return

        # Extract time and memory values
        times = [t for t, _ in memory_history]
        memories = [m for _, m in memory_history]

        # Calculate growth rate
        start_time, start_memory = times[0], memories[0]
        end_time, end_memory = times[-1], memories[-1]
        duration_minutes = (end_time - start_time) / 60.0

        if duration_minutes <= 0:
            return

        growth_rate = (end_memory - start_memory) / duration_minutes
        total_growth = end_memory - start_memory

        # Check for significant growth
        if total_growth >= self.config.growth_threshold_mb and growth_rate >= self.config.growth_rate_threshold_mb_per_min:
            # Calculate trend strength
            trend_slope, correlation = self._calculate_trend_strength(times, memories)

            # Create growth pattern
            pattern = MemoryGrowthPattern(
                cache_name=cache_name,
                start_time=start_time,
                end_time=end_time,
                start_memory_mb=start_memory,
                end_memory_mb=end_memory,
                growth_rate_mb_per_min=growth_rate,
                confidence=min(correlation, 1.0),
                data_points=len(memory_history),
                trend_slope=trend_slope,
                correlation_coefficient=correlation,
            )

            self.growth_patterns[cache_name].append(pattern)

            # Generate leak evidence if pattern is strong
            if correlation > 0.7:  # Strong positive correlation
                self._generate_growth_leak_evidence(pattern)

    def _calculate_trend_strength(self, times: list[float], values: list[float]) -> tuple[float, float]:
        """Calculate trend strength using linear regression."""
        if len(times) < 2:
            return 0.0, 0.0

        # Normalize times to start from 0
        normalized_times = [t - times[0] for t in times]

        # Calculate means
        mean_time = mean(normalized_times)
        mean_value = mean(values)

        # Calculate slope and correlation
        numerator = sum((t - mean_time) * (v - mean_value) for t, v in zip(normalized_times, values, strict=False))
        denominator_time = sum((t - mean_time) ** 2 for t in normalized_times)
        denominator_value = sum((v - mean_value) ** 2 for v in values)

        if denominator_time == 0:
            return 0.0, 0.0

        slope = numerator / denominator_time
        correlation = numerator / (denominator_time * denominator_value) ** 0.5 if denominator_value > 0 else 0.0

        return slope, correlation

    def _generate_growth_leak_evidence(self, pattern: MemoryGrowthPattern) -> None:
        """Generate leak evidence for growth pattern."""
        # Determine severity based on growth rate
        if pattern.growth_rate_mb_per_min >= 10:
            severity = LeakSeverity.CRITICAL
        elif pattern.growth_rate_mb_per_min >= 5:
            severity = LeakSeverity.HIGH
        elif pattern.growth_rate_mb_per_min >= 2:
            severity = LeakSeverity.MEDIUM
        else:
            severity = LeakSeverity.LOW

        # Generate evidence
        evidence = LeakEvidence(
            timestamp=pattern.end_time,
            leak_type=LeakType.GRADUAL_GROWTH,
            severity=severity,
            cache_name=pattern.cache_name,
            description=f"Gradual memory growth detected: {pattern.total_growth_mb:.1f}MB over {pattern.duration_minutes:.1f} minutes",
            metrics={
                "growth_rate_mb_per_min": pattern.growth_rate_mb_per_min,
                "total_growth_mb": pattern.total_growth_mb,
                "duration_minutes": pattern.duration_minutes,
                "confidence": pattern.confidence,
                "correlation_coefficient": pattern.correlation_coefficient,
            },
            suggested_actions=[
                "Review cache eviction policies",
                "Check for objects not being properly released",
                "Investigate allocation patterns",
                "Consider reducing cache size limits",
            ],
            confidence_score=pattern.confidence,
        )

        self._record_leak_evidence(evidence)

    def _detect_sudden_spike(self, cache_name: str, current_memory: float) -> None:
        """Detect sudden memory spikes."""
        if len(self.memory_history) < 2:
            return

        # Get previous memory usage
        prev_time, prev_memory = self.memory_history[-2]
        prev_cache_memory = prev_memory.get(cache_name, 0)

        # Check for sudden spike
        memory_increase = current_memory - prev_cache_memory
        time_diff = time.time() - prev_time

        if memory_increase >= 100 and time_diff < 300:  # 100MB increase in less than 5 minutes
            severity = LeakSeverity.CRITICAL if memory_increase >= 500 else LeakSeverity.HIGH

            evidence = LeakEvidence(
                timestamp=time.time(),
                leak_type=LeakType.SUDDEN_SPIKE,
                severity=severity,
                cache_name=cache_name,
                description=f"Sudden memory spike detected: {memory_increase:.1f}MB increase in {time_diff:.1f} seconds",
                metrics={
                    "memory_increase_mb": memory_increase,
                    "time_seconds": time_diff,
                    "rate_mb_per_sec": memory_increase / time_diff if time_diff > 0 else 0,
                },
                suggested_actions=[
                    "Investigate recent cache operations",
                    "Check for bulk insertions without proper cleanup",
                    "Review application logic for memory leaks",
                    "Consider emergency cache clearing",
                ],
                confidence_score=0.9,
            )

            self._record_leak_evidence(evidence)

    def _analyze_object_retention(self, cache_name: str) -> None:
        """Analyze object retention patterns."""
        if cache_name not in self.object_lifecycles:
            return

        current_time = time.time()
        threshold_seconds = self.config.retention_threshold_minutes * 60

        objects = self.object_lifecycles[cache_name]
        retention_times = [(current_time - creation_time) / 60.0 for creation_time in objects.values()]

        if not retention_times:
            return

        # Calculate retention metrics
        retained_objects = sum(1 for rt in retention_times if rt * 60 >= threshold_seconds)
        retention_ratio = retained_objects / len(retention_times)
        avg_retention_minutes = mean(retention_times)
        max_retention_minutes = max(retention_times)

        # Count old objects
        old_objects = sum(1 for rt in retention_times if rt >= 60)  # 1 hour
        very_old_objects = sum(1 for rt in retention_times if rt >= 240)  # 4 hours

        # Create retention analysis
        analysis = RetentionAnalysis(
            cache_name=cache_name,
            total_objects=len(retention_times),
            retained_objects=retained_objects,
            retention_ratio=retention_ratio,
            avg_retention_time_minutes=avg_retention_minutes,
            max_retention_time_minutes=max_retention_minutes,
            retention_threshold_minutes=self.config.retention_threshold_minutes,
            old_objects_count=old_objects,
            very_old_objects_count=very_old_objects,
        )

        self.retention_analyses[cache_name] = analysis

        # Check for retention leak
        if analysis.is_leak_suspected:
            self._generate_retention_leak_evidence(analysis)

    def _generate_retention_leak_evidence(self, analysis: RetentionAnalysis) -> None:
        """Generate leak evidence for retention analysis."""
        severity = LeakSeverity.HIGH if analysis.retention_ratio > 0.9 else LeakSeverity.MEDIUM

        evidence = LeakEvidence(
            timestamp=time.time(),
            leak_type=LeakType.RETENTION_LEAK,
            severity=severity,
            cache_name=analysis.cache_name,
            description=f"Object retention leak detected: {analysis.retention_ratio:.1%} objects retained beyond threshold",
            metrics={
                "retention_ratio": analysis.retention_ratio,
                "avg_retention_minutes": analysis.avg_retention_time_minutes,
                "max_retention_minutes": analysis.max_retention_time_minutes,
                "old_objects_count": analysis.old_objects_count,
                "very_old_objects_count": analysis.very_old_objects_count,
            },
            suggested_actions=[
                "Review cache expiration policies",
                "Check for objects with missing TTL",
                "Investigate reference counting issues",
                "Consider more aggressive eviction policies",
            ],
            confidence_score=0.8,
        )

        self._record_leak_evidence(evidence)

    def _detect_fragmentation_leak(self, cache_name: str) -> None:
        """Detect memory fragmentation issues."""
        # This would typically require more detailed memory analysis
        # For now, we'll use a heuristic based on allocation patterns

        # Get recent allocation patterns from memory profiler if available
        try:
            from .cache_memory_profiler import get_memory_profiler

            profiler = get_memory_profiler()
            if profiler:
                patterns = profiler.get_allocation_patterns(cache_name, window_minutes=30)
                if patterns and not patterns.get("error"):
                    # Calculate fragmentation indicator
                    alloc_count = patterns["allocations"]["count"]
                    dealloc_count = patterns["deallocations"]["count"]

                    if alloc_count > 0 and dealloc_count > 0:
                        fragmentation_ratio = 1.0 - (dealloc_count / alloc_count)

                        if fragmentation_ratio > self.config.fragmentation_threshold:
                            severity = LeakSeverity.MEDIUM if fragmentation_ratio > 0.5 else LeakSeverity.LOW

                            evidence = LeakEvidence(
                                timestamp=time.time(),
                                leak_type=LeakType.FRAGMENTATION,
                                severity=severity,
                                cache_name=cache_name,
                                description=f"Memory fragmentation detected: {fragmentation_ratio:.1%} fragmentation ratio",
                                metrics={
                                    "fragmentation_ratio": fragmentation_ratio,
                                    "allocation_count": alloc_count,
                                    "deallocation_count": dealloc_count,
                                },
                                suggested_actions=[
                                    "Consider cache compaction",
                                    "Review allocation patterns",
                                    "Implement memory pooling",
                                    "Optimize object sizes",
                                ],
                                confidence_score=0.6,
                            )

                            self._record_leak_evidence(evidence)
        except ImportError:
            pass

    def _record_leak_evidence(self, evidence: LeakEvidence) -> None:
        """Record leak evidence and manage alerts."""
        with self.evidence_lock:
            self.leak_evidence.append(evidence)

        # Update statistics
        self.detection_stats["total_leaks_detected"] += 1
        self.detection_stats["leaks_by_type"][evidence.leak_type.value] += 1
        self.detection_stats["leaks_by_severity"][evidence.severity.value] += 1

        # Check if we should send an alert
        if self._should_send_alert(evidence):
            self._send_leak_alert(evidence)

        # Track memory event
        track_cache_memory_event(
            evidence.cache_name,
            CacheMemoryEvent.PRESSURE,
            0.0,
            {
                "leak_type": evidence.leak_type.value,
                "severity": evidence.severity.value,
                "confidence": evidence.confidence_score,
            },
        )

    def _should_send_alert(self, evidence: LeakEvidence) -> bool:
        """Check if an alert should be sent for this evidence."""
        current_time = time.time()
        cache_name = evidence.cache_name

        # Check cooldown period
        cooldown_seconds = self.config.alert_cooldown_minutes * 60
        recent_alerts = [(t, name) for t, name in self.recent_alerts if current_time - t < cooldown_seconds and name == cache_name]

        if recent_alerts:
            return False

        # Check hourly rate limit
        hour_seconds = 3600
        hour_alerts = [(t, name) for t, name in self.recent_alerts if current_time - t < hour_seconds and name == cache_name]

        if len(hour_alerts) >= self.config.max_alerts_per_hour:
            return False

        # Check severity threshold
        if evidence.severity in [LeakSeverity.CRITICAL, LeakSeverity.HIGH]:
            return True

        return False

    def _send_leak_alert(self, evidence: LeakEvidence) -> None:
        """Send a leak alert."""
        current_time = time.time()
        self.recent_alerts.append((current_time, evidence.cache_name))

        # Log the alert
        self.logger.warning(
            f"MEMORY LEAK DETECTED: {evidence.leak_type.value} in cache '{evidence.cache_name}' "
            f"(Severity: {evidence.severity.value}, Confidence: {evidence.confidence_score:.2f})"
        )
        self.logger.warning(f"Description: {evidence.description}")
        self.logger.warning(f"Suggested actions: {', '.join(evidence.suggested_actions)}")

        # Here you could integrate with external alerting systems
        # e.g., send to Slack, email, PagerDuty, etc.

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Get current memory usage
                cache_memory = get_total_cache_memory_usage()

                # Record snapshot (this would be integrated with the profiler)
                cache_breakdown = {"total": cache_memory}
                self.record_memory_snapshot(cache_breakdown)

                # Check for sudden spikes
                if cache_breakdown:
                    for cache_name, memory_mb in cache_breakdown.items():
                        self._detect_sudden_spike(cache_name, memory_mb)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    async def _analysis_loop(self) -> None:
        """Background analysis loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.fragmentation_check_interval_minutes * 60)

                # Analyze object retention for all caches
                for cache_name in self.object_lifecycles:
                    self._analyze_object_retention(cache_name)

                # Check for fragmentation
                for cache_name in self.object_lifecycles:
                    self._detect_fragmentation_leak(cache_name)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")

    def get_leak_summary(self, cache_name: str | None = None) -> dict[str, Any]:
        """Get summary of detected leaks."""
        with self.evidence_lock:
            relevant_evidence = [e for e in self.leak_evidence if cache_name is None or e.cache_name == cache_name]

        if not relevant_evidence:
            return {"total_leaks": 0, "leaks_by_type": {}, "leaks_by_severity": {}}

        # Aggregate by type and severity
        leaks_by_type = defaultdict(int)
        leaks_by_severity = defaultdict(int)

        for evidence in relevant_evidence:
            leaks_by_type[evidence.leak_type.value] += 1
            leaks_by_severity[evidence.severity.value] += 1

        return {
            "total_leaks": len(relevant_evidence),
            "leaks_by_type": dict(leaks_by_type),
            "leaks_by_severity": dict(leaks_by_severity),
            "recent_leaks": [
                {
                    "timestamp": e.timestamp,
                    "type": e.leak_type.value,
                    "severity": e.severity.value,
                    "cache_name": e.cache_name,
                    "description": e.description,
                    "confidence": e.confidence_score,
                }
                for e in sorted(relevant_evidence, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
        }

    def get_detection_stats(self) -> dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_leaks_detected": self.detection_stats["total_leaks_detected"],
            "leaks_by_type": dict(self.detection_stats["leaks_by_type"]),
            "leaks_by_severity": dict(self.detection_stats["leaks_by_severity"]),
            "false_positives": self.detection_stats["false_positives"],
            "confirmed_leaks": self.detection_stats["confirmed_leaks"],
            "detection_accuracy": (self.detection_stats["confirmed_leaks"] / max(1, self.detection_stats["total_leaks_detected"])),
        }

    def get_retention_analysis(self, cache_name: str) -> dict[str, Any] | None:
        """Get retention analysis for a cache."""
        analysis = self.retention_analyses.get(cache_name)
        if not analysis:
            return None

        return {
            "cache_name": analysis.cache_name,
            "total_objects": analysis.total_objects,
            "retained_objects": analysis.retained_objects,
            "retention_ratio": analysis.retention_ratio,
            "avg_retention_time_minutes": analysis.avg_retention_time_minutes,
            "max_retention_time_minutes": analysis.max_retention_time_minutes,
            "old_objects_count": analysis.old_objects_count,
            "very_old_objects_count": analysis.very_old_objects_count,
            "is_leak_suspected": analysis.is_leak_suspected,
        }

    def mark_false_positive(self, evidence_timestamp: float) -> bool:
        """Mark a leak detection as false positive."""
        with self.evidence_lock:
            for evidence in self.leak_evidence:
                if evidence.timestamp == evidence_timestamp:
                    # Update statistics
                    self.detection_stats["false_positives"] += 1
                    return True
        return False

    def mark_confirmed_leak(self, evidence_timestamp: float) -> bool:
        """Mark a leak detection as confirmed."""
        with self.evidence_lock:
            for evidence in self.leak_evidence:
                if evidence.timestamp == evidence_timestamp:
                    # Update statistics
                    self.detection_stats["confirmed_leaks"] += 1
                    return True
        return False

    def clear_leak_evidence(self, older_than_hours: int = 24) -> int:
        """Clear old leak evidence."""
        cutoff_time = time.time() - (older_than_hours * 3600)

        with self.evidence_lock:
            old_count = len(self.leak_evidence)
            self.leak_evidence = deque([e for e in self.leak_evidence if e.timestamp >= cutoff_time], maxlen=500)
            cleared_count = old_count - len(self.leak_evidence)

        self.logger.info(f"Cleared {cleared_count} old leak evidence entries")
        return cleared_count


# Global leak detector instance
_leak_detector: CacheMemoryLeakDetector | None = None


async def get_leak_detector(config: LeakDetectionConfig | None = None) -> CacheMemoryLeakDetector:
    """Get the global leak detector instance."""
    global _leak_detector
    if _leak_detector is None:
        _leak_detector = CacheMemoryLeakDetector(config)
        await _leak_detector.initialize()
    return _leak_detector


async def shutdown_leak_detector() -> None:
    """Shutdown the global leak detector."""
    global _leak_detector
    if _leak_detector:
        await _leak_detector.shutdown()
        _leak_detector = None
