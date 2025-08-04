"""
Performance Bottleneck Identification and Analysis Tool - Wave 5.0 Implementation.

Provides comprehensive bottleneck detection and analysis capabilities including:
- Automated bottleneck detection across system components
- Performance hotspot identification and analysis
- Resource utilization analysis and optimization suggestions
- Dependency chain analysis for performance impact
- Real-time bottleneck monitoring and alerting
- Historical bottleneck pattern analysis
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

from src.services.performance_monitor import (
    PerformanceMetricType,
    get_performance_monitor,
)
from src.utils.performance_monitor import get_cache_performance_monitor


class BottleneckType(Enum):
    """Types of performance bottlenecks."""

    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    CACHE_MISS = "cache_miss"
    NETWORK_LATENCY = "network_latency"
    DATABASE_QUERY = "database_query"
    ALGORITHM_COMPLEXITY = "algorithm_complexity"
    RESOURCE_CONTENTION = "resource_contention"
    DEPENDENCY_CHAIN = "dependency_chain"
    CUSTOM = "custom"


class BottleneckSeverity(Enum):
    """Severity levels for bottlenecks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BottleneckEvidence:
    """Evidence supporting a bottleneck identification."""

    metric_name: str
    current_value: float
    threshold_value: float
    impact_score: float  # 0-100 score indicating impact severity
    confidence: float  # 0-1 confidence in this evidence
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def deviation_percent(self) -> float:
        """Calculate deviation from threshold as percentage."""
        if self.threshold_value == 0:
            return 0.0
        return ((self.current_value - self.threshold_value) / self.threshold_value) * 100


@dataclass
class PerformanceBottleneck:
    """Represents an identified performance bottleneck."""

    bottleneck_id: str
    bottleneck_type: BottleneckType
    component: str
    severity: BottleneckSeverity

    # Detection details
    title: str
    description: str
    impact_description: str
    first_detected: float
    last_detected: float
    detection_count: int = 1

    # Evidence and metrics
    evidence: list[BottleneckEvidence] = field(default_factory=list)
    affected_metrics: list[str] = field(default_factory=list)

    # Impact analysis
    estimated_impact_percent: float = 0.0  # Estimated performance impact
    affected_components: list[str] = field(default_factory=list)
    dependency_chain: list[str] = field(default_factory=list)

    # Resolution suggestions
    resolution_suggestions: list[str] = field(default_factory=list)
    estimated_effort: str = "medium"  # low, medium, high
    priority_score: float = 0.0  # 0-100 priority for resolution

    # Status tracking
    acknowledged: bool = False
    in_progress: bool = False
    resolved: bool = False
    resolution_notes: str | None = None

    def add_evidence(self, evidence: BottleneckEvidence):
        """Add supporting evidence to the bottleneck."""
        self.evidence.append(evidence)
        self.last_detected = time.time()
        self.detection_count += 1

    def calculate_priority_score(self):
        """Calculate priority score based on severity, impact, and evidence."""
        severity_weights = {
            BottleneckSeverity.LOW: 25,
            BottleneckSeverity.MEDIUM: 50,
            BottleneckSeverity.HIGH: 75,
            BottleneckSeverity.CRITICAL: 100,
        }

        base_score = severity_weights.get(self.severity, 50)
        impact_multiplier = self.estimated_impact_percent / 100
        evidence_strength = min(len(self.evidence) / 5.0, 1.0)  # Up to 5 pieces of evidence

        self.priority_score = base_score * (1 + impact_multiplier) * (0.5 + 0.5 * evidence_strength)
        self.priority_score = min(self.priority_score, 100.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bottleneck_id": self.bottleneck_id,
            "bottleneck_type": self.bottleneck_type.value,
            "component": self.component,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "impact_description": self.impact_description,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "detection_count": self.detection_count,
            "evidence": [
                {
                    "metric_name": e.metric_name,
                    "current_value": e.current_value,
                    "threshold_value": e.threshold_value,
                    "impact_score": e.impact_score,
                    "confidence": e.confidence,
                    "deviation_percent": e.deviation_percent,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                }
                for e in self.evidence
            ],
            "affected_metrics": self.affected_metrics,
            "estimated_impact_percent": self.estimated_impact_percent,
            "affected_components": self.affected_components,
            "dependency_chain": self.dependency_chain,
            "resolution_suggestions": self.resolution_suggestions,
            "estimated_effort": self.estimated_effort,
            "priority_score": self.priority_score,
            "acknowledged": self.acknowledged,
            "in_progress": self.in_progress,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class BottleneckAnalysisConfig:
    """Configuration for bottleneck analysis."""

    # Detection thresholds
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    response_time_threshold_ms: float = 15000.0
    cache_hit_rate_threshold: float = 70.0
    error_rate_threshold: float = 5.0

    # Analysis settings
    analysis_window_minutes: int = 30
    evidence_confidence_threshold: float = 0.6
    minimum_impact_score: float = 30.0

    # Detection sensitivity
    detection_sensitivity: str = "medium"  # low, medium, high
    require_multiple_evidence: bool = True
    evidence_correlation_threshold: float = 0.7

    # Reporting
    enable_real_time_detection: bool = True
    enable_historical_analysis: bool = True
    generate_resolution_suggestions: bool = True


class PerformanceBottleneckAnalyzer:
    """
    Comprehensive performance bottleneck identification and analysis system.

    Provides automated detection of performance bottlenecks across system components
    with detailed analysis, impact assessment, and resolution recommendations.
    """

    def __init__(self, config: BottleneckAnalysisConfig | None = None):
        """
        Initialize the bottleneck analyzer.

        Args:
            config: Analysis configuration settings
        """
        self.config = config or BottleneckAnalysisConfig()
        self.logger = logging.getLogger(__name__)

        # Performance monitoring integration
        self.performance_monitor = get_performance_monitor()
        self.cache_monitor = get_cache_performance_monitor()

        # Bottleneck tracking
        self._detected_bottlenecks: dict[str, PerformanceBottleneck] = {}
        self._bottleneck_history: deque = deque(maxlen=1000)
        self._analysis_cache: dict[str, Any] = {}

        # Analysis state
        self._analysis_task: asyncio.Task | None = None
        self._is_analyzing = False

        # Component mapping and dependencies
        self._component_dependencies: dict[str, list[str]] = {}
        self._performance_baselines: dict[str, dict[str, float]] = {}

        # Detection patterns
        self._bottleneck_patterns = {
            BottleneckType.CPU_BOUND: self._detect_cpu_bottleneck,
            BottleneckType.MEMORY_BOUND: self._detect_memory_bottleneck,
            BottleneckType.IO_BOUND: self._detect_io_bottleneck,
            BottleneckType.CACHE_MISS: self._detect_cache_bottleneck,
            BottleneckType.NETWORK_LATENCY: self._detect_network_bottleneck,
            BottleneckType.ALGORITHM_COMPLEXITY: self._detect_algorithm_bottleneck,
            BottleneckType.RESOURCE_CONTENTION: self._detect_resource_contention,
            BottleneckType.DEPENDENCY_CHAIN: self._detect_dependency_bottleneck,
        }

        # Initialize component dependencies
        self._initialize_component_dependencies()

        self.logger.info("PerformanceBottleneckAnalyzer initialized")

    def _initialize_component_dependencies(self):
        """Initialize known component dependencies."""
        self._component_dependencies = {
            "cache_system": ["memory_system", "storage_system"],
            "query_router": ["cache_system", "database"],
            "search_service": ["cache_system", "embedding_service"],
            "embedding_service": ["memory_system", "cpu_system"],
            "indexing_service": ["storage_system", "memory_system", "cpu_system"],
            "graph_service": ["cache_system", "memory_system"],
        }

    async def start_analysis(self):
        """Start real-time bottleneck analysis."""
        if self._is_analyzing:
            return {"status": "already_running", "message": "Bottleneck analysis already running"}

        try:
            self._is_analyzing = True

            # Start analysis loop
            if self.config.enable_real_time_detection:
                self._analysis_task = asyncio.create_task(self._analysis_loop())

            # Capture baseline metrics
            await self._capture_performance_baselines()

            self.logger.info("Performance bottleneck analysis started")
            return {"status": "started", "message": "Bottleneck analysis activated"}

        except Exception as e:
            self.logger.error(f"Error starting bottleneck analysis: {e}")
            self._is_analyzing = False
            return {"status": "error", "message": f"Failed to start analysis: {e}"}

    async def stop_analysis(self):
        """Stop real-time bottleneck analysis."""
        self._is_analyzing = False

        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance bottleneck analysis stopped")
        return {"status": "stopped", "message": "Bottleneck analysis deactivated"}

    async def _analysis_loop(self):
        """Main analysis loop for real-time bottleneck detection."""
        self.logger.info("Bottleneck analysis loop started")

        while self._is_analyzing:
            try:
                # Perform bottleneck analysis
                await self._analyze_current_performance()

                # Wait before next analysis
                await asyncio.sleep(60.0)  # Analyze every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(60.0)

        self.logger.info("Bottleneck analysis loop stopped")

    async def _capture_performance_baselines(self):
        """Capture baseline performance metrics for comparison."""
        try:
            current_metrics = self.performance_monitor.get_current_metrics()

            # Group metrics by component
            for metric in current_metrics:
                component = metric.component
                metric_name = metric.metric_type.value

                if component not in self._performance_baselines:
                    self._performance_baselines[component] = {}

                self._performance_baselines[component][metric_name] = metric.value

            self.logger.info(f"Captured performance baselines for {len(self._performance_baselines)} components")

        except Exception as e:
            self.logger.error(f"Error capturing performance baselines: {e}")

    async def _analyze_current_performance(self):
        """Analyze current performance metrics for bottlenecks."""
        try:
            current_time = time.time()

            # Get recent performance metrics
            recent_metrics = self.performance_monitor.get_current_metrics()

            if not recent_metrics:
                return

            # Group metrics by component
            metrics_by_component = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_component[metric.component].append(metric)

            # Analyze each component for bottlenecks
            for component, metrics in metrics_by_component.items():
                await self._analyze_component_bottlenecks(component, metrics, current_time)

            # Perform cross-component analysis
            await self._analyze_system_wide_bottlenecks(metrics_by_component, current_time)

            # Update bottleneck priorities
            self._update_bottleneck_priorities()

        except Exception as e:
            self.logger.error(f"Error analyzing current performance: {e}")

    async def _analyze_component_bottlenecks(self, component: str, metrics: list, current_time: float):
        """Analyze a specific component for bottlenecks."""
        try:
            # Convert metrics to dict for easier access
            metric_values = {}
            for metric in metrics:
                metric_values[metric.metric_type.value] = metric.value

            # Run all bottleneck detection patterns
            for bottleneck_type, detection_func in self._bottleneck_patterns.items():
                try:
                    await detection_func(component, metric_values, current_time)
                except Exception as e:
                    self.logger.error(f"Error in {bottleneck_type.value} detection for {component}: {e}")

        except Exception as e:
            self.logger.error(f"Error analyzing component {component}: {e}")

    async def _detect_cpu_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect CPU-bound bottlenecks."""
        cpu_usage = metric_values.get("cpu_usage", 0)
        response_time = metric_values.get("response_time", 0)

        if cpu_usage > self.config.cpu_threshold_percent:
            # High CPU usage detected
            evidence = BottleneckEvidence(
                metric_name="cpu_usage",
                current_value=cpu_usage,
                threshold_value=self.config.cpu_threshold_percent,
                impact_score=min((cpu_usage - self.config.cpu_threshold_percent) * 2, 100),
                confidence=0.8,
                timestamp=current_time,
                metadata={"response_time": response_time},
            )

            # Check if this correlates with high response time
            if response_time > self.config.response_time_threshold_ms:
                evidence.confidence = 0.9
                evidence.impact_score = min(evidence.impact_score * 1.5, 100)

            await self._record_bottleneck(
                bottleneck_type=BottleneckType.CPU_BOUND,
                component=component,
                evidence=evidence,
                title=f"High CPU Usage in {component}",
                description=f"CPU usage ({cpu_usage:.1f}%) exceeds threshold ({self.config.cpu_threshold_percent}%)",
                impact_description="High CPU usage may cause slow response times and reduced throughput",
                resolution_suggestions=[
                    "Optimize CPU-intensive algorithms",
                    "Implement parallel processing",
                    "Add CPU resource scaling",
                    "Cache computation results",
                ],
            )

    async def _detect_memory_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect memory-bound bottlenecks."""
        memory_usage = metric_values.get("memory_usage", 0)

        if memory_usage > 0:  # Only analyze if we have memory data
            # Get system memory info for context
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_usage / (system_memory.total / 1024 / 1024)) * 100

            if memory_percent > self.config.memory_threshold_percent:
                evidence = BottleneckEvidence(
                    metric_name="memory_usage",
                    current_value=memory_usage,
                    threshold_value=(system_memory.total / 1024 / 1024) * (self.config.memory_threshold_percent / 100),
                    impact_score=min((memory_percent - self.config.memory_threshold_percent) * 3, 100),
                    confidence=0.85,
                    timestamp=current_time,
                    metadata={"memory_percent": memory_percent},
                )

                await self._record_bottleneck(
                    bottleneck_type=BottleneckType.MEMORY_BOUND,
                    component=component,
                    evidence=evidence,
                    title=f"High Memory Usage in {component}",
                    description=f"Memory usage ({memory_usage:.1f}MB, {memory_percent:.1f}%) exceeds threshold",
                    impact_description="High memory usage may cause garbage collection pressure and system slowdown",
                    resolution_suggestions=[
                        "Implement memory pooling",
                        "Optimize data structures",
                        "Add memory-based cache eviction",
                        "Reduce object allocations",
                    ],
                )

    async def _detect_io_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect I/O-bound bottlenecks."""
        response_time = metric_values.get("response_time", 0)
        disk_io = metric_values.get("disk_io", 0)

        # High response time might indicate I/O bottleneck
        if response_time > self.config.response_time_threshold_ms * 1.5:  # 1.5x threshold for I/O
            # Check if this is likely I/O bound vs CPU bound
            cpu_usage = metric_values.get("cpu_usage", 0)

            if cpu_usage < 50:  # Low CPU but high response time suggests I/O
                evidence = BottleneckEvidence(
                    metric_name="response_time",
                    current_value=response_time,
                    threshold_value=self.config.response_time_threshold_ms,
                    impact_score=min((response_time - self.config.response_time_threshold_ms) / 1000, 100),
                    confidence=0.7,
                    timestamp=current_time,
                    metadata={"cpu_usage": cpu_usage, "likely_io_bound": True},
                )

                await self._record_bottleneck(
                    bottleneck_type=BottleneckType.IO_BOUND,
                    component=component,
                    evidence=evidence,
                    title=f"I/O Bottleneck in {component}",
                    description=f"High response time ({response_time:.0f}ms) with low CPU usage suggests I/O bottleneck",
                    impact_description="I/O bottlenecks cause delays in data access and processing",
                    resolution_suggestions=[
                        "Implement asynchronous I/O",
                        "Add connection pooling",
                        "Optimize database queries",
                        "Implement data caching",
                    ],
                )

    async def _detect_cache_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect cache-related bottlenecks."""
        cache_hit_rate = metric_values.get("cache_hit_rate", 100)  # Default to 100% if no data

        if cache_hit_rate < self.config.cache_hit_rate_threshold:
            # Check cache metrics for more context
            cache_metrics = self.cache_monitor.get_aggregated_metrics()
            cache_context = {}

            if cache_metrics and "summary" in cache_metrics:
                summary = cache_metrics["summary"]
                cache_context = {
                    "total_operations": summary.get("total_operations", 0),
                    "cache_size_mb": summary.get("total_size_mb", 0),
                    "error_rate": summary.get("overall_error_rate", 0),
                }

            evidence = BottleneckEvidence(
                metric_name="cache_hit_rate",
                current_value=cache_hit_rate,
                threshold_value=self.config.cache_hit_rate_threshold,
                impact_score=min((self.config.cache_hit_rate_threshold - cache_hit_rate) * 2, 100),
                confidence=0.9,
                timestamp=current_time,
                metadata=cache_context,
            )

            await self._record_bottleneck(
                bottleneck_type=BottleneckType.CACHE_MISS,
                component=component,
                evidence=evidence,
                title=f"Low Cache Hit Rate in {component}",
                description=f"Cache hit rate ({cache_hit_rate:.1f}%) below threshold ({self.config.cache_hit_rate_threshold}%)",
                impact_description="Low cache hit rates increase response times and resource usage",
                resolution_suggestions=[
                    "Increase cache size limits",
                    "Optimize cache key generation",
                    "Implement cache warming strategies",
                    "Review cache TTL settings",
                ],
            )

    async def _detect_network_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect network-related bottlenecks."""
        response_time = metric_values.get("response_time", 0)
        network_io = metric_values.get("network_io", 0)
        error_rate = metric_values.get("error_rate", 0)

        # High error rate might indicate network issues
        if error_rate > self.config.error_rate_threshold:
            evidence = BottleneckEvidence(
                metric_name="error_rate",
                current_value=error_rate,
                threshold_value=self.config.error_rate_threshold,
                impact_score=min(error_rate * 10, 100),
                confidence=0.75,
                timestamp=current_time,
                metadata={"response_time": response_time},
            )

            await self._record_bottleneck(
                bottleneck_type=BottleneckType.NETWORK_LATENCY,
                component=component,
                evidence=evidence,
                title=f"High Error Rate in {component}",
                description=f"Error rate ({error_rate:.1f}%) exceeds threshold ({self.config.error_rate_threshold}%)",
                impact_description="High error rates may indicate network connectivity or latency issues",
                resolution_suggestions=[
                    "Implement retry mechanisms",
                    "Add connection health checks",
                    "Optimize network timeouts",
                    "Implement circuit breakers",
                ],
            )

    async def _detect_algorithm_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect algorithm complexity bottlenecks."""
        response_time = metric_values.get("response_time", 0)
        throughput = metric_values.get("throughput", 0)

        # High response time with low resource usage might indicate algorithmic issues
        cpu_usage = metric_values.get("cpu_usage", 0)
        memory_usage = metric_values.get("memory_usage", 0)

        if (
            response_time > self.config.response_time_threshold_ms * 2  # Very high response time
            and cpu_usage < 60  # Moderate CPU usage
            and memory_usage < 1024
        ):  # Moderate memory usage

            evidence = BottleneckEvidence(
                metric_name="response_time",
                current_value=response_time,
                threshold_value=self.config.response_time_threshold_ms,
                impact_score=min((response_time - self.config.response_time_threshold_ms) / 2000, 100),
                confidence=0.6,  # Lower confidence as this is harder to detect
                timestamp=current_time,
                metadata={"cpu_usage": cpu_usage, "memory_usage": memory_usage, "likely_algorithm_issue": True},
            )

            await self._record_bottleneck(
                bottleneck_type=BottleneckType.ALGORITHM_COMPLEXITY,
                component=component,
                evidence=evidence,
                title=f"Potential Algorithm Bottleneck in {component}",
                description=f"High response time ({response_time:.0f}ms) with moderate resource usage suggests algorithmic complexity",
                impact_description="Inefficient algorithms cause poor performance regardless of hardware resources",
                resolution_suggestions=[
                    "Profile and optimize algorithms",
                    "Implement more efficient data structures",
                    "Add algorithmic optimizations",
                    "Consider parallel processing",
                ],
            )

    async def _detect_resource_contention(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect resource contention bottlenecks."""
        # Look for patterns that suggest contention
        response_time = metric_values.get("response_time", 0)
        queue_size = metric_values.get("queue_size", 0)

        # High queue size with moderate response times suggests contention
        if queue_size > 10 and response_time > self.config.response_time_threshold_ms * 0.7:
            evidence = BottleneckEvidence(
                metric_name="queue_size",
                current_value=queue_size,
                threshold_value=5,  # Threshold for queue size
                impact_score=min(queue_size * 5, 100),
                confidence=0.8,
                timestamp=current_time,
                metadata={"response_time": response_time},
            )

            await self._record_bottleneck(
                bottleneck_type=BottleneckType.RESOURCE_CONTENTION,
                component=component,
                evidence=evidence,
                title=f"Resource Contention in {component}",
                description=f"High queue size ({queue_size}) indicates resource contention",
                impact_description="Resource contention causes requests to queue up and increases latency",
                resolution_suggestions=[
                    "Increase resource pool sizes",
                    "Implement load balancing",
                    "Add resource scaling",
                    "Optimize resource usage patterns",
                ],
            )

    async def _detect_dependency_bottleneck(self, component: str, metric_values: dict[str, float], current_time: float):
        """Detect bottlenecks caused by dependency chains."""
        response_time = metric_values.get("response_time", 0)

        if component in self._component_dependencies:
            dependencies = self._component_dependencies[component]

            # Check if dependencies have issues
            dependency_issues = []
            for dep in dependencies:
                if dep in self._detected_bottlenecks:
                    for bottleneck in self._detected_bottlenecks.values():
                        if bottleneck.component == dep and not bottleneck.resolved:
                            dependency_issues.append(dep)
                            break

            if dependency_issues and response_time > self.config.response_time_threshold_ms:
                evidence = BottleneckEvidence(
                    metric_name="response_time",
                    current_value=response_time,
                    threshold_value=self.config.response_time_threshold_ms,
                    impact_score=min(len(dependency_issues) * 30, 100),
                    confidence=0.85,
                    timestamp=current_time,
                    metadata={"dependency_issues": dependency_issues},
                )

                await self._record_bottleneck(
                    bottleneck_type=BottleneckType.DEPENDENCY_CHAIN,
                    component=component,
                    evidence=evidence,
                    title=f"Dependency Chain Bottleneck in {component}",
                    description=f"Performance issues in dependencies ({', '.join(dependency_issues)}) affecting {component}",
                    impact_description="Dependency bottlenecks cascade and multiply performance impact",
                    resolution_suggestions=[
                        "Optimize dependency performance",
                        "Implement dependency caching",
                        "Add circuit breakers for dependencies",
                        "Consider dependency alternatives",
                    ],
                )

    async def _analyze_system_wide_bottlenecks(self, metrics_by_component: dict[str, list], current_time: float):
        """Analyze system-wide patterns for bottlenecks."""
        try:
            # Look for system-wide patterns
            all_response_times = []
            all_cpu_usage = []
            all_memory_usage = []

            for component, metrics in metrics_by_component.items():
                for metric in metrics:
                    if metric.metric_type.value == "response_time":
                        all_response_times.append(metric.value)
                    elif metric.metric_type.value == "cpu_usage":
                        all_cpu_usage.append(metric.value)
                    elif metric.metric_type.value == "memory_usage":
                        all_memory_usage.append(metric.value)

            # Check for system-wide performance degradation
            if all_response_times:
                avg_response_time = statistics.mean(all_response_times)
                if avg_response_time > self.config.response_time_threshold_ms * 1.5:
                    # System-wide performance issue
                    await self._record_system_wide_bottleneck(
                        "System-wide Performance Degradation",
                        f"Average response time ({avg_response_time:.0f}ms) indicates system-wide performance issues",
                        current_time,
                    )

        except Exception as e:
            self.logger.error(f"Error analyzing system-wide bottlenecks: {e}")

    async def _record_bottleneck(
        self,
        bottleneck_type: BottleneckType,
        component: str,
        evidence: BottleneckEvidence,
        title: str,
        description: str,
        impact_description: str,
        resolution_suggestions: list[str],
    ):
        """Record a detected bottleneck."""
        try:
            # Check if evidence meets confidence threshold
            if evidence.confidence < self.config.evidence_confidence_threshold:
                return

            # Check if impact meets minimum threshold
            if evidence.impact_score < self.config.minimum_impact_score:
                return

            # Generate bottleneck ID
            bottleneck_key = f"{component}_{bottleneck_type.value}"

            # Check if this bottleneck already exists
            if bottleneck_key in self._detected_bottlenecks:
                # Update existing bottleneck
                existing = self._detected_bottlenecks[bottleneck_key]
                existing.add_evidence(evidence)

                # Update severity based on evidence
                if evidence.impact_score > 80:
                    existing.severity = BottleneckSeverity.CRITICAL
                elif evidence.impact_score > 60:
                    existing.severity = BottleneckSeverity.HIGH
                elif evidence.impact_score > 40:
                    existing.severity = BottleneckSeverity.MEDIUM
                else:
                    existing.severity = BottleneckSeverity.LOW

                existing.calculate_priority_score()

            else:
                # Create new bottleneck
                severity = BottleneckSeverity.LOW
                if evidence.impact_score > 80:
                    severity = BottleneckSeverity.CRITICAL
                elif evidence.impact_score > 60:
                    severity = BottleneckSeverity.HIGH
                elif evidence.impact_score > 40:
                    severity = BottleneckSeverity.MEDIUM

                bottleneck = PerformanceBottleneck(
                    bottleneck_id=f"bottleneck_{int(evidence.timestamp)}_{bottleneck_key}",
                    bottleneck_type=bottleneck_type,
                    component=component,
                    severity=severity,
                    title=title,
                    description=description,
                    impact_description=impact_description,
                    first_detected=evidence.timestamp,
                    last_detected=evidence.timestamp,
                    evidence=[evidence],
                    affected_metrics=[evidence.metric_name],
                    resolution_suggestions=resolution_suggestions,
                    estimated_impact_percent=min(evidence.impact_score, 100),
                )

                # Add dependency chain if applicable
                if component in self._component_dependencies:
                    bottleneck.dependency_chain = self._component_dependencies[component]

                bottleneck.calculate_priority_score()
                self._detected_bottlenecks[bottleneck_key] = bottleneck

                self.logger.warning(f"New bottleneck detected: {title} (Impact: {evidence.impact_score:.1f})")

        except Exception as e:
            self.logger.error(f"Error recording bottleneck: {e}")

    async def _record_system_wide_bottleneck(self, title: str, description: str, current_time: float):
        """Record a system-wide bottleneck."""
        try:
            evidence = BottleneckEvidence(
                metric_name="system_performance",
                current_value=100,  # System-wide issue
                threshold_value=80,
                impact_score=90,
                confidence=0.8,
                timestamp=current_time,
                metadata={"system_wide": True},
            )

            await self._record_bottleneck(
                bottleneck_type=BottleneckType.RESOURCE_CONTENTION,
                component="system",
                evidence=evidence,
                title=title,
                description=description,
                impact_description="System-wide performance issues affect all components",
                resolution_suggestions=[
                    "Investigate infrastructure capacity",
                    "Check for system-wide resource constraints",
                    "Review overall system architecture",
                    "Consider horizontal scaling",
                ],
            )

        except Exception as e:
            self.logger.error(f"Error recording system-wide bottleneck: {e}")

    def _update_bottleneck_priorities(self):
        """Update priority scores for all detected bottlenecks."""
        try:
            for bottleneck in self._detected_bottlenecks.values():
                if not bottleneck.resolved:
                    bottleneck.calculate_priority_score()

        except Exception as e:
            self.logger.error(f"Error updating bottleneck priorities: {e}")

    async def perform_manual_analysis(self, component: str | None = None, time_window_minutes: int = 30) -> dict[str, Any]:
        """
        Perform manual bottleneck analysis for a specific timeframe.

        Args:
            component: Specific component to analyze (None for all)
            time_window_minutes: Analysis time window

        Returns:
            Analysis results dictionary
        """
        try:
            analysis_start = time.time()
            cutoff_time = analysis_start - (time_window_minutes * 60)

            # Get metrics for the specified time window
            all_metrics = [m for m in self.performance_monitor._metrics if m.timestamp > cutoff_time]

            if component:
                all_metrics = [m for m in all_metrics if m.component == component]

            if not all_metrics:
                return {
                    "analysis_timestamp": analysis_start,
                    "time_window_minutes": time_window_minutes,
                    "component": component,
                    "bottlenecks_found": 0,
                    "message": "No metrics found for analysis period",
                }

            # Group metrics by component
            metrics_by_component = defaultdict(list)
            for metric in all_metrics:
                metrics_by_component[metric.component].append(metric)

            # Analyze each component
            found_bottlenecks = []
            for comp, metrics in metrics_by_component.items():
                # Calculate average values for analysis
                metric_averages = defaultdict(list)
                for metric in metrics:
                    metric_averages[metric.metric_type.value].append(metric.value)

                avg_metrics = {name: statistics.mean(values) for name, values in metric_averages.items()}

                # Run bottleneck detection
                temp_bottlenecks = {}
                original_bottlenecks = self._detected_bottlenecks.copy()
                self._detected_bottlenecks = {}

                await self._analyze_component_bottlenecks(comp, metrics, analysis_start)

                # Collect found bottlenecks
                for bottleneck in self._detected_bottlenecks.values():
                    found_bottlenecks.append(bottleneck.to_dict())

                # Restore original bottlenecks
                self._detected_bottlenecks = original_bottlenecks

            return {
                "analysis_timestamp": analysis_start,
                "time_window_minutes": time_window_minutes,
                "component": component,
                "metrics_analyzed": len(all_metrics),
                "components_analyzed": list(metrics_by_component.keys()),
                "bottlenecks_found": len(found_bottlenecks),
                "bottlenecks": found_bottlenecks,
                "analysis_duration_seconds": time.time() - analysis_start,
            }

        except Exception as e:
            self.logger.error(f"Error performing manual analysis: {e}")
            return {"error": str(e), "analysis_timestamp": time.time(), "component": component}

    def get_detected_bottlenecks(
        self, severity: BottleneckSeverity | None = None, component: str | None = None, resolved: bool | None = None
    ) -> list[dict[str, Any]]:
        """
        Get detected bottlenecks with optional filtering.

        Args:
            severity: Filter by severity level
            component: Filter by component name
            resolved: Filter by resolution status

        Returns:
            List of bottleneck dictionaries
        """
        try:
            bottlenecks = list(self._detected_bottlenecks.values())

            # Apply filters
            if severity:
                bottlenecks = [b for b in bottlenecks if b.severity == severity]

            if component:
                bottlenecks = [b for b in bottlenecks if b.component == component]

            if resolved is not None:
                bottlenecks = [b for b in bottlenecks if b.resolved == resolved]

            # Sort by priority score (highest first)
            bottlenecks.sort(key=lambda b: b.priority_score, reverse=True)

            return [b.to_dict() for b in bottlenecks]

        except Exception as e:
            self.logger.error(f"Error getting detected bottlenecks: {e}")
            return []

    def get_bottleneck_summary(self) -> dict[str, Any]:
        """Get a summary of detected bottlenecks."""
        try:
            bottlenecks = list(self._detected_bottlenecks.values())
            unresolved = [b for b in bottlenecks if not b.resolved]

            # Count by severity
            severity_counts = {
                "critical": len([b for b in unresolved if b.severity == BottleneckSeverity.CRITICAL]),
                "high": len([b for b in unresolved if b.severity == BottleneckSeverity.HIGH]),
                "medium": len([b for b in unresolved if b.severity == BottleneckSeverity.MEDIUM]),
                "low": len([b for b in unresolved if b.severity == BottleneckSeverity.LOW]),
            }

            # Count by type
            type_counts = defaultdict(int)
            for bottleneck in unresolved:
                type_counts[bottleneck.bottleneck_type.value] += 1

            # Count by component
            component_counts = defaultdict(int)
            for bottleneck in unresolved:
                component_counts[bottleneck.component] += 1

            # Get top priority bottlenecks
            top_bottlenecks = sorted(unresolved, key=lambda b: b.priority_score, reverse=True)[:5]

            return {
                "analysis_timestamp": time.time(),
                "total_bottlenecks": len(bottlenecks),
                "unresolved_bottlenecks": len(unresolved),
                "resolved_bottlenecks": len(bottlenecks) - len(unresolved),
                "severity_breakdown": severity_counts,
                "type_breakdown": dict(type_counts),
                "component_breakdown": dict(component_counts),
                "top_priority_bottlenecks": [
                    {
                        "bottleneck_id": b.bottleneck_id,
                        "title": b.title,
                        "component": b.component,
                        "severity": b.severity.value,
                        "priority_score": b.priority_score,
                        "estimated_impact_percent": b.estimated_impact_percent,
                    }
                    for b in top_bottlenecks
                ],
                "analysis_status": "active" if self._is_analyzing else "inactive",
            }

        except Exception as e:
            self.logger.error(f"Error getting bottleneck summary: {e}")
            return {"error": str(e)}

    async def acknowledge_bottleneck(self, bottleneck_id: str, user: str = "system") -> dict[str, Any]:
        """Acknowledge a detected bottleneck."""
        try:
            for bottleneck in self._detected_bottlenecks.values():
                if bottleneck.bottleneck_id == bottleneck_id:
                    bottleneck.acknowledged = True
                    self.logger.info(f"Bottleneck {bottleneck_id} acknowledged by {user}")

                    return {
                        "success": True,
                        "message": f"Bottleneck acknowledged by {user}",
                        "bottleneck_id": bottleneck_id,
                        "timestamp": time.time(),
                    }

            return {"success": False, "message": "Bottleneck not found", "bottleneck_id": bottleneck_id}

        except Exception as e:
            self.logger.error(f"Error acknowledging bottleneck: {e}")
            return {"success": False, "message": str(e)}

    async def resolve_bottleneck(self, bottleneck_id: str, resolution_notes: str = "", user: str = "system") -> dict[str, Any]:
        """Mark a bottleneck as resolved."""
        try:
            for bottleneck in self._detected_bottlenecks.values():
                if bottleneck.bottleneck_id == bottleneck_id:
                    bottleneck.resolved = True
                    bottleneck.resolution_notes = resolution_notes
                    self.logger.info(f"Bottleneck {bottleneck_id} resolved by {user}")

                    # Move to history
                    self._bottleneck_history.append(bottleneck.to_dict())

                    return {
                        "success": True,
                        "message": f"Bottleneck resolved by {user}",
                        "bottleneck_id": bottleneck_id,
                        "resolution_notes": resolution_notes,
                        "timestamp": time.time(),
                    }

            return {"success": False, "message": "Bottleneck not found", "bottleneck_id": bottleneck_id}

        except Exception as e:
            self.logger.error(f"Error resolving bottleneck: {e}")
            return {"success": False, "message": str(e)}

    async def shutdown(self):
        """Shutdown the bottleneck analyzer."""
        self.logger.info("Shutting down PerformanceBottleneckAnalyzer")
        await self.stop_analysis()

        # Clear state
        self._detected_bottlenecks.clear()
        self._bottleneck_history.clear()
        self._analysis_cache.clear()
        self._performance_baselines.clear()

        self.logger.info("PerformanceBottleneckAnalyzer shutdown complete")


# Global analyzer instance
_bottleneck_analyzer: PerformanceBottleneckAnalyzer | None = None


def get_bottleneck_analyzer() -> PerformanceBottleneckAnalyzer:
    """Get the global bottleneck analyzer instance."""
    global _bottleneck_analyzer
    if _bottleneck_analyzer is None:
        _bottleneck_analyzer = PerformanceBottleneckAnalyzer()
    return _bottleneck_analyzer
