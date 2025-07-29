"""
Performance Optimization Recommendations Engine - Wave 5.0 Implementation.

Provides intelligent performance optimization recommendations including:
- AI-driven performance analysis and recommendations
- Pattern-based optimization suggestions
- Historical performance trend analysis
- Automated optimization opportunity detection
- Resource optimization recommendations
- Performance best practices enforcement
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

from src.services.performance_monitor import get_performance_monitor
from src.tools.performance.bottleneck_analyzer import get_bottleneck_analyzer
from src.tools.performance.data_collector import get_data_collector
from src.utils.performance_monitor import get_cache_performance_monitor


class RecommendationType(Enum):
    """Types of optimization recommendations."""

    CACHE_OPTIMIZATION = "cache_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    RESOURCE_SCALING = "resource_scaling"
    CONFIGURATION_TUNING = "configuration_tuning"
    ARCHITECTURE_IMPROVEMENT = "architecture_improvement"
    MONITORING_IMPROVEMENT = "monitoring_improvement"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    CUSTOM = "custom"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationStatus(Enum):
    """Status of recommendation implementation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    OBSOLETE = "obsolete"


@dataclass
class OptimizationRecommendation:
    """Represents a performance optimization recommendation."""

    recommendation_id: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority

    # Basic information
    title: str
    description: str
    rationale: str

    # Impact analysis
    expected_improvement_percent: float = 0.0
    confidence_score: float = 0.0  # 0-1 confidence in the recommendation
    implementation_effort: str = "medium"  # low, medium, high

    # Supporting evidence
    affected_components: list[str] = field(default_factory=list)
    supporting_metrics: list[str] = field(default_factory=list)
    baseline_metrics: dict[str, float] = field(default_factory=dict)

    # Implementation details
    implementation_steps: list[str] = field(default_factory=list)
    required_resources: list[str] = field(default_factory=list)
    estimated_cost: str = "low"  # low, medium, high
    estimated_time_hours: float = 0.0

    # Risk assessment
    risk_level: str = "low"  # low, medium, high
    potential_side_effects: list[str] = field(default_factory=list)
    rollback_plan: str = ""

    # Tracking
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: RecommendationStatus = RecommendationStatus.PENDING
    implementation_notes: str = ""

    # Related recommendations
    depends_on: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)

    def update_status(self, status: RecommendationStatus, notes: str = ""):
        """Update the recommendation status."""
        self.status = status
        self.updated_at = time.time()
        if notes:
            self.implementation_notes = notes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_id": self.recommendation_id,
            "recommendation_type": self.recommendation_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "expected_improvement_percent": self.expected_improvement_percent,
            "confidence_score": self.confidence_score,
            "implementation_effort": self.implementation_effort,
            "affected_components": self.affected_components,
            "supporting_metrics": self.supporting_metrics,
            "baseline_metrics": self.baseline_metrics,
            "implementation_steps": self.implementation_steps,
            "required_resources": self.required_resources,
            "estimated_cost": self.estimated_cost,
            "estimated_time_hours": self.estimated_time_hours,
            "risk_level": self.risk_level,
            "potential_side_effects": self.potential_side_effects,
            "rollback_plan": self.rollback_plan,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "implementation_notes": self.implementation_notes,
            "depends_on": self.depends_on,
            "conflicts_with": self.conflicts_with,
        }


@dataclass
class OptimizationPattern:
    """Defines a pattern for generating optimization recommendations."""

    pattern_id: str
    name: str
    description: str
    recommendation_type: RecommendationType

    # Detection criteria
    trigger_conditions: dict[str, Any] = field(default_factory=dict)
    metric_thresholds: dict[str, float] = field(default_factory=dict)
    pattern_indicators: list[str] = field(default_factory=list)

    # Recommendation template
    title_template: str = ""
    description_template: str = ""
    rationale_template: str = ""

    # Default values
    default_priority: RecommendationPriority = RecommendationPriority.MEDIUM
    default_confidence: float = 0.7
    default_effort: str = "medium"

    def matches_condition(self, metrics: dict[str, float], context: dict[str, Any]) -> bool:
        """Check if current metrics match this pattern's conditions."""
        try:
            # Check metric thresholds
            for metric_name, threshold in self.metric_thresholds.items():
                if metric_name in metrics:
                    comparison = self.trigger_conditions.get(f"{metric_name}_comparison", "gt")
                    if comparison == "gt" and metrics[metric_name] <= threshold:
                        return False
                    elif comparison == "lt" and metrics[metric_name] >= threshold:
                        return False
                    elif comparison == "eq" and metrics[metric_name] != threshold:
                        return False
                else:
                    return False  # Required metric not available

            # Check additional conditions
            for condition_key, condition_value in self.trigger_conditions.items():
                if not condition_key.endswith("_comparison"):
                    if condition_key in context:
                        if context[condition_key] != condition_value:
                            return False
                    else:
                        return False

            return True

        except Exception:
            return False


class PerformanceOptimizationEngine:
    """
    Intelligent performance optimization recommendations engine.

    Analyzes performance data, identifies optimization opportunities,
    and generates actionable recommendations with impact assessment.
    """

    def __init__(self):
        """Initialize the optimization engine."""
        self.logger = logging.getLogger(__name__)

        # Integration with monitoring systems
        self.performance_monitor = get_performance_monitor()
        self.cache_monitor = get_cache_performance_monitor()
        self.bottleneck_analyzer = get_bottleneck_analyzer()
        self.data_collector = get_data_collector()

        # Recommendation management
        self._recommendations: dict[str, OptimizationRecommendation] = {}
        self._recommendation_history: deque = deque(maxlen=1000)
        self._optimization_patterns: list[OptimizationPattern] = []

        # Analysis state
        self._analysis_task: asyncio.Task | None = None
        self._is_analyzing = False
        self._last_analysis_time = 0.0

        # Performance baselines and trends
        self._performance_baselines: dict[str, dict[str, float]] = {}
        self._performance_trends: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Recommendation statistics
        self._recommendation_stats = {
            "total_generated": 0,
            "implemented": 0,
            "rejected": 0,
            "average_confidence": 0.0,
            "average_improvement": 0.0,
        }

        # Initialize optimization patterns
        self._initialize_optimization_patterns()

        self.logger.info("PerformanceOptimizationEngine initialized")

    def _initialize_optimization_patterns(self):
        """Initialize built-in optimization patterns."""
        patterns = [
            # Cache optimization patterns
            OptimizationPattern(
                pattern_id="low_cache_hit_rate",
                name="Low Cache Hit Rate",
                description="Detects low cache hit rates and recommends cache optimization",
                recommendation_type=RecommendationType.CACHE_OPTIMIZATION,
                metric_thresholds={"cache_hit_rate": 70.0},
                trigger_conditions={"cache_hit_rate_comparison": "lt"},
                title_template="Improve Cache Hit Rate for {component}",
                description_template="Cache hit rate is {hit_rate:.1f}%, below optimal threshold of 70%",
                rationale_template="Low cache hit rates increase response times and resource usage",
                default_priority=RecommendationPriority.HIGH,
                default_confidence=0.9,
            ),
            # Memory optimization patterns
            OptimizationPattern(
                pattern_id="high_memory_usage",
                name="High Memory Usage",
                description="Detects high memory usage and recommends memory optimization",
                recommendation_type=RecommendationType.MEMORY_OPTIMIZATION,
                metric_thresholds={"memory_usage": 2048.0},  # 2GB
                trigger_conditions={"memory_usage_comparison": "gt"},
                title_template="Optimize Memory Usage for {component}",
                description_template="Memory usage is {memory_usage:.0f}MB, exceeding recommended limits",
                rationale_template="High memory usage can cause garbage collection pressure and system slowdown",
                default_priority=RecommendationPriority.HIGH,
                default_confidence=0.8,
            ),
            # Response time patterns
            OptimizationPattern(
                pattern_id="slow_response_time",
                name="Slow Response Time",
                description="Detects slow response times and recommends optimization",
                recommendation_type=RecommendationType.ALGORITHM_OPTIMIZATION,
                metric_thresholds={"response_time": 15000.0},  # 15 seconds
                trigger_conditions={"response_time_comparison": "gt"},
                title_template="Improve Response Time for {component}",
                description_template="Response time is {response_time:.0f}ms, exceeding target of 15 seconds",
                rationale_template="Slow response times impact user experience and system throughput",
                default_priority=RecommendationPriority.HIGH,
                default_confidence=0.8,
            ),
            # CPU optimization patterns
            OptimizationPattern(
                pattern_id="high_cpu_usage",
                name="High CPU Usage",
                description="Detects high CPU usage and recommends optimization",
                recommendation_type=RecommendationType.CPU_OPTIMIZATION,
                metric_thresholds={"cpu_usage": 80.0},
                trigger_conditions={"cpu_usage_comparison": "gt"},
                title_template="Optimize CPU Usage for {component}",
                description_template="CPU usage is {cpu_usage:.1f}%, above recommended threshold",
                rationale_template="High CPU usage can cause performance bottlenecks and reduced throughput",
                default_priority=RecommendationPriority.MEDIUM,
                default_confidence=0.7,
            ),
            # Error rate patterns
            OptimizationPattern(
                pattern_id="high_error_rate",
                name="High Error Rate",
                description="Detects high error rates and recommends reliability improvements",
                recommendation_type=RecommendationType.MONITORING_IMPROVEMENT,
                metric_thresholds={"error_rate": 5.0},
                trigger_conditions={"error_rate_comparison": "gt"},
                title_template="Reduce Error Rate for {component}",
                description_template="Error rate is {error_rate:.1f}%, above acceptable threshold",
                rationale_template="High error rates indicate system reliability issues",
                default_priority=RecommendationPriority.CRITICAL,
                default_confidence=0.9,
            ),
        ]

        self._optimization_patterns.extend(patterns)

    async def start_analysis(self):
        """Start continuous optimization analysis."""
        if self._is_analyzing:
            return {"status": "already_running", "message": "Optimization analysis already running"}

        try:
            self._is_analyzing = True

            # Start analysis loop
            self._analysis_task = asyncio.create_task(self._analysis_loop())

            # Capture initial baselines
            await self._capture_performance_baselines()

            self.logger.info("Performance optimization analysis started")
            return {"status": "started", "message": "Optimization analysis activated"}

        except Exception as e:
            self.logger.error(f"Error starting optimization analysis: {e}")
            self._is_analyzing = False
            return {"status": "error", "message": f"Failed to start analysis: {e}"}

    async def stop_analysis(self):
        """Stop continuous optimization analysis."""
        self._is_analyzing = False

        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Performance optimization analysis stopped")
        return {"status": "stopped", "message": "Optimization analysis deactivated"}

    async def _analysis_loop(self):
        """Main optimization analysis loop."""
        self.logger.info("Optimization analysis loop started")

        while self._is_analyzing:
            try:
                # Perform optimization analysis
                await self._analyze_optimization_opportunities()

                # Update trends
                await self._update_performance_trends()

                # Clean up obsolete recommendations
                await self._cleanup_obsolete_recommendations()

                # Wait before next analysis
                await asyncio.sleep(300.0)  # Analyze every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization analysis loop: {e}")
                await asyncio.sleep(300.0)

        self.logger.info("Optimization analysis loop stopped")

    async def _capture_performance_baselines(self):
        """Capture baseline performance metrics."""
        try:
            current_metrics = self.performance_monitor.get_current_metrics()

            for metric in current_metrics:
                component = metric.component
                metric_name = metric.metric_type.value

                if component not in self._performance_baselines:
                    self._performance_baselines[component] = {}

                self._performance_baselines[component][metric_name] = metric.value

            self.logger.info(f"Captured performance baselines for {len(self._performance_baselines)} components")

        except Exception as e:
            self.logger.error(f"Error capturing performance baselines: {e}")

    async def _analyze_optimization_opportunities(self):
        """Analyze current performance for optimization opportunities."""
        try:
            current_time = time.time()
            self._last_analysis_time = current_time

            # Get current performance data
            current_metrics = self.performance_monitor.get_current_metrics()
            cache_metrics = self.cache_monitor.get_aggregated_metrics()
            bottlenecks = self.bottleneck_analyzer.get_detected_bottlenecks()

            # Convert metrics to analysis format
            metrics_by_component = self._prepare_metrics_for_analysis(current_metrics, cache_metrics)

            # Generate recommendations based on patterns
            for component, metrics in metrics_by_component.items():
                await self._analyze_component_optimization(component, metrics, bottlenecks)

            # Analyze system-wide optimization opportunities
            await self._analyze_system_wide_optimization(metrics_by_component, bottlenecks)

            # Analyze historical trends for predictive recommendations
            await self._analyze_trend_based_optimization()

        except Exception as e:
            self.logger.error(f"Error analyzing optimization opportunities: {e}")

    def _prepare_metrics_for_analysis(self, current_metrics: list, cache_metrics: dict[str, Any] | None) -> dict[str, dict[str, float]]:
        """Prepare metrics for optimization analysis."""
        try:
            metrics_by_component = defaultdict(dict)

            # Process performance monitor metrics
            for metric in current_metrics:
                metrics_by_component[metric.component][metric.metric_type.value] = metric.value

            # Add cache metrics
            if cache_metrics and "summary" in cache_metrics:
                summary = cache_metrics["summary"]
                cache_component_metrics = {
                    "cache_hit_rate": summary.get("overall_hit_rate", 0.0) * 100,
                    "cache_operations": summary.get("total_operations", 0),
                    "cache_size": summary.get("total_size_mb", 0.0),
                    "cache_error_rate": summary.get("overall_error_rate", 0.0) * 100,
                    "cache_response_time": summary.get("average_response_time_ms", 0.0),
                }
                metrics_by_component["cache_system"].update(cache_component_metrics)

            return dict(metrics_by_component)

        except Exception as e:
            self.logger.error(f"Error preparing metrics for analysis: {e}")
            return {}

    async def _analyze_component_optimization(self, component: str, metrics: dict[str, float], bottlenecks: list[dict[str, Any]]):
        """Analyze optimization opportunities for a specific component."""
        try:
            # Check each optimization pattern
            for pattern in self._optimization_patterns:
                context = {"component": component, "bottlenecks": bottlenecks}

                if pattern.matches_condition(metrics, context):
                    await self._generate_pattern_recommendation(pattern, component, metrics, context)

            # Check for component-specific bottlenecks
            component_bottlenecks = [b for b in bottlenecks if b.get("component") == component]

            for bottleneck in component_bottlenecks:
                await self._generate_bottleneck_recommendation(bottleneck, metrics)

        except Exception as e:
            self.logger.error(f"Error analyzing component optimization for {component}: {e}")

    async def _generate_pattern_recommendation(
        self, pattern: OptimizationPattern, component: str, metrics: dict[str, float], context: dict[str, Any]
    ):
        """Generate a recommendation based on a detected pattern."""
        try:
            recommendation_id = f"{pattern.pattern_id}_{component}_{int(time.time())}"

            # Check if similar recommendation already exists
            existing_rec = self._find_similar_recommendation(pattern.pattern_id, component)
            if existing_rec and existing_rec.status == RecommendationStatus.PENDING:
                return  # Don't create duplicate recommendations

            # Format template strings
            title = pattern.title_template.format(component=component, **metrics)
            description = pattern.description_template.format(component=component, **metrics)
            rationale = pattern.rationale_template.format(component=component, **metrics)

            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(pattern, metrics)

            # Generate implementation steps
            implementation_steps = self._generate_implementation_steps(pattern, component, metrics)

            # Create recommendation
            recommendation = OptimizationRecommendation(
                recommendation_id=recommendation_id,
                recommendation_type=pattern.recommendation_type,
                priority=pattern.default_priority,
                title=title,
                description=description,
                rationale=rationale,
                expected_improvement_percent=expected_improvement,
                confidence_score=pattern.default_confidence,
                implementation_effort=pattern.default_effort,
                affected_components=[component],
                supporting_metrics=list(pattern.metric_thresholds.keys()),
                baseline_metrics=metrics.copy(),
                implementation_steps=implementation_steps,
                required_resources=self._determine_required_resources(pattern),
                estimated_cost=self._estimate_implementation_cost(pattern),
                estimated_time_hours=self._estimate_implementation_time(pattern),
                risk_level=self._assess_risk_level(pattern),
                potential_side_effects=self._identify_potential_side_effects(pattern),
                rollback_plan=self._generate_rollback_plan(pattern),
            )

            # Store recommendation
            self._recommendations[recommendation_id] = recommendation
            self._recommendation_stats["total_generated"] += 1

            self.logger.info(f"Generated optimization recommendation: {title}")

        except Exception as e:
            self.logger.error(f"Error generating pattern recommendation: {e}")

    def _find_similar_recommendation(self, pattern_id: str, component: str) -> OptimizationRecommendation | None:
        """Find similar existing recommendation."""
        for rec in self._recommendations.values():
            if (
                pattern_id in rec.recommendation_id
                and component in rec.affected_components
                and rec.status in [RecommendationStatus.PENDING, RecommendationStatus.IN_PROGRESS]
            ):
                return rec
        return None

    def _calculate_expected_improvement(self, pattern: OptimizationPattern, metrics: dict[str, float]) -> float:
        """Calculate expected performance improvement percentage."""
        try:
            # Base improvement estimates by pattern type
            base_improvements = {
                RecommendationType.CACHE_OPTIMIZATION: 25.0,
                RecommendationType.MEMORY_OPTIMIZATION: 15.0,
                RecommendationType.CPU_OPTIMIZATION: 20.0,
                RecommendationType.ALGORITHM_OPTIMIZATION: 30.0,
                RecommendationType.RESOURCE_SCALING: 40.0,
                RecommendationType.CONFIGURATION_TUNING: 10.0,
            }

            base_improvement = base_improvements.get(pattern.recommendation_type, 15.0)

            # Adjust based on severity of the issue
            severity_multiplier = 1.0
            for metric_name, threshold in pattern.metric_thresholds.items():
                if metric_name in metrics:
                    current_value = metrics[metric_name]
                    comparison = pattern.trigger_conditions.get(f"{metric_name}_comparison", "gt")

                    if comparison == "gt" and current_value > threshold:
                        severity_multiplier = max(severity_multiplier, current_value / threshold)
                    elif comparison == "lt" and current_value < threshold:
                        severity_multiplier = max(severity_multiplier, threshold / current_value)

            # Cap the improvement estimate
            improvement = min(base_improvement * min(severity_multiplier, 2.0), 80.0)

            return round(improvement, 1)

        except Exception:
            return 15.0  # Default improvement estimate

    def _generate_implementation_steps(self, pattern: OptimizationPattern, component: str, metrics: dict[str, float]) -> list[str]:
        """Generate implementation steps for a recommendation."""
        try:
            steps = []

            if pattern.recommendation_type == RecommendationType.CACHE_OPTIMIZATION:
                steps = [
                    "Analyze current cache configuration and usage patterns",
                    "Increase cache size limits if memory allows",
                    "Optimize cache key generation strategy",
                    "Implement cache warming for frequently accessed data",
                    "Review and adjust cache TTL settings",
                    "Monitor cache hit rate improvements",
                ]

            elif pattern.recommendation_type == RecommendationType.MEMORY_OPTIMIZATION:
                steps = [
                    "Profile memory usage to identify hotspots",
                    "Implement memory pooling for frequently allocated objects",
                    "Optimize data structures to reduce memory footprint",
                    "Add memory-based cache eviction policies",
                    "Review object lifecycle and implement proper cleanup",
                    "Monitor memory usage trends",
                ]

            elif pattern.recommendation_type == RecommendationType.CPU_OPTIMIZATION:
                steps = [
                    "Profile CPU usage to identify bottlenecks",
                    "Optimize CPU-intensive algorithms",
                    "Implement parallel processing where applicable",
                    "Add CPU resource scaling mechanisms",
                    "Cache computation results to avoid redundant work",
                    "Monitor CPU utilization improvements",
                ]

            elif pattern.recommendation_type == RecommendationType.ALGORITHM_OPTIMIZATION:
                steps = [
                    "Profile application to identify slow operations",
                    "Analyze algorithm complexity and data access patterns",
                    "Implement more efficient algorithms and data structures",
                    "Add request batching and optimization",
                    "Implement asynchronous processing where possible",
                    "Measure performance improvements",
                ]

            else:
                steps = [
                    f"Analyze {component} performance metrics",
                    "Identify specific optimization opportunities",
                    "Implement targeted optimizations",
                    "Test performance improvements",
                    "Monitor ongoing performance",
                ]

            return steps

        except Exception as e:
            self.logger.error(f"Error generating implementation steps: {e}")
            return ["Analyze and optimize component performance"]

    def _determine_required_resources(self, pattern: OptimizationPattern) -> list[str]:
        """Determine resources required for implementation."""
        base_resources = ["Development time", "Testing environment"]

        if pattern.recommendation_type == RecommendationType.RESOURCE_SCALING:
            base_resources.extend(["Additional server capacity", "Load balancer configuration"])

        if pattern.recommendation_type == RecommendationType.CACHE_OPTIMIZATION:
            base_resources.append("Additional memory allocation")

        if pattern.recommendation_type == RecommendationType.ALGORITHM_OPTIMIZATION:
            base_resources.extend(["Performance profiling tools", "Code review"])

        return base_resources

    def _estimate_implementation_cost(self, pattern: OptimizationPattern) -> str:
        """Estimate implementation cost."""
        cost_mapping = {
            RecommendationType.CONFIGURATION_TUNING: "low",
            RecommendationType.CACHE_OPTIMIZATION: "low",
            RecommendationType.MEMORY_OPTIMIZATION: "medium",
            RecommendationType.CPU_OPTIMIZATION: "medium",
            RecommendationType.ALGORITHM_OPTIMIZATION: "high",
            RecommendationType.RESOURCE_SCALING: "medium",
            RecommendationType.ARCHITECTURE_IMPROVEMENT: "high",
        }

        return cost_mapping.get(pattern.recommendation_type, "medium")

    def _estimate_implementation_time(self, pattern: OptimizationPattern) -> float:
        """Estimate implementation time in hours."""
        time_mapping = {
            RecommendationType.CONFIGURATION_TUNING: 2.0,
            RecommendationType.CACHE_OPTIMIZATION: 8.0,
            RecommendationType.MEMORY_OPTIMIZATION: 16.0,
            RecommendationType.CPU_OPTIMIZATION: 24.0,
            RecommendationType.ALGORITHM_OPTIMIZATION: 40.0,
            RecommendationType.RESOURCE_SCALING: 16.0,
            RecommendationType.ARCHITECTURE_IMPROVEMENT: 80.0,
        }

        return time_mapping.get(pattern.recommendation_type, 16.0)

    def _assess_risk_level(self, pattern: OptimizationPattern) -> str:
        """Assess implementation risk level."""
        risk_mapping = {
            RecommendationType.CONFIGURATION_TUNING: "low",
            RecommendationType.CACHE_OPTIMIZATION: "low",
            RecommendationType.MEMORY_OPTIMIZATION: "medium",
            RecommendationType.CPU_OPTIMIZATION: "medium",
            RecommendationType.ALGORITHM_OPTIMIZATION: "high",
            RecommendationType.RESOURCE_SCALING: "medium",
            RecommendationType.ARCHITECTURE_IMPROVEMENT: "high",
        }

        return risk_mapping.get(pattern.recommendation_type, "medium")

    def _identify_potential_side_effects(self, pattern: OptimizationPattern) -> list[str]:
        """Identify potential side effects of implementation."""
        side_effects_mapping = {
            RecommendationType.CACHE_OPTIMIZATION: ["Increased memory usage", "Potential cache invalidation issues"],
            RecommendationType.MEMORY_OPTIMIZATION: ["Possible performance impact during implementation", "Changed memory access patterns"],
            RecommendationType.CPU_OPTIMIZATION: ["Potential impact on other processes", "Changed CPU utilization patterns"],
            RecommendationType.ALGORITHM_OPTIMIZATION: ["Risk of introducing bugs", "Changed system behavior patterns"],
            RecommendationType.RESOURCE_SCALING: ["Increased infrastructure costs", "Potential network latency changes"],
        }

        return side_effects_mapping.get(pattern.recommendation_type, ["Minimal side effects expected"])

    def _generate_rollback_plan(self, pattern: OptimizationPattern) -> str:
        """Generate rollback plan for the optimization."""
        rollback_plans = {
            RecommendationType.CONFIGURATION_TUNING: "Revert configuration settings to previous values",
            RecommendationType.CACHE_OPTIMIZATION: "Restore original cache configuration and clear cache",
            RecommendationType.MEMORY_OPTIMIZATION: "Revert memory management changes and restart services",
            RecommendationType.CPU_OPTIMIZATION: "Disable optimization flags and revert to original algorithms",
            RecommendationType.ALGORITHM_OPTIMIZATION: "Deploy previous version of optimized components",
            RecommendationType.RESOURCE_SCALING: "Scale back resources to original capacity",
        }

        return rollback_plans.get(pattern.recommendation_type, "Document current state and implement gradual rollback procedures")

    async def _generate_bottleneck_recommendation(self, bottleneck: dict[str, Any], metrics: dict[str, float]):
        """Generate recommendations based on detected bottlenecks."""
        try:
            recommendation_id = f"bottleneck_{bottleneck['bottleneck_id']}_{int(time.time())}"

            # Map bottleneck types to recommendation types
            type_mapping = {
                "cpu_bound": RecommendationType.CPU_OPTIMIZATION,
                "memory_bound": RecommendationType.MEMORY_OPTIMIZATION,
                "cache_miss": RecommendationType.CACHE_OPTIMIZATION,
                "algorithm_complexity": RecommendationType.ALGORITHM_OPTIMIZATION,
                "resource_contention": RecommendationType.RESOURCE_SCALING,
            }

            rec_type = type_mapping.get(bottleneck.get("bottleneck_type", ""), RecommendationType.CONFIGURATION_TUNING)

            # Map severity to priority
            severity_to_priority = {
                "critical": RecommendationPriority.CRITICAL,
                "high": RecommendationPriority.HIGH,
                "medium": RecommendationPriority.MEDIUM,
                "low": RecommendationPriority.LOW,
            }

            priority = severity_to_priority.get(bottleneck.get("severity", "medium"), RecommendationPriority.MEDIUM)

            recommendation = OptimizationRecommendation(
                recommendation_id=recommendation_id,
                recommendation_type=rec_type,
                priority=priority,
                title=f"Resolve {bottleneck.get('title', 'Performance Bottleneck')}",
                description=bottleneck.get("description", "Address detected performance bottleneck"),
                rationale=bottleneck.get("impact_description", "Bottleneck is impacting system performance"),
                expected_improvement_percent=bottleneck.get("estimated_impact_percent", 20.0),
                confidence_score=0.8,
                implementation_effort="medium",
                affected_components=[bottleneck.get("component", "unknown")],
                supporting_metrics=bottleneck.get("affected_metrics", []),
                baseline_metrics=metrics.copy(),
                implementation_steps=bottleneck.get("resolution_suggestions", []),
                required_resources=["Development time", "Analysis tools"],
                estimated_cost="medium",
                estimated_time_hours=16.0,
                risk_level="medium",
                potential_side_effects=["Temporary performance impact during implementation"],
                rollback_plan="Monitor performance and revert changes if issues arise",
            )

            self._recommendations[recommendation_id] = recommendation
            self._recommendation_stats["total_generated"] += 1

            self.logger.info(f"Generated bottleneck recommendation: {recommendation.title}")

        except Exception as e:
            self.logger.error(f"Error generating bottleneck recommendation: {e}")

    async def _analyze_system_wide_optimization(self, metrics_by_component: dict[str, dict[str, float]], bottlenecks: list[dict[str, Any]]):
        """Analyze system-wide optimization opportunities."""
        try:
            # Check for system-wide patterns
            all_response_times = []
            all_memory_usage = []
            all_error_rates = []

            for component, metrics in metrics_by_component.items():
                if "response_time" in metrics:
                    all_response_times.append(metrics["response_time"])
                if "memory_usage" in metrics:
                    all_memory_usage.append(metrics["memory_usage"])
                if "error_rate" in metrics:
                    all_error_rates.append(metrics["error_rate"])

            # System-wide performance degradation
            if all_response_times and statistics.mean(all_response_times) > 20000:  # 20 seconds
                await self._generate_system_wide_recommendation(
                    "System-wide Performance Optimization",
                    "Multiple components showing slow response times",
                    RecommendationType.ARCHITECTURE_IMPROVEMENT,
                    RecommendationPriority.HIGH,
                )

            # System-wide memory pressure
            if all_memory_usage and statistics.mean(all_memory_usage) > 1500:  # 1.5GB average
                await self._generate_system_wide_recommendation(
                    "System-wide Memory Optimization",
                    "High memory usage across multiple components",
                    RecommendationType.MEMORY_OPTIMIZATION,
                    RecommendationPriority.HIGH,
                )

            # Multiple bottlenecks
            if len(bottlenecks) >= 3:
                await self._generate_system_wide_recommendation(
                    "Comprehensive Performance Review",
                    f"Multiple bottlenecks detected ({len(bottlenecks)} total)",
                    RecommendationType.ARCHITECTURE_IMPROVEMENT,
                    RecommendationPriority.CRITICAL,
                )

        except Exception as e:
            self.logger.error(f"Error analyzing system-wide optimization: {e}")

    async def _generate_system_wide_recommendation(
        self, title: str, description: str, rec_type: RecommendationType, priority: RecommendationPriority
    ):
        """Generate a system-wide optimization recommendation."""
        try:
            recommendation_id = f"system_wide_{int(time.time())}"

            recommendation = OptimizationRecommendation(
                recommendation_id=recommendation_id,
                recommendation_type=rec_type,
                priority=priority,
                title=title,
                description=description,
                rationale="System-wide performance issues require comprehensive optimization",
                expected_improvement_percent=35.0,
                confidence_score=0.7,
                implementation_effort="high",
                affected_components=["system"],
                supporting_metrics=["response_time", "memory_usage", "error_rate"],
                baseline_metrics={},
                implementation_steps=[
                    "Conduct comprehensive performance audit",
                    "Identify architectural bottlenecks",
                    "Develop optimization roadmap",
                    "Implement high-impact optimizations first",
                    "Monitor system-wide performance improvements",
                ],
                required_resources=["Senior development team", "Performance testing tools", "Monitoring infrastructure"],
                estimated_cost="high",
                estimated_time_hours=120.0,
                risk_level="medium",
                potential_side_effects=["Temporary service disruptions during implementation"],
                rollback_plan="Phased implementation with rollback points at each stage",
            )

            self._recommendations[recommendation_id] = recommendation
            self._recommendation_stats["total_generated"] += 1

            self.logger.info(f"Generated system-wide recommendation: {title}")

        except Exception as e:
            self.logger.error(f"Error generating system-wide recommendation: {e}")

    async def _analyze_trend_based_optimization(self):
        """Analyze performance trends for predictive recommendations."""
        try:
            # Update trends with recent data
            recent_data = self.data_collector.get_raw_data(hours=24)

            for data_point in recent_data:
                trend_key = f"{data_point['component']}_{data_point['metric_name']}"
                self._performance_trends[trend_key].append(data_point["value"])

            # Analyze trends for early warning recommendations
            for trend_key, values in self._performance_trends.items():
                if len(values) >= 20:  # Need sufficient data points
                    await self._analyze_trend_pattern(trend_key, values)

        except Exception as e:
            self.logger.error(f"Error analyzing trend-based optimization: {e}")

    async def _analyze_trend_pattern(self, trend_key: str, values: list[float]):
        """Analyze a specific trend pattern for optimization opportunities."""
        try:
            if not values:
                return

            component, metric_name = trend_key.rsplit("_", 1)
            recent_values = values[-10:]  # Last 10 values
            older_values = values[-20:-10]  # Previous 10 values

            if len(recent_values) < 5 or len(older_values) < 5:
                return

            recent_avg = statistics.mean(recent_values)
            older_avg = statistics.mean(older_values)

            # Calculate trend
            if older_avg > 0:
                trend_percent = ((recent_avg - older_avg) / older_avg) * 100
            else:
                trend_percent = 0

            # Generate predictive recommendations
            if abs(trend_percent) > 15:  # Significant trend
                await self._generate_trend_recommendation(component, metric_name, trend_percent, recent_avg)

        except Exception as e:
            self.logger.error(f"Error analyzing trend pattern for {trend_key}: {e}")

    async def _generate_trend_recommendation(self, component: str, metric_name: str, trend_percent: float, current_value: float):
        """Generate recommendation based on performance trend."""
        try:
            recommendation_id = f"trend_{component}_{metric_name}_{int(time.time())}"

            # Determine recommendation type based on metric and trend
            if "memory" in metric_name and trend_percent > 0:
                rec_type = RecommendationType.MEMORY_OPTIMIZATION
                title = f"Proactive Memory Optimization for {component}"
                description = f"Memory usage trending upward by {trend_percent:.1f}% - proactive optimization recommended"
            elif "response_time" in metric_name and trend_percent > 0:
                rec_type = RecommendationType.ALGORITHM_OPTIMIZATION
                title = f"Proactive Performance Optimization for {component}"
                description = f"Response time trending upward by {trend_percent:.1f}% - proactive optimization recommended"
            elif "error_rate" in metric_name and trend_percent > 0:
                rec_type = RecommendationType.MONITORING_IMPROVEMENT
                title = f"Proactive Reliability Improvement for {component}"
                description = f"Error rate trending upward by {trend_percent:.1f}% - proactive measures recommended"
            else:
                return  # Don't generate recommendation for this trend

            priority = RecommendationPriority.MEDIUM
            if abs(trend_percent) > 30:
                priority = RecommendationPriority.HIGH

            recommendation = OptimizationRecommendation(
                recommendation_id=recommendation_id,
                recommendation_type=rec_type,
                priority=priority,
                title=title,
                description=description,
                rationale="Trending performance degradation detected - early intervention recommended",
                expected_improvement_percent=15.0,
                confidence_score=0.6,  # Lower confidence for predictive recommendations
                implementation_effort="medium",
                affected_components=[component],
                supporting_metrics=[metric_name],
                baseline_metrics={metric_name: current_value},
                implementation_steps=[
                    "Investigate root cause of performance trend",
                    "Implement targeted optimizations",
                    "Monitor trend reversal",
                    "Establish preventive measures",
                ],
                required_resources=["Development time", "Monitoring tools"],
                estimated_cost="low",
                estimated_time_hours=8.0,
                risk_level="low",
                potential_side_effects=["Minimal impact expected"],
                rollback_plan="Simple configuration rollback if needed",
            )

            self._recommendations[recommendation_id] = recommendation
            self._recommendation_stats["total_generated"] += 1

            self.logger.info(f"Generated trend-based recommendation: {title}")

        except Exception as e:
            self.logger.error(f"Error generating trend recommendation: {e}")

    async def _update_performance_trends(self):
        """Update performance trend data."""
        try:
            current_metrics = self.performance_monitor.get_current_metrics()

            for metric in current_metrics:
                trend_key = f"{metric.component}_{metric.metric_type.value}"
                self._performance_trends[trend_key].append(metric.value)

        except Exception as e:
            self.logger.error(f"Error updating performance trends: {e}")

    async def _cleanup_obsolete_recommendations(self):
        """Clean up obsolete recommendations."""
        try:
            current_time = time.time()
            obsolete_ids = []

            for rec_id, recommendation in self._recommendations.items():
                # Mark as obsolete if very old and not implemented
                age_hours = (current_time - recommendation.created_at) / 3600

                if age_hours > 168 and recommendation.status == RecommendationStatus.PENDING:  # Older than 1 week
                    recommendation.update_status(RecommendationStatus.OBSOLETE, "Marked obsolete due to age")
                    obsolete_ids.append(rec_id)

            # Move obsolete recommendations to history
            for rec_id in obsolete_ids:
                if rec_id in self._recommendations:
                    self._recommendation_history.append(self._recommendations[rec_id].to_dict())
                    del self._recommendations[rec_id]

            if obsolete_ids:
                self.logger.info(f"Cleaned up {len(obsolete_ids)} obsolete recommendations")

        except Exception as e:
            self.logger.error(f"Error cleaning up obsolete recommendations: {e}")

    def get_recommendations(
        self,
        priority: RecommendationPriority | None = None,
        rec_type: RecommendationType | None = None,
        component: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get optimization recommendations with optional filtering.

        Args:
            priority: Filter by priority level
            rec_type: Filter by recommendation type
            component: Filter by affected component
            status: Filter by implementation status

        Returns:
            List of recommendation dictionaries
        """
        try:
            recommendations = list(self._recommendations.values())

            # Apply filters
            if priority:
                recommendations = [r for r in recommendations if r.priority == priority]

            if rec_type:
                recommendations = [r for r in recommendations if r.recommendation_type == rec_type]

            if component:
                recommendations = [r for r in recommendations if component in r.affected_components]

            if status:
                recommendations = [r for r in recommendations if r.status == status]

            # Sort by priority and confidence
            priority_weights = {
                RecommendationPriority.CRITICAL: 4,
                RecommendationPriority.HIGH: 3,
                RecommendationPriority.MEDIUM: 2,
                RecommendationPriority.LOW: 1,
            }

            recommendations.sort(key=lambda r: (priority_weights.get(r.priority, 0), r.confidence_score), reverse=True)

            return [r.to_dict() for r in recommendations]

        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []

    def get_recommendation_summary(self) -> dict[str, Any]:
        """Get a summary of optimization recommendations."""
        try:
            recommendations = list(self._recommendations.values())

            # Count by priority
            priority_counts = {
                "critical": len([r for r in recommendations if r.priority == RecommendationPriority.CRITICAL]),
                "high": len([r for r in recommendations if r.priority == RecommendationPriority.HIGH]),
                "medium": len([r for r in recommendations if r.priority == RecommendationPriority.MEDIUM]),
                "low": len([r for r in recommendations if r.priority == RecommendationPriority.LOW]),
            }

            # Count by type
            type_counts = defaultdict(int)
            for rec in recommendations:
                type_counts[rec.recommendation_type.value] += 1

            # Count by status
            status_counts = defaultdict(int)
            for rec in recommendations:
                status_counts[rec.status.value] += 1

            # Calculate averages
            if recommendations:
                avg_confidence = statistics.mean(r.confidence_score for r in recommendations)
                avg_improvement = statistics.mean(r.expected_improvement_percent for r in recommendations)
            else:
                avg_confidence = 0.0
                avg_improvement = 0.0

            # Top recommendations
            top_recommendations = sorted(recommendations, key=lambda r: (r.priority.value, r.confidence_score), reverse=True)[:5]

            return {
                "analysis_timestamp": time.time(),
                "last_analysis_time": self._last_analysis_time,
                "total_recommendations": len(recommendations),
                "priority_breakdown": priority_counts,
                "type_breakdown": dict(type_counts),
                "status_breakdown": dict(status_counts),
                "average_confidence": round(avg_confidence, 2),
                "average_expected_improvement": round(avg_improvement, 1),
                "recommendation_stats": self._recommendation_stats.copy(),
                "top_recommendations": [
                    {
                        "recommendation_id": r.recommendation_id,
                        "title": r.title,
                        "priority": r.priority.value,
                        "expected_improvement": r.expected_improvement_percent,
                        "confidence": r.confidence_score,
                        "status": r.status.value,
                    }
                    for r in top_recommendations
                ],
                "analysis_status": "active" if self._is_analyzing else "inactive",
            }

        except Exception as e:
            self.logger.error(f"Error getting recommendation summary: {e}")
            return {"error": str(e)}

    async def implement_recommendation(self, recommendation_id: str, user: str = "system", notes: str = "") -> dict[str, Any]:
        """Mark a recommendation as implemented."""
        try:
            if recommendation_id in self._recommendations:
                recommendation = self._recommendations[recommendation_id]
                recommendation.update_status(RecommendationStatus.IMPLEMENTED, notes)

                # Update statistics
                self._recommendation_stats["implemented"] += 1

                # Move to history
                self._recommendation_history.append(recommendation.to_dict())
                del self._recommendations[recommendation_id]

                self.logger.info(f"Recommendation {recommendation_id} marked as implemented by {user}")

                return {
                    "success": True,
                    "message": f"Recommendation marked as implemented by {user}",
                    "recommendation_id": recommendation_id,
                    "timestamp": time.time(),
                }

            return {"success": False, "message": "Recommendation not found", "recommendation_id": recommendation_id}

        except Exception as e:
            self.logger.error(f"Error implementing recommendation: {e}")
            return {"success": False, "message": str(e)}

    async def reject_recommendation(self, recommendation_id: str, reason: str = "", user: str = "system") -> dict[str, Any]:
        """Reject a recommendation."""
        try:
            if recommendation_id in self._recommendations:
                recommendation = self._recommendations[recommendation_id]
                recommendation.update_status(RecommendationStatus.REJECTED, reason)

                # Update statistics
                self._recommendation_stats["rejected"] += 1

                # Move to history
                self._recommendation_history.append(recommendation.to_dict())
                del self._recommendations[recommendation_id]

                self.logger.info(f"Recommendation {recommendation_id} rejected by {user}: {reason}")

                return {
                    "success": True,
                    "message": f"Recommendation rejected by {user}",
                    "recommendation_id": recommendation_id,
                    "reason": reason,
                    "timestamp": time.time(),
                }

            return {"success": False, "message": "Recommendation not found", "recommendation_id": recommendation_id}

        except Exception as e:
            self.logger.error(f"Error rejecting recommendation: {e}")
            return {"success": False, "message": str(e)}

    async def generate_optimization_report(self) -> dict[str, Any]:
        """Generate a comprehensive optimization report."""
        try:
            current_time = time.time()

            # Get all recommendations
            all_recommendations = self.get_recommendations()

            # Get performance summary
            performance_summary = self.performance_monitor.get_performance_summary()

            # Get bottleneck summary
            bottleneck_summary = self.bottleneck_analyzer.get_bottleneck_summary()

            # Calculate potential improvements
            total_potential_improvement = sum(r["expected_improvement_percent"] for r in all_recommendations if r["status"] == "pending")

            high_confidence_improvements = sum(
                r["expected_improvement_percent"] for r in all_recommendations if r["status"] == "pending" and r["confidence_score"] > 0.8
            )

            report = {
                "report_timestamp": current_time,
                "report_period": "current_state",
                "executive_summary": {
                    "total_recommendations": len(all_recommendations),
                    "high_priority_recommendations": len(
                        [r for r in all_recommendations if r["priority"] in ["high", "critical"] and r["status"] == "pending"]
                    ),
                    "total_potential_improvement": round(total_potential_improvement, 1),
                    "high_confidence_improvement": round(high_confidence_improvements, 1),
                    "active_bottlenecks": bottleneck_summary.get("unresolved_bottlenecks", 0),
                    "optimization_status": "active" if self._is_analyzing else "inactive",
                },
                "performance_status": performance_summary,
                "bottleneck_analysis": bottleneck_summary,
                "recommendation_summary": self.get_recommendation_summary(),
                "detailed_recommendations": all_recommendations,
                "optimization_roadmap": self._generate_optimization_roadmap(all_recommendations),
                "risk_assessment": self._generate_risk_assessment(all_recommendations),
                "resource_requirements": self._calculate_resource_requirements(all_recommendations),
                "implementation_timeline": self._generate_implementation_timeline(all_recommendations),
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating optimization report: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def _generate_optimization_roadmap(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate an optimization implementation roadmap."""
        try:
            pending_recs = [r for r in recommendations if r["status"] == "pending"]

            # Prioritize recommendations
            critical_recs = [r for r in pending_recs if r["priority"] == "critical"]
            high_recs = [r for r in pending_recs if r["priority"] == "high"]
            medium_recs = [r for r in pending_recs if r["priority"] == "medium"]
            low_recs = [r for r in pending_recs if r["priority"] == "low"]

            roadmap = {
                "phase_1_immediate": {
                    "description": "Critical and high-priority optimizations",
                    "recommendations": critical_recs + high_recs[:3],  # Top 3 high priority
                    "estimated_duration_weeks": 2,
                    "expected_improvement": sum(r["expected_improvement_percent"] for r in critical_recs + high_recs[:3]),
                },
                "phase_2_short_term": {
                    "description": "Remaining high-priority and top medium-priority optimizations",
                    "recommendations": high_recs[3:] + medium_recs[:5],
                    "estimated_duration_weeks": 4,
                    "expected_improvement": sum(r["expected_improvement_percent"] for r in high_recs[3:] + medium_recs[:5]),
                },
                "phase_3_medium_term": {
                    "description": "Medium and low-priority optimizations",
                    "recommendations": medium_recs[5:] + low_recs,
                    "estimated_duration_weeks": 8,
                    "expected_improvement": sum(r["expected_improvement_percent"] for r in medium_recs[5:] + low_recs),
                },
            }

            return roadmap

        except Exception as e:
            self.logger.error(f"Error generating optimization roadmap: {e}")
            return {}

    def _generate_risk_assessment(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate risk assessment for recommendations."""
        try:
            pending_recs = [r for r in recommendations if r["status"] == "pending"]

            risk_levels = {"low": 0, "medium": 0, "high": 0}
            high_risk_items = []

            for rec in pending_recs:
                risk_level = rec.get("risk_level", "medium")
                risk_levels[risk_level] += 1

                if risk_level == "high":
                    high_risk_items.append(
                        {
                            "title": rec["title"],
                            "risk_factors": rec.get("potential_side_effects", []),
                            "mitigation": rec.get("rollback_plan", ""),
                        }
                    )

            return {
                "overall_risk": "high" if risk_levels["high"] > 0 else "medium" if risk_levels["medium"] > 2 else "low",
                "risk_distribution": risk_levels,
                "high_risk_recommendations": len(high_risk_items),
                "high_risk_details": high_risk_items,
                "mitigation_strategies": [
                    "Implement changes in isolated environments first",
                    "Monitor performance metrics closely during implementation",
                    "Maintain rollback procedures for all changes",
                    "Implement changes gradually with validation at each step",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {e}")
            return {}

    def _calculate_resource_requirements(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate total resource requirements."""
        try:
            pending_recs = [r for r in recommendations if r["status"] == "pending"]

            total_time = sum(r.get("estimated_time_hours", 0) for r in pending_recs)

            cost_distribution = {"low": 0, "medium": 0, "high": 0}
            effort_distribution = {"low": 0, "medium": 0, "high": 0}

            for rec in pending_recs:
                cost_distribution[rec.get("estimated_cost", "medium")] += 1
                effort_distribution[rec.get("implementation_effort", "medium")] += 1

            # Collect unique required resources
            all_resources = set()
            for rec in pending_recs:
                all_resources.update(rec.get("required_resources", []))

            return {
                "total_estimated_hours": total_time,
                "total_estimated_weeks": round(total_time / 40, 1),  # Assuming 40-hour work weeks
                "cost_distribution": cost_distribution,
                "effort_distribution": effort_distribution,
                "required_resources": list(all_resources),
                "resource_recommendations": [
                    "Allocate dedicated performance optimization team",
                    "Set up performance testing environment",
                    "Ensure monitoring and rollback capabilities",
                    "Plan for gradual implementation phases",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error calculating resource requirements: {e}")
            return {}

    def _generate_implementation_timeline(self, recommendations: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate implementation timeline."""
        try:
            pending_recs = [r for r in recommendations if r["status"] == "pending"]

            # Sort by priority and estimated time
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            sorted_recs = sorted(pending_recs, key=lambda r: (priority_order.get(r["priority"], 3), r.get("estimated_time_hours", 0)))

            timeline = {}
            current_week = 1
            weekly_capacity = 40  # hours per week
            current_week_hours = 0

            for rec in sorted_recs:
                estimated_hours = rec.get("estimated_time_hours", 8)

                if current_week_hours + estimated_hours > weekly_capacity:
                    current_week += 1
                    current_week_hours = 0

                if f"week_{current_week}" not in timeline:
                    timeline[f"week_{current_week}"] = []

                timeline[f"week_{current_week}"].append(
                    {
                        "title": rec["title"],
                        "priority": rec["priority"],
                        "estimated_hours": estimated_hours,
                        "expected_improvement": rec["expected_improvement_percent"],
                    }
                )

                current_week_hours += estimated_hours

            return {
                "total_weeks": current_week,
                "weekly_schedule": timeline,
                "assumptions": [
                    "40-hour work weeks assumed",
                    "Single developer/team working on optimizations",
                    "No parallel implementation assumed",
                    "Time estimates include testing and validation",
                ],
            }

        except Exception as e:
            self.logger.error(f"Error generating implementation timeline: {e}")
            return {}

    async def shutdown(self):
        """Shutdown the optimization engine."""
        self.logger.info("Shutting down PerformanceOptimizationEngine")
        await self.stop_analysis()

        # Clear state
        self._recommendations.clear()
        self._recommendation_history.clear()
        self._performance_baselines.clear()
        self._performance_trends.clear()

        self.logger.info("PerformanceOptimizationEngine shutdown complete")


# Global optimization engine instance
_optimization_engine: PerformanceOptimizationEngine | None = None


def get_optimization_engine() -> PerformanceOptimizationEngine:
    """Get the global optimization engine instance."""
    global _optimization_engine
    if _optimization_engine is None:
        _optimization_engine = PerformanceOptimizationEngine()
    return _optimization_engine
