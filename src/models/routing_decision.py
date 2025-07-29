"""
Routing Decision Models for Wave 4.0 Intelligent Query Routing

This module defines the data models for intelligent query routing decisions,
including routing strategies, performance expectations, and decision rationale.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RoutingStrategy(Enum):
    """Available routing strategies for query processing."""

    LOCAL = "local"  # Entity-focused, direct retrieval
    GLOBAL = "global"  # Relationship-focused, broad search
    HYBRID = "hybrid"  # Balanced approach
    MIX = "mix"  # Adaptive multi-modal
    CUSTOM = "custom"  # Custom routing configuration


class RoutingPriority(Enum):
    """Priority levels for query routing."""

    LOW = "low"  # Background processing
    NORMAL = "normal"  # Standard priority
    HIGH = "high"  # Expedited processing
    CRITICAL = "critical"  # Immediate processing


class RoutingConfidence(Enum):
    """Confidence levels in routing decisions."""

    VERY_LOW = "very_low"  # < 0.3
    LOW = "low"  # 0.3 - 0.5
    MODERATE = "moderate"  # 0.5 - 0.7
    HIGH = "high"  # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class RoutingMetrics:
    """Performance metrics and expectations for routing decisions."""

    # Performance expectations
    expected_latency_ms: float = 0.0
    expected_throughput: float = 0.0
    expected_accuracy: float = 0.0
    expected_recall: float = 0.0

    # Resource requirements
    cpu_intensity: float = 0.5  # 0.0-1.0 scale
    memory_requirements_mb: float = 0.0
    io_operations: int = 0
    network_requests: int = 0

    # Historical performance (if available)
    historical_latency_avg: float = 0.0
    historical_success_rate: float = 0.0
    historical_user_satisfaction: float = 0.0

    # Quality indicators
    result_diversity_score: float = 0.0
    relevance_score: float = 0.0
    coverage_score: float = 0.0


@dataclass
class RoutingConstraints:
    """Constraints and limitations for query routing."""

    # Time constraints
    max_processing_time_ms: float | None = None
    timeout_ms: float = 30000.0

    # Resource constraints
    max_memory_mb: float | None = None
    max_cpu_usage: float | None = None
    max_concurrent_operations: int = 10

    # Quality constraints
    min_confidence_threshold: float = 0.3
    min_result_count: int = 1
    max_result_count: int = 50

    # Functional constraints
    allowed_modes: list[str] = field(default_factory=lambda: ["local", "global", "hybrid", "mix"])
    excluded_modes: list[str] = field(default_factory=list)
    require_cached_results: bool = False
    allow_fallback: bool = True


@dataclass
class RoutingDecisionFactor:
    """Individual factor contributing to routing decision."""

    factor_name: str
    factor_weight: float  # Importance weight (0.0-1.0)
    factor_value: float  # Factor value/score
    factor_confidence: float  # Confidence in this factor
    factor_description: str = ""

    def get_weighted_contribution(self) -> float:
        """Get the weighted contribution of this factor."""
        return self.factor_weight * self.factor_value * self.factor_confidence


@dataclass
class RoutingAlternative:
    """Alternative routing option with scoring."""

    mode: str
    confidence: float
    expected_performance: RoutingMetrics
    decision_factors: list[RoutingDecisionFactor] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)

    def get_total_score(self) -> float:
        """Calculate total score based on decision factors."""
        if not self.decision_factors:
            return self.confidence

        return sum(factor.get_weighted_contribution() for factor in self.decision_factors)


@dataclass
class RoutingDecision:
    """Comprehensive intelligent routing decision with full context."""

    # === PRIMARY DECISION ===
    selected_mode: str
    selection_confidence: float
    selection_rationale: str = ""

    # === DECISION ANALYSIS ===
    decision_factors: list[RoutingDecisionFactor] = field(default_factory=list)
    alternatives: list[RoutingAlternative] = field(default_factory=list)
    fallback_sequence: list[str] = field(default_factory=list)

    # === PERFORMANCE EXPECTATIONS ===
    performance_metrics: RoutingMetrics = field(default_factory=RoutingMetrics)
    routing_constraints: RoutingConstraints = field(default_factory=RoutingConstraints)

    # === CONFIGURATION RECOMMENDATIONS ===
    config_adjustments: dict[str, Any] = field(default_factory=dict)
    parameter_tuning: dict[str, float] = field(default_factory=dict)
    optimization_suggestions: list[str] = field(default_factory=list)

    # === DECISION CONTEXT ===
    decision_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    decision_version: str = "4.0.0"
    decision_complexity: str = "standard"  # simple, standard, complex, expert

    # === MONITORING AND FEEDBACK ===
    success_indicators: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)
    monitoring_metrics: list[str] = field(default_factory=list)

    # === METADATA ===
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_confidence_level(self) -> RoutingConfidence:
        """Get the confidence level category."""
        if self.selection_confidence < 0.3:
            return RoutingConfidence.VERY_LOW
        elif self.selection_confidence < 0.5:
            return RoutingConfidence.LOW
        elif self.selection_confidence < 0.7:
            return RoutingConfidence.MODERATE
        elif self.selection_confidence < 0.9:
            return RoutingConfidence.HIGH
        else:
            return RoutingConfidence.VERY_HIGH

    def get_decision_summary(self) -> dict[str, Any]:
        """Get a summary of the routing decision."""
        return {
            "selected_mode": self.selected_mode,
            "confidence": self.selection_confidence,
            "confidence_level": self.get_confidence_level().value,
            "rationale": self.selection_rationale,
            "alternatives_count": len(self.alternatives),
            "has_fallback": len(self.fallback_sequence) > 0,
            "expected_latency_ms": self.performance_metrics.expected_latency_ms,
            "decision_complexity": self.decision_complexity,
            "timestamp": self.decision_timestamp,
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of performance expectations."""
        return {
            "expected_latency_ms": self.performance_metrics.expected_latency_ms,
            "expected_accuracy": self.performance_metrics.expected_accuracy,
            "cpu_intensity": self.performance_metrics.cpu_intensity,
            "memory_requirements_mb": self.performance_metrics.memory_requirements_mb,
            "io_operations": self.performance_metrics.io_operations,
            "success_rate_estimate": self.performance_metrics.historical_success_rate,
        }

    def get_top_decision_factors(self, n: int = 3) -> list[RoutingDecisionFactor]:
        """Get the top N most important decision factors."""
        return sorted(self.decision_factors, key=lambda f: f.get_weighted_contribution(), reverse=True)[:n]

    def add_decision_factor(self, name: str, weight: float, value: float, confidence: float = 1.0, description: str = "") -> None:
        """Add a decision factor to the routing decision."""
        factor = RoutingDecisionFactor(
            factor_name=name, factor_weight=weight, factor_value=value, factor_confidence=confidence, factor_description=description
        )
        self.decision_factors.append(factor)

    def add_alternative(
        self,
        mode: str,
        confidence: float,
        performance: RoutingMetrics | None = None,
        pros: list[str] | None = None,
        cons: list[str] | None = None,
    ) -> None:
        """Add an alternative routing option."""
        alternative = RoutingAlternative(
            mode=mode, confidence=confidence, expected_performance=performance or RoutingMetrics(), pros=pros or [], cons=cons or []
        )
        self.alternatives.append(alternative)

    def validate_decision(self) -> tuple[bool, list[str]]:
        """Validate the routing decision for consistency and completeness."""
        issues = []

        # Basic validation
        if not self.selected_mode:
            issues.append("No mode selected")

        if self.selection_confidence < 0.0 or self.selection_confidence > 1.0:
            issues.append("Selection confidence out of range [0,1]")

        # Decision factor validation
        if self.decision_factors:
            total_weight = sum(f.factor_weight for f in self.decision_factors)
            if abs(total_weight - 1.0) > 0.1:  # Allow some tolerance
                issues.append(f"Decision factor weights sum to {total_weight:.2f}, expected ~1.0")

        # Performance expectations validation
        if self.performance_metrics.expected_latency_ms < 0:
            issues.append("Negative expected latency")

        if self.performance_metrics.cpu_intensity < 0 or self.performance_metrics.cpu_intensity > 1:
            issues.append("CPU intensity out of range [0,1]")

        # Alternatives validation
        if self.alternatives and self.selected_mode not in [alt.mode for alt in self.alternatives]:
            issues.append("Selected mode not present in alternatives")

        return len(issues) == 0, issues

    def to_dict(self) -> dict[str, Any]:
        """Convert routing decision to dictionary for serialization."""
        return {
            "selected_mode": self.selected_mode,
            "selection_confidence": self.selection_confidence,
            "selection_rationale": self.selection_rationale,
            "decision_summary": self.get_decision_summary(),
            "performance_summary": self.get_performance_summary(),
            "top_factors": [
                {
                    "name": f.factor_name,
                    "weight": f.factor_weight,
                    "value": f.factor_value,
                    "contribution": f.get_weighted_contribution(),
                    "description": f.factor_description,
                }
                for f in self.get_top_decision_factors()
            ],
            "alternatives": [
                {"mode": alt.mode, "confidence": alt.confidence, "pros": alt.pros, "cons": alt.cons} for alt in self.alternatives
            ],
            "fallback_sequence": self.fallback_sequence,
            "config_adjustments": self.config_adjustments,
            "metadata": {
                "timestamp": self.decision_timestamp,
                "version": self.decision_version,
                "complexity": self.decision_complexity,
                **self.metadata,
            },
        }


@dataclass
class RoutingHistory:
    """Historical routing decisions for learning and optimization."""

    query_hash: str
    routing_decisions: list[RoutingDecision] = field(default_factory=list)
    performance_history: list[dict[str, Any]] = field(default_factory=list)
    success_rate: float = 0.0
    average_latency_ms: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_decision(self, decision: RoutingDecision, performance_result: dict[str, Any] | None = None) -> None:
        """Add a routing decision to history."""
        self.routing_decisions.append(decision)

        if performance_result:
            self.performance_history.append(
                {"timestamp": datetime.now().isoformat(), "decision_mode": decision.selected_mode, "actual_performance": performance_result}
            )

        self.last_updated = datetime.now().isoformat()
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update aggregate statistics."""
        if not self.performance_history:
            return

        # Calculate success rate
        successful = sum(1 for p in self.performance_history if p.get("actual_performance", {}).get("success", False))
        self.success_rate = successful / len(self.performance_history)

        # Calculate average latency
        latencies = [
            p.get("actual_performance", {}).get("latency_ms", 0)
            for p in self.performance_history
            if "latency_ms" in p.get("actual_performance", {})
        ]

        if latencies:
            self.average_latency_ms = sum(latencies) / len(latencies)

    def get_mode_preferences(self) -> dict[str, float]:
        """Get historical mode preferences with success rates."""
        mode_stats = {}

        for decision in self.routing_decisions:
            mode = decision.selected_mode
            if mode not in mode_stats:
                mode_stats[mode] = {"count": 0, "success": 0}
            mode_stats[mode]["count"] += 1

        # Add success information from performance history
        for perf in self.performance_history:
            mode = perf.get("decision_mode")
            success = perf.get("actual_performance", {}).get("success", False)
            if mode in mode_stats and success:
                mode_stats[mode]["success"] += 1

        # Calculate preference scores
        preferences = {}
        for mode, stats in mode_stats.items():
            if stats["count"] > 0:
                success_rate = stats["success"] / stats["count"]
                frequency = stats["count"] / len(self.routing_decisions)
                preferences[mode] = success_rate * 0.7 + frequency * 0.3

        return preferences

    def get_learning_insights(self) -> dict[str, Any]:
        """Extract learning insights from routing history."""
        return {
            "total_decisions": len(self.routing_decisions),
            "success_rate": self.success_rate,
            "average_latency_ms": self.average_latency_ms,
            "mode_preferences": self.get_mode_preferences(),
            "most_successful_mode": (
                max(self.get_mode_preferences().items(), key=lambda x: x[1])[0] if self.get_mode_preferences() else None
            ),
            "confidence_trends": self._analyze_confidence_trends(),
            "performance_trends": self._analyze_performance_trends(),
        }

    def _analyze_confidence_trends(self) -> dict[str, Any]:
        """Analyze confidence trends over time."""
        if not self.routing_decisions:
            return {}

        confidences = [d.selection_confidence for d in self.routing_decisions]

        return {
            "average_confidence": sum(confidences) / len(confidences),
            "confidence_trend": "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable",
            "high_confidence_rate": sum(1 for c in confidences if c > 0.7) / len(confidences),
        }

    def _analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends over time."""
        if not self.performance_history:
            return {}

        recent_performance = self.performance_history[-10:]  # Last 10 decisions

        latencies = [p.get("actual_performance", {}).get("latency_ms", 0) for p in recent_performance]
        successes = [p.get("actual_performance", {}).get("success", False) for p in recent_performance]

        return {
            "recent_average_latency": sum(latencies) / len(latencies) if latencies else 0,
            "recent_success_rate": sum(successes) / len(successes) if successes else 0,
            "performance_stability": "stable",  # Could be more sophisticated
            "trend_direction": "improving",  # Could be more sophisticated
        }
