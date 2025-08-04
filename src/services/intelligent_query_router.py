"""
Intelligent Query Router for Wave 4.0 Query Analysis and Routing System

This service implements advanced query routing using comprehensive analysis of query
complexity, intent, keywords, and historical patterns to select optimal processing strategies.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models.query_features import QueryComplexity, QueryFeatures, QueryType
from models.routing_decision import (
    RoutingAlternative,
    RoutingConfidence,
    RoutingConstraints,
    RoutingDecision,
    RoutingDecisionFactor,
    RoutingHistory,
    RoutingMetrics,
    RoutingStrategy,
)
from services.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


class IntelligentQueryRouter:
    """
    Advanced intelligent query router that analyzes queries comprehensively
    and makes optimal routing decisions based on multiple factors.
    """

    def __init__(self, query_analyzer: QueryAnalyzer | None = None):
        self.query_analyzer = query_analyzer
        self.logger = logging.getLogger(__name__)

        # Routing history for learning
        self.routing_history: dict[str, RoutingHistory] = {}

        # Performance statistics
        self.routing_stats = {
            "total_decisions": 0,
            "decisions_by_mode": {"local": 0, "global": 0, "hybrid": 0, "mix": 0},
            "average_confidence": 0.0,
            "average_decision_time_ms": 0.0,
            "success_rate": 0.0,
        }

        # Decision weights and thresholds
        self._initialize_routing_configuration()

        # Cache for frequent queries
        self.decision_cache: dict[str, tuple[RoutingDecision, datetime]] = {}
        self.cache_ttl_minutes = 60

    def _initialize_routing_configuration(self):
        """Initialize routing configuration and decision weights."""

        # Decision factor weights (should sum to 1.0)
        self.decision_weights = {
            "complexity_analysis": 0.25,  # Query complexity factors
            "intent_analysis": 0.20,  # Intent clarity and type
            "keyword_analysis": 0.15,  # Keyword characteristics
            "historical_performance": 0.15,  # Past performance for similar queries
            "resource_constraints": 0.10,  # Available resources
            "contextual_factors": 0.10,  # Context and domain
            "user_preferences": 0.05,  # User/system preferences
        }

        # Mode selection thresholds
        self.mode_thresholds = {
            "local": {"entity_density_min": 0.3, "complexity_max": QueryComplexity.MODERATE, "relationship_density_max": 0.2},
            "global": {"relationship_density_min": 0.3, "conceptual_focus_min": 0.4, "complexity_min": QueryComplexity.MODERATE},
            "hybrid": {"balanced_score_min": 0.4, "multi_intent_preference": True},
            "mix": {"complexity_min": QueryComplexity.COMPLEX, "multi_faceted_preference": True, "low_confidence_threshold": 0.6},
        }

        # Performance expectations by mode
        self.mode_performance_profiles = {
            "local": RoutingMetrics(
                expected_latency_ms=500,
                expected_accuracy=0.85,
                expected_recall=0.75,
                cpu_intensity=0.3,
                memory_requirements_mb=100,
                io_operations=2,
            ),
            "global": RoutingMetrics(
                expected_latency_ms=1200,
                expected_accuracy=0.80,
                expected_recall=0.90,
                cpu_intensity=0.7,
                memory_requirements_mb=300,
                io_operations=8,
            ),
            "hybrid": RoutingMetrics(
                expected_latency_ms=800,
                expected_accuracy=0.82,
                expected_recall=0.85,
                cpu_intensity=0.5,
                memory_requirements_mb=200,
                io_operations=5,
            ),
            "mix": RoutingMetrics(
                expected_latency_ms=1500,
                expected_accuracy=0.87,
                expected_recall=0.92,
                cpu_intensity=0.8,
                memory_requirements_mb=400,
                io_operations=12,
            ),
        }

    async def route_query(self, query_features: QueryFeatures, constraints: RoutingConstraints | None = None) -> RoutingDecision:
        """
        Make an intelligent routing decision for a query.

        Args:
            query_features: Comprehensive query analysis results
            constraints: Optional routing constraints

        Returns:
            RoutingDecision with complete routing analysis
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Making routing decision for query: {query_features.query_hash}")

            # Check cache first
            cached_decision = self._check_decision_cache(query_features.query_hash)
            if cached_decision:
                self.logger.debug(f"Using cached routing decision for query: {query_features.query_hash}")
                return cached_decision

            # Apply default constraints if none provided
            if constraints is None:
                constraints = RoutingConstraints()

            # Analyze all decision factors
            decision_factors = await self._analyze_decision_factors(query_features, constraints)

            # Generate routing alternatives
            alternatives = await self._generate_routing_alternatives(query_features, decision_factors)

            # Select the best routing option
            selected_mode, selection_confidence = self._select_optimal_mode(alternatives, decision_factors, constraints)

            # Build comprehensive routing decision
            routing_decision = await self._build_routing_decision(
                selected_mode, selection_confidence, query_features, decision_factors, alternatives, constraints
            )

            # Cache the decision
            self._cache_decision(query_features.query_hash, routing_decision)

            # Update routing history
            self._update_routing_history(query_features.query_hash, routing_decision)

            # Update statistics
            decision_time_ms = (time.time() - start_time) * 1000
            self._update_routing_stats(routing_decision, decision_time_ms)

            self.logger.debug(
                f"Routing decision complete: mode={selected_mode}, "
                f"confidence={selection_confidence:.2f}, "
                f"time={decision_time_ms:.2f}ms"
            )

            return routing_decision

        except Exception as e:
            self.logger.error(f"Error in intelligent query routing: {e}")
            # Return fallback decision
            return await self._create_fallback_decision(query_features, constraints)

    async def _analyze_decision_factors(
        self, query_features: QueryFeatures, constraints: RoutingConstraints
    ) -> list[RoutingDecisionFactor]:
        """Analyze all factors that influence routing decision."""
        factors = []

        # 1. Complexity Analysis Factor
        complexity_factor = self._analyze_complexity_factor(query_features)
        factors.append(complexity_factor)

        # 2. Intent Analysis Factor
        intent_factor = self._analyze_intent_factor(query_features)
        factors.append(intent_factor)

        # 3. Keyword Analysis Factor
        keyword_factor = self._analyze_keyword_factor(query_features)
        factors.append(keyword_factor)

        # 4. Historical Performance Factor
        historical_factor = await self._analyze_historical_factor(query_features)
        factors.append(historical_factor)

        # 5. Resource Constraints Factor
        resource_factor = self._analyze_resource_factor(constraints)
        factors.append(resource_factor)

        # 6. Contextual Factors
        contextual_factor = self._analyze_contextual_factor(query_features)
        factors.append(contextual_factor)

        # 7. User Preferences Factor
        preference_factor = self._analyze_preference_factor(query_features)
        factors.append(preference_factor)

        return factors

    def _analyze_complexity_factor(self, query_features: QueryFeatures) -> RoutingDecisionFactor:
        """Analyze complexity-based routing preferences."""
        complexity_score = 0.0
        confidence = 0.8

        if query_features.complexity_analysis:
            complexity_score = query_features.complexity_analysis.complexity_score
            confidence = query_features.complexity_analysis.complexity_confidence
        else:
            # Fallback complexity scoring
            complexity_map = {
                QueryComplexity.SIMPLE: 0.2,
                QueryComplexity.MODERATE: 0.4,
                QueryComplexity.COMPLEX: 0.7,
                QueryComplexity.MULTI_FACETED: 0.9,
            }
            complexity_score = complexity_map.get(query_features.complexity, 0.5)

        return RoutingDecisionFactor(
            factor_name="complexity_analysis",
            factor_weight=self.decision_weights["complexity_analysis"],
            factor_value=complexity_score,
            factor_confidence=confidence,
            factor_description=f"Query complexity: {query_features.complexity.value} (score: {complexity_score:.2f})",
        )

    def _analyze_intent_factor(self, query_features: QueryFeatures) -> RoutingDecisionFactor:
        """Analyze intent-based routing preferences."""
        intent_score = 0.5
        confidence = query_features.confidence_score

        if query_features.intent_analysis:
            # Use intent clarity and specificity
            intent_score = query_features.intent_analysis.intent_clarity * 0.6 + query_features.intent_analysis.intent_specificity * 0.4
            confidence = query_features.intent_analysis.intent_confidence

        # Adjust score based on query type
        intent_preferences = {
            QueryType.ENTITY_FOCUSED: 0.3,  # Prefer local
            QueryType.RELATIONSHIP_FOCUSED: 0.8,  # Prefer global
            QueryType.CONCEPTUAL: 0.7,  # Prefer global
            QueryType.IMPLEMENTATION_FOCUSED: 0.4,  # Prefer hybrid
            QueryType.PATTERN_FOCUSED: 0.6,  # Prefer hybrid
            QueryType.EXPLORATION: 0.9,  # Prefer mix
            QueryType.TROUBLESHOOTING: 0.5,  # Prefer hybrid
        }

        type_preference = intent_preferences.get(query_features.query_type, 0.5)
        intent_score = (intent_score + type_preference) / 2

        return RoutingDecisionFactor(
            factor_name="intent_analysis",
            factor_weight=self.decision_weights["intent_analysis"],
            factor_value=intent_score,
            factor_confidence=confidence,
            factor_description=f"Intent: {query_features.query_type.value} (score: {intent_score:.2f})",
        )

    def _analyze_keyword_factor(self, query_features: QueryFeatures) -> RoutingDecisionFactor:
        """Analyze keyword-based routing preferences."""
        keyword_stats = query_features.keywords.get_keyword_statistics()

        # Calculate keyword-based routing preference
        entity_ratio = keyword_stats["level_distribution"]["entities"] / max(1, keyword_stats["total_keywords"])
        relationship_ratio = keyword_stats["level_distribution"]["relationships"] / max(1, keyword_stats["total_keywords"])
        concept_ratio = keyword_stats["level_distribution"]["concepts"] / max(1, keyword_stats["total_keywords"])

        # Score based on keyword characteristics
        if entity_ratio > 0.4:
            keyword_score = 0.3  # Favor local
        elif relationship_ratio > 0.3 or concept_ratio > 0.3:
            keyword_score = 0.8  # Favor global
        elif entity_ratio > 0.2 and relationship_ratio > 0.2:
            keyword_score = 0.5  # Favor hybrid
        else:
            keyword_score = 0.6  # Favor mix

        confidence = min(keyword_stats["keyword_diversity"] + 0.3, 1.0)

        return RoutingDecisionFactor(
            factor_name="keyword_analysis",
            factor_weight=self.decision_weights["keyword_analysis"],
            factor_value=keyword_score,
            factor_confidence=confidence,
            factor_description=f"Keywords: {keyword_stats['total_keywords']} total, "
            f"entities: {entity_ratio:.1%}, relationships: {relationship_ratio:.1%}",
        )

    async def _analyze_historical_factor(self, query_features: QueryFeatures) -> RoutingDecisionFactor:
        """Analyze historical performance for similar queries."""
        query_hash = query_features.query_hash

        if query_hash in self.routing_history:
            history = self.routing_history[query_hash]
            insights = history.get_learning_insights()

            # Use historical success rate as the factor value
            historical_score = insights.get("success_rate", 0.5)
            confidence = min(insights.get("total_decisions", 0) / 10.0, 1.0)  # More decisions = higher confidence

            description = f"Historical success rate: {historical_score:.1%} " f"({insights.get('total_decisions', 0)} decisions)"
        else:
            # No history available - use neutral score
            historical_score = 0.5
            confidence = 0.2
            description = "No historical data available"

        return RoutingDecisionFactor(
            factor_name="historical_performance",
            factor_weight=self.decision_weights["historical_performance"],
            factor_value=historical_score,
            factor_confidence=confidence,
            factor_description=description,
        )

    def _analyze_resource_factor(self, constraints: RoutingConstraints) -> RoutingDecisionFactor:
        """Analyze resource constraints impact on routing."""
        resource_score = 0.5
        confidence = 0.7

        # Analyze constraints to determine resource availability
        constraint_factors = []

        if constraints.max_processing_time_ms:
            time_pressure = min(constraints.max_processing_time_ms / 2000.0, 1.0)
            constraint_factors.append(time_pressure)

        if constraints.max_memory_mb:
            memory_availability = min(constraints.max_memory_mb / 500.0, 1.0)
            constraint_factors.append(memory_availability)

        if constraints.max_cpu_usage:
            cpu_availability = constraints.max_cpu_usage
            constraint_factors.append(cpu_availability)

        if constraint_factors:
            resource_score = sum(constraint_factors) / len(constraint_factors)
            confidence = 0.9

        return RoutingDecisionFactor(
            factor_name="resource_constraints",
            factor_weight=self.decision_weights["resource_constraints"],
            factor_value=resource_score,
            factor_confidence=confidence,
            factor_description=f"Resource availability score: {resource_score:.2f}",
        )

    def _analyze_contextual_factor(self, query_features: QueryFeatures) -> RoutingDecisionFactor:
        """Analyze contextual factors for routing."""
        contextual_score = 0.5
        confidence = 0.6

        # Consider domain hints and technology stack
        domain_count = len(query_features.detected_domains)
        tech_count = len(query_features.technology_stack)

        if domain_count > 0 or tech_count > 0:
            # More domain/tech context suggests more complex routing needs
            contextual_score = min(0.3 + (domain_count + tech_count) * 0.1, 1.0)
            confidence = 0.8

        # Consider semantic depth and abstraction level
        if hasattr(query_features, "semantic_depth"):
            depth_influence = query_features.semantic_depth * 0.3
            contextual_score = (contextual_score + depth_influence) / 2

        return RoutingDecisionFactor(
            factor_name="contextual_factors",
            factor_weight=self.decision_weights["contextual_factors"],
            factor_value=contextual_score,
            factor_confidence=confidence,
            factor_description=f"Context richness: {domain_count} domains, {tech_count} technologies",
        )

    def _analyze_preference_factor(self, query_features: QueryFeatures) -> RoutingDecisionFactor:
        """Analyze user/system preferences."""
        # This would typically integrate with user preference systems
        # For now, use system defaults

        preference_score = 0.5  # Neutral preference
        confidence = 0.3

        # Could be enhanced with:
        # - User's historical mode preferences
        # - System-wide performance preferences
        # - Time-of-day preferences
        # - Load balancing preferences

        return RoutingDecisionFactor(
            factor_name="user_preferences",
            factor_weight=self.decision_weights["user_preferences"],
            factor_value=preference_score,
            factor_confidence=confidence,
            factor_description="Using system default preferences",
        )

    async def _generate_routing_alternatives(
        self, query_features: QueryFeatures, decision_factors: list[RoutingDecisionFactor]
    ) -> list[RoutingAlternative]:
        """Generate and score routing alternatives."""
        alternatives = []

        # Evaluate each possible routing mode
        for mode in ["local", "global", "hybrid", "mix"]:
            score = self._calculate_mode_score(mode, query_features, decision_factors)
            performance = self.mode_performance_profiles[mode]

            # Generate pros and cons for this mode
            pros, cons = self._generate_mode_pros_cons(mode, query_features)

            alternative = RoutingAlternative(mode=mode, confidence=score, expected_performance=performance, pros=pros, cons=cons)

            alternatives.append(alternative)

        # Sort by confidence score
        alternatives.sort(key=lambda x: x.confidence, reverse=True)

        return alternatives

    def _calculate_mode_score(self, mode: str, query_features: QueryFeatures, decision_factors: list[RoutingDecisionFactor]) -> float:
        """Calculate suitability score for a specific routing mode."""
        base_score = 0.5

        # Mode-specific scoring logic
        if mode == "local":
            score = self._score_local_mode(query_features, decision_factors)
        elif mode == "global":
            score = self._score_global_mode(query_features, decision_factors)
        elif mode == "hybrid":
            score = self._score_hybrid_mode(query_features, decision_factors)
        elif mode == "mix":
            score = self._score_mix_mode(query_features, decision_factors)
        else:
            score = base_score

        return min(max(score, 0.0), 1.0)

    def _score_local_mode(self, query_features: QueryFeatures, decision_factors: list[RoutingDecisionFactor]) -> float:
        """Score local mode suitability."""
        score = 0.3  # Base score

        # Favor local for entity-focused queries
        if query_features.query_type == QueryType.ENTITY_FOCUSED:
            score += 0.4

        # Favor local for simple queries
        if query_features.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]:
            score += 0.3

        # Favor local for high entity density
        entity_count = len(query_features.keywords.entity_names)
        total_keywords = len(query_features.keywords.get_all_keywords())
        if total_keywords > 0 and entity_count / total_keywords > 0.4:
            score += 0.2

        # Penalize for relationship-heavy queries
        if query_features.has_relationships:
            score -= 0.2

        return score

    def _score_global_mode(self, query_features: QueryFeatures, decision_factors: list[RoutingDecisionFactor]) -> float:
        """Score global mode suitability."""
        score = 0.3  # Base score

        # Favor global for relationship-focused queries
        if query_features.query_type == QueryType.RELATIONSHIP_FOCUSED:
            score += 0.4

        # Favor global for conceptual queries
        if query_features.query_type == QueryType.CONCEPTUAL:
            score += 0.3

        # Favor global for queries with relationships
        if query_features.has_relationships:
            score += 0.3

        # Favor global for complex queries
        if query_features.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_FACETED]:
            score += 0.2

        return score

    def _score_hybrid_mode(self, query_features: QueryFeatures, decision_factors: list[RoutingDecisionFactor]) -> float:
        """Score hybrid mode suitability."""
        score = 0.5  # Base score (good default)

        # Favor hybrid for implementation-focused queries
        if query_features.query_type == QueryType.IMPLEMENTATION_FOCUSED:
            score += 0.3

        # Favor hybrid for balanced entity/relationship queries
        has_entities = query_features.has_specific_entities
        has_relationships = query_features.has_relationships
        if has_entities and has_relationships:
            score += 0.3

        # Favor hybrid for moderate complexity
        if query_features.complexity == QueryComplexity.MODERATE:
            score += 0.2

        return score

    def _score_mix_mode(self, query_features: QueryFeatures, decision_factors: list[RoutingDecisionFactor]) -> float:
        """Score mix mode suitability."""
        score = 0.2  # Lower base score (more expensive)

        # Favor mix for exploratory queries
        if query_features.query_type == QueryType.EXPLORATION:
            score += 0.5

        # Favor mix for multi-faceted complexity
        if query_features.complexity == QueryComplexity.MULTI_FACETED:
            score += 0.4

        # Favor mix for low confidence classifications
        if query_features.confidence_score < 0.6:
            score += 0.3

        # Favor mix for queries with multiple intents
        if query_features.intent_analysis and query_features.intent_analysis.multi_intent:
            score += 0.3

        return score

    def _generate_mode_pros_cons(self, mode: str, query_features: QueryFeatures) -> tuple[list[str], list[str]]:
        """Generate pros and cons for a routing mode."""
        pros_cons = {
            "local": {
                "pros": ["Fast response time", "Low resource usage", "Direct entity lookup"],
                "cons": ["Limited scope", "May miss relationships", "Poor for complex queries"],
            },
            "global": {
                "pros": ["Comprehensive coverage", "Good for relationships", "Handles complexity well"],
                "cons": ["Higher latency", "More resource intensive", "May return too many results"],
            },
            "hybrid": {
                "pros": ["Balanced approach", "Good default choice", "Reasonable performance"],
                "cons": ["May not excel in specific scenarios", "Medium resource usage"],
            },
            "mix": {
                "pros": ["Most comprehensive", "Handles uncertainty well", "Best coverage"],
                "cons": ["Highest latency", "Most resource intensive", "Complex processing"],
            },
        }

        mode_info = pros_cons.get(mode, {"pros": [], "cons": []})
        return mode_info["pros"], mode_info["cons"]

    def _select_optimal_mode(
        self, alternatives: list[RoutingAlternative], decision_factors: list[RoutingDecisionFactor], constraints: RoutingConstraints
    ) -> tuple[str, float]:
        """Select the optimal routing mode from alternatives."""

        # Filter alternatives based on constraints
        valid_alternatives = []
        for alt in alternatives:
            if self._meets_constraints(alt, constraints):
                valid_alternatives.append(alt)

        if not valid_alternatives:
            # If no alternatives meet constraints, use the best available
            valid_alternatives = alternatives

        # Select the highest scoring valid alternative
        best_alternative = valid_alternatives[0]  # Already sorted by confidence

        return best_alternative.mode, best_alternative.confidence

    def _meets_constraints(self, alternative: RoutingAlternative, constraints: RoutingConstraints) -> bool:
        """Check if an alternative meets the routing constraints."""

        # Check mode restrictions
        if alternative.mode in constraints.excluded_modes:
            return False

        if constraints.allowed_modes and alternative.mode not in constraints.allowed_modes:
            return False

        # Check performance constraints
        perf = alternative.expected_performance

        if constraints.max_processing_time_ms and perf.expected_latency_ms > constraints.max_processing_time_ms:
            return False

        if constraints.max_memory_mb and perf.memory_requirements_mb > constraints.max_memory_mb:
            return False

        # Check confidence threshold
        if alternative.confidence < constraints.min_confidence_threshold:
            return False

        return True

    async def _build_routing_decision(
        self,
        selected_mode: str,
        selection_confidence: float,
        query_features: QueryFeatures,
        decision_factors: list[RoutingDecisionFactor],
        alternatives: list[RoutingAlternative],
        constraints: RoutingConstraints,
    ) -> RoutingDecision:
        """Build comprehensive routing decision object."""

        # Generate rationale
        rationale = self._generate_decision_rationale(selected_mode, decision_factors, query_features)

        # Create fallback sequence
        fallback_sequence = [alt.mode for alt in alternatives[1:4]]  # Top 3 alternatives after selected

        # Get performance expectations
        performance_metrics = self.mode_performance_profiles[selected_mode]

        # Generate config adjustments
        config_adjustments = self._generate_config_adjustments(selected_mode, query_features)

        # Build the decision
        decision = RoutingDecision(
            selected_mode=selected_mode,
            selection_confidence=selection_confidence,
            selection_rationale=rationale,
            decision_factors=decision_factors,
            alternatives=alternatives,
            fallback_sequence=fallback_sequence,
            performance_metrics=performance_metrics,
            routing_constraints=constraints,
            config_adjustments=config_adjustments,
            decision_complexity="standard",
            success_indicators=self._generate_success_indicators(selected_mode),
            failure_indicators=self._generate_failure_indicators(selected_mode),
            monitoring_metrics=self._generate_monitoring_metrics(selected_mode),
        )

        return decision

    def _generate_decision_rationale(
        self, selected_mode: str, decision_factors: list[RoutingDecisionFactor], query_features: QueryFeatures
    ) -> str:
        """Generate human-readable rationale for the routing decision."""

        top_factors = sorted(decision_factors, key=lambda f: f.get_weighted_contribution(), reverse=True)[:3]

        rationale_parts = [
            f"Selected {selected_mode} mode based on:",
            f"Query type: {query_features.query_type.value}",
            f"Complexity: {query_features.complexity.value}",
        ]

        for factor in top_factors:
            contribution = factor.get_weighted_contribution()
            rationale_parts.append(f"{factor.factor_name}: {contribution:.2f} - {factor.factor_description}")

        return "; ".join(rationale_parts)

    def _generate_config_adjustments(self, selected_mode: str, query_features: QueryFeatures) -> dict[str, Any]:
        """Generate configuration adjustments for the selected mode."""
        adjustments = {}

        # Adjust based on query complexity
        if query_features.complexity == QueryComplexity.COMPLEX:
            adjustments["max_results"] = 30
            adjustments["timeout_ms"] = 45000
        elif query_features.complexity == QueryComplexity.SIMPLE:
            adjustments["max_results"] = 10
            adjustments["timeout_ms"] = 15000

        # Adjust based on query type
        if query_features.query_type == QueryType.ENTITY_FOCUSED:
            adjustments["entity_boost"] = 1.2
        elif query_features.query_type == QueryType.RELATIONSHIP_FOCUSED:
            adjustments["relationship_boost"] = 1.3

        return adjustments

    def _generate_success_indicators(self, mode: str) -> list[str]:
        """Generate success indicators for monitoring."""
        common_indicators = ["response_time_within_expected", "results_returned_count_adequate", "no_errors_occurred"]

        mode_specific = {
            "local": ["entity_match_found", "precise_results"],
            "global": ["relationship_coverage_good", "comprehensive_results"],
            "hybrid": ["balanced_result_types", "satisfactory_coverage"],
            "mix": ["multi_perspective_coverage", "uncertainty_handled"],
        }

        return common_indicators + mode_specific.get(mode, [])

    def _generate_failure_indicators(self, mode: str) -> list[str]:
        """Generate failure indicators for monitoring."""
        return ["timeout_exceeded", "no_results_returned", "error_rate_high", "user_dissatisfaction_indicated"]

    def _generate_monitoring_metrics(self, mode: str) -> list[str]:
        """Generate metrics to monitor for this routing decision."""
        return ["response_time_ms", "result_count", "error_count", "user_satisfaction_score", "resource_usage", "cache_hit_rate"]

    async def _create_fallback_decision(self, query_features: QueryFeatures, constraints: RoutingConstraints | None) -> RoutingDecision:
        """Create a fallback routing decision when analysis fails."""
        return RoutingDecision(
            selected_mode="hybrid",
            selection_confidence=0.3,
            selection_rationale="Fallback decision due to analysis failure",
            decision_complexity="fallback",
            performance_metrics=self.mode_performance_profiles["hybrid"],
            routing_constraints=constraints or RoutingConstraints(),
        )

    def _check_decision_cache(self, query_hash: str) -> RoutingDecision | None:
        """Check if we have a cached decision for this query."""
        if query_hash in self.decision_cache:
            decision, timestamp = self.decision_cache[query_hash]

            # Check if cache entry is still valid
            if datetime.now() - timestamp < timedelta(minutes=self.cache_ttl_minutes):
                return decision
            else:
                # Remove expired entry
                del self.decision_cache[query_hash]

        return None

    def _cache_decision(self, query_hash: str, decision: RoutingDecision) -> None:
        """Cache a routing decision."""
        self.decision_cache[query_hash] = (decision, datetime.now())

        # Simple cache size management
        if len(self.decision_cache) > 1000:
            # Remove oldest entries
            sorted_entries = sorted(self.decision_cache.items(), key=lambda x: x[1][1])
            for query_hash, _ in sorted_entries[:100]:
                del self.decision_cache[query_hash]

    def _update_routing_history(self, query_hash: str, decision: RoutingDecision) -> None:
        """Update routing history with new decision."""
        if query_hash not in self.routing_history:
            self.routing_history[query_hash] = RoutingHistory(query_hash=query_hash)

        self.routing_history[query_hash].add_decision(decision)

    def _update_routing_stats(self, decision: RoutingDecision, decision_time_ms: float) -> None:
        """Update routing statistics."""
        self.routing_stats["total_decisions"] += 1
        self.routing_stats["decisions_by_mode"][decision.selected_mode] += 1

        # Update average confidence
        total = self.routing_stats["total_decisions"]
        current_avg = self.routing_stats["average_confidence"]
        new_avg = ((current_avg * (total - 1)) + decision.selection_confidence) / total
        self.routing_stats["average_confidence"] = new_avg

        # Update average decision time
        current_time_avg = self.routing_stats["average_decision_time_ms"]
        new_time_avg = ((current_time_avg * (total - 1)) + decision_time_ms) / total
        self.routing_stats["average_decision_time_ms"] = new_time_avg

    def get_routing_statistics(self) -> dict[str, Any]:
        """Get current routing statistics."""
        return self.routing_stats.copy()

    def get_routing_history_summary(self) -> dict[str, Any]:
        """Get summary of routing history."""
        if not self.routing_history:
            return {"total_queries": 0, "histories": []}

        summaries = []
        for query_hash, history in self.routing_history.items():
            insights = history.get_learning_insights()
            summaries.append(
                {
                    "query_hash": query_hash,
                    "total_decisions": insights["total_decisions"],
                    "success_rate": insights["success_rate"],
                    "most_successful_mode": insights.get("most_successful_mode"),
                    "last_updated": history.last_updated,
                }
            )

        return {"total_queries": len(self.routing_history), "histories": summaries}


# Factory function
_intelligent_router_instance = None


async def get_intelligent_query_router(query_analyzer: QueryAnalyzer | None = None) -> IntelligentQueryRouter:
    """Get or create an IntelligentQueryRouter instance."""
    global _intelligent_router_instance
    if _intelligent_router_instance is None:
        if query_analyzer is None:
            from services.query_analyzer import get_query_analyzer

            query_analyzer = await get_query_analyzer()
        _intelligent_router_instance = IntelligentQueryRouter(query_analyzer)
    return _intelligent_router_instance
