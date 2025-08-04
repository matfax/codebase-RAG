"""
Query Features Data Models for Multi-Modal Retrieval System

This module defines the data models for analyzing and classifying search queries
to determine the optimal retrieval strategy for the LightRAG-inspired multi-modal system.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class QueryType(Enum):
    """Type of query based on intent analysis."""

    ENTITY_FOCUSED = "entity_focused"  # Queries about specific entities
    RELATIONSHIP_FOCUSED = "relationship_focused"  # Queries about connections
    CONCEPTUAL = "conceptual"  # High-level conceptual queries
    IMPLEMENTATION_FOCUSED = "implementation_focused"  # Code implementation queries
    PATTERN_FOCUSED = "pattern_focused"  # Architectural pattern queries
    EXPLORATION = "exploration"  # Exploratory queries
    TROUBLESHOOTING = "troubleshooting"  # Debugging/troubleshooting queries


class QueryComplexity(Enum):
    """Complexity level of the query."""

    SIMPLE = "simple"  # Single concept, straightforward
    MODERATE = "moderate"  # Multiple concepts, some relationships
    COMPLEX = "complex"  # Multiple entities, complex relationships
    MULTI_FACETED = "multi_faceted"  # Multiple dimensions, requires hybrid approach


class KeywordLevel(Enum):
    """Level of keywords extracted from query."""

    LOW_LEVEL = "low_level"  # Specific entities, names, functions
    HIGH_LEVEL = "high_level"  # Abstract concepts, patterns, relationships


@dataclass
class KeywordExtraction:
    """Enhanced keywords extracted from a query at multiple levels."""

    low_level_keywords: list[str] = field(default_factory=list)
    high_level_keywords: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    concept_terms: list[str] = field(default_factory=list)
    technical_terms: list[str] = field(default_factory=list)
    relationship_indicators: list[str] = field(default_factory=list)

    # Enhanced multi-level classification
    micro_level_keywords: list[str] = field(default_factory=list)  # Very specific identifiers
    macro_level_keywords: list[str] = field(default_factory=list)  # Broad architectural terms
    domain_specific_keywords: list[str] = field(default_factory=list)  # Domain-specific terminology
    contextual_keywords: list[str] = field(default_factory=list)  # Context-dependent terms

    # Advanced analysis metadata
    keyword_confidence: dict[str, float] = field(default_factory=dict)  # Confidence per keyword
    keyword_levels: dict[str, list[str]] = field(default_factory=dict)  # Semantic levels per keyword
    keyword_clusters: dict[str, list[str]] = field(default_factory=dict)  # Related keyword groups
    semantic_expansion: dict[str, list[str]] = field(default_factory=dict)  # Potential related terms

    # Quality metrics
    extraction_quality: dict[str, float] = field(default_factory=dict)

    def get_all_keywords(self) -> list[str]:
        """Get all keywords combined."""
        return (
            self.low_level_keywords
            + self.high_level_keywords
            + self.entity_names
            + self.concept_terms
            + self.technical_terms
            + self.relationship_indicators
            + self.micro_level_keywords
            + self.macro_level_keywords
            + self.domain_specific_keywords
            + self.contextual_keywords
        )

    def get_keywords_by_level(self, level: KeywordLevel) -> list[str]:
        """Get keywords filtered by semantic level."""
        if level == KeywordLevel.LOW_LEVEL:
            return self.low_level_keywords + self.micro_level_keywords
        elif level == KeywordLevel.HIGH_LEVEL:
            return self.high_level_keywords + self.macro_level_keywords
        else:
            return self.get_all_keywords()

    def get_keyword_statistics(self) -> dict[str, Any]:
        """Get statistical information about extracted keywords."""
        all_keywords = self.get_all_keywords()
        unique_keywords = list(set(all_keywords))

        return {
            "total_keywords": len(all_keywords),
            "unique_keywords": len(unique_keywords),
            "keyword_diversity": len(unique_keywords) / max(1, len(all_keywords)),
            "level_distribution": {
                "low_level": len(self.low_level_keywords + self.micro_level_keywords),
                "high_level": len(self.high_level_keywords + self.macro_level_keywords),
                "entities": len(self.entity_names),
                "concepts": len(self.concept_terms),
                "technical": len(self.technical_terms),
                "relationships": len(self.relationship_indicators),
            },
            "average_confidence": sum(self.keyword_confidence.values()) / max(1, len(self.keyword_confidence)),
            "cluster_count": len(self.keyword_clusters),
        }


@dataclass
class QueryComplexityAnalysis:
    """Detailed complexity analysis results."""

    overall_complexity: QueryComplexity
    complexity_score: float  # 0.0-1.0 overall complexity score
    complexity_confidence: float  # Confidence in complexity assessment

    # Detailed complexity factors
    lexical_complexity: float = 0.0
    syntactic_complexity: float = 0.0
    semantic_complexity: float = 0.0
    conceptual_depth: float = 0.0
    relationship_complexity: float = 0.0
    domain_specificity: float = 0.0

    # Complexity breakdown by categories
    linguistic_complexity: float = 0.0
    conceptual_complexity: float = 0.0
    domain_complexity: float = 0.0

    # Supporting metrics
    complexity_factors: dict[str, float] = field(default_factory=dict)
    complexity_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryIntentAnalysis:
    """Detailed intent analysis results."""

    primary_intent: QueryType
    intent_confidence: float  # Confidence in primary intent
    secondary_intent: QueryType | None = None

    # Intent scoring and distribution
    intent_scores: dict[str, float] = field(default_factory=dict)
    intent_distribution: dict[str, float] = field(default_factory=dict)

    # Intent characteristics
    intent_clarity: float = 0.0  # How clear/unambiguous the intent is
    intent_specificity: float = 0.0  # How specific vs general the intent is
    multi_intent: bool = False  # Whether query has multiple strong intents
    detected_intents: list[str] = field(default_factory=list)  # All detected strong intents

    # Supporting evidence
    intent_evidence: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class QueryProcessingContext:
    """Context information for query processing and routing."""

    # Processing environment
    processing_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: float = 0.0
    processor_version: str = "4.0.0"

    # Query characteristics for routing
    query_signature: str = ""  # Hash or signature for caching
    query_category: str = "general"  # Broad category classification
    priority_level: str = "normal"  # Processing priority

    # Performance indicators
    expected_processing_time_ms: float = 0.0
    resource_requirements: dict[str, str] = field(default_factory=dict)

    # Historical context (if available)
    similar_queries_count: int = 0
    previous_query_success_rate: float = 0.0


@dataclass
class QueryRoutingDecision:
    """Decision information for query routing."""

    # Primary routing decision
    recommended_mode: str = "hybrid"  # local, global, hybrid, or mix
    mode_confidence: float = 0.5  # Confidence in mode recommendation

    # Alternative modes with scores
    mode_scores: dict[str, float] = field(default_factory=dict)
    fallback_modes: list[str] = field(default_factory=list)

    # Routing rationale
    decision_factors: dict[str, float] = field(default_factory=dict)
    decision_reasoning: str = ""

    # Configuration recommendations
    suggested_config_adjustments: dict[str, Any] = field(default_factory=dict)
    performance_expectations: dict[str, float] = field(default_factory=dict)


@dataclass
class QueryFeatures:
    """Unified and comprehensive features extracted from a search query."""

    # === BASIC QUERY INFORMATION ===
    original_query: str
    normalized_query: str
    query_length: int
    word_count: int

    # === CLASSIFICATION RESULTS ===
    query_type: QueryType
    complexity: QueryComplexity
    confidence_score: float  # Overall classification confidence (0.0-1.0)

    # === FIELDS WITH DEFAULTS ===
    query_hash: str = ""  # For caching and deduplication

    # === ENHANCED ANALYSIS RESULTS ===
    complexity_analysis: QueryComplexityAnalysis | None = None
    intent_analysis: QueryIntentAnalysis | None = None
    processing_context: QueryProcessingContext | None = None
    routing_decision: QueryRoutingDecision | None = None

    # === KEYWORD ANALYSIS ===
    keywords: KeywordExtraction = field(default_factory=KeywordExtraction)

    # === SEMANTIC ANALYSIS ===
    has_specific_entities: bool = False
    has_relationships: bool = False
    has_patterns: bool = False
    has_implementation_focus: bool = False
    has_conceptual_focus: bool = False

    # Enhanced semantic indicators
    semantic_depth: float = 0.0  # 0.0=surface, 1.0=deep conceptual
    abstraction_level: float = 0.0  # 0.0=concrete, 1.0=abstract
    technical_density: float = 0.0  # Density of technical terms

    # === STRUCTURAL INDICATORS ===
    mentions_functions: bool = False
    mentions_classes: bool = False
    mentions_files: bool = False
    mentions_modules: bool = False
    mentions_patterns: bool = False

    # Enhanced structural analysis
    has_code_syntax: bool = False
    has_natural_language: bool = True
    structural_complexity: float = 0.0

    # === CONTEXTUAL INDICATORS ===
    language_hints: list[str] = field(default_factory=list)
    framework_hints: list[str] = field(default_factory=list)
    domain_hints: list[str] = field(default_factory=list)

    # Enhanced contextual analysis
    detected_domains: dict[str, float] = field(default_factory=dict)  # Domain -> confidence
    technology_stack: list[str] = field(default_factory=list)
    use_case_category: str = "general"

    # === ROUTING AND PERFORMANCE ===
    recommended_mode: str = "hybrid"  # Backward compatibility
    mode_confidence: float = 0.5  # Backward compatibility

    # === METADATA AND DIAGNOSTICS ===
    processing_time_ms: float = 0.0
    analysis_timestamp: str | None = None
    analysis_version: str = "4.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    # === QUALITY AND VALIDATION ===
    analysis_quality_score: float = 0.0  # Overall quality of analysis
    validation_flags: list[str] = field(default_factory=list)  # Any issues detected

    def __post_init__(self):
        """Post-initialization processing."""
        # Generate query hash for caching
        import hashlib

        self.query_hash = hashlib.md5(self.original_query.encode()).hexdigest()[:16]

        # Set analysis timestamp if not provided
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()

        # Ensure routing decision is populated (backward compatibility)
        if self.routing_decision is None:
            self.routing_decision = QueryRoutingDecision(recommended_mode=self.recommended_mode, mode_confidence=self.mode_confidence)

    def get_complexity_summary(self) -> dict[str, Any]:
        """Get a summary of complexity analysis."""
        if self.complexity_analysis:
            return {
                "overall_complexity": self.complexity_analysis.overall_complexity.value,
                "complexity_score": self.complexity_analysis.complexity_score,
                "primary_factors": {
                    "lexical": self.complexity_analysis.lexical_complexity,
                    "syntactic": self.complexity_analysis.syntactic_complexity,
                    "semantic": self.complexity_analysis.semantic_complexity,
                    "conceptual": self.complexity_analysis.conceptual_depth,
                },
                "confidence": self.complexity_analysis.complexity_confidence,
            }
        else:
            return {"overall_complexity": self.complexity.value, "complexity_score": 0.5, "confidence": 0.5}

    def get_intent_summary(self) -> dict[str, Any]:
        """Get a summary of intent analysis."""
        if self.intent_analysis:
            return {
                "primary_intent": self.intent_analysis.primary_intent.value,
                "intent_confidence": self.intent_analysis.intent_confidence,
                "secondary_intent": self.intent_analysis.secondary_intent.value if self.intent_analysis.secondary_intent else None,
                "multi_intent": self.intent_analysis.multi_intent,
                "intent_clarity": self.intent_analysis.intent_clarity,
                "specificity": self.intent_analysis.intent_specificity,
            }
        else:
            return {"primary_intent": self.query_type.value, "intent_confidence": self.confidence_score, "multi_intent": False}

    def get_routing_summary(self) -> dict[str, Any]:
        """Get a summary of routing decision."""
        if self.routing_decision:
            return {
                "recommended_mode": self.routing_decision.recommended_mode,
                "mode_confidence": self.routing_decision.mode_confidence,
                "alternative_modes": list(self.routing_decision.mode_scores.keys()),
                "decision_reasoning": self.routing_decision.decision_reasoning,
                "performance_expectations": self.routing_decision.performance_expectations,
            }
        else:
            return {"recommended_mode": self.recommended_mode, "mode_confidence": self.mode_confidence}

    def get_keyword_summary(self) -> dict[str, Any]:
        """Get a summary of keyword analysis."""
        return {
            "total_keywords": len(self.keywords.get_all_keywords()),
            "entities": len(self.keywords.entity_names),
            "concepts": len(self.keywords.concept_terms),
            "technical_terms": len(self.keywords.technical_terms),
            "relationships": len(self.keywords.relationship_indicators),
            "keyword_statistics": self.keywords.get_keyword_statistics(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_hash": self.query_hash,
            "original_query": self.original_query,
            "query_type": self.query_type.value,
            "complexity": self.complexity.value,
            "confidence_score": self.confidence_score,
            "complexity_summary": self.get_complexity_summary(),
            "intent_summary": self.get_intent_summary(),
            "routing_summary": self.get_routing_summary(),
            "keyword_summary": self.get_keyword_summary(),
            "semantic_indicators": {
                "has_entities": self.has_specific_entities,
                "has_relationships": self.has_relationships,
                "has_patterns": self.has_patterns,
                "semantic_depth": self.semantic_depth,
                "technical_density": self.technical_density,
            },
            "context": {"domains": self.detected_domains, "technology_stack": self.technology_stack, "use_case": self.use_case_category},
            "analysis_metadata": {
                "processing_time_ms": self.processing_time_ms,
                "timestamp": self.analysis_timestamp,
                "version": self.analysis_version,
                "quality_score": self.analysis_quality_score,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryFeatures":
        """Create QueryFeatures from dictionary."""
        # This would be implemented for deserialization
        # Simplified version for now
        return cls(
            original_query=data.get("original_query", ""),
            normalized_query=data.get("original_query", "").lower(),
            query_length=len(data.get("original_query", "")),
            word_count=len(data.get("original_query", "").split()),
            query_type=QueryType(data.get("query_type", "exploration")),
            complexity=QueryComplexity(data.get("complexity", "moderate")),
            confidence_score=data.get("confidence_score", 0.5),
        )


@dataclass
class RetrievalModeConfig:
    """Configuration for a specific retrieval mode."""

    mode_name: str
    weight_local: float = 0.5  # Weight for local/entity-focused retrieval
    weight_global: float = 0.5  # Weight for global/relationship-focused retrieval
    max_results: int = 20
    similarity_threshold: float = 0.3
    expansion_depth: int = 2
    token_allocation: dict[str, float] = field(
        default_factory=lambda: {"entity_tokens": 0.4, "relationship_tokens": 0.4, "context_tokens": 0.2}
    )

    # Performance parameters
    timeout_seconds: float = 30.0
    enable_caching: bool = True
    enable_graph_expansion: bool = True

    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0."""
        total_weight = self.weight_local + self.weight_global
        return abs(total_weight - 1.0) < 0.01

    def validate_token_allocation(self) -> bool:
        """Validate that token allocation sums to 1.0."""
        total_allocation = sum(self.token_allocation.values())
        return abs(total_allocation - 1.0) < 0.01


@dataclass
class LocalModeConfig(RetrievalModeConfig):
    """Configuration specific to Local mode retrieval."""

    mode_name: str = "local"
    weight_local: float = 0.8
    weight_global: float = 0.2
    expansion_depth: int = 1  # Limited expansion for focused search
    token_allocation: dict[str, float] = field(
        default_factory=lambda: {"entity_tokens": 0.7, "relationship_tokens": 0.2, "context_tokens": 0.1}
    )


@dataclass
class GlobalModeConfig(RetrievalModeConfig):
    """Configuration specific to Global mode retrieval."""

    mode_name: str = "global"
    weight_local: float = 0.2
    weight_global: float = 0.8
    expansion_depth: int = 3  # Deeper expansion for broad search
    token_allocation: dict[str, float] = field(
        default_factory=lambda: {"entity_tokens": 0.2, "relationship_tokens": 0.6, "context_tokens": 0.2}
    )


@dataclass
class HybridModeConfig(RetrievalModeConfig):
    """Configuration specific to Hybrid mode retrieval."""

    mode_name: str = "hybrid"
    weight_local: float = 0.5
    weight_global: float = 0.5
    expansion_depth: int = 2  # Balanced expansion
    token_allocation: dict[str, float] = field(
        default_factory=lambda: {"entity_tokens": 0.4, "relationship_tokens": 0.4, "context_tokens": 0.2}
    )


@dataclass
class MixModeConfig(RetrievalModeConfig):
    """Configuration for Mix mode with adaptive parameters."""

    mode_name: str = "mix"
    weight_local: float = 0.5  # Default balanced configuration
    weight_global: float = 0.5
    expansion_depth: int = 2
    token_allocation: dict[str, float] = field(
        default_factory=lambda: {"entity_tokens": 0.4, "relationship_tokens": 0.4, "context_tokens": 0.2}
    )

    # Adaptive parameters based on query analysis
    adaptive_weights: dict[str, float] = field(default_factory=dict)
    confidence_threshold: float = 0.7  # Threshold for mode selection confidence
    fallback_mode: str = "hybrid"


@dataclass
class RetrievalResult:
    """Result from a multi-modal retrieval operation."""

    query: str
    mode_used: str
    config_used: RetrievalModeConfig
    results: list[dict[str, Any]]

    # Performance metrics
    total_execution_time_ms: float
    query_analysis_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    post_processing_time_ms: float = 0.0

    # Quality metrics
    total_results: int = 0
    average_confidence: float = 0.0
    result_diversity_score: float = 0.0

    # Mode-specific metrics
    local_results_count: int = 0
    global_results_count: int = 0
    hybrid_results_count: int = 0

    # Additional information
    fallback_used: bool = False
    cache_hit: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for retrieval modes."""

    mode_name: str
    query_count: int = 0
    total_execution_time_ms: float = 0.0
    average_execution_time_ms: float = 0.0

    # Success metrics
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0

    # Quality metrics
    average_result_count: float = 0.0
    average_confidence: float = 0.0
    average_diversity: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Resource usage
    peak_memory_mb: float = 0.0
    average_cpu_usage: float = 0.0

    def update_metrics(self, result: RetrievalResult):
        """Update metrics with a new result."""
        self.query_count += 1
        self.total_execution_time_ms += result.total_execution_time_ms
        self.average_execution_time_ms = self.total_execution_time_ms / self.query_count

        if result.error_message is None:
            self.successful_queries += 1
            self.average_result_count = (
                self.average_result_count * (self.successful_queries - 1) + result.total_results
            ) / self.successful_queries
            self.average_confidence = (
                self.average_confidence * (self.successful_queries - 1) + result.average_confidence
            ) / self.successful_queries
            self.average_diversity = (
                self.average_diversity * (self.successful_queries - 1) + result.result_diversity_score
            ) / self.successful_queries
        else:
            self.failed_queries += 1

        self.success_rate = self.successful_queries / self.query_count if self.query_count > 0 else 0.0

        if result.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        self.cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
