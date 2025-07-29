"""
Enhanced Complexity Detection for Wave 4.0 Query Analysis System

This module provides advanced complexity analysis capabilities with fine-tuned metrics,
dynamic thresholds, and comprehensive complexity profiling for optimal query routing.
"""

import logging
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ..models.query_features import KeywordExtraction, QueryComplexity
except ImportError:
    from models.query_features import KeywordExtraction, QueryComplexity


logger = logging.getLogger(__name__)


class ComplexityDimension(Enum):
    """Different dimensions of query complexity."""

    LEXICAL = "lexical"  # Vocabulary complexity
    SYNTACTIC = "syntactic"  # Grammatical structure complexity
    SEMANTIC = "semantic"  # Meaning and concept complexity
    PRAGMATIC = "pragmatic"  # Context and intent complexity
    COMPUTATIONAL = "computational"  # Processing requirement complexity
    DOMAIN_SPECIFIC = "domain_specific"  # Technical domain complexity


@dataclass
class ComplexityProfile:
    """Comprehensive complexity profile for a query."""

    # Overall metrics
    overall_complexity: QueryComplexity
    complexity_score: float  # 0.0-1.0 normalized score
    confidence: float  # Confidence in assessment

    # Dimensional scores (0.0-1.0 each)
    lexical_complexity: float = 0.0
    syntactic_complexity: float = 0.0
    semantic_complexity: float = 0.0
    pragmatic_complexity: float = 0.0
    computational_complexity: float = 0.0
    domain_complexity: float = 0.0

    # Advanced metrics
    complexity_variance: float = 0.0  # How varied complexity is across dimensions
    complexity_stability: float = 0.0  # How consistent complexity indicators are
    processing_difficulty: float = 0.0  # Expected processing difficulty

    # Component analysis
    word_complexity_distribution: dict[str, float] = field(default_factory=dict)
    phrase_complexity_scores: list[float] = field(default_factory=list)
    structural_patterns: list[str] = field(default_factory=list)

    # Quality indicators
    analysis_quality: float = 0.0
    missing_dimensions: list[str] = field(default_factory=list)
    uncertainty_factors: list[str] = field(default_factory=list)


class EnhancedComplexityDetector:
    """
    Advanced complexity detection system with multi-dimensional analysis,
    dynamic threshold adjustment, and comprehensive profiling capabilities.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Enhanced complexity patterns and indicators
        self._initialize_complexity_patterns()

        # Dynamic threshold configuration
        self._initialize_dynamic_thresholds()

        # Analysis cache for performance
        self.analysis_cache: dict[str, ComplexityProfile] = {}

        # Complexity learning system
        self.complexity_feedback: dict[str, list[float]] = {}

        # Performance tracking
        self.analysis_stats = {
            "analyses_performed": 0,
            "cache_hits": 0,
            "average_analysis_time_ms": 0.0,
            "complexity_distribution": {"simple": 0, "moderate": 0, "complex": 0, "multi_faceted": 0},
        }

    def _initialize_complexity_patterns(self):
        """Initialize enhanced complexity detection patterns."""

        # Lexical complexity patterns
        self.lexical_patterns = {
            "technical_terms": {
                "patterns": [
                    r"\b\w*(?:tion|sion|ment|ness|ity|ing|ed)\b",  # Complex suffixes
                    r"\b(?:pre|post|anti|proto|meta|micro|macro)\w+",  # Complex prefixes
                    r"\b\w{10,}\b",  # Long words
                    r"\b[A-Z]{2,}\b",  # Acronyms
                ],
                "weight": 0.3,
            },
            "rare_vocabulary": {
                "patterns": [
                    r"\b(?:paradigm|methodology|infrastructure|architecture|optimization)\b",
                    r"\b(?:instantiate|encapsulation|polymorphism|inheritance|abstraction)\b",
                ],
                "weight": 0.4,
            },
            "domain_jargon": {
                "patterns": [
                    r"\b(?:async|await|callback|closure|decorator|middleware)\b",
                    r"\b(?:microservice|monolith|scalability|throughput|latency)\b",
                ],
                "weight": 0.3,
            },
        }

        # Syntactic complexity patterns
        self.syntactic_patterns = {
            "nested_structures": {
                "patterns": [
                    r"\([^()]*\([^()]*\)[^()]*\)",  # Nested parentheses
                    r"\{[^{}]*\{[^{}]*\}[^{}]*\}",  # Nested braces
                    r"\[[^\[\]]*\[[^\[\]]*\][^\[\]]*\]",  # Nested brackets
                ],
                "weight": 0.4,
            },
            "complex_clauses": {
                "patterns": [
                    r"\b(?:although|however|nevertheless|furthermore|moreover|consequently)\b",
                    r"\b(?:which|that|where|when|because|since|while|whereas)\b.*\b(?:and|or|but)\b",
                ],
                "weight": 0.3,
            },
            "conditional_logic": {
                "patterns": [
                    r"\bif\b.*\bthen\b.*\belse\b",
                    r"\bunless\b.*\botherwise\b",
                    r"\bprovided\b.*\bgiven\b",
                ],
                "weight": 0.3,
            },
        }

        # Semantic complexity patterns
        self.semantic_patterns = {
            "abstract_concepts": {
                "patterns": [
                    r"\b(?:abstraction|conceptualization|generalization|specification)\b",
                    r"\b(?:paradigm|framework|methodology|approach|strategy)\b",
                ],
                "weight": 0.4,
            },
            "multi_domain": {"indicators": ["cross_domain_terms", "interdisciplinary_concepts"], "weight": 0.3},
            "ambiguous_references": {
                "patterns": [
                    r"\bit\b(?!\s+is\b)",  # Ambiguous "it"
                    r"\bthis\b(?!\s+\w+)",  # Ambiguous "this"
                    r"\bthat\b(?!\s+\w+)",  # Ambiguous "that"
                ],
                "weight": 0.3,
            },
        }

        # Computational complexity indicators
        self.computational_patterns = {
            "algorithmic_complexity": {
                "patterns": [
                    r"\b(?:recursive|iterative|optimize|algorithm|complexity)\b",
                    r"\b(?:search|sort|traverse|parse|analyze)\b",
                ],
                "weight": 0.4,
            },
            "data_processing": {
                "patterns": [
                    r"\b(?:process|transform|aggregate|filter|map|reduce)\b",
                    r"\b(?:database|query|index|schema|migration)\b",
                ],
                "weight": 0.3,
            },
            "system_integration": {
                "patterns": [
                    r"\b(?:integrate|connect|interface|communicate|synchronize)\b",
                    r"\b(?:api|service|endpoint|protocol|authentication)\b",
                ],
                "weight": 0.3,
            },
        }

    def _initialize_dynamic_thresholds(self):
        """Initialize dynamic threshold system."""

        # Base thresholds (will be adjusted based on context)
        self.base_thresholds = {
            QueryComplexity.SIMPLE: {"score_range": (0.0, 0.25), "max_dimensions": 2, "max_word_count": 5, "max_technical_terms": 1},
            QueryComplexity.MODERATE: {"score_range": (0.25, 0.50), "max_dimensions": 4, "max_word_count": 12, "max_technical_terms": 3},
            QueryComplexity.COMPLEX: {"score_range": (0.50, 0.75), "max_dimensions": 5, "max_word_count": 20, "max_technical_terms": 6},
            QueryComplexity.MULTI_FACETED: {
                "score_range": (0.75, 1.0),
                "max_dimensions": 6,
                "max_word_count": float("inf"),
                "max_technical_terms": float("inf"),
            },
        }

        # Context-specific threshold adjustments
        self.threshold_adjustments = {
            "programming_context": {"technical_term_weight": 1.2, "domain_complexity_boost": 0.1},
            "academic_context": {"semantic_complexity_boost": 0.15, "abstract_concept_weight": 1.3},
            "business_context": {"pragmatic_complexity_boost": 0.1, "multi_domain_weight": 1.1},
        }

    async def analyze_complexity(
        self, query: str, keywords: KeywordExtraction | None = None, context_hints: dict[str, Any] | None = None
    ) -> ComplexityProfile:
        """
        Perform comprehensive complexity analysis of a query.

        Args:
            query: The query string to analyze
            keywords: Pre-extracted keywords (optional)
            context_hints: Additional context for analysis

        Returns:
            ComplexityProfile with detailed complexity analysis
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context_hints)
            if cache_key in self.analysis_cache:
                self.analysis_stats["cache_hits"] += 1
                return self.analysis_cache[cache_key]

            self.logger.debug(f"Analyzing complexity for query: '{query[:50]}...'")

            # Normalize query for analysis
            normalized_query = self._normalize_query(query)

            # Multi-dimensional complexity analysis
            dimensions = await self._analyze_all_dimensions(normalized_query, keywords, context_hints)

            # Calculate overall complexity score
            overall_score, confidence = self._calculate_overall_complexity(dimensions, normalized_query)

            # Determine complexity level with dynamic thresholds
            complexity_level = self._determine_complexity_level(overall_score, dimensions, context_hints)

            # Advanced metrics calculation
            advanced_metrics = self._calculate_advanced_metrics(dimensions, normalized_query)

            # Build comprehensive profile
            profile = ComplexityProfile(
                overall_complexity=complexity_level, complexity_score=overall_score, confidence=confidence, **dimensions, **advanced_metrics
            )

            # Quality assessment
            self._assess_analysis_quality(profile, query)

            # Cache the result
            self.analysis_cache[cache_key] = profile

            # Update statistics
            analysis_time = (time.time() - start_time) * 1000
            self._update_analysis_stats(profile, analysis_time)

            self.logger.debug(
                f"Complexity analysis complete: {complexity_level.value} "
                f"(score: {overall_score:.3f}, confidence: {confidence:.3f}, "
                f"time: {analysis_time:.2f}ms)"
            )

            return profile

        except Exception as e:
            self.logger.error(f"Error in complexity analysis: {e}")
            # Return fallback profile
            return self._create_fallback_profile(query)

    async def _analyze_all_dimensions(
        self, query: str, keywords: KeywordExtraction | None, context_hints: dict[str, Any] | None
    ) -> dict[str, float]:
        """Analyze complexity across all dimensions."""

        dimensions = {}

        # Lexical complexity analysis
        dimensions["lexical_complexity"] = await self._analyze_lexical_complexity(query)

        # Syntactic complexity analysis
        dimensions["syntactic_complexity"] = await self._analyze_syntactic_complexity(query)

        # Semantic complexity analysis
        dimensions["semantic_complexity"] = await self._analyze_semantic_complexity(query, keywords)

        # Pragmatic complexity analysis
        dimensions["pragmatic_complexity"] = await self._analyze_pragmatic_complexity(query, context_hints)

        # Computational complexity analysis
        dimensions["computational_complexity"] = await self._analyze_computational_complexity(query, keywords)

        # Domain-specific complexity analysis
        dimensions["domain_complexity"] = await self._analyze_domain_complexity(query, keywords, context_hints)

        return dimensions

    async def _analyze_lexical_complexity(self, query: str) -> float:
        """Analyze lexical complexity (vocabulary sophistication)."""

        words = query.lower().split()
        if not words:
            return 0.0

        complexity_score = 0.0

        # Word length complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = min(avg_word_length / 8.0, 1.0)  # Normalize to 0-1
        complexity_score += length_score * 0.25

        # Vocabulary diversity
        unique_words = set(words)
        diversity_score = len(unique_words) / len(words)
        complexity_score += diversity_score * 0.2

        # Technical term density
        technical_score = 0.0
        for pattern_group in self.lexical_patterns.values():
            pattern_matches = 0
            for pattern in pattern_group["patterns"]:
                pattern_matches += len(re.findall(pattern, query, re.IGNORECASE))

            if pattern_matches > 0:
                density = pattern_matches / len(words)
                technical_score += density * pattern_group["weight"]

        complexity_score += min(technical_score, 1.0) * 0.35

        # Rare word detection (simplified heuristic)
        rare_word_count = sum(1 for word in words if len(word) > 7 and word.isalpha())
        rare_score = min(rare_word_count / max(len(words), 1), 1.0)
        complexity_score += rare_score * 0.2

        return min(complexity_score, 1.0)

    async def _analyze_syntactic_complexity(self, query: str) -> float:
        """Analyze syntactic complexity (grammatical structure)."""

        complexity_score = 0.0
        word_count = len(query.split())

        # Sentence length complexity
        length_complexity = min(word_count / 20.0, 1.0)
        complexity_score += length_complexity * 0.3

        # Nested structure complexity
        nesting_score = 0.0
        for pattern_group in self.syntactic_patterns.values():
            pattern_matches = 0
            for pattern in pattern_group["patterns"]:
                pattern_matches += len(re.findall(pattern, query, re.IGNORECASE))

            if pattern_matches > 0:
                density = pattern_matches / max(word_count, 1)
                nesting_score += density * pattern_group["weight"]

        complexity_score += min(nesting_score, 1.0) * 0.4

        # Punctuation complexity
        complex_punctuation = len(re.findall(r'[()[\]{}<>"\';,]', query))
        punct_score = min(complex_punctuation / max(len(query), 1) * 10, 1.0)
        complexity_score += punct_score * 0.2

        # Question complexity
        question_words = len(re.findall(r"\b(what|how|why|where|when|which|who)\b", query, re.IGNORECASE))
        question_score = min(question_words / max(word_count, 1) * 3, 1.0)
        complexity_score += question_score * 0.1

        return min(complexity_score, 1.0)

    async def _analyze_semantic_complexity(self, query: str, keywords: KeywordExtraction | None) -> float:
        """Analyze semantic complexity (meaning and concepts)."""

        complexity_score = 0.0

        # Abstract concept density
        abstract_score = 0.0
        for pattern_group in self.semantic_patterns.values():
            if "patterns" in pattern_group:
                pattern_matches = 0
                for pattern in pattern_group["patterns"]:
                    pattern_matches += len(re.findall(pattern, query, re.IGNORECASE))

                if pattern_matches > 0:
                    abstract_score += pattern_matches * pattern_group["weight"]

        complexity_score += min(abstract_score / max(len(query.split()), 1), 1.0) * 0.4

        # Concept diversity (using keywords if available)
        if keywords:
            concept_categories = [keywords.concept_terms, keywords.technical_terms, keywords.entity_names, keywords.relationship_indicators]

            non_empty_categories = sum(1 for cat in concept_categories if cat)
            diversity_score = non_empty_categories / len(concept_categories)
            complexity_score += diversity_score * 0.3

            # Semantic field analysis
            all_keywords = keywords.get_all_keywords()
            if all_keywords:
                semantic_density = len(set(all_keywords)) / max(len(all_keywords), 1)
                complexity_score += semantic_density * 0.3
        else:
            # Fallback semantic analysis without keywords
            complexity_score += 0.3  # Neutral score

        return min(complexity_score, 1.0)

    async def _analyze_pragmatic_complexity(self, query: str, context_hints: dict[str, Any] | None) -> float:
        """Analyze pragmatic complexity (context and intent)."""

        complexity_score = 0.0

        # Intent clarity analysis
        question_indicators = len(re.findall(r"\b(what|how|why|where|when|which|who)\b", query, re.IGNORECASE))
        action_indicators = len(re.findall(r"\b(find|show|create|implement|analyze|compare)\b", query, re.IGNORECASE))

        if question_indicators == 0 and action_indicators == 0:
            # Unclear intent increases complexity
            complexity_score += 0.3
        elif question_indicators > 1 or action_indicators > 1:
            # Multiple intents increase complexity
            complexity_score += 0.2

        # Context dependency analysis
        ambiguous_refs = len(re.findall(r"\b(it|this|that|they|them)\b(?!\s+\w+)", query, re.IGNORECASE))
        context_dependency = min(ambiguous_refs / max(len(query.split()), 1) * 5, 1.0)
        complexity_score += context_dependency * 0.4

        # Domain context complexity
        if context_hints:
            domain_count = len(context_hints.get("domains", []))
            tech_count = len(context_hints.get("technology_stack", []))

            if domain_count > 1 or tech_count > 2:
                complexity_score += 0.3

        # Implicit knowledge requirements
        implicit_indicators = len(re.findall(r"\b(obviously|clearly|naturally|of course|as usual|typically)\b", query, re.IGNORECASE))
        implicit_score = min(implicit_indicators / max(len(query.split()), 1) * 5, 1.0)
        complexity_score += implicit_score * 0.1

        return min(complexity_score, 1.0)

    async def _analyze_computational_complexity(self, query: str, keywords: KeywordExtraction | None) -> float:
        """Analyze computational complexity (processing requirements)."""

        complexity_score = 0.0

        # Algorithmic operation indicators
        algo_score = 0.0
        for pattern_group in self.computational_patterns.values():
            pattern_matches = 0
            for pattern in pattern_group["patterns"]:
                pattern_matches += len(re.findall(pattern, query, re.IGNORECASE))

            if pattern_matches > 0:
                density = pattern_matches / max(len(query.split()), 1)
                algo_score += density * pattern_group["weight"]

        complexity_score += min(algo_score, 1.0) * 0.5

        # Data processing complexity
        data_ops = len(re.findall(r"\b(all|every|each|multiple|many|various|different|complex|large|massive)\b", query, re.IGNORECASE))
        data_complexity = min(data_ops / max(len(query.split()), 1) * 3, 1.0)
        complexity_score += data_complexity * 0.3

        # Integration complexity
        integration_terms = len(re.findall(r"\b(between|across|through|via|using|with|from.*to|integrate|connect)\b", query, re.IGNORECASE))
        integration_score = min(integration_terms / max(len(query.split()), 1) * 2, 1.0)
        complexity_score += integration_score * 0.2

        return min(complexity_score, 1.0)

    async def _analyze_domain_complexity(
        self, query: str, keywords: KeywordExtraction | None, context_hints: dict[str, Any] | None
    ) -> float:
        """Analyze domain-specific complexity."""

        complexity_score = 0.0

        # Technical domain density
        if keywords:
            technical_density = len(keywords.technical_terms) / max(len(keywords.get_all_keywords()), 1)
            complexity_score += technical_density * 0.4

            # Domain-specific terminology
            domain_terms = keywords.domain_specific_keywords
            domain_density = len(domain_terms) / max(len(keywords.get_all_keywords()), 1)
            complexity_score += domain_density * 0.3

        # Context-specific complexity boost
        if context_hints:
            domains = context_hints.get("detected_domains", {})
            if len(domains) > 1:
                # Multi-domain complexity
                complexity_score += 0.2

            # High-confidence domain indicators
            max_confidence = max(domains.values()) if domains else 0
            if max_confidence > 0.8:
                complexity_score += 0.1

        # Programming-specific complexity
        code_patterns = len(re.findall(r"[()\[\]{}._]|\w+\(\)|\w+\.\w+", query))
        code_complexity = min(code_patterns / max(len(query), 1) * 20, 1.0)
        complexity_score += code_complexity * 0.3

        return min(complexity_score, 1.0)

    def _calculate_overall_complexity(self, dimensions: dict[str, float], query: str) -> tuple[float, float]:
        """Calculate overall complexity score and confidence."""

        # Weighted combination of dimensions
        weights = {
            "lexical_complexity": 0.15,
            "syntactic_complexity": 0.20,
            "semantic_complexity": 0.25,
            "pragmatic_complexity": 0.15,
            "computational_complexity": 0.15,
            "domain_complexity": 0.10,
        }

        overall_score = sum(dimensions[dim] * weight for dim, weight in weights.items())

        # Calculate confidence based on dimension consistency
        dimension_values = list(dimensions.values())
        if len(dimension_values) > 1:
            variance = statistics.variance(dimension_values)
            confidence = max(0.3, 1.0 - variance)  # Lower variance = higher confidence
        else:
            confidence = 0.5

        # Adjust confidence based on query length and clarity
        word_count = len(query.split())
        if word_count < 3:
            confidence *= 0.8  # Less confident for very short queries
        elif word_count > 25:
            confidence *= 0.9  # Slightly less confident for very long queries

        return min(overall_score, 1.0), min(confidence, 1.0)

    def _determine_complexity_level(
        self, score: float, dimensions: dict[str, float], context_hints: dict[str, Any] | None
    ) -> QueryComplexity:
        """Determine complexity level using dynamic thresholds."""

        # Apply context-specific threshold adjustments
        adjusted_thresholds = self._adjust_thresholds_for_context(context_hints)

        # Check each complexity level
        for complexity_level, thresholds in adjusted_thresholds.items():
            score_min, score_max = thresholds["score_range"]

            if score_min <= score <= score_max:
                # Additional validation checks
                if self._validate_complexity_level(complexity_level, dimensions, context_hints):
                    return complexity_level

        # Fallback based on pure score ranges
        if score <= 0.25:
            return QueryComplexity.SIMPLE
        elif score <= 0.50:
            return QueryComplexity.MODERATE
        elif score <= 0.75:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.MULTI_FACETED

    def _adjust_thresholds_for_context(self, context_hints: dict[str, Any] | None) -> dict[QueryComplexity, dict[str, Any]]:
        """Adjust complexity thresholds based on context."""

        adjusted = {}

        for level, thresholds in self.base_thresholds.items():
            adjusted[level] = thresholds.copy()

            if context_hints:
                # Programming context adjustments
                if any(lang in str(context_hints).lower() for lang in ["python", "javascript", "java"]):
                    score_min, score_max = thresholds["score_range"]
                    # Slightly lower thresholds for programming context
                    adjustment = self.threshold_adjustments.get("programming_context", {})
                    boost = adjustment.get("domain_complexity_boost", 0)
                    adjusted[level]["score_range"] = (max(0, score_min - boost), min(1.0, score_max - boost))

        return adjusted

    def _validate_complexity_level(
        self, level: QueryComplexity, dimensions: dict[str, float], context_hints: dict[str, Any] | None
    ) -> bool:
        """Validate that the assigned complexity level makes sense."""

        # Simple validation rules
        if level == QueryComplexity.SIMPLE:
            # Simple queries shouldn't have high semantic or computational complexity
            if dimensions.get("semantic_complexity", 0) > 0.7 or dimensions.get("computational_complexity", 0) > 0.6:
                return False

        elif level == QueryComplexity.MULTI_FACETED:
            # Multi-faceted queries should have complexity in multiple dimensions
            high_complexity_dims = sum(1 for score in dimensions.values() if score > 0.6)
            if high_complexity_dims < 3:
                return False

        return True

    def _calculate_advanced_metrics(self, dimensions: dict[str, float], query: str) -> dict[str, Any]:
        """Calculate advanced complexity metrics."""

        dimension_values = list(dimensions.values())

        # Complexity variance (how varied complexity is across dimensions)
        complexity_variance = statistics.variance(dimension_values) if len(dimension_values) > 1 else 0.0

        # Complexity stability (inverse of variance, normalized)
        complexity_stability = max(0.0, 1.0 - complexity_variance)

        # Processing difficulty (weighted combination emphasizing hard-to-process aspects)
        processing_difficulty = (
            dimensions.get("computational_complexity", 0) * 0.4
            + dimensions.get("semantic_complexity", 0) * 0.3
            + dimensions.get("domain_complexity", 0) * 0.2
            + dimensions.get("pragmatic_complexity", 0) * 0.1
        )

        # Word-level complexity analysis
        words = query.lower().split()
        word_complexity_dist = {}

        for word in words:
            # Simple word complexity heuristic
            complexity = (
                min(len(word) / 10.0, 1.0) * 0.5  # Length factor
                + (1.0 if re.match(r"[a-z_]*[A-Z]", word) else 0.0) * 0.3  # CamelCase factor
                + (1.0 if len(re.findall(r"[^a-zA-Z0-9]", word)) > 0 else 0.0) * 0.2  # Special chars
            )
            word_complexity_dist[word] = complexity

        # Phrase complexity (simplified)
        phrases = re.split(r"[.!?;,]", query)
        phrase_complexity_scores = []

        for phrase in phrases:
            if phrase.strip():
                phrase_words = len(phrase.split())
                phrase_complexity = min(phrase_words / 10.0, 1.0)
                phrase_complexity_scores.append(phrase_complexity)

        # Structural patterns detected
        structural_patterns = []
        if re.search(r"\([^)]*\)", query):
            structural_patterns.append("parenthetical_expressions")
        if re.search(r"\b\w+\.\w+", query):
            structural_patterns.append("dot_notation")
        if re.search(r"\b\w+\(\)", query):
            structural_patterns.append("function_calls")
        if re.search(r"\b(and|or|but)\b.*\b(and|or|but)\b", query, re.IGNORECASE):
            structural_patterns.append("complex_conjunction")

        return {
            "complexity_variance": complexity_variance,
            "complexity_stability": complexity_stability,
            "processing_difficulty": processing_difficulty,
            "word_complexity_distribution": word_complexity_dist,
            "phrase_complexity_scores": phrase_complexity_scores,
            "structural_patterns": structural_patterns,
        }

    def _assess_analysis_quality(self, profile: ComplexityProfile, query: str) -> None:
        """Assess the quality of complexity analysis."""

        quality_factors = []
        missing_dimensions = []
        uncertainty_factors = []

        # Check for missing or low-quality dimensions
        dimensions = {
            "lexical_complexity": profile.lexical_complexity,
            "syntactic_complexity": profile.syntactic_complexity,
            "semantic_complexity": profile.semantic_complexity,
            "pragmatic_complexity": profile.pragmatic_complexity,
            "computational_complexity": profile.computational_complexity,
            "domain_complexity": profile.domain_complexity,
        }

        for dim_name, dim_score in dimensions.items():
            if dim_score == 0.0:
                missing_dimensions.append(dim_name)
                quality_factors.append(0.7)  # Penalize missing dimensions
            elif dim_score < 0.1:
                uncertainty_factors.append(f"low_{dim_name}")
                quality_factors.append(0.9)
            else:
                quality_factors.append(1.0)

        # Query length quality factor
        word_count = len(query.split())
        if word_count < 2:
            quality_factors.append(0.5)  # Very short queries are hard to analyze
            uncertainty_factors.append("very_short_query")
        elif word_count > 50:
            quality_factors.append(0.8)  # Very long queries may have noise
            uncertainty_factors.append("very_long_query")
        else:
            quality_factors.append(1.0)

        # Calculate overall quality
        analysis_quality = sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

        # Update profile
        profile.analysis_quality = analysis_quality
        profile.missing_dimensions = missing_dimensions
        profile.uncertainty_factors = uncertainty_factors

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent analysis."""

        # Basic normalization
        normalized = query.strip()

        # Remove excessive whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Handle common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "it's": "it is",
            "that's": "that is",
            "here's": "here is",
        }

        for contraction, expansion in contractions.items():
            normalized = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, normalized, flags=re.IGNORECASE)

        return normalized

    def _generate_cache_key(self, query: str, context_hints: dict[str, Any] | None) -> str:
        """Generate cache key for complexity analysis."""
        import hashlib

        key_components = [query.lower().strip()]

        if context_hints:
            # Include relevant context in cache key
            context_str = str(sorted(context_hints.items()))
            key_components.append(context_str)

        combined = "|".join(key_components)
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def _create_fallback_profile(self, query: str) -> ComplexityProfile:
        """Create fallback complexity profile when analysis fails."""

        # Simple heuristic-based fallback
        word_count = len(query.split())

        if word_count <= 3:
            complexity = QueryComplexity.SIMPLE
            score = 0.2
        elif word_count <= 10:
            complexity = QueryComplexity.MODERATE
            score = 0.4
        elif word_count <= 20:
            complexity = QueryComplexity.COMPLEX
            score = 0.6
        else:
            complexity = QueryComplexity.MULTI_FACETED
            score = 0.8

        return ComplexityProfile(
            overall_complexity=complexity,
            complexity_score=score,
            confidence=0.3,  # Low confidence for fallback
            analysis_quality=0.2,
            uncertainty_factors=["analysis_fallback"],
        )

    def _update_analysis_stats(self, profile: ComplexityProfile, analysis_time: float) -> None:
        """Update analysis statistics."""

        self.analysis_stats["analyses_performed"] += 1

        # Update average analysis time
        current_avg = self.analysis_stats["average_analysis_time_ms"]
        count = self.analysis_stats["analyses_performed"]
        new_avg = ((current_avg * (count - 1)) + analysis_time) / count
        self.analysis_stats["average_analysis_time_ms"] = new_avg

        # Update complexity distribution
        complexity_key = profile.overall_complexity.value
        self.analysis_stats["complexity_distribution"][complexity_key] += 1

    def get_analysis_statistics(self) -> dict[str, Any]:
        """Get current analysis statistics."""

        stats = self.analysis_stats.copy()

        # Add cache statistics
        stats["cache_size"] = len(self.analysis_cache)
        stats["cache_hit_rate"] = self.analysis_stats["cache_hits"] / max(self.analysis_stats["analyses_performed"], 1)

        return stats

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        self.logger.info("Complexity analysis cache cleared")

    async def batch_analyze_complexity(
        self, queries: list[str], context_hints: list[dict[str, Any]] | None = None
    ) -> list[ComplexityProfile]:
        """Perform batch complexity analysis for multiple queries."""

        results = []

        for i, query in enumerate(queries):
            hints = context_hints[i] if context_hints and i < len(context_hints) else None
            profile = await self.analyze_complexity(query, context_hints=hints)
            results.append(profile)

        return results


# Factory function
_complexity_detector_instance = None


def get_complexity_detector() -> EnhancedComplexityDetector:
    """Get or create a ComplexityDetector instance."""
    global _complexity_detector_instance
    if _complexity_detector_instance is None:
        _complexity_detector_instance = EnhancedComplexityDetector()
    return _complexity_detector_instance
