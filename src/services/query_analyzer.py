"""
Query Analysis Service for Multi-Modal Retrieval System

This service provides comprehensive query analysis and classification to determine
the optimal retrieval strategy for the LightRAG-inspired multi-modal system.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from ..models.query_features import (
        GlobalModeConfig,
        HybridModeConfig,
        KeywordExtraction,
        LocalModeConfig,
        MixModeConfig,
        QueryComplexity,
        QueryComplexityAnalysis,
        QueryFeatures,
        QueryIntentAnalysis,
        QueryType,
        RetrievalModeConfig,
    )
    from ..utils.complexity_detector import get_complexity_detector
    from ..utils.keyword_extractor import get_keyword_extractor
except ImportError:
    # Fallback for standalone testing
    from models.query_features import (
        GlobalModeConfig,
        HybridModeConfig,
        KeywordExtraction,
        LocalModeConfig,
        MixModeConfig,
        QueryComplexity,
        QueryComplexityAnalysis,
        QueryFeatures,
        QueryIntentAnalysis,
        QueryType,
        RetrievalModeConfig,
    )
    from utils.complexity_detector import get_complexity_detector
    from utils.keyword_extractor import get_keyword_extractor


logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Comprehensive query analyzer that classifies queries and recommends
    optimal retrieval strategies for the multi-modal retrieval system.
    """

    def __init__(self):
        self.keyword_extractor = get_keyword_extractor()
        self.complexity_detector = get_complexity_detector()
        self.logger = logging.getLogger(__name__)

        # Classification patterns and rules
        self._initialize_classification_rules()

        # Mode configurations
        self.mode_configs = {
            "local": LocalModeConfig(),
            "global": GlobalModeConfig(),
            "hybrid": HybridModeConfig(),
            "mix": MixModeConfig(),
        }

        # Performance tracking
        self.analysis_stats = {
            "queries_analyzed": 0,
            "average_analysis_time_ms": 0.0,
            "mode_recommendations": {"local": 0, "global": 0, "hybrid": 0, "mix": 0},
            "enhanced_complexity_analyses": 0,
            "complexity_accuracy_feedback": {"correct": 0, "incorrect": 0},
        }

    def _initialize_classification_rules(self):
        """Initialize patterns and rules for query classification."""

        # Query type indicators
        self.type_indicators = {
            QueryType.ENTITY_FOCUSED: [
                r"\b(function|method|class|variable|attribute)\s+\w+",
                r"\b\w+\(\)",  # function calls
                r"\b[A-Z][a-z]*[A-Z]\w*\b",  # PascalCase
                r"\b[a-z_]\w*[A-Z]\w*\b",  # camelCase
                r"\bfind\s+\w+",
                r"\bshow\s+me\s+\w+",
                r"\bwhat\s+is\s+\w+",
            ],
            QueryType.RELATIONSHIP_FOCUSED: [
                r"\b(connects?|links?|relates?|between|among)",
                r"\b(inherits?|extends?|implements?|uses?|calls?)",
                r"\b(depends?|requires?|imports?|includes?)",
                r"\b(parent|child|base|derived|super|sub)",
                r"\bhow.*connect",
                r"\brelationship.*between",
                r"\bconnection.*to",
            ],
            QueryType.CONCEPTUAL: [
                r"\b(pattern|design|architecture|approach|concept)",
                r"\b(why|how|what.*purpose|what.*goal)",
                r"\b(strategy|methodology|principle|paradigm)",
                r"\b(overview|summary|explanation|description)",
                r"\bexplain.*concept",
                r"\bunderstand.*idea",
            ],
            QueryType.IMPLEMENTATION_FOCUSED: [
                r"\b(implement|code|write|create|build)",
                r"\b(example|sample|demo|tutorial)",
                r"\bhow.*to.*code",
                r"\bshow.*implementation",
                r"\bwrite.*function",
                r"\bcreate.*class",
            ],
            QueryType.PATTERN_FOCUSED: [
                r"\b(pattern|template|boilerplate|scaffold)",
                r"\b(best.*practice|common.*approach|standard.*way)",
                r"\b(design.*pattern|architectural.*pattern)",
                r"\btypical.*implementation",
                r"\busual.*way",
            ],
            QueryType.EXPLORATION: [
                r"\b(explore|browse|discover|find.*similar)",
                r"\b(what.*else|alternatives?|options?)",
                r"\b(similar.*to|like.*this|comparable)",
                r"\bshow.*all",
                r"\blist.*everything",
            ],
            QueryType.TROUBLESHOOTING: [
                r"\b(error|bug|issue|problem|fix|debug)",
                r"\b(why.*not.*work|not.*working|broken)",
                r"\b(troubleshoot|diagnose|solve)",
                r"\bwhat.*wrong",
                r"\bhow.*fix",
            ],
        }

        # Complexity indicators
        self.complexity_indicators = {
            "simple_markers": [
                r"^\w+$",  # single word
                r"^\w+\s+\w+$",  # two words
                r"\bshow\s+\w+",
                r"\bfind\s+\w+",
                r"\bwhat\s+is\s+\w+",
            ],
            "moderate_markers": [
                r"\b(and|or|but)\b",  # logical connectors
                r"\b\w+.*\w+.*\w+",  # three or more concepts
                r"\b(between|among|with|for)\b",  # relational prepositions
            ],
            "complex_markers": [
                r"\b(implement|integrate|combine|compare)\b",
                r"\bmultiple\b",
                r"\bdifferent.*types?",
                r"\bvarious.*approaches?",
                r"\bcombination.*of",
            ],
            "multi_faceted_markers": [
                r"\b(architecture|system|framework|infrastructure)\b",
                r"\bend.*to.*end",
                r"\bfull.*stack",
                r"\bcomprehensive",
                r"\beverything.*about",
                r"\ball.*aspects?",
            ],
        }

        # Language and framework hints
        self.context_hints = {
            "languages": [
                "python",
                "javascript",
                "typescript",
                "java",
                "cpp",
                "c++",
                "rust",
                "go",
                "php",
                "ruby",
                "swift",
                "kotlin",
                "scala",
            ],
            "frameworks": [
                "react",
                "vue",
                "angular",
                "django",
                "flask",
                "fastapi",
                "spring",
                "express",
                "laravel",
                "rails",
                "tensorflow",
                "pytorch",
                "pandas",
                "numpy",
            ],
            "domains": [
                "web",
                "mobile",
                "desktop",
                "api",
                "database",
                "ml",
                "ai",
                "data",
                "analytics",
                "security",
                "devops",
                "frontend",
                "backend",
                "fullstack",
            ],
        }

    async def analyze_query(self, query: str) -> QueryFeatures:
        """
        Perform comprehensive analysis of a search query.

        Args:
            query: The search query to analyze

        Returns:
            QueryFeatures with complete analysis results
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Analyzing query: '{query[:50]}...'")

            # Basic normalization and preprocessing
            normalized_query = self._normalize_query(query)

            # Extract keywords at different levels
            keywords = self.keyword_extractor.extract_keywords(query)

            # Classify query type
            query_type, type_confidence = self._classify_query_type(normalized_query, keywords)

            # Determine complexity
            complexity = await self._determine_complexity(normalized_query, keywords)

            # Extract semantic indicators
            semantic_indicators = self._analyze_semantic_indicators(normalized_query, keywords)

            # Extract context hints
            context_hints = self._extract_context_hints(normalized_query)

            # Recommend retrieval mode
            recommended_mode, mode_confidence = self._recommend_retrieval_mode(query_type, complexity, keywords, semantic_indicators)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Get advanced complexity analysis
            complexity_analysis = await self._analyze_query_complexity_advanced(normalized_query, keywords)

            # Get advanced intent analysis
            intent_analysis = self._analyze_query_intent_advanced(normalized_query, keywords)

            # Build enhanced complexity and intent analysis objects
            complexity_analysis_obj = QueryComplexityAnalysis(
                overall_complexity=complexity_analysis["overall_complexity"],
                complexity_score=complexity_analysis["complexity_score"],
                complexity_confidence=complexity_analysis["complexity_confidence"],
                lexical_complexity=complexity_analysis["complexity_factors"]["lexical_complexity"],
                syntactic_complexity=complexity_analysis["complexity_factors"]["syntactic_complexity"],
                semantic_complexity=complexity_analysis["complexity_factors"]["semantic_complexity"],
                conceptual_depth=complexity_analysis["complexity_factors"]["conceptual_depth"],
                relationship_complexity=complexity_analysis["complexity_factors"]["relationship_complexity"],
                domain_specificity=complexity_analysis["complexity_factors"]["domain_specificity"],
                linguistic_complexity=complexity_analysis["complexity_breakdown"]["linguistic_complexity"],
                conceptual_complexity=complexity_analysis["complexity_breakdown"]["conceptual_complexity"],
                domain_complexity=complexity_analysis["complexity_breakdown"]["domain_complexity"],
                complexity_factors=complexity_analysis["complexity_factors"],
                complexity_metrics=complexity_analysis["metrics"],
            )

            intent_analysis_obj = QueryIntentAnalysis(
                primary_intent=intent_analysis["primary_intent"],
                intent_confidence=intent_analysis["intent_confidence"],
                secondary_intent=intent_analysis.get("secondary_intent"),
                intent_scores=intent_analysis["intent_scores"],
                intent_distribution=intent_analysis["intent_distribution"],
                intent_clarity=intent_analysis["intent_clarity"],
                intent_specificity=intent_analysis["intent_specificity"],
                multi_intent=intent_analysis["multi_intent"],
                detected_intents=intent_analysis["detected_intents"],
                intent_evidence=intent_analysis["intent_evidence"],
            )

            # Build QueryFeatures object with enhanced complexity and intent information
            features = QueryFeatures(
                original_query=query,
                normalized_query=normalized_query,
                query_length=len(query),
                word_count=len(normalized_query.split()),
                query_type=query_type,
                complexity=complexity,
                confidence_score=type_confidence,
                keywords=keywords,
                complexity_analysis=complexity_analysis_obj,
                intent_analysis=intent_analysis_obj,
                **semantic_indicators,
                **context_hints,
                recommended_mode=recommended_mode,
                mode_confidence=mode_confidence,
                processing_time_ms=processing_time_ms,
                analysis_timestamp=datetime.now().isoformat(),
                metadata={
                    "enhanced_complexity_detector_used": True,
                    "analysis_quality": complexity_analysis.get("advanced_metrics", {}).get("analysis_quality", 0.0),
                    "structural_patterns": complexity_analysis.get("advanced_metrics", {}).get("structural_patterns", []),
                    "uncertainty_factors": complexity_analysis.get("advanced_metrics", {}).get("uncertainty_factors", []),
                },
            )

            # Update statistics
            self._update_stats(features)

            self.logger.debug(
                f"Query analysis complete: type={query_type.value}, "
                f"complexity={complexity.value}, mode={recommended_mode}, "
                f"intent_confidence={intent_analysis['intent_confidence']:.2f}, "
                f"complexity_score={complexity_analysis['complexity_score']:.2f}, "
                f"time={processing_time_ms:.2f}ms"
            )

            return features

        except Exception as e:
            self.logger.error(f"Error analyzing query '{query}': {e}")
            # Return minimal features on error
            return QueryFeatures(
                original_query=query,
                normalized_query=query.lower().strip(),
                query_length=len(query),
                word_count=len(query.split()),
                query_type=QueryType.EXPLORATION,
                complexity=QueryComplexity.MODERATE,
                confidence_score=0.0,
                keywords=KeywordExtraction(),
                processing_time_ms=(time.time() - start_time) * 1000,
                analysis_timestamp=datetime.now().isoformat(),
            )

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent analysis."""
        import re

        # Basic normalization
        normalized = query.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Handle common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
        }

        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)

        return normalized

    def _classify_query_type(self, query: str, keywords: KeywordExtraction) -> tuple[QueryType, float]:
        """Enhanced query type classification with advanced intent analysis."""
        import re

        # Get comprehensive intent analysis
        intent_analysis = self._analyze_query_intent_advanced(query, keywords)

        # Extract primary intent and confidence
        return intent_analysis["primary_intent"], intent_analysis["intent_confidence"]

    def _analyze_query_intent_advanced(self, query: str, keywords: KeywordExtraction) -> dict[str, any]:
        """Advanced query intent analysis with multi-dimensional classification."""
        import re

        # Initialize intent scoring system
        intent_scores = dict.fromkeys(QueryType, 0.0)
        intent_evidence = {intent_type: [] for intent_type in QueryType}

        # Pattern-based scoring with weighted evidence
        for query_type, patterns in self.type_indicators.items():
            pattern_matches = 0
            pattern_evidence = []

            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    pattern_matches += len(matches)
                    pattern_evidence.extend(matches)

            # Weighted scoring based on pattern strength
            if len(patterns) > 0:
                base_score = pattern_matches / len(patterns)
                intent_scores[query_type] += base_score * 0.6  # 60% weight for patterns
                intent_evidence[query_type].extend(pattern_evidence)

        # Advanced semantic analysis
        semantic_scores = self._analyze_semantic_intent(query, keywords)
        for intent_type, score in semantic_scores.items():
            intent_scores[intent_type] += score * 0.3  # 30% weight for semantics

        # Structural analysis
        structural_scores = self._analyze_structural_intent(query, keywords)
        for intent_type, score in structural_scores.items():
            intent_scores[intent_type] += score * 0.1  # 10% weight for structure

        # Context-aware adjustments
        context_adjustments = self._apply_context_adjustments(query, keywords, intent_scores)
        for intent_type, adjustment in context_adjustments.items():
            intent_scores[intent_type] += adjustment

        # Find primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)

        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        secondary_intent = sorted_intents[1][0] if len(sorted_intents) > 1 else None
        secondary_score = sorted_intents[1][1] if len(sorted_intents) > 1 else 0.0

        # Calculate confidence based on score separation
        if secondary_score > 0:
            confidence = min(primary_score / (primary_score + secondary_score), 1.0)
        else:
            confidence = min(primary_score, 1.0)

        # Intent clarity and specificity analysis
        intent_clarity = self._calculate_intent_clarity(intent_scores)
        intent_specificity = self._calculate_intent_specificity(query, keywords)

        # Multi-intent detection
        multi_intents = self._detect_multi_intents(intent_scores, threshold=0.3)

        return {
            "primary_intent": primary_intent,
            "intent_confidence": confidence,
            "secondary_intent": secondary_intent,
            "intent_scores": dict(intent_scores),
            "intent_evidence": {k.value: v for k, v in intent_evidence.items()},
            "intent_clarity": intent_clarity,
            "intent_specificity": intent_specificity,
            "multi_intent": len(multi_intents) > 1,
            "detected_intents": [intent.value for intent in multi_intents],
            "intent_distribution": self._calculate_intent_distribution(intent_scores),
        }

    def _analyze_semantic_intent(self, query: str, keywords: KeywordExtraction) -> dict[QueryType, float]:
        """Analyze semantic indicators for intent classification."""
        semantic_scores = dict.fromkeys(QueryType, 0.0)

        # Entity-focused indicators
        if keywords.entity_names:
            entity_ratio = len(keywords.entity_names) / max(1, len(keywords.get_all_keywords()))
            semantic_scores[QueryType.ENTITY_FOCUSED] += entity_ratio * 0.8

        # Relationship-focused indicators
        if keywords.relationship_indicators:
            rel_ratio = len(keywords.relationship_indicators) / max(1, len(query.split()))
            semantic_scores[QueryType.RELATIONSHIP_FOCUSED] += rel_ratio * 1.0

        # Conceptual indicators
        if keywords.concept_terms:
            concept_ratio = len(keywords.concept_terms) / max(1, len(keywords.get_all_keywords()))
            semantic_scores[QueryType.CONCEPTUAL] += concept_ratio * 0.7

        # Technical implementation focus
        technical_density = len(keywords.technical_terms) / max(1, len(query.split()))
        if technical_density > 0.2:
            semantic_scores[QueryType.IMPLEMENTATION_FOCUSED] += technical_density * 0.6

        # Question word analysis
        question_intents = self._analyze_question_words(query)
        for intent_type, score in question_intents.items():
            semantic_scores[intent_type] += score

        return semantic_scores

    def _analyze_structural_intent(self, query: str, keywords: KeywordExtraction) -> dict[QueryType, float]:
        """Analyze structural patterns for intent classification."""
        import re

        structural_scores = dict.fromkeys(QueryType, 0.0)

        # Query length and complexity patterns
        word_count = len(query.split())

        if word_count <= 3:
            # Short queries often entity-focused
            structural_scores[QueryType.ENTITY_FOCUSED] += 0.3
        elif word_count > 15:
            # Long queries often conceptual or exploratory
            structural_scores[QueryType.CONCEPTUAL] += 0.2
            structural_scores[QueryType.EXPLORATION] += 0.2

        # Code-like patterns
        code_patterns = r"[()\[\]{}._]|\b\w+\(\)|\b[a-z_]+[A-Z]\w*\b"
        if re.search(code_patterns, query):
            structural_scores[QueryType.IMPLEMENTATION_FOCUSED] += 0.4
            structural_scores[QueryType.ENTITY_FOCUSED] += 0.2

        # Problem/solution structure
        problem_patterns = r"\b(problem|issue|error|bug|fix|solve|debug)\b"
        if re.search(problem_patterns, query, re.IGNORECASE):
            structural_scores[QueryType.TROUBLESHOOTING] += 0.5

        # Comparison structure
        comparison_patterns = r"\b(compare|vs|versus|difference|better|worse|similar|different)\b"
        if re.search(comparison_patterns, query, re.IGNORECASE):
            structural_scores[QueryType.PATTERN_FOCUSED] += 0.3
            structural_scores[QueryType.EXPLORATION] += 0.2

        return structural_scores

    def _analyze_question_words(self, query: str) -> dict[QueryType, float]:
        """Analyze question words to determine intent."""
        import re

        question_scores = dict.fromkeys(QueryType, 0.0)

        # What questions - often entity or conceptual
        if re.search(r"\bwhat\b", query, re.IGNORECASE):
            if re.search(r"what.*is|what.*are", query, re.IGNORECASE):
                question_scores[QueryType.ENTITY_FOCUSED] += 0.4
                question_scores[QueryType.CONCEPTUAL] += 0.3
            elif re.search(r"what.*does|what.*happens", query, re.IGNORECASE):
                question_scores[QueryType.CONCEPTUAL] += 0.5

        # How questions - often implementation or conceptual
        if re.search(r"\bhow\b", query, re.IGNORECASE):
            if re.search(r"how.*to|how.*do", query, re.IGNORECASE):
                question_scores[QueryType.IMPLEMENTATION_FOCUSED] += 0.6
            elif re.search(r"how.*work|how.*connect", query, re.IGNORECASE):
                question_scores[QueryType.RELATIONSHIP_FOCUSED] += 0.5
                question_scores[QueryType.CONCEPTUAL] += 0.3

        # Why questions - almost always conceptual
        if re.search(r"\bwhy\b", query, re.IGNORECASE):
            question_scores[QueryType.CONCEPTUAL] += 0.7

        # Where questions - often structural or entity-focused
        if re.search(r"\bwhere\b", query, re.IGNORECASE):
            question_scores[QueryType.ENTITY_FOCUSED] += 0.4
            question_scores[QueryType.EXPLORATION] += 0.3

        # Which questions - often pattern or exploration
        if re.search(r"\bwhich\b", query, re.IGNORECASE):
            question_scores[QueryType.PATTERN_FOCUSED] += 0.4
            question_scores[QueryType.EXPLORATION] += 0.3

        return question_scores

    def _apply_context_adjustments(
        self, query: str, keywords: KeywordExtraction, intent_scores: dict[QueryType, float]
    ) -> dict[QueryType, float]:
        """Apply context-aware adjustments to intent scores."""
        adjustments = dict.fromkeys(QueryType, 0.0)

        # Technical context boosts implementation and troubleshooting
        if keywords.technical_terms:
            tech_density = len(keywords.technical_terms) / max(1, len(query.split()))
            adjustments[QueryType.IMPLEMENTATION_FOCUSED] += tech_density * 0.2
            adjustments[QueryType.TROUBLESHOOTING] += tech_density * 0.1

        # Abstract concepts boost conceptual intent
        abstract_terms = ["pattern", "design", "architecture", "approach", "strategy"]
        abstract_count = sum(1 for term in abstract_terms if term in query.lower())
        if abstract_count > 0:
            adjustments[QueryType.CONCEPTUAL] += abstract_count * 0.1
            adjustments[QueryType.PATTERN_FOCUSED] += abstract_count * 0.1

        # Multiple entities suggest relationship focus
        if len(keywords.entity_names) > 1:
            adjustments[QueryType.RELATIONSHIP_FOCUSED] += 0.2
            adjustments[QueryType.EXPLORATION] += 0.1

        # Uncertain language suggests exploration
        uncertain_terms = ["might", "maybe", "possibly", "could", "alternatives", "options"]
        uncertain_count = sum(1 for term in uncertain_terms if term in query.lower())
        if uncertain_count > 0:
            adjustments[QueryType.EXPLORATION] += uncertain_count * 0.15

        return adjustments

    def _calculate_intent_clarity(self, intent_scores: dict[QueryType, float]) -> float:
        """Calculate how clear/unambiguous the intent is."""
        sorted_scores = sorted(intent_scores.values(), reverse=True)

        if len(sorted_scores) < 2 or sorted_scores[0] == 0:
            return 0.0

        # Clarity is higher when there's a clear winner
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]

        clarity = (top_score - second_score) / max(top_score, 0.1)
        return min(clarity, 1.0)

    def _calculate_intent_specificity(self, query: str, keywords: KeywordExtraction) -> float:
        """Calculate how specific vs general the query intent is."""
        # Specific indicators
        specific_factors = [
            len(keywords.entity_names) > 0,  # Has specific entities
            len(keywords.technical_terms) > 2,  # Has multiple technical terms
            any(char in query for char in '()[]{}"'),  # Has specific syntax
            len(query.split()) > 1 and len(query.split()) < 8,  # Focused length
        ]

        # General indicators
        general_factors = [
            any(word in query.lower() for word in ["general", "overview", "all", "everything"]),
            len(keywords.concept_terms) > len(keywords.entity_names),
            "how" in query.lower() and "to" not in query.lower(),
        ]

        specificity = (sum(specific_factors) - sum(general_factors)) / max(1, len(specific_factors))
        return max(0.0, min(1.0, (specificity + 1) / 2))  # Normalize to 0-1

    def _detect_multi_intents(self, intent_scores: dict[QueryType, float], threshold: float = 0.3) -> list[QueryType]:
        """Detect queries with multiple strong intents."""
        strong_intents = []
        max_score = max(intent_scores.values()) if intent_scores.values() else 0

        if max_score > 0:
            for intent_type, score in intent_scores.items():
                if score >= threshold and score >= max_score * 0.6:  # At least 60% of max score
                    strong_intents.append(intent_type)

        return strong_intents

    def _calculate_intent_distribution(self, intent_scores: dict[QueryType, float]) -> dict[str, float]:
        """Calculate the distribution of intent scores as percentages."""
        total_score = sum(intent_scores.values())

        if total_score == 0:
            return {intent.value: 0.0 for intent in QueryType}

        return {intent_type.value: score / total_score for intent_type, score in intent_scores.items()}

    async def _determine_complexity(self, query: str, keywords: KeywordExtraction) -> QueryComplexity:
        """Enhanced complexity determination using advanced complexity detector."""

        # Use the enhanced complexity detector for sophisticated analysis
        complexity_profile = await self.complexity_detector.analyze_complexity(
            query=query,
            keywords=keywords,
            context_hints=None,  # Will be enhanced in future iterations
        )

        self.analysis_stats["enhanced_complexity_analyses"] += 1

        return complexity_profile.overall_complexity

    async def _analyze_query_complexity_advanced(self, query: str, keywords: KeywordExtraction) -> dict[str, any]:
        """Advanced query complexity analysis using enhanced complexity detector."""

        # Use the enhanced complexity detector for comprehensive analysis
        complexity_profile = await self.complexity_detector.analyze_complexity(
            query=query,
            keywords=keywords,
            context_hints=None,  # Can be enhanced with additional context
        )

        # Convert ComplexityProfile to the expected dictionary format
        analysis = {
            "overall_complexity": complexity_profile.overall_complexity,
            "complexity_score": complexity_profile.complexity_score,
            "complexity_confidence": complexity_profile.confidence,
            "complexity_factors": {
                "lexical_complexity": complexity_profile.lexical_complexity,
                "syntactic_complexity": complexity_profile.syntactic_complexity,
                "semantic_complexity": complexity_profile.semantic_complexity,
                "conceptual_depth": complexity_profile.pragmatic_complexity,
                "relationship_complexity": complexity_profile.computational_complexity,
                "domain_specificity": complexity_profile.domain_complexity,
            },
            "metrics": {
                "word_count": len(query.split()),
                "entity_count": len(keywords.entity_names),
                "concept_count": len(keywords.concept_terms),
                "relationship_count": len(keywords.relationship_indicators),
                "technical_count": len(keywords.technical_terms),
                "complexity_variance": complexity_profile.complexity_variance,
                "complexity_stability": complexity_profile.complexity_stability,
                "processing_difficulty": complexity_profile.processing_difficulty,
            },
            "complexity_breakdown": {
                "linguistic_complexity": (complexity_profile.lexical_complexity + complexity_profile.syntactic_complexity) / 2,
                "semantic_complexity": complexity_profile.semantic_complexity,
                "conceptual_complexity": (complexity_profile.pragmatic_complexity + complexity_profile.computational_complexity) / 2,
                "domain_complexity": complexity_profile.domain_complexity,
            },
            "advanced_metrics": {
                "analysis_quality": complexity_profile.analysis_quality,
                "structural_patterns": complexity_profile.structural_patterns,
                "uncertainty_factors": complexity_profile.uncertainty_factors,
                "missing_dimensions": complexity_profile.missing_dimensions,
            },
        }

        return analysis

    def _calculate_lexical_complexity(self, query: str) -> float:
        """Calculate lexical complexity based on vocabulary diversity and word characteristics."""
        import re

        words = query.lower().split()
        if not words:
            return 0.0

        # Unique word ratio
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_complexity = min(avg_word_length / 10.0, 1.0)

        # Technical vocabulary density
        technical_words = sum(
            1
            for word in words
            if len(word) > 6
            and (word.endswith(("tion", "sion", "ment", "ness", "ity")) or word.startswith(("pre", "post", "anti", "proto")))
        )
        tech_density = min(technical_words / max(1, len(words)), 1.0)

        # Complex punctuation and symbols
        complex_chars = len(re.findall(r'[()\[\]{}<>"\'\_\-\.]', query))
        symbol_complexity = min(complex_chars / max(1, len(query)) * 10, 1.0)

        return diversity_ratio * 0.3 + length_complexity * 0.3 + tech_density * 0.2 + symbol_complexity * 0.2

    def _calculate_syntactic_complexity(self, query: str) -> float:
        """Calculate syntactic complexity based on sentence structure."""
        import re

        # Sentence length (normalized)
        word_count = len(query.split())
        length_complexity = min(word_count / 20.0, 1.0)

        # Clause complexity (conjunctions, relative pronouns)
        clause_markers = len(re.findall(r"\b(and|or|but|which|that|where|when|because|although|however|therefore)\b", query, re.IGNORECASE))
        clause_complexity = min(clause_markers / max(1, word_count) * 5, 1.0)

        # Question complexity
        question_markers = len(re.findall(r"\b(what|how|why|where|when|which|who)\b", query, re.IGNORECASE))
        question_complexity = min(question_markers / max(1, word_count) * 3, 1.0)

        # Nested structure indicators
        nesting_indicators = len(re.findall(r"[()\[\]{}]", query))
        nesting_complexity = min(nesting_indicators / max(1, len(query)) * 20, 1.0)

        return length_complexity * 0.3 + clause_complexity * 0.3 + question_complexity * 0.2 + nesting_complexity * 0.2

    def _calculate_semantic_complexity(self, keywords: KeywordExtraction) -> float:
        """Calculate semantic complexity based on keyword diversity and relationships."""
        all_keywords = keywords.get_all_keywords()
        if not all_keywords:
            return 0.0

        # Keyword diversity across categories
        categories = [keywords.entity_names, keywords.concept_terms, keywords.technical_terms, keywords.relationship_indicators]

        non_empty_categories = sum(1 for cat in categories if cat)
        category_diversity = non_empty_categories / len(categories)

        # Semantic field diversity (different domains)
        semantic_fields = self._identify_semantic_fields(keywords)
        field_diversity = min(len(semantic_fields) / 5.0, 1.0)

        # Abstract vs concrete balance
        abstract_terms = len(keywords.concept_terms)
        concrete_terms = len(keywords.entity_names) + len(keywords.technical_terms)
        total_terms = abstract_terms + concrete_terms

        if total_terms > 0:
            abstraction_balance = 1.0 - abs(abstract_terms - concrete_terms) / total_terms
        else:
            abstraction_balance = 0.0

        return category_diversity * 0.4 + field_diversity * 0.4 + abstraction_balance * 0.2

    def _calculate_conceptual_depth(self, keywords: KeywordExtraction, query: str) -> float:
        """Calculate conceptual depth based on hierarchical and abstract thinking."""
        import re

        # Hierarchical indicators
        hierarchy_patterns = r"\b(parent|child|base|derived|super|sub|inherit|extend|implement)\b"
        hierarchy_matches = len(re.findall(hierarchy_patterns, query, re.IGNORECASE))
        hierarchy_depth = min(hierarchy_matches / max(1, len(query.split())) * 10, 1.0)

        # Abstract concept density
        abstract_concepts = len(keywords.concept_terms)
        total_keywords = len(keywords.get_all_keywords())
        concept_density = abstract_concepts / max(1, total_keywords)

        # Meta-cognitive indicators (thinking about thinking)
        meta_patterns = r"\b(understand|analyze|compare|evaluate|design|architect|pattern|approach)\b"
        meta_matches = len(re.findall(meta_patterns, query, re.IGNORECASE))
        meta_complexity = min(meta_matches / max(1, len(query.split())) * 5, 1.0)

        # Multi-level reasoning indicators
        reasoning_patterns = r"\b(because|therefore|however|although|while|whereas|given|assuming)\b"
        reasoning_matches = len(re.findall(reasoning_patterns, query, re.IGNORECASE))
        reasoning_complexity = min(reasoning_matches / max(1, len(query.split())) * 8, 1.0)

        return hierarchy_depth * 0.25 + concept_density * 0.25 + meta_complexity * 0.25 + reasoning_complexity * 0.25

    def _calculate_relationship_complexity(self, keywords: KeywordExtraction, query: str) -> float:
        """Calculate relationship complexity based on connection types and density."""
        import re

        # Direct relationship indicators
        relationship_count = len(keywords.relationship_indicators)
        total_keywords = len(keywords.get_all_keywords())
        relationship_density = relationship_count / max(1, total_keywords)

        # Multiple entity relationships
        entity_count = len(keywords.entity_names)
        if entity_count >= 2:
            # Potential relationships grow quadratically
            potential_relationships = entity_count * (entity_count - 1) / 2
            entity_relationship_complexity = min(potential_relationships / 10.0, 1.0)
        else:
            entity_relationship_complexity = 0.0

        # Temporal relationships
        temporal_patterns = r"\b(before|after|during|while|then|next|previous|following)\b"
        temporal_matches = len(re.findall(temporal_patterns, query, re.IGNORECASE))
        temporal_complexity = min(temporal_matches / max(1, len(query.split())) * 5, 1.0)

        # Causal relationships
        causal_patterns = r"\b(cause|effect|result|lead|trigger|because|due|since)\b"
        causal_matches = len(re.findall(causal_patterns, query, re.IGNORECASE))
        causal_complexity = min(causal_matches / max(1, len(query.split())) * 5, 1.0)

        return relationship_density * 0.4 + entity_relationship_complexity * 0.3 + temporal_complexity * 0.15 + causal_complexity * 0.15

    def _calculate_domain_specificity(self, keywords: KeywordExtraction, query: str) -> float:
        """Calculate domain specificity and technical depth."""
        # Technical term density
        technical_count = len(keywords.technical_terms)
        total_keywords = len(keywords.get_all_keywords())
        technical_density = technical_count / max(1, total_keywords)

        # Domain-specific vocabulary
        domain_terms = self._count_domain_terms(query)
        domain_density = domain_terms / max(1, len(query.split()))

        # Jargon and acronyms
        jargon_count = self._count_jargon_terms(query)
        jargon_density = jargon_count / max(1, len(query.split()))

        # Entity specificity (named entities vs generic terms)
        entity_count = len(keywords.entity_names)
        entity_specificity = min(entity_count / max(1, len(query.split())) * 2, 1.0)

        return technical_density * 0.3 + domain_density * 0.3 + jargon_density * 0.2 + entity_specificity * 0.2

    def _identify_semantic_fields(self, keywords: KeywordExtraction) -> list[str]:
        """Identify semantic fields represented in the keywords."""
        semantic_fields = set()

        # Programming and software development
        if any(
            term in " ".join(keywords.get_all_keywords()).lower()
            for term in ["code", "function", "class", "method", "variable", "programming", "software"]
        ):
            semantic_fields.add("programming")

        # System architecture
        if any(
            term in " ".join(keywords.get_all_keywords()).lower() for term in ["architecture", "system", "design", "pattern", "structure"]
        ):
            semantic_fields.add("architecture")

        # Data and databases
        if any(term in " ".join(keywords.get_all_keywords()).lower() for term in ["data", "database", "query", "table", "schema"]):
            semantic_fields.add("data")

        # Web development
        if any(
            term in " ".join(keywords.get_all_keywords()).lower() for term in ["web", "html", "css", "javascript", "frontend", "backend"]
        ):
            semantic_fields.add("web")

        # Machine learning/AI
        if any(
            term in " ".join(keywords.get_all_keywords()).lower() for term in ["machine", "learning", "model", "neural", "ai", "algorithm"]
        ):
            semantic_fields.add("ml_ai")

        return list(semantic_fields)

    def _count_domain_terms(self, query: str) -> int:
        """Count domain-specific technical terms."""
        domain_terms = {
            "api",
            "sdk",
            "framework",
            "library",
            "module",
            "package",
            "namespace",
            "interface",
            "protocol",
            "service",
            "microservice",
            "middleware",
            "database",
            "schema",
            "index",
            "query",
            "transaction",
            "migration",
            "authentication",
            "authorization",
            "security",
            "encryption",
            "hash",
            "deployment",
            "docker",
            "kubernetes",
            "cloud",
            "serverless",
            "performance",
            "optimization",
            "scalability",
            "concurrency",
            "parallelism",
        }

        query_words = set(query.lower().split())
        return len(query_words.intersection(domain_terms))

    def _count_jargon_terms(self, query: str) -> int:
        """Count jargon and acronym usage."""
        import re

        # Common tech acronyms
        acronyms = {
            "api",
            "sdk",
            "orm",
            "mvc",
            "mvp",
            "mvvm",
            "crud",
            "rest",
            "soap",
            "json",
            "xml",
            "html",
            "css",
            "sql",
            "nosql",
            "http",
            "https",
            "tcp",
            "udp",
            "dns",
            "cdn",
            "aws",
            "gcp",
            "azure",
            "ci",
            "cd",
        }

        # All caps words (potential acronyms)
        caps_words = set(re.findall(r"\b[A-Z]{2,}\b", query))

        query_words = set(query.lower().split())
        known_acronyms = query_words.intersection(acronyms)

        return len(known_acronyms) + len(caps_words)

    def _analyze_semantic_indicators(self, query: str, keywords: KeywordExtraction) -> dict[str, bool]:
        """Analyze semantic indicators in the query."""
        indicators = {
            "has_specific_entities": len(keywords.entity_names) > 0,
            "has_relationships": len(keywords.relationship_indicators) > 0,
            "has_patterns": any(term in keywords.concept_terms for term in ["pattern", "design", "architecture"]),
            "has_implementation_focus": any(word in query for word in ["implement", "code", "write", "create", "build"]),
            "has_conceptual_focus": len(keywords.concept_terms) > len(keywords.entity_names),
        }

        # Structural indicators
        indicators.update(
            {
                "mentions_functions": any(word in query for word in ["function", "method", "def", "()"]),
                "mentions_classes": any(word in query for word in ["class", "object", "instance"]),
                "mentions_files": any(word in query for word in ["file", "module", "script"]),
                "mentions_modules": any(word in query for word in ["module", "package", "library"]),
                "mentions_patterns": any(word in query for word in ["pattern", "template", "design"]),
            }
        )

        return indicators

    def _extract_context_hints(self, query: str) -> dict[str, list[str]]:
        """Extract language, framework, and domain hints from the query."""
        hints = {
            "language_hints": [],
            "framework_hints": [],
            "domain_hints": [],
        }

        query_lower = query.lower()

        # Language hints
        for lang in self.context_hints["languages"]:
            if lang in query_lower:
                hints["language_hints"].append(lang)

        # Framework hints
        for framework in self.context_hints["frameworks"]:
            if framework in query_lower:
                hints["framework_hints"].append(framework)

        # Domain hints
        for domain in self.context_hints["domains"]:
            if domain in query_lower:
                hints["domain_hints"].append(domain)

        return hints

    def _recommend_retrieval_mode(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        keywords: KeywordExtraction,
        semantic_indicators: dict[str, bool],
    ) -> tuple[str, float]:
        """Recommend the best retrieval mode based on analysis."""

        # Mode selection logic based on query characteristics

        # Local mode preference
        if (
            query_type == QueryType.ENTITY_FOCUSED
            and semantic_indicators["has_specific_entities"]
            and not semantic_indicators["has_relationships"]
        ):
            return "local", 0.9

        # Global mode preference
        if (
            query_type == QueryType.RELATIONSHIP_FOCUSED
            or query_type == QueryType.CONCEPTUAL
            or semantic_indicators["has_relationships"]
            and semantic_indicators["has_conceptual_focus"]
        ):
            return "global", 0.8

        # Mix mode for complex queries
        if complexity == QueryComplexity.MULTI_FACETED:
            return "mix", 0.7

        # Hybrid mode for balanced queries
        if (
            semantic_indicators["has_specific_entities"]
            and semantic_indicators["has_relationships"]
            or query_type == QueryType.IMPLEMENTATION_FOCUSED
            or complexity == QueryComplexity.COMPLEX
        ):
            return "hybrid", 0.6

        # Default to hybrid with moderate confidence
        return "hybrid", 0.5

    def _update_stats(self, features: QueryFeatures):
        """Update analysis statistics."""
        self.analysis_stats["queries_analyzed"] += 1

        # Update average analysis time
        current_avg = self.analysis_stats["average_analysis_time_ms"]
        current_count = self.analysis_stats["queries_analyzed"]
        new_avg = ((current_avg * (current_count - 1)) + features.processing_time_ms) / current_count
        self.analysis_stats["average_analysis_time_ms"] = new_avg

        # Update mode recommendation counts
        self.analysis_stats["mode_recommendations"][features.recommended_mode] += 1

    def get_mode_config(self, mode_name: str) -> RetrievalModeConfig:
        """Get configuration for a specific retrieval mode."""
        return self.mode_configs.get(mode_name, self.mode_configs["hybrid"])

    def adapt_mode_config(
        self,
        base_config: RetrievalModeConfig,
        query_features: QueryFeatures,
    ) -> RetrievalModeConfig:
        """Adapt mode configuration based on query features."""
        adapted_config = base_config

        # Adjust based on query complexity
        if query_features.complexity == QueryComplexity.COMPLEX:
            adapted_config.max_results = min(adapted_config.max_results + 10, 50)
            adapted_config.expansion_depth = min(adapted_config.expansion_depth + 1, 4)
        elif query_features.complexity == QueryComplexity.SIMPLE:
            adapted_config.max_results = max(adapted_config.max_results - 5, 5)
            adapted_config.expansion_depth = max(adapted_config.expansion_depth - 1, 1)

        # Adjust based on entity focus
        if query_features.has_specific_entities:
            adapted_config.token_allocation["entity_tokens"] = min(adapted_config.token_allocation["entity_tokens"] + 0.1, 0.8)
            adapted_config.token_allocation["relationship_tokens"] = max(adapted_config.token_allocation["relationship_tokens"] - 0.05, 0.1)

        # Adjust based on relationship focus
        if query_features.has_relationships:
            adapted_config.token_allocation["relationship_tokens"] = min(adapted_config.token_allocation["relationship_tokens"] + 0.1, 0.8)
            adapted_config.token_allocation["entity_tokens"] = max(adapted_config.token_allocation["entity_tokens"] - 0.05, 0.1)

        return adapted_config

    def get_analysis_stats(self) -> dict[str, any]:
        """Get current analysis statistics."""
        return self.analysis_stats.copy()


# Factory function
_query_analyzer_instance = None


async def get_query_analyzer() -> QueryAnalyzer:
    """Get or create a QueryAnalyzer instance."""
    global _query_analyzer_instance
    if _query_analyzer_instance is None:
        _query_analyzer_instance = QueryAnalyzer()
    return _query_analyzer_instance
