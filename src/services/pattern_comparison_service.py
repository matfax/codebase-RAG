"""
Pattern Comparison Service for Graph RAG enhancement.

This service provides comprehensive architectural pattern comparison and analysis
capabilities, enabling developers to compare patterns across projects, identify
best practices, and generate insights for architectural improvements.

Built on top of Wave 3's pattern recognition and implementation chain services.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .implementation_chain_service import ChainType, ImplementationChain, ImplementationChainService
from .pattern_recognition_service import PatternMatch, PatternRecognitionService, PatternType


class ComparisonType(Enum):
    """Types of pattern comparisons that can be performed."""

    CROSS_PROJECT = "cross_project"  # Compare patterns across different projects
    EVOLUTION_ANALYSIS = "evolution_analysis"  # Analyze pattern evolution over time
    QUALITY_BENCHMARKING = "quality_benchmarking"  # Benchmark pattern quality against standards
    IMPLEMENTATION_VARIANCE = "implementation_variance"  # Analyze variance in pattern implementations
    ARCHITECTURAL_ALIGNMENT = "architectural_alignment"  # Check alignment with architectural principles
    BEST_PRACTICE_ANALYSIS = "best_practice_analysis"  # Identify best practice implementations
    COMPLEXITY_ANALYSIS = "complexity_analysis"  # Analyze pattern complexity across implementations
    CONSISTENCY_ANALYSIS = "consistency_analysis"  # Analyze pattern consistency within projects


@dataclass
class PatternComparisonMetric:
    """Metric for comparing patterns."""

    metric_name: str
    metric_type: str  # "quality", "complexity", "consistency", "completeness"
    value: float  # Normalized value (0.0-1.0)
    raw_value: float  # Original raw value
    description: str

    # Contextual information
    measurement_source: str = ""  # How this metric was measured
    confidence: float = 1.0  # Confidence in the measurement
    benchmark_value: float | None = None  # Benchmark value for comparison


@dataclass
class PatternComparisonResult:
    """Result of comparing two patterns."""

    pattern1: PatternMatch
    pattern2: PatternMatch
    comparison_type: ComparisonType
    overall_similarity: float  # Overall similarity score (0.0-1.0)

    # Detailed comparison metrics
    structural_similarity: float
    quality_similarity: float
    complexity_similarity: float
    implementation_similarity: float

    # Metric comparisons
    metrics_comparison: dict[str, tuple[float, float]] = None  # metric_name -> (value1, value2)

    # Insights and recommendations
    strengths_pattern1: list[str] = None
    strengths_pattern2: list[str] = None
    improvement_suggestions: list[str] = None

    # Supporting evidence
    similar_components: list[str] = None
    different_aspects: list[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.metrics_comparison is None:
            self.metrics_comparison = {}
        if self.strengths_pattern1 is None:
            self.strengths_pattern1 = []
        if self.strengths_pattern2 is None:
            self.strengths_pattern2 = []
        if self.improvement_suggestions is None:
            self.improvement_suggestions = []
        if self.similar_components is None:
            self.similar_components = []
        if self.different_aspects is None:
            self.different_aspects = []


@dataclass
class ArchitecturalAnalysisResult:
    """Result of comprehensive architectural analysis."""

    analysis_scope: str
    projects_analyzed: list[str]
    patterns_analyzed: int
    analysis_time_ms: float

    # Pattern distribution analysis
    pattern_frequency: dict[PatternType, int] = None
    pattern_quality_distribution: dict[str, int] = None  # "high", "medium", "low"
    pattern_complexity_trends: dict[str, float] = None

    # Cross-project insights
    common_patterns: list[PatternType] = None  # Patterns found across multiple projects
    unique_patterns: dict[str, list[PatternType]] = None  # Project-specific patterns
    pattern_variations: dict[PatternType, int] = None  # Number of variations per pattern type

    # Quality insights
    best_implementations: dict[PatternType, PatternMatch] = None  # Best implementation per pattern
    worst_implementations: dict[PatternType, PatternMatch] = None  # Worst implementation per pattern
    quality_benchmarks: dict[PatternType, float] = None  # Quality benchmarks per pattern

    # Recommendations
    architecture_recommendations: list[str] = None
    standardization_opportunities: list[str] = None
    improvement_priorities: list[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.pattern_frequency is None:
            self.pattern_frequency = {}
        if self.pattern_quality_distribution is None:
            self.pattern_quality_distribution = {}
        if self.pattern_complexity_trends is None:
            self.pattern_complexity_trends = {}
        if self.common_patterns is None:
            self.common_patterns = []
        if self.unique_patterns is None:
            self.unique_patterns = {}
        if self.pattern_variations is None:
            self.pattern_variations = {}
        if self.best_implementations is None:
            self.best_implementations = {}
        if self.worst_implementations is None:
            self.worst_implementations = {}
        if self.quality_benchmarks is None:
            self.quality_benchmarks = {}
        if self.architecture_recommendations is None:
            self.architecture_recommendations = []
        if self.standardization_opportunities is None:
            self.standardization_opportunities = []
        if self.improvement_priorities is None:
            self.improvement_priorities = []


@dataclass
class PatternEvolutionAnalysis:
    """Analysis of how patterns evolve or change."""

    pattern_type: PatternType
    baseline_implementation: PatternMatch  # Reference implementation
    compared_implementations: list[PatternMatch]

    # Evolution metrics
    complexity_trend: str  # "increasing", "decreasing", "stable"
    quality_trend: str  # "improving", "declining", "stable"
    consistency_trend: str  # "more_consistent", "less_consistent", "stable"

    # Evolutionary insights
    common_improvements: list[str] = None
    common_degradations: list[str] = None
    emerging_practices: list[str] = None
    deprecated_practices: list[str] = None

    # Recommendations based on evolution
    evolution_recommendations: list[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.common_improvements is None:
            self.common_improvements = []
        if self.common_degradations is None:
            self.common_degradations = []
        if self.emerging_practices is None:
            self.emerging_practices = []
        if self.deprecated_practices is None:
            self.deprecated_practices = []
        if self.evolution_recommendations is None:
            self.evolution_recommendations = []


class PatternComparisonService:
    """
    Service for comparing and analyzing architectural patterns across codebases.

    This service provides comprehensive pattern analysis capabilities including
    cross-project comparisons, quality benchmarking, and architectural insights.
    """

    def __init__(
        self,
        pattern_recognition_service: PatternRecognitionService,
        implementation_chain_service: ImplementationChainService,
    ):
        """Initialize the pattern comparison service.

        Args:
            pattern_recognition_service: Service for pattern recognition
            implementation_chain_service: Service for implementation chain tracking
        """
        self.pattern_recognition_service = pattern_recognition_service
        self.implementation_chain_service = implementation_chain_service
        self.logger = logging.getLogger(__name__)

        # Initialize quality benchmarks and standards
        self.quality_benchmarks = self._initialize_quality_benchmarks()
        self.architectural_principles = self._initialize_architectural_principles()

    async def compare_patterns(
        self,
        pattern1: PatternMatch,
        pattern2: PatternMatch,
        comparison_type: ComparisonType = ComparisonType.CROSS_PROJECT,
    ) -> PatternComparisonResult:
        """
        Compare two pattern implementations.

        Args:
            pattern1: First pattern to compare
            pattern2: Second pattern to compare
            comparison_type: Type of comparison to perform

        Returns:
            PatternComparisonResult with detailed comparison
        """
        try:
            self.logger.info(f"Comparing patterns: {pattern1.pattern_type.value} vs {pattern2.pattern_type.value}")

            # Calculate similarity metrics
            structural_similarity = self._calculate_structural_similarity(pattern1, pattern2)
            quality_similarity = self._calculate_quality_similarity(pattern1, pattern2)
            complexity_similarity = self._calculate_complexity_similarity(pattern1, pattern2)
            implementation_similarity = self._calculate_implementation_similarity(pattern1, pattern2)

            # Calculate overall similarity
            overall_similarity = (
                structural_similarity * 0.3 + quality_similarity * 0.3 + complexity_similarity * 0.2 + implementation_similarity * 0.2
            )

            # Extract detailed metrics
            metrics_comparison = await self._extract_detailed_metrics(pattern1, pattern2)

            # Generate insights and recommendations
            strengths1, strengths2 = self._identify_pattern_strengths(pattern1, pattern2)
            improvements = self._generate_improvement_suggestions(pattern1, pattern2, comparison_type)

            # Find similarities and differences
            similar_components = self._find_similar_components(pattern1, pattern2)
            different_aspects = self._identify_different_aspects(pattern1, pattern2)

            result = PatternComparisonResult(
                pattern1=pattern1,
                pattern2=pattern2,
                comparison_type=comparison_type,
                overall_similarity=overall_similarity,
                structural_similarity=structural_similarity,
                quality_similarity=quality_similarity,
                complexity_similarity=complexity_similarity,
                implementation_similarity=implementation_similarity,
                metrics_comparison=metrics_comparison,
                strengths_pattern1=strengths1,
                strengths_pattern2=strengths2,
                improvement_suggestions=improvements,
                similar_components=similar_components,
                different_aspects=different_aspects,
            )

            self.logger.info(f"Pattern comparison completed. Overall similarity: {overall_similarity:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Error comparing patterns: {e}")
            return PatternComparisonResult(
                pattern1=pattern1,
                pattern2=pattern2,
                comparison_type=comparison_type,
                overall_similarity=0.0,
                structural_similarity=0.0,
                quality_similarity=0.0,
                complexity_similarity=0.0,
                implementation_similarity=0.0,
            )

    async def analyze_architectural_patterns(
        self,
        projects: list[str],
        pattern_types: list[PatternType] | None = None,
        min_quality_threshold: float = 0.5,
    ) -> ArchitecturalAnalysisResult:
        """
        Perform comprehensive architectural pattern analysis across projects.

        Args:
            projects: List of project names to analyze
            pattern_types: Specific pattern types to analyze (all if None)
            min_quality_threshold: Minimum quality threshold for analysis

        Returns:
            ArchitecturalAnalysisResult with comprehensive analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting architectural analysis across {len(projects)} projects")

            # Collect patterns from all projects
            all_patterns = []
            for project_name in projects:
                try:
                    project_analysis = await self.pattern_recognition_service.analyze_project_patterns(
                        project_name, min_confidence=min_quality_threshold
                    )

                    # Filter by pattern types if specified
                    project_patterns = project_analysis.patterns_found
                    if pattern_types:
                        project_patterns = [p for p in project_patterns if p.pattern_type in pattern_types]

                    all_patterns.extend(project_patterns)

                except Exception as e:
                    self.logger.error(f"Error analyzing project {project_name}: {e}")
                    continue

            if not all_patterns:
                self.logger.warning("No patterns found for architectural analysis")
                return ArchitecturalAnalysisResult(
                    analysis_scope="cross-project",
                    projects_analyzed=projects,
                    patterns_analyzed=0,
                    analysis_time_ms=(time.time() - start_time) * 1000,
                )

            # Analyze pattern distribution
            pattern_frequency = self._analyze_pattern_frequency(all_patterns)
            quality_distribution = self._analyze_quality_distribution(all_patterns)
            complexity_trends = self._analyze_complexity_trends(all_patterns)

            # Cross-project insights
            common_patterns = self._identify_common_patterns(all_patterns, projects)
            unique_patterns = self._identify_unique_patterns(all_patterns, projects)
            pattern_variations = self._analyze_pattern_variations(all_patterns)

            # Quality insights
            best_implementations = self._identify_best_implementations(all_patterns)
            worst_implementations = self._identify_worst_implementations(all_patterns)
            quality_benchmarks = self._calculate_quality_benchmarks(all_patterns)

            # Generate recommendations
            arch_recommendations = self._generate_architectural_recommendations(all_patterns, projects)
            standardization_opportunities = self._identify_standardization_opportunities(all_patterns, projects)
            improvement_priorities = self._prioritize_improvements(all_patterns)

            analysis_time_ms = (time.time() - start_time) * 1000

            result = ArchitecturalAnalysisResult(
                analysis_scope="cross-project",
                projects_analyzed=projects,
                patterns_analyzed=len(all_patterns),
                analysis_time_ms=analysis_time_ms,
                pattern_frequency=pattern_frequency,
                pattern_quality_distribution=quality_distribution,
                pattern_complexity_trends=complexity_trends,
                common_patterns=common_patterns,
                unique_patterns=unique_patterns,
                pattern_variations=pattern_variations,
                best_implementations=best_implementations,
                worst_implementations=worst_implementations,
                quality_benchmarks=quality_benchmarks,
                architecture_recommendations=arch_recommendations,
                standardization_opportunities=standardization_opportunities,
                improvement_priorities=improvement_priorities,
            )

            self.logger.info(
                f"Architectural analysis completed in {analysis_time_ms:.2f}ms. "
                f"Analyzed {len(all_patterns)} patterns across {len(projects)} projects."
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in architectural pattern analysis: {e}")
            analysis_time_ms = (time.time() - start_time) * 1000
            return ArchitecturalAnalysisResult(
                analysis_scope="cross-project",
                projects_analyzed=projects,
                patterns_analyzed=0,
                analysis_time_ms=analysis_time_ms,
            )

    async def analyze_pattern_evolution(
        self,
        pattern_type: PatternType,
        baseline_project: str,
        comparison_projects: list[str],
    ) -> PatternEvolutionAnalysis:
        """
        Analyze how a pattern type evolves across different implementations.

        Args:
            pattern_type: Type of pattern to analyze evolution for
            baseline_project: Project to use as baseline
            comparison_projects: Projects to compare against baseline

        Returns:
            PatternEvolutionAnalysis with evolution insights
        """
        try:
            self.logger.info(f"Analyzing evolution of {pattern_type.value} pattern")

            # Get baseline implementation
            baseline_analysis = await self.pattern_recognition_service.analyze_project_patterns(baseline_project)
            baseline_patterns = [p for p in baseline_analysis.patterns_found if p.pattern_type == pattern_type]

            if not baseline_patterns:
                self.logger.warning(f"No {pattern_type.value} patterns found in baseline project: {baseline_project}")
                return PatternEvolutionAnalysis(
                    pattern_type=pattern_type,
                    baseline_implementation=None,
                    compared_implementations=[],
                    complexity_trend="unknown",
                    quality_trend="unknown",
                    consistency_trend="unknown",
                )

            baseline_implementation = max(baseline_patterns, key=lambda p: p.pattern_quality)

            # Get comparison implementations
            comparison_implementations = []
            for project_name in comparison_projects:
                try:
                    project_analysis = await self.pattern_recognition_service.analyze_project_patterns(project_name)
                    project_patterns = [p for p in project_analysis.patterns_found if p.pattern_type == pattern_type]

                    if project_patterns:
                        best_pattern = max(project_patterns, key=lambda p: p.pattern_quality)
                        comparison_implementations.append(best_pattern)

                except Exception as e:
                    self.logger.error(f"Error analyzing {project_name} for evolution: {e}")
                    continue

            # Analyze trends
            complexity_trend = self._analyze_complexity_trend(baseline_implementation, comparison_implementations)
            quality_trend = self._analyze_quality_trend(baseline_implementation, comparison_implementations)
            consistency_trend = self._analyze_consistency_trend(baseline_implementation, comparison_implementations)

            # Extract evolutionary insights
            common_improvements = self._identify_common_improvements(baseline_implementation, comparison_implementations)
            common_degradations = self._identify_common_degradations(baseline_implementation, comparison_implementations)
            emerging_practices = self._identify_emerging_practices(comparison_implementations)
            deprecated_practices = self._identify_deprecated_practices(baseline_implementation, comparison_implementations)

            # Generate evolution-based recommendations
            evolution_recommendations = self._generate_evolution_recommendations(
                baseline_implementation, comparison_implementations, complexity_trend, quality_trend
            )

            result = PatternEvolutionAnalysis(
                pattern_type=pattern_type,
                baseline_implementation=baseline_implementation,
                compared_implementations=comparison_implementations,
                complexity_trend=complexity_trend,
                quality_trend=quality_trend,
                consistency_trend=consistency_trend,
                common_improvements=common_improvements,
                common_degradations=common_degradations,
                emerging_practices=emerging_practices,
                deprecated_practices=deprecated_practices,
                evolution_recommendations=evolution_recommendations,
            )

            self.logger.info(f"Pattern evolution analysis completed for {pattern_type.value}")

            return result

        except Exception as e:
            self.logger.error(f"Error in pattern evolution analysis: {e}")
            return PatternEvolutionAnalysis(
                pattern_type=pattern_type,
                baseline_implementation=None,
                compared_implementations=[],
                complexity_trend="error",
                quality_trend="error",
                consistency_trend="error",
            )

    async def benchmark_pattern_quality(
        self,
        pattern: PatternMatch,
        reference_projects: list[str],
    ) -> dict[str, Any]:
        """
        Benchmark a pattern against quality standards and reference implementations.

        Args:
            pattern: Pattern to benchmark
            reference_projects: Projects to use for reference comparisons

        Returns:
            Dictionary with benchmarking results and recommendations
        """
        try:
            self.logger.info(f"Benchmarking {pattern.pattern_type.value} pattern quality")

            benchmark_result = {
                "pattern_id": f"{pattern.project_name}_{pattern.pattern_type.value}",
                "overall_score": pattern.pattern_quality,
                "benchmark_scores": {},
                "reference_comparisons": [],
                "quality_ranking": "unknown",
                "improvement_areas": [],
                "strengths": [],
            }

            # Benchmark against standards
            benchmark_scores = {}
            quality_benchmark = self.quality_benchmarks.get(pattern.pattern_type, 0.7)

            benchmark_scores["quality_vs_standard"] = pattern.pattern_quality / quality_benchmark
            benchmark_scores["completeness_score"] = pattern.pattern_completeness
            benchmark_scores["complexity_score"] = 1.0 - pattern.pattern_complexity  # Lower complexity is better

            # Compare against reference implementations
            reference_comparisons = []
            for project_name in reference_projects:
                try:
                    project_analysis = await self.pattern_recognition_service.analyze_project_patterns(project_name)
                    reference_patterns = [p for p in project_analysis.patterns_found if p.pattern_type == pattern.pattern_type]

                    for ref_pattern in reference_patterns:
                        comparison = await self.compare_patterns(pattern, ref_pattern, ComparisonType.QUALITY_BENCHMARKING)
                        reference_comparisons.append(
                            {
                                "reference_project": project_name,
                                "similarity": comparison.overall_similarity,
                                "quality_comparison": ref_pattern.pattern_quality - pattern.pattern_quality,
                                "strengths": comparison.strengths_pattern2,
                                "improvements": comparison.improvement_suggestions,
                            }
                        )

                except Exception as e:
                    self.logger.error(f"Error comparing with reference project {project_name}: {e}")
                    continue

            # Determine quality ranking
            if reference_comparisons:
                avg_quality_comparison = sum(rc["quality_comparison"] for rc in reference_comparisons) / len(reference_comparisons)

                if avg_quality_comparison <= -0.1:
                    quality_ranking = "excellent"
                elif avg_quality_comparison <= 0.0:
                    quality_ranking = "good"
                elif avg_quality_comparison <= 0.1:
                    quality_ranking = "average"
                else:
                    quality_ranking = "below_average"
            else:
                # Fallback to absolute quality score
                if pattern.pattern_quality >= 0.8:
                    quality_ranking = "excellent"
                elif pattern.pattern_quality >= 0.7:
                    quality_ranking = "good"
                elif pattern.pattern_quality >= 0.5:
                    quality_ranking = "average"
                else:
                    quality_ranking = "below_average"

            # Identify improvement areas and strengths
            improvement_areas = []
            strengths = []

            if pattern.pattern_completeness < 0.8:
                improvement_areas.append("Pattern completeness - consider implementing missing components")
            if pattern.pattern_complexity > 0.7:
                improvement_areas.append("Pattern complexity - consider simplifying the implementation")
            if pattern.confidence < 0.8:
                improvement_areas.append("Pattern confidence - strengthen pattern adherence")

            if pattern.pattern_quality >= 0.8:
                strengths.append("High overall pattern quality")
            if pattern.pattern_completeness >= 0.8:
                strengths.append("Complete pattern implementation")
            if pattern.pattern_complexity <= 0.3:
                strengths.append("Simple and maintainable implementation")

            # Update result
            benchmark_result.update(
                {
                    "benchmark_scores": benchmark_scores,
                    "reference_comparisons": reference_comparisons,
                    "quality_ranking": quality_ranking,
                    "improvement_areas": improvement_areas,
                    "strengths": strengths,
                }
            )

            self.logger.info(f"Pattern benchmarking completed. Quality ranking: {quality_ranking}")

            return benchmark_result

        except Exception as e:
            self.logger.error(f"Error benchmarking pattern quality: {e}")
            return {
                "pattern_id": f"{pattern.project_name}_{pattern.pattern_type.value}",
                "overall_score": pattern.pattern_quality,
                "quality_ranking": "error",
                "error": str(e),
            }

    def _calculate_structural_similarity(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """Calculate structural similarity between patterns."""
        try:
            if pattern1.pattern_type != pattern2.pattern_type:
                return 0.0

            similarity_factors = []

            # Component count similarity
            count1 = len(pattern1.components)
            count2 = len(pattern2.components)
            count_similarity = 1.0 - abs(count1 - count2) / max(count1, count2, 1)
            similarity_factors.append(count_similarity)

            # Structural evidence similarity
            evidence1 = set(pattern1.structural_evidence.keys())
            evidence2 = set(pattern2.structural_evidence.keys())
            evidence_overlap = len(evidence1 & evidence2) / len(evidence1 | evidence2) if (evidence1 | evidence2) else 0.0
            similarity_factors.append(evidence_overlap)

            # Breadcrumb scope similarity
            scope1_parts = pattern1.breadcrumb_scope.split(".") if pattern1.breadcrumb_scope else []
            scope2_parts = pattern2.breadcrumb_scope.split(".") if pattern2.breadcrumb_scope else []

            common_scope_parts = 0
            for i in range(min(len(scope1_parts), len(scope2_parts))):
                if scope1_parts[i] == scope2_parts[i]:
                    common_scope_parts += 1
                else:
                    break

            scope_similarity = common_scope_parts / max(len(scope1_parts), len(scope2_parts), 1)
            similarity_factors.append(scope_similarity)

            return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating structural similarity: {e}")
            return 0.0

    def _calculate_quality_similarity(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """Calculate quality similarity between patterns."""
        try:
            # Quality score similarity
            quality_diff = abs(pattern1.pattern_quality - pattern2.pattern_quality)
            quality_similarity = 1.0 - quality_diff

            # Completeness similarity
            completeness_diff = abs(pattern1.pattern_completeness - pattern2.pattern_completeness)
            completeness_similarity = 1.0 - completeness_diff

            # Confidence similarity
            confidence_diff = abs(pattern1.confidence - pattern2.confidence)
            confidence_similarity = 1.0 - confidence_diff

            return quality_similarity * 0.5 + completeness_similarity * 0.3 + confidence_similarity * 0.2

        except Exception as e:
            self.logger.error(f"Error calculating quality similarity: {e}")
            return 0.0

    def _calculate_complexity_similarity(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """Calculate complexity similarity between patterns."""
        try:
            complexity_diff = abs(pattern1.pattern_complexity - pattern2.pattern_complexity)
            return 1.0 - complexity_diff

        except Exception as e:
            self.logger.error(f"Error calculating complexity similarity: {e}")
            return 0.0

    def _calculate_implementation_similarity(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """Calculate implementation similarity between patterns."""
        try:
            similarity_factors = []

            # Naming evidence similarity
            naming1 = set(pattern1.naming_evidence)
            naming2 = set(pattern2.naming_evidence)
            naming_overlap = len(naming1 & naming2) / len(naming1 | naming2) if (naming1 | naming2) else 0.0
            similarity_factors.append(naming_overlap)

            # Behavioral evidence similarity
            behavioral1 = set(pattern1.behavioral_evidence)
            behavioral2 = set(pattern2.behavioral_evidence)
            behavioral_overlap = len(behavioral1 & behavioral2) / len(behavioral1 | behavioral2) if (behavioral1 | behavioral2) else 0.0
            similarity_factors.append(behavioral_overlap)

            # File path similarity (same type of files)
            files1 = set(pattern1.file_paths)
            files2 = set(pattern2.file_paths)

            # Compare file extensions
            ext1 = {f.split(".")[-1] for f in files1 if "." in f}
            ext2 = {f.split(".")[-1] for f in files2 if "." in f}
            ext_overlap = len(ext1 & ext2) / len(ext1 | ext2) if (ext1 | ext2) else 0.0
            similarity_factors.append(ext_overlap)

            return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating implementation similarity: {e}")
            return 0.0

    async def _extract_detailed_metrics(self, pattern1: PatternMatch, pattern2: PatternMatch) -> dict[str, tuple[float, float]]:
        """Extract detailed metrics for comparison."""
        metrics = {}

        try:
            metrics["quality"] = (pattern1.pattern_quality, pattern2.pattern_quality)
            metrics["completeness"] = (pattern1.pattern_completeness, pattern2.pattern_completeness)
            metrics["complexity"] = (pattern1.pattern_complexity, pattern2.pattern_complexity)
            metrics["confidence"] = (pattern1.confidence, pattern2.confidence)
            metrics["component_count"] = (len(pattern1.components), len(pattern2.components))
            metrics["file_count"] = (len(pattern1.file_paths), len(pattern2.file_paths))

            return metrics

        except Exception as e:
            self.logger.error(f"Error extracting detailed metrics: {e}")
            return {}

    def _identify_pattern_strengths(self, pattern1: PatternMatch, pattern2: PatternMatch) -> tuple[list[str], list[str]]:
        """Identify strengths of each pattern."""
        strengths1 = []
        strengths2 = []

        try:
            # Compare quality aspects
            if pattern1.pattern_quality > pattern2.pattern_quality + 0.1:
                strengths1.append("Higher overall pattern quality")
            elif pattern2.pattern_quality > pattern1.pattern_quality + 0.1:
                strengths2.append("Higher overall pattern quality")

            if pattern1.pattern_completeness > pattern2.pattern_completeness + 0.1:
                strengths1.append("More complete pattern implementation")
            elif pattern2.pattern_completeness > pattern1.pattern_completeness + 0.1:
                strengths2.append("More complete pattern implementation")

            if pattern1.pattern_complexity < pattern2.pattern_complexity - 0.1:
                strengths1.append("Simpler and more maintainable implementation")
            elif pattern2.pattern_complexity < pattern1.pattern_complexity - 0.1:
                strengths2.append("Simpler and more maintainable implementation")

            # Compare evidence quality
            if len(pattern1.naming_evidence) > len(pattern2.naming_evidence):
                strengths1.append("Better naming convention adherence")
            elif len(pattern2.naming_evidence) > len(pattern1.naming_evidence):
                strengths2.append("Better naming convention adherence")

            if len(pattern1.behavioral_evidence) > len(pattern2.behavioral_evidence):
                strengths1.append("Stronger behavioral pattern adherence")
            elif len(pattern2.behavioral_evidence) > len(pattern1.behavioral_evidence):
                strengths2.append("Stronger behavioral pattern adherence")

            return strengths1, strengths2

        except Exception as e:
            self.logger.error(f"Error identifying pattern strengths: {e}")
            return [], []

    def _generate_improvement_suggestions(
        self,
        pattern1: PatternMatch,
        pattern2: PatternMatch,
        comparison_type: ComparisonType,
    ) -> list[str]:
        """Generate improvement suggestions based on comparison."""
        suggestions = []

        try:
            # Quality-based suggestions
            if pattern1.pattern_quality < pattern2.pattern_quality:
                suggestions.append(f"Consider adopting quality practices from {pattern2.project_name}")

            if pattern1.pattern_completeness < pattern2.pattern_completeness:
                suggestions.append("Implement missing pattern components to improve completeness")

            if pattern1.pattern_complexity > pattern2.pattern_complexity + 0.2:
                suggestions.append("Consider simplifying the implementation to reduce complexity")

            # Evidence-based suggestions
            missing_naming = set(pattern2.naming_evidence) - set(pattern1.naming_evidence)
            if missing_naming:
                suggestions.append(f"Consider adopting naming conventions: {', '.join(list(missing_naming)[:3])}")

            missing_behavioral = set(pattern2.behavioral_evidence) - set(pattern1.behavioral_evidence)
            if missing_behavioral:
                suggestions.append(f"Consider implementing behavioral aspects: {', '.join(list(missing_behavioral)[:3])}")

            # Structural suggestions
            if len(pattern1.components) < len(pattern2.components):
                suggestions.append("Consider adding additional components to strengthen the pattern")

            # Project-specific suggestions
            if comparison_type == ComparisonType.CROSS_PROJECT:
                suggestions.append(f"Study implementation in {pattern2.project_name} for alternative approaches")

            return suggestions[:5]  # Limit to top 5 suggestions

        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {e}")
            return ["Unable to generate specific suggestions due to analysis error"]

    def _find_similar_components(self, pattern1: PatternMatch, pattern2: PatternMatch) -> list[str]:
        """Find similar components between patterns."""
        try:
            similar_components = []

            # Compare component names
            names1 = {comp.name for comp in pattern1.components if comp.name}
            names2 = {comp.name for comp in pattern2.components if comp.name}
            common_names = names1 & names2
            similar_components.extend(f"Common component: {name}" for name in common_names)

            # Compare component types
            types1 = {comp.chunk_type for comp in pattern1.components}
            types2 = {comp.chunk_type for comp in pattern2.components}
            common_types = types1 & types2
            similar_components.extend(f"Common type: {ctype.value}" for ctype in common_types)

            # Compare naming evidence
            common_naming = set(pattern1.naming_evidence) & set(pattern2.naming_evidence)
            similar_components.extend(f"Common naming: {naming}" for naming in common_naming)

            return similar_components[:10]  # Limit to top 10

        except Exception as e:
            self.logger.error(f"Error finding similar components: {e}")
            return []

    def _identify_different_aspects(self, pattern1: PatternMatch, pattern2: PatternMatch) -> list[str]:
        """Identify different aspects between patterns."""
        try:
            differences = []

            # Quality differences
            quality_diff = abs(pattern1.pattern_quality - pattern2.pattern_quality)
            if quality_diff > 0.1:
                differences.append(f"Quality difference: {quality_diff:.2f}")

            # Complexity differences
            complexity_diff = abs(pattern1.pattern_complexity - pattern2.pattern_complexity)
            if complexity_diff > 0.1:
                differences.append(f"Complexity difference: {complexity_diff:.2f}")

            # Component count differences
            count_diff = abs(len(pattern1.components) - len(pattern2.components))
            if count_diff > 0:
                differences.append(f"Component count difference: {count_diff}")

            # Different naming evidence
            naming_diff = set(pattern1.naming_evidence) ^ set(pattern2.naming_evidence)  # Symmetric difference
            if naming_diff:
                differences.append(f"Different naming approaches: {len(naming_diff)} unique aspects")

            # Different projects
            if pattern1.project_name != pattern2.project_name:
                differences.append(f"Different projects: {pattern1.project_name} vs {pattern2.project_name}")

            return differences[:8]  # Limit to top 8 differences

        except Exception as e:
            self.logger.error(f"Error identifying different aspects: {e}")
            return []

    def _analyze_pattern_frequency(self, patterns: list[PatternMatch]) -> dict[PatternType, int]:
        """Analyze frequency of pattern types."""
        frequency = defaultdict(int)
        for pattern in patterns:
            frequency[pattern.pattern_type] += 1
        return dict(frequency)

    def _analyze_quality_distribution(self, patterns: list[PatternMatch]) -> dict[str, int]:
        """Analyze quality distribution of patterns."""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for pattern in patterns:
            if pattern.pattern_quality >= 0.8:
                distribution["high"] += 1
            elif pattern.pattern_quality >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _analyze_complexity_trends(self, patterns: list[PatternMatch]) -> dict[str, float]:
        """Analyze complexity trends across patterns."""
        if not patterns:
            return {}

        # Group by pattern type and calculate average complexity
        complexity_by_type = defaultdict(list)
        for pattern in patterns:
            complexity_by_type[pattern.pattern_type].append(pattern.pattern_complexity)

        trends = {}
        for pattern_type, complexities in complexity_by_type.items():
            avg_complexity = sum(complexities) / len(complexities)
            trends[pattern_type.value] = avg_complexity

        return trends

    def _identify_common_patterns(self, patterns: list[PatternMatch], projects: list[str]) -> list[PatternType]:
        """Identify patterns that appear across multiple projects."""
        pattern_projects = defaultdict(set)

        for pattern in patterns:
            pattern_projects[pattern.pattern_type].add(pattern.project_name)

        # Find patterns that appear in multiple projects
        common_patterns = []
        for pattern_type, project_set in pattern_projects.items():
            if len(project_set) >= min(3, len(projects) // 2):  # At least 3 projects or half of all projects
                common_patterns.append(pattern_type)

        return common_patterns

    def _identify_unique_patterns(self, patterns: list[PatternMatch], projects: list[str]) -> dict[str, list[PatternType]]:
        """Identify patterns unique to specific projects."""
        pattern_projects = defaultdict(set)

        for pattern in patterns:
            pattern_projects[pattern.pattern_type].add(pattern.project_name)

        unique_patterns = defaultdict(list)
        for pattern_type, project_set in pattern_projects.items():
            if len(project_set) == 1:  # Pattern appears in only one project
                project_name = next(iter(project_set))
                unique_patterns[project_name].append(pattern_type)

        return dict(unique_patterns)

    def _analyze_pattern_variations(self, patterns: list[PatternMatch]) -> dict[PatternType, int]:
        """Analyze number of variations per pattern type."""
        variations = defaultdict(int)

        # Group patterns by type and project
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)

        for pattern_type, type_patterns in pattern_groups.items():
            # Count unique implementations (different projects or significantly different quality)
            unique_implementations = set()
            for pattern in type_patterns:
                # Use project name and quality tier as variation identifier
                quality_tier = "high" if pattern.pattern_quality >= 0.8 else "medium" if pattern.pattern_quality >= 0.6 else "low"
                unique_implementations.add(f"{pattern.project_name}_{quality_tier}")

            variations[pattern_type] = len(unique_implementations)

        return dict(variations)

    def _identify_best_implementations(self, patterns: list[PatternMatch]) -> dict[PatternType, PatternMatch]:
        """Identify best implementation for each pattern type."""
        best_implementations = {}

        # Group by pattern type
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)

        # Find best implementation for each type
        for pattern_type, type_patterns in pattern_groups.items():
            best_pattern = max(type_patterns, key=lambda p: p.pattern_quality)
            best_implementations[pattern_type] = best_pattern

        return best_implementations

    def _identify_worst_implementations(self, patterns: list[PatternMatch]) -> dict[PatternType, PatternMatch]:
        """Identify worst implementation for each pattern type."""
        worst_implementations = {}

        # Group by pattern type
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)

        # Find worst implementation for each type
        for pattern_type, type_patterns in pattern_groups.items():
            worst_pattern = min(type_patterns, key=lambda p: p.pattern_quality)
            worst_implementations[pattern_type] = worst_pattern

        return worst_implementations

    def _calculate_quality_benchmarks(self, patterns: list[PatternMatch]) -> dict[PatternType, float]:
        """Calculate quality benchmarks for each pattern type."""
        benchmarks = {}

        # Group by pattern type
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)

        # Calculate benchmark (75th percentile) for each type
        for pattern_type, type_patterns in pattern_groups.items():
            qualities = [p.pattern_quality for p in type_patterns]
            qualities.sort()

            if qualities:
                # Use 75th percentile as benchmark
                percentile_75_index = int(len(qualities) * 0.75)
                benchmark = qualities[min(percentile_75_index, len(qualities) - 1)]
                benchmarks[pattern_type] = benchmark

        return benchmarks

    def _generate_architectural_recommendations(self, patterns: list[PatternMatch], projects: list[str]) -> list[str]:
        """Generate architectural recommendations based on analysis."""
        recommendations = []

        try:
            # Analyze overall pattern quality
            avg_quality = sum(p.pattern_quality for p in patterns) / len(patterns) if patterns else 0.0

            if avg_quality < 0.7:
                recommendations.append(
                    "Overall pattern quality is below recommended threshold. Focus on improving pattern implementations."
                )

            # Analyze pattern coverage
            pattern_types_found = {p.pattern_type for p in patterns}
            essential_patterns = {PatternType.SERVICE_LAYER, PatternType.REPOSITORY, PatternType.MVC}
            missing_essential = essential_patterns - pattern_types_found

            if missing_essential:
                recommendations.append(f"Consider implementing essential patterns: {', '.join(p.value for p in missing_essential)}")

            # Analyze consistency
            complexity_variance = 0.0
            if patterns:
                complexities = [p.pattern_complexity for p in patterns]
                avg_complexity = sum(complexities) / len(complexities)
                complexity_variance = sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)

            if complexity_variance > 0.1:
                recommendations.append("High variance in pattern complexity. Consider standardizing implementation approaches.")

            # Project-specific recommendations
            if len(projects) > 1:
                recommendations.append("Consider creating cross-project pattern guidelines to improve consistency.")

            return recommendations[:5]  # Limit to top 5 recommendations

        except Exception as e:
            self.logger.error(f"Error generating architectural recommendations: {e}")
            return ["Enable to generate recommendations due to analysis error"]

    def _identify_standardization_opportunities(self, patterns: list[PatternMatch], projects: list[str]) -> list[str]:
        """Identify opportunities for pattern standardization."""
        opportunities = []

        try:
            # Group patterns by type
            pattern_groups = defaultdict(list)
            for pattern in patterns:
                pattern_groups[pattern.pattern_type].append(pattern)

            for pattern_type, type_patterns in pattern_groups.items():
                if len(type_patterns) > 1:
                    # Check quality variance
                    qualities = [p.pattern_quality for p in type_patterns]
                    quality_variance = sum((q - sum(qualities) / len(qualities)) ** 2 for q in qualities) / len(qualities)

                    if quality_variance > 0.05:  # High variance threshold
                        best_project = max(type_patterns, key=lambda p: p.pattern_quality).project_name
                        opportunities.append(f"Standardize {pattern_type.value} pattern based on {best_project} implementation")

                    # Check naming consistency
                    all_naming = set()
                    for pattern in type_patterns:
                        all_naming.update(pattern.naming_evidence)

                    if len(all_naming) > 5:  # Too many different naming approaches
                        opportunities.append(f"Standardize naming conventions for {pattern_type.value} pattern")

            return opportunities[:5]  # Limit to top 5 opportunities

        except Exception as e:
            self.logger.error(f"Error identifying standardization opportunities: {e}")
            return []

    def _prioritize_improvements(self, patterns: list[PatternMatch]) -> list[str]:
        """Prioritize improvement areas based on impact and effort."""
        priorities = []

        try:
            # Find patterns with low quality but high frequency (high impact)
            pattern_frequency = defaultdict(int)
            pattern_quality = defaultdict(list)

            for pattern in patterns:
                pattern_frequency[pattern.pattern_type] += 1
                pattern_quality[pattern.pattern_type].append(pattern.pattern_quality)

            # Calculate priority scores
            priority_scores = []
            for pattern_type, freq in pattern_frequency.items():
                avg_quality = sum(pattern_quality[pattern_type]) / len(pattern_quality[pattern_type])

                # High frequency + low quality = high priority
                priority_score = freq * (1.0 - avg_quality)
                priority_scores.append((pattern_type, priority_score, freq, avg_quality))

            # Sort by priority score (highest first)
            priority_scores.sort(key=lambda x: x[1], reverse=True)

            for pattern_type, score, freq, avg_quality in priority_scores[:5]:
                priorities.append(
                    f"High priority: Improve {pattern_type.value} pattern " f"(appears {freq} times, avg quality: {avg_quality:.2f})"
                )

            return priorities

        except Exception as e:
            self.logger.error(f"Error prioritizing improvements: {e}")
            return []

    def _analyze_complexity_trend(self, baseline: PatternMatch, comparisons: list[PatternMatch]) -> str:
        """Analyze complexity trend across implementations."""
        if not comparisons:
            return "unknown"

        complexities = [p.pattern_complexity for p in comparisons]
        avg_complexity = sum(complexities) / len(complexities)

        if avg_complexity > baseline.pattern_complexity + 0.1:
            return "increasing"
        elif avg_complexity < baseline.pattern_complexity - 0.1:
            return "decreasing"
        else:
            return "stable"

    def _analyze_quality_trend(self, baseline: PatternMatch, comparisons: list[PatternMatch]) -> str:
        """Analyze quality trend across implementations."""
        if not comparisons:
            return "unknown"

        qualities = [p.pattern_quality for p in comparisons]
        avg_quality = sum(qualities) / len(qualities)

        if avg_quality > baseline.pattern_quality + 0.1:
            return "improving"
        elif avg_quality < baseline.pattern_quality - 0.1:
            return "declining"
        else:
            return "stable"

    def _analyze_consistency_trend(self, baseline: PatternMatch, comparisons: list[PatternMatch]) -> str:
        """Analyze consistency trend across implementations."""
        if not comparisons:
            return "unknown"

        # Calculate variance in quality
        qualities = [p.pattern_quality for p in comparisons]
        if len(qualities) < 2:
            return "stable"

        avg_quality = sum(qualities) / len(qualities)
        variance = sum((q - avg_quality) ** 2 for q in qualities) / len(qualities)

        # Compare with baseline (assuming baseline represents earlier state)
        # Lower variance = more consistent
        if variance < 0.05:  # Low variance threshold
            return "more_consistent"
        elif variance > 0.2:  # High variance threshold
            return "less_consistent"
        else:
            return "stable"

    def _identify_common_improvements(self, baseline: PatternMatch, comparisons: list[PatternMatch]) -> list[str]:
        """Identify common improvements across implementations."""
        improvements = []

        try:
            better_implementations = [p for p in comparisons if p.pattern_quality > baseline.pattern_quality + 0.1]

            if not better_implementations:
                return improvements

            # Look for common naming evidence in better implementations
            common_naming = set(better_implementations[0].naming_evidence)
            for impl in better_implementations[1:]:
                common_naming &= set(impl.naming_evidence)

            baseline_naming = set(baseline.naming_evidence)
            new_naming = common_naming - baseline_naming

            if new_naming:
                improvements.extend(f"Adopt naming convention: {naming}" for naming in list(new_naming)[:3])

            # Look for common structural improvements
            if all(len(impl.components) > len(baseline.components) for impl in better_implementations):
                improvements.append("Add additional components to strengthen pattern")

            if all(impl.pattern_completeness > baseline.pattern_completeness + 0.1 for impl in better_implementations):
                improvements.append("Improve pattern completeness")

            return improvements[:5]  # Limit to top 5

        except Exception as e:
            self.logger.error(f"Error identifying common improvements: {e}")
            return []

    def _identify_common_degradations(self, baseline: PatternMatch, comparisons: list[PatternMatch]) -> list[str]:
        """Identify common degradations across implementations."""
        degradations = []

        try:
            worse_implementations = [p for p in comparisons if p.pattern_quality < baseline.pattern_quality - 0.1]

            if not worse_implementations:
                return degradations

            # Look for common issues in worse implementations
            if all(impl.pattern_complexity > baseline.pattern_complexity + 0.1 for impl in worse_implementations):
                degradations.append("Increased pattern complexity")

            if all(impl.pattern_completeness < baseline.pattern_completeness - 0.1 for impl in worse_implementations):
                degradations.append("Reduced pattern completeness")

            if all(len(impl.components) < len(baseline.components) for impl in worse_implementations):
                degradations.append("Missing pattern components")

            return degradations[:3]  # Limit to top 3

        except Exception as e:
            self.logger.error(f"Error identifying common degradations: {e}")
            return []

    def _identify_emerging_practices(self, comparisons: list[PatternMatch]) -> list[str]:
        """Identify emerging practices in recent implementations."""
        practices = []

        try:
            if not comparisons:
                return practices

            # Look for naming evidence that appears in most implementations
            all_naming = defaultdict(int)
            for impl in comparisons:
                for naming in impl.naming_evidence:
                    all_naming[naming] += 1

            # Find naming that appears in majority of implementations
            majority_threshold = len(comparisons) // 2 + 1
            emerging_naming = [naming for naming, count in all_naming.items() if count >= majority_threshold]

            practices.extend(f"Emerging naming practice: {naming}" for naming in emerging_naming[:3])

            # Look for structural patterns
            if sum(len(impl.components) for impl in comparisons) / len(comparisons) > 5:
                practices.append("Trend towards more comprehensive pattern implementations")

            return practices[:5]  # Limit to top 5

        except Exception as e:
            self.logger.error(f"Error identifying emerging practices: {e}")
            return []

    def _identify_deprecated_practices(self, baseline: PatternMatch, comparisons: list[PatternMatch]) -> list[str]:
        """Identify practices that are becoming deprecated."""
        deprecated = []

        try:
            # Look for baseline practices that rarely appear in comparisons
            baseline_naming = set(baseline.naming_evidence)

            for naming in baseline_naming:
                appearance_count = sum(1 for impl in comparisons if naming in impl.naming_evidence)
                appearance_ratio = appearance_count / len(comparisons) if comparisons else 0

                if appearance_ratio < 0.3:  # Appears in less than 30% of implementations
                    deprecated.append(f"Deprecated naming practice: {naming}")

            return deprecated[:3]  # Limit to top 3

        except Exception as e:
            self.logger.error(f"Error identifying deprecated practices: {e}")
            return []

    def _generate_evolution_recommendations(
        self,
        baseline: PatternMatch,
        comparisons: list[PatternMatch],
        complexity_trend: str,
        quality_trend: str,
    ) -> list[str]:
        """Generate recommendations based on evolution analysis."""
        recommendations = []

        try:
            if quality_trend == "declining":
                recommendations.append("Pattern quality is declining. Review and strengthen implementation standards.")
            elif quality_trend == "improving":
                recommendations.append("Pattern quality is improving. Document best practices for continued improvement.")

            if complexity_trend == "increasing":
                recommendations.append("Pattern complexity is increasing. Consider simplification strategies.")
            elif complexity_trend == "decreasing":
                recommendations.append("Pattern complexity is decreasing. Good trend - maintain simplicity focus.")

            # Find the best implementation for recommendations
            if comparisons:
                best_impl = max(comparisons, key=lambda p: p.pattern_quality)
                if best_impl.pattern_quality > baseline.pattern_quality + 0.1:
                    recommendations.append(f"Consider adopting practices from {best_impl.project_name} implementation.")

            return recommendations[:5]  # Limit to top 5

        except Exception as e:
            self.logger.error(f"Error generating evolution recommendations: {e}")
            return []

    def _initialize_quality_benchmarks(self) -> dict[PatternType, float]:
        """Initialize quality benchmarks for different pattern types."""
        return {
            PatternType.SINGLETON: 0.8,
            PatternType.FACTORY: 0.7,
            PatternType.OBSERVER: 0.75,
            PatternType.MVC: 0.8,
            PatternType.REPOSITORY: 0.75,
            PatternType.SERVICE_LAYER: 0.7,
            PatternType.DECORATOR: 0.75,
            PatternType.STRATEGY: 0.7,
            # Add more as needed...
        }

    def _initialize_architectural_principles(self) -> dict[str, dict[str, Any]]:
        """Initialize architectural principles for evaluation."""
        return {
            "separation_of_concerns": {
                "description": "Each component should have a single responsibility",
                "evaluation_criteria": ["component_cohesion", "interface_clarity"],
            },
            "loose_coupling": {
                "description": "Components should have minimal dependencies",
                "evaluation_criteria": ["dependency_count", "interface_abstraction"],
            },
            "high_cohesion": {
                "description": "Related functionality should be grouped together",
                "evaluation_criteria": ["functional_grouping", "module_organization"],
            },
            # Add more principles as needed...
        }


# Factory function for dependency injection
_pattern_comparison_service_instance = None


def get_pattern_comparison_service(
    pattern_recognition_service: PatternRecognitionService = None,
    implementation_chain_service: ImplementationChainService = None,
) -> PatternComparisonService:
    """
    Get or create a PatternComparisonService instance.

    Args:
        pattern_recognition_service: Pattern recognition service instance
        implementation_chain_service: Implementation chain service instance

    Returns:
        PatternComparisonService instance
    """
    global _pattern_comparison_service_instance

    if _pattern_comparison_service_instance is None:
        if pattern_recognition_service is None:
            from .pattern_recognition_service import get_pattern_recognition_service

            pattern_recognition_service = get_pattern_recognition_service()

        if implementation_chain_service is None:
            from .implementation_chain_service import get_implementation_chain_service

            implementation_chain_service = get_implementation_chain_service()

        _pattern_comparison_service_instance = PatternComparisonService(
            pattern_recognition_service=pattern_recognition_service,
            implementation_chain_service=implementation_chain_service,
        )

    return _pattern_comparison_service_instance
