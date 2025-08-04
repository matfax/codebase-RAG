"""
Path Importance Scoring Service for Wave 2.0 Task 2.6 - Multi-factor Path Scoring.

This service implements sophisticated algorithms to score path importance based on
information density, relevance, complexity, and architectural significance. It provides
weighted scoring mechanisms for prioritization in retrieval and pruning operations.
"""

import asyncio
import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathNode,
    PathType,
    RelationalPathCollection,
)


class ScoringMethod(Enum):
    """Available scoring methods."""

    INFORMATION_DENSITY = "information_density"  # Based on information content
    ARCHITECTURAL_SIGNIFICANCE = "architectural"  # Based on architectural importance
    COMPLEXITY_WEIGHTED = "complexity_weighted"  # Weighted by complexity
    USAGE_FREQUENCY = "usage_frequency"  # Based on estimated usage
    HYBRID_WEIGHTED = "hybrid_weighted"  # Combination of multiple factors


class ImportanceCategory(Enum):
    """Categories of path importance."""

    CRITICAL = "critical"  # Essential paths (>0.8)
    HIGH = "high"  # High importance (0.6-0.8)
    MEDIUM = "medium"  # Medium importance (0.4-0.6)
    LOW = "low"  # Low importance (0.2-0.4)
    MINIMAL = "minimal"  # Minimal importance (<0.2)


@dataclass
class ScoringFactors:
    """Individual scoring factors for path importance."""

    # Information-based factors
    information_density: float = 0.0  # Density of information content
    semantic_richness: float = 0.0  # Richness of semantic information
    structural_uniqueness: float = 0.0  # Uniqueness of structure

    # Architectural factors
    architectural_role: float = 0.0  # Role in architecture
    connectivity_centrality: float = 0.0  # Position in connectivity graph
    pattern_significance: float = 0.0  # Significance in design patterns

    # Complexity factors
    structural_complexity: float = 0.0  # Complexity of structure
    cognitive_load: float = 0.0  # Mental effort required to understand
    maintenance_impact: float = 0.0  # Impact on maintenance

    # Usage factors
    execution_frequency: float = 0.0  # How often path is executed
    access_patterns: float = 0.0  # Quality of access patterns
    error_proneness: float = 0.0  # Likelihood of containing errors

    # Quality factors
    code_quality: float = 0.0  # Overall code quality
    documentation_quality: float = 0.0  # Quality of documentation
    testing_coverage: float = 0.0  # Test coverage metrics

    def get_weighted_score(self, weights: dict[str, float]) -> float:
        """Calculate weighted score using provided weights."""
        factors = {
            "information_density": self.information_density,
            "semantic_richness": self.semantic_richness,
            "structural_uniqueness": self.structural_uniqueness,
            "architectural_role": self.architectural_role,
            "connectivity_centrality": self.connectivity_centrality,
            "pattern_significance": self.pattern_significance,
            "structural_complexity": self.structural_complexity,
            "cognitive_load": self.cognitive_load,
            "maintenance_impact": self.maintenance_impact,
            "execution_frequency": self.execution_frequency,
            "access_patterns": self.access_patterns,
            "error_proneness": self.error_proneness,
            "code_quality": self.code_quality,
            "documentation_quality": self.documentation_quality,
            "testing_coverage": self.testing_coverage,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for factor_name, factor_value in factors.items():
            if factor_name in weights:
                weight = weights[factor_name]
                weighted_sum += factor_value * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


@dataclass
class PathImportanceScore:
    """Complete importance score for a path."""

    # Core identification
    path_id: str  # Path identifier
    path_type: PathType  # Type of path

    # Score components
    overall_score: float  # Final weighted score (0-1)
    scoring_factors: ScoringFactors  # Detailed factor breakdown
    confidence: float  # Confidence in score (0-1)

    # Classification
    importance_category: ImportanceCategory  # Categorical importance
    priority_rank: int = 0  # Relative priority ranking

    # Context information
    scoring_method: ScoringMethod  # Method used for scoring
    context_factors: dict[str, Any] = field(default_factory=dict)  # Additional context

    # Quality indicators
    score_stability: float = 1.0  # How stable the score is
    interpretability: float = 1.0  # How interpretable the score is

    def is_high_importance(self) -> bool:
        """Check if path has high importance."""
        return self.importance_category in {ImportanceCategory.CRITICAL, ImportanceCategory.HIGH}

    def is_critical(self) -> bool:
        """Check if path is critical."""
        return self.importance_category == ImportanceCategory.CRITICAL

    def get_normalized_score(self, min_score: float = 0.0, max_score: float = 1.0) -> float:
        """Get score normalized to specified range."""
        if max_score <= min_score:
            return min_score

        return min_score + (self.overall_score * (max_score - min_score))


@dataclass
class ScoringResult:
    """Result of path importance scoring operation."""

    # Scoring results
    path_scores: list[PathImportanceScore]  # Individual path scores
    score_distribution: dict[ImportanceCategory, int]  # Count by category

    # Statistics
    average_score: float  # Average importance score
    score_variance: float  # Variance in scores
    highest_scoring_paths: list[str]  # IDs of highest scoring paths
    lowest_scoring_paths: list[str]  # IDs of lowest scoring paths

    # Quality metrics
    scoring_consistency: float  # Consistency of scoring
    discriminative_power: float  # How well scores discriminate
    calibration_quality: float  # How well calibrated scores are

    # Processing metadata
    processing_time_ms: float  # Time taken for scoring
    paths_processed: int  # Number of paths scored
    scoring_method: ScoringMethod  # Method used

    # Insights
    scoring_insights: list[str] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)

    def get_paths_by_category(self, category: ImportanceCategory) -> list[PathImportanceScore]:
        """Get paths of a specific importance category."""
        return [score for score in self.path_scores if score.importance_category == category]

    def get_top_paths(self, n: int = 10) -> list[PathImportanceScore]:
        """Get top N highest scoring paths."""
        sorted_scores = sorted(self.path_scores, key=lambda s: s.overall_score, reverse=True)
        return sorted_scores[:n]

    def is_well_distributed(self) -> bool:
        """Check if scores are well distributed across categories."""
        non_empty_categories = sum(1 for count in self.score_distribution.values() if count > 0)
        return non_empty_categories >= 3 and self.discriminative_power > 0.5


@dataclass
class ScoringConfig:
    """Configuration for path importance scoring."""

    # Scoring method and weights
    scoring_method: ScoringMethod = ScoringMethod.HYBRID_WEIGHTED

    # Factor weights (sum should be 1.0)
    information_weights: dict[str, float] = field(
        default_factory=lambda: {"information_density": 0.25, "semantic_richness": 0.15, "structural_uniqueness": 0.10}
    )

    architectural_weights: dict[str, float] = field(
        default_factory=lambda: {"architectural_role": 0.15, "connectivity_centrality": 0.10, "pattern_significance": 0.05}
    )

    complexity_weights: dict[str, float] = field(
        default_factory=lambda: {"structural_complexity": 0.08, "cognitive_load": 0.07, "maintenance_impact": 0.05}
    )

    # Category thresholds
    critical_threshold: float = 0.8  # Threshold for critical importance
    high_threshold: float = 0.6  # Threshold for high importance
    medium_threshold: float = 0.4  # Threshold for medium importance
    low_threshold: float = 0.2  # Threshold for low importance

    # Quality settings
    min_confidence_threshold: float = 0.5  # Minimum confidence for reliable scores
    enable_score_normalization: bool = True  # Normalize scores across collection
    adjust_for_path_type: bool = True  # Adjust scoring based on path type

    # Performance settings
    max_scoring_time_ms: float = 30000  # Maximum time for scoring operation
    enable_parallel_scoring: bool = True  # Enable parallel factor calculation
    cache_factor_calculations: bool = True  # Cache expensive calculations

    # Advanced options
    dynamic_weight_adjustment: bool = False  # Dynamically adjust weights
    contextual_scoring: bool = True  # Consider path context
    temporal_factors: bool = False  # Consider temporal aspects (future)


class PathImportanceScorer:
    """
    Advanced service that calculates comprehensive importance scores for paths
    based on multiple factors including information density, architectural significance,
    complexity, and usage patterns. Provides configurable scoring methods and
    detailed factor analysis.

    Key features:
    - Multi-factor scoring with configurable weights
    - Path type-aware scoring adjustments
    - Architectural significance analysis
    - Information density calculation
    - Quality assessment and calibration
    - Performance optimization with caching
    """

    def __init__(self, config: ScoringConfig | None = None):
        """
        Initialize the path importance scorer.

        Args:
            config: Scoring configuration options
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ScoringConfig()

        # Performance tracking
        self._scoring_stats = {
            "total_scoring_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time_ms": 0.0,
            "average_paths_per_operation": 0.0,
            "total_paths_scored": 0,
            "cache_hit_rate": 0.0,
        }

        # Caching for expensive calculations
        self._factor_cache: dict[str, ScoringFactors] = {}
        self._score_cache: dict[str, PathImportanceScore] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def score_path_collection(
        self, collection: RelationalPathCollection, context_info: dict[str, Any] | None = None
    ) -> ScoringResult:
        """
        Score all paths in a collection for importance.

        Args:
            collection: Path collection to score
            context_info: Optional contextual information

        Returns:
            ScoringResult with importance scores and analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Scoring path collection: {collection.collection_name}")

            # Get all paths from collection
            all_paths = collection.execution_paths + collection.data_flow_paths + collection.dependency_paths

            if not all_paths:
                return self._create_empty_result(collection, "No paths to score")

            # Score individual paths
            path_scores = await self._score_paths(all_paths, collection, context_info or {})

            # Normalize scores if enabled
            if self.config.enable_score_normalization:
                path_scores = await self._normalize_scores(path_scores)

            # Assign priority rankings
            path_scores = await self._assign_priority_rankings(path_scores)

            # Calculate result statistics
            processing_time_ms = (time.time() - start_time) * 1000
            result_stats = await self._calculate_result_statistics(path_scores)

            # Create scoring result
            result = ScoringResult(
                path_scores=path_scores,
                score_distribution=result_stats["distribution"],
                average_score=result_stats["average_score"],
                score_variance=result_stats["score_variance"],
                highest_scoring_paths=result_stats["highest_scoring"],
                lowest_scoring_paths=result_stats["lowest_scoring"],
                scoring_consistency=result_stats["consistency"],
                discriminative_power=result_stats["discriminative_power"],
                calibration_quality=result_stats["calibration_quality"],
                processing_time_ms=processing_time_ms,
                paths_processed=len(all_paths),
                scoring_method=self.config.scoring_method,
            )

            # Add insights and warnings
            await self._add_scoring_insights(result, collection)

            # Update performance statistics
            self._update_performance_stats(processing_time_ms, len(all_paths), True)

            self.logger.info(
                f"Scoring completed: {len(path_scores)} paths scored in {processing_time_ms:.2f}ms, "
                f"avg score: {result.average_score:.3f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Path scoring failed: {str(e)}")
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time_ms, len(all_paths) if "all_paths" in locals() else 0, False)

            return self._create_empty_result(collection, f"Scoring failed: {str(e)}")

    async def score_single_path(
        self, path: AnyPath, context_paths: list[AnyPath] | None = None, context_info: dict[str, Any] | None = None
    ) -> PathImportanceScore:
        """
        Score a single path for importance.

        Args:
            path: Path to score
            context_paths: Optional context paths for relative scoring
            context_info: Optional contextual information

        Returns:
            PathImportanceScore for the path
        """
        # Use cache if available
        cache_key = self._generate_cache_key(path, context_info or {})
        if self.config.cache_factor_calculations and cache_key in self._score_cache:
            self._cache_hits += 1
            return self._score_cache[cache_key]

        self._cache_misses += 1

        # Calculate scoring factors
        factors = await self._calculate_scoring_factors(path, context_paths or [], context_info or {})

        # Calculate overall score
        overall_score = await self._calculate_overall_score(factors, path.path_type)

        # Determine importance category
        category = self._determine_importance_category(overall_score)

        # Calculate confidence
        confidence = await self._calculate_score_confidence(factors, path)

        # Create importance score
        importance_score = PathImportanceScore(
            path_id=path.path_id,
            path_type=path.path_type,
            overall_score=overall_score,
            scoring_factors=factors,
            confidence=confidence,
            importance_category=category,
            scoring_method=self.config.scoring_method,
            context_factors=context_info or {},
        )

        # Cache result
        if self.config.cache_factor_calculations:
            self._score_cache[cache_key] = importance_score

        return importance_score

    async def _score_paths(
        self, paths: list[AnyPath], collection: RelationalPathCollection, context_info: dict[str, Any]
    ) -> list[PathImportanceScore]:
        """Score multiple paths efficiently."""
        # Prepare context information for all paths
        enriched_context = await self._enrich_context_info(context_info, collection, paths)

        # Score paths
        if self.config.enable_parallel_scoring and len(paths) > 5:
            # Parallel scoring for better performance
            scoring_tasks = [self.score_single_path(path, paths, enriched_context) for path in paths]
            path_scores = await asyncio.gather(*scoring_tasks, return_exceptions=True)

            # Filter out exceptions
            valid_scores = []
            for score in path_scores:
                if isinstance(score, PathImportanceScore):
                    valid_scores.append(score)
                else:
                    self.logger.warning(f"Path scoring failed: {str(score)}")

            return valid_scores
        else:
            # Sequential scoring
            path_scores = []
            for path in paths:
                try:
                    score = await self.score_single_path(path, paths, enriched_context)
                    path_scores.append(score)
                except Exception as e:
                    self.logger.warning(f"Failed to score path {path.path_id}: {str(e)}")

            return path_scores

    async def _calculate_scoring_factors(self, path: AnyPath, context_paths: list[AnyPath], context_info: dict[str, Any]) -> ScoringFactors:
        """Calculate all scoring factors for a path."""
        factors = ScoringFactors()

        # Information factors
        factors.information_density = await self._calculate_information_density(path)
        factors.semantic_richness = await self._calculate_semantic_richness(path)
        factors.structural_uniqueness = await self._calculate_structural_uniqueness(path, context_paths)

        # Architectural factors
        factors.architectural_role = await self._calculate_architectural_role(path, context_info)
        factors.connectivity_centrality = await self._calculate_connectivity_centrality(path, context_paths)
        factors.pattern_significance = await self._calculate_pattern_significance(path)

        # Complexity factors
        factors.structural_complexity = await self._calculate_structural_complexity(path)
        factors.cognitive_load = await self._calculate_cognitive_load(path)
        factors.maintenance_impact = await self._calculate_maintenance_impact(path)

        # Usage factors
        factors.execution_frequency = await self._calculate_execution_frequency(path)
        factors.access_patterns = await self._calculate_access_patterns(path)
        factors.error_proneness = await self._calculate_error_proneness(path)

        # Quality factors
        factors.code_quality = await self._calculate_code_quality(path)
        factors.documentation_quality = await self._calculate_documentation_quality(path)
        factors.testing_coverage = await self._calculate_testing_coverage(path)

        return factors

    async def _calculate_information_density(self, path: AnyPath) -> float:
        """Calculate information density of a path."""
        if not path.nodes:
            return 0.0

        # Node type diversity
        unique_types = len({node.chunk_type for node in path.nodes})
        max_possible_types = 8  # Estimate of maximum chunk types
        type_diversity = unique_types / max_possible_types

        # Breadcrumb complexity
        breadcrumb_depths = [len(node.breadcrumb.split(".")) for node in path.nodes if node.breadcrumb]
        avg_depth = sum(breadcrumb_depths) / len(breadcrumb_depths) if breadcrumb_depths else 0
        depth_score = min(1.0, avg_depth / 5.0)  # Normalize to 5 levels

        # Node importance distribution
        node_importances = [node.importance_score for node in path.nodes]
        importance_variance = np.var(node_importances) if len(node_importances) > 1 else 0.0

        # Path length factor (normalized)
        length_factor = min(1.0, len(path.nodes) / 15.0)  # Normalize to 15 nodes

        # Combine factors
        density = type_diversity * 0.3 + depth_score * 0.25 + importance_variance * 0.2 + length_factor * 0.25

        return min(1.0, density)

    async def _calculate_semantic_richness(self, path: AnyPath) -> float:
        """Calculate semantic richness of a path."""
        richness_factors = []

        # Node name diversity
        node_names = [node.name.lower() for node in path.nodes if node.name]
        unique_names = len(set(node_names))
        name_diversity = unique_names / max(1, len(node_names))
        richness_factors.append(name_diversity)

        # Semantic context from nodes
        contexts = [node.semantic_summary for node in path.nodes if node.semantic_summary]
        context_availability = len(contexts) / max(1, len(path.nodes))
        richness_factors.append(context_availability)

        # Path-specific semantic content
        if isinstance(path, ExecutionPath):
            use_cases = getattr(path, "use_cases", [])
            semantic_content = len(use_cases) / 5.0  # Normalize to 5 use cases
            richness_factors.append(min(1.0, semantic_content))
        elif isinstance(path, DataFlowPath):
            transformations = getattr(path, "transformations", [])
            semantic_content = len(transformations) / 10.0  # Normalize to 10 transformations
            richness_factors.append(min(1.0, semantic_content))
        elif isinstance(path, DependencyPath):
            modules = getattr(path, "required_modules", [])
            semantic_content = len(modules) / 10.0  # Normalize to 10 modules
            richness_factors.append(min(1.0, semantic_content))

        return sum(richness_factors) / len(richness_factors) if richness_factors else 0.0

    async def _calculate_structural_uniqueness(self, path: AnyPath, context_paths: list[AnyPath]) -> float:
        """Calculate how structurally unique a path is."""
        if not context_paths:
            return 0.5  # Default when no context available

        # Calculate similarity to other paths
        similarities = []
        path_breadcrumbs = {node.breadcrumb for node in path.nodes}

        for other_path in context_paths:
            if other_path.path_id == path.path_id:
                continue

            other_breadcrumbs = {node.breadcrumb for node in other_path.nodes}

            if not path_breadcrumbs or not other_breadcrumbs:
                similarity = 0.0
            else:
                intersection = path_breadcrumbs.intersection(other_breadcrumbs)
                union = path_breadcrumbs.union(other_breadcrumbs)
                similarity = len(intersection) / len(union)

            similarities.append(similarity)

        # Uniqueness is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        uniqueness = 1.0 - avg_similarity

        return max(0.0, min(1.0, uniqueness))

    async def _calculate_architectural_role(self, path: AnyPath, context_info: dict[str, Any]) -> float:
        """Calculate architectural significance of a path."""
        role_score = 0.0

        # Check if path is an entry point
        entry_points = context_info.get("primary_entry_points", [])
        if isinstance(path, ExecutionPath) and hasattr(path, "entry_points"):
            path_entries = getattr(path, "entry_points", [])
            if any(entry in entry_points for entry in path_entries):
                role_score += 0.3

        # Check architectural patterns
        patterns = context_info.get("architectural_patterns", [])
        if patterns:
            # Simple heuristic: paths with pattern-related names are more significant
            pattern_keywords = ["factory", "builder", "observer", "strategy", "adapter", "facade"]
            node_names = [node.name.lower() for node in path.nodes]

            for keyword in pattern_keywords:
                if any(keyword in name for name in node_names):
                    role_score += 0.2
                    break

        # Check if path represents main business logic
        business_keywords = ["process", "handle", "execute", "compute", "calculate", "manage"]
        node_names = [node.name.lower() for node in path.nodes]

        for keyword in business_keywords:
            if any(keyword in name for name in node_names):
                role_score += 0.1

        # Path type significance
        if isinstance(path, ExecutionPath):
            role_score += 0.2  # Execution paths are architecturally significant
        elif isinstance(path, DependencyPath):
            role_score += 0.15  # Dependencies define architecture

        return min(1.0, role_score)

    async def _calculate_connectivity_centrality(self, path: AnyPath, context_paths: list[AnyPath]) -> float:
        """Calculate connectivity centrality of a path."""
        if not context_paths:
            return 0.0

        path_breadcrumbs = {node.breadcrumb for node in path.nodes}
        connections = 0

        # Count connections to other paths
        for other_path in context_paths:
            if other_path.path_id == path.path_id:
                continue

            other_breadcrumbs = {node.breadcrumb for node in other_path.nodes}

            # Check for shared breadcrumbs (connections)
            shared = path_breadcrumbs.intersection(other_breadcrumbs)
            if shared:
                connections += len(shared)

        # Normalize by maximum possible connections
        max_connections = len(context_paths) * len(path_breadcrumbs)
        centrality = connections / max(1, max_connections)

        return min(1.0, centrality * 5.0)  # Scale up for better distribution

    async def _calculate_pattern_significance(self, path: AnyPath) -> float:
        """Calculate significance in design patterns."""
        significance = 0.0

        # Check for design pattern indicators in node names
        pattern_indicators = {
            "factory": 0.3,
            "builder": 0.3,
            "singleton": 0.25,
            "observer": 0.25,
            "strategy": 0.25,
            "adapter": 0.2,
            "facade": 0.2,
            "proxy": 0.2,
            "decorator": 0.2,
            "command": 0.15,
            "state": 0.15,
            "visitor": 0.15,
        }

        node_names = [node.name.lower() for node in path.nodes]

        for pattern, weight in pattern_indicators.items():
            if any(pattern in name for name in node_names):
                significance += weight

        # Check for abstract/interface patterns
        abstract_indicators = ["abstract", "interface", "base", "template"]
        for indicator in abstract_indicators:
            if any(indicator in name for name in node_names):
                significance += 0.1

        return min(1.0, significance)

    async def _calculate_structural_complexity(self, path: AnyPath) -> float:
        """Calculate structural complexity of a path."""
        if hasattr(path, "complexity_score"):
            return getattr(path, "complexity_score", 0.0)

        # Estimate complexity from path structure
        complexity_factors = []

        # Node count factor
        node_count_factor = min(1.0, len(path.nodes) / 20.0)  # Normalize to 20 nodes
        complexity_factors.append(node_count_factor)

        # Node type diversity
        unique_types = len({node.chunk_type for node in path.nodes})
        type_diversity = min(1.0, unique_types / 6.0)  # Normalize to 6 types
        complexity_factors.append(type_diversity)

        # Breadcrumb depth variance
        depths = [len(node.breadcrumb.split(".")) for node in path.nodes if node.breadcrumb]
        if len(depths) > 1:
            depth_variance = np.var(depths)
            normalized_variance = min(1.0, depth_variance / 4.0)  # Normalize
            complexity_factors.append(normalized_variance)

        # Path-specific complexity
        if isinstance(path, ExecutionPath):
            # Check for complex execution characteristics
            complex_features = 0
            if hasattr(path, "has_loops") and getattr(path, "has_loops", False):
                complex_features += 1
            if hasattr(path, "has_exceptions") and getattr(path, "has_exceptions", False):
                complex_features += 1
            if hasattr(path, "is_async") and getattr(path, "is_async", False):
                complex_features += 1

            feature_complexity = complex_features / 3.0
            complexity_factors.append(feature_complexity)

        return sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.0

    async def _calculate_cognitive_load(self, path: AnyPath) -> float:
        """Calculate cognitive load required to understand the path."""
        load_factors = []

        # Length-based cognitive load
        length_load = min(1.0, len(path.nodes) / 15.0)  # Normalize to 15 nodes
        load_factors.append(length_load)

        # Naming complexity
        node_names = [node.name for node in path.nodes if node.name]
        avg_name_length = sum(len(name) for name in node_names) / max(1, len(node_names))
        name_complexity = min(1.0, (avg_name_length - 5) / 20.0)  # Names longer than 5 chars add load
        load_factors.append(max(0.0, name_complexity))

        # Abstraction level variance
        breadcrumb_depths = [len(node.breadcrumb.split(".")) for node in path.nodes if node.breadcrumb]
        if len(breadcrumb_depths) > 1:
            depth_range = max(breadcrumb_depths) - min(breadcrumb_depths)
            abstraction_load = min(1.0, depth_range / 5.0)  # Normalize to 5 levels
            load_factors.append(abstraction_load)

        return sum(load_factors) / len(load_factors) if load_factors else 0.0

    async def _calculate_maintenance_impact(self, path: AnyPath) -> float:
        """Calculate maintenance impact of the path."""
        impact_factors = []

        # Complexity impact
        structural_complexity = await self._calculate_structural_complexity(path)
        impact_factors.append(structural_complexity)

        # Connectivity impact (estimated from node centrality)
        node_importance_sum = sum(node.importance_score for node in path.nodes)
        avg_importance = node_importance_sum / max(1, len(path.nodes))
        impact_factors.append(avg_importance)

        # Path type impact
        type_impact = 0.5  # Default
        if isinstance(path, ExecutionPath):
            type_impact = 0.7  # High impact for execution paths
        elif isinstance(path, DependencyPath):
            type_impact = 0.8  # Higher impact for dependencies
        elif isinstance(path, DataFlowPath):
            type_impact = 0.6  # Medium-high for data flows

        impact_factors.append(type_impact)

        return sum(impact_factors) / len(impact_factors)

    async def _calculate_execution_frequency(self, path: AnyPath) -> float:
        """Estimate execution frequency of the path."""
        # This is a simplified estimation - in real implementation,
        # this could be based on profiling data or static analysis

        frequency_indicators = []

        # Check for common execution patterns
        if isinstance(path, ExecutionPath):
            # Entry points are executed more frequently
            if hasattr(path, "entry_points") and getattr(path, "entry_points", []):
                frequency_indicators.append(0.8)

            # Main/process functions are frequently executed
            node_names = [node.name.lower() for node in path.nodes]
            frequent_patterns = ["main", "process", "handle", "run", "execute", "start"]
            for pattern in frequent_patterns:
                if any(pattern in name for name in node_names):
                    frequency_indicators.append(0.6)
                    break

        # Simple heuristic based on path position in hierarchy
        if path.nodes:
            avg_depth = sum(len(node.breadcrumb.split(".")) for node in path.nodes) / len(path.nodes)
            # Shallow paths (closer to entry points) are executed more frequently
            depth_frequency = max(0.0, 1.0 - (avg_depth - 1) / 5.0)
            frequency_indicators.append(depth_frequency)

        return sum(frequency_indicators) / len(frequency_indicators) if frequency_indicators else 0.3

    async def _calculate_access_patterns(self, path: AnyPath) -> float:
        """Calculate quality of access patterns."""
        # Simplified analysis based on node roles and structure

        if not path.nodes:
            return 0.0

        # Check for balanced access patterns
        role_counts = defaultdict(int)
        for node in path.nodes:
            role_counts[node.role_in_path] += 1

        # Good access patterns have clear source/target relationships
        has_source = role_counts.get("source", 0) > 0
        has_target = role_counts.get("target", 0) > 0
        has_intermediates = role_counts.get("intermediate", 0) > 0

        pattern_quality = 0.0
        if has_source:
            pattern_quality += 0.3
        if has_target:
            pattern_quality += 0.3
        if has_intermediates:
            pattern_quality += 0.2

        # Penalize excessive branch points
        branch_points = role_counts.get("branch_point", 0)
        if branch_points > len(path.nodes) * 0.3:  # More than 30% branch points
            pattern_quality *= 0.7

        return min(1.0, pattern_quality + 0.2)  # Base quality of 0.2

    async def _calculate_error_proneness(self, path: AnyPath) -> float:
        """Estimate error proneness of the path."""
        risk_factors = []

        # Complexity-based risk
        complexity = await self._calculate_structural_complexity(path)
        risk_factors.append(complexity)

        # Check for error-prone patterns
        if isinstance(path, ExecutionPath):
            risk_score = 0.0

            # Async operations increase risk
            if hasattr(path, "is_async") and getattr(path, "is_async", False):
                risk_score += 0.2

            # Exception handling might indicate error-prone areas
            if hasattr(path, "has_exceptions") and getattr(path, "has_exceptions", False):
                risk_score += 0.15

            # Loops can be error-prone
            if hasattr(path, "has_loops") and getattr(path, "has_loops", False):
                risk_score += 0.1

            risk_factors.append(risk_score)

        # Deep nesting indicates complexity and potential errors
        if path.nodes:
            max_depth = max(len(node.breadcrumb.split(".")) for node in path.nodes)
            depth_risk = min(1.0, (max_depth - 2) / 8.0)  # Risk increases after depth 2
            risk_factors.append(max(0.0, depth_risk))

        return sum(risk_factors) / len(risk_factors) if risk_factors else 0.3

    async def _calculate_code_quality(self, path: AnyPath) -> float:
        """Estimate code quality of the path."""
        # This is a simplified estimation - real implementation would
        # analyze actual code quality metrics

        quality_indicators = []

        # Node importance as quality proxy
        if path.nodes:
            avg_importance = sum(node.importance_score for node in path.nodes) / len(path.nodes)
            quality_indicators.append(avg_importance)

        # Documentation availability
        documented_nodes = sum(1 for node in path.nodes if node.semantic_summary)
        documentation_ratio = documented_nodes / max(1, len(path.nodes))
        quality_indicators.append(documentation_ratio)

        # Naming quality (simple heuristic)
        if path.nodes:
            node_names = [node.name for node in path.nodes if node.name]
            avg_name_length = sum(len(name) for name in node_names) / max(1, len(node_names))
            # Good names are neither too short nor too long
            optimal_length = 12
            length_quality = 1.0 - abs(avg_name_length - optimal_length) / optimal_length
            quality_indicators.append(max(0.0, length_quality))

        return sum(quality_indicators) / len(quality_indicators) if quality_indicators else 0.5

    async def _calculate_documentation_quality(self, path: AnyPath) -> float:
        """Calculate documentation quality of the path."""
        if not path.nodes:
            return 0.0

        # Check for semantic summaries
        documented_nodes = sum(1 for node in path.nodes if node.semantic_summary)
        basic_documentation = documented_nodes / len(path.nodes)

        # Check for path-level documentation
        path_documentation = 0.0
        if isinstance(path, ExecutionPath) and hasattr(path, "semantic_description"):
            if path.semantic_description:
                path_documentation = 0.3

        # Check for contextual information
        contextual_info = 0.0
        nodes_with_context = sum(1 for node in path.nodes if node.local_context)
        if nodes_with_context > 0:
            contextual_info = (nodes_with_context / len(path.nodes)) * 0.2

        total_quality = basic_documentation * 0.5 + path_documentation + contextual_info
        return min(1.0, total_quality)

    async def _calculate_testing_coverage(self, path: AnyPath) -> float:
        """Estimate testing coverage for the path."""
        # This is a placeholder - real implementation would integrate
        # with actual test coverage tools

        # Simple heuristic based on path characteristics
        coverage_estimate = 0.5  # Default moderate coverage

        # Important paths likely have better coverage
        if path.nodes:
            avg_importance = sum(node.importance_score for node in path.nodes) / len(path.nodes)
            coverage_estimate = (coverage_estimate + avg_importance) / 2.0

        # Entry points often have good test coverage
        if isinstance(path, ExecutionPath) and hasattr(path, "entry_points"):
            if getattr(path, "entry_points", []):
                coverage_estimate = min(1.0, coverage_estimate + 0.2)

        return coverage_estimate

    async def _calculate_overall_score(self, factors: ScoringFactors, path_type: PathType) -> float:
        """Calculate overall importance score from factors."""
        # Combine all weights
        all_weights = {}
        all_weights.update(self.config.information_weights)
        all_weights.update(self.config.architectural_weights)
        all_weights.update(self.config.complexity_weights)

        # Get weighted score
        base_score = factors.get_weighted_score(all_weights)

        # Apply path type adjustments if enabled
        if self.config.adjust_for_path_type:
            type_multiplier = self._get_path_type_multiplier(path_type)
            base_score *= type_multiplier

        return min(1.0, max(0.0, base_score))

    def _get_path_type_multiplier(self, path_type: PathType) -> float:
        """Get scoring multiplier based on path type."""
        multipliers = {
            PathType.EXECUTION_PATH: 1.1,  # Slightly favor execution paths
            PathType.DEPENDENCY_PATH: 1.05,  # Dependencies are important
            PathType.DATA_FLOW: 1.0,  # Neutral
            PathType.CONTROL_FLOW: 1.0,  # Neutral
            PathType.API_USAGE: 0.95,  # Slightly less important
            PathType.DESIGN_PATTERN: 1.15,  # Favor design patterns
            PathType.ARCHITECTURAL_PATH: 1.2,  # Favor architectural paths
        }

        return multipliers.get(path_type, 1.0)

    def _determine_importance_category(self, overall_score: float) -> ImportanceCategory:
        """Determine importance category from overall score."""
        if overall_score >= self.config.critical_threshold:
            return ImportanceCategory.CRITICAL
        elif overall_score >= self.config.high_threshold:
            return ImportanceCategory.HIGH
        elif overall_score >= self.config.medium_threshold:
            return ImportanceCategory.MEDIUM
        elif overall_score >= self.config.low_threshold:
            return ImportanceCategory.LOW
        else:
            return ImportanceCategory.MINIMAL

    async def _calculate_score_confidence(self, factors: ScoringFactors, path: AnyPath) -> float:
        """Calculate confidence in the importance score."""
        confidence_factors = []

        # Factor completeness (more factors = higher confidence)
        factor_values = [
            factors.information_density,
            factors.semantic_richness,
            factors.structural_uniqueness,
            factors.architectural_role,
            factors.connectivity_centrality,
            factors.pattern_significance,
            factors.structural_complexity,
            factors.cognitive_load,
            factors.maintenance_impact,
            factors.execution_frequency,
            factors.access_patterns,
            factors.error_proneness,
            factors.code_quality,
            factors.documentation_quality,
            factors.testing_coverage,
        ]

        non_zero_factors = sum(1 for f in factor_values if f > 0.0)
        completeness = non_zero_factors / len(factor_values)
        confidence_factors.append(completeness)

        # Node availability (more nodes = more data = higher confidence)
        node_availability = min(1.0, len(path.nodes) / 5.0)  # Normalize to 5 nodes
        confidence_factors.append(node_availability)

        # Factor consistency (less variance = higher confidence)
        if factor_values:
            factor_variance = np.var([f for f in factor_values if f > 0.0])
            consistency = 1.0 - min(1.0, factor_variance)
            confidence_factors.append(consistency)

        return sum(confidence_factors) / len(confidence_factors)

    async def _enrich_context_info(
        self, context_info: dict[str, Any], collection: RelationalPathCollection, paths: list[AnyPath]
    ) -> dict[str, Any]:
        """Enrich context information for better scoring."""
        enriched = dict(context_info)

        # Add collection-level information
        enriched["collection_name"] = collection.collection_name
        enriched["total_paths"] = len(paths)
        enriched["path_types"] = list({path.path_type for path in paths})
        enriched["primary_entry_points"] = collection.primary_entry_points
        enriched["architectural_patterns"] = collection.architectural_patterns

        # Add statistical information
        if paths:
            path_lengths = [len(path.nodes) for path in paths]
            enriched["avg_path_length"] = sum(path_lengths) / len(path_lengths)
            enriched["max_path_length"] = max(path_lengths)
            enriched["min_path_length"] = min(path_lengths)

        return enriched

    async def _normalize_scores(self, path_scores: list[PathImportanceScore]) -> list[PathImportanceScore]:
        """Normalize scores across the collection."""
        if not path_scores:
            return path_scores

        # Get score range
        scores = [score.overall_score for score in path_scores]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score - min_score < 1e-6:
            return path_scores

        # Normalize to 0-1 range
        normalized_scores = []
        for path_score in path_scores:
            normalized_value = (path_score.overall_score - min_score) / (max_score - min_score)

            # Update the score
            path_score.overall_score = normalized_value
            path_score.importance_category = self._determine_importance_category(normalized_value)
            normalized_scores.append(path_score)

        return normalized_scores

    async def _assign_priority_rankings(self, path_scores: list[PathImportanceScore]) -> list[PathImportanceScore]:
        """Assign priority rankings to path scores."""
        # Sort by overall score (descending)
        sorted_scores = sorted(path_scores, key=lambda s: s.overall_score, reverse=True)

        # Assign rankings
        for i, path_score in enumerate(sorted_scores, 1):
            path_score.priority_rank = i

        return path_scores

    async def _calculate_result_statistics(self, path_scores: list[PathImportanceScore]) -> dict[str, Any]:
        """Calculate statistics for the scoring result."""
        if not path_scores:
            return {
                "distribution": {},
                "average_score": 0.0,
                "score_variance": 0.0,
                "highest_scoring": [],
                "lowest_scoring": [],
                "consistency": 0.0,
                "discriminative_power": 0.0,
                "calibration_quality": 0.0,
            }

        scores = [score.overall_score for score in path_scores]

        # Score distribution by category
        distribution = defaultdict(int)
        for path_score in path_scores:
            distribution[path_score.importance_category] += 1

        # Statistical measures
        average_score = sum(scores) / len(scores)
        score_variance = np.var(scores)

        # Top and bottom paths
        sorted_scores = sorted(path_scores, key=lambda s: s.overall_score, reverse=True)
        highest_scoring = [s.path_id for s in sorted_scores[:5]]
        lowest_scoring = [s.path_id for s in sorted_scores[-5:]]

        # Consistency (based on confidence)
        confidences = [score.confidence for score in path_scores]
        consistency = sum(confidences) / len(confidences)

        # Discriminative power (score range)
        min_score = min(scores)
        max_score = max(scores)
        discriminative_power = max_score - min_score

        # Calibration quality (how well scores match categories)
        calibration_errors = []
        for path_score in path_scores:
            expected_score = self._get_expected_score_for_category(path_score.importance_category)
            error = abs(path_score.overall_score - expected_score)
            calibration_errors.append(error)

        calibration_quality = 1.0 - (sum(calibration_errors) / len(calibration_errors))

        return {
            "distribution": dict(distribution),
            "average_score": average_score,
            "score_variance": score_variance,
            "highest_scoring": highest_scoring,
            "lowest_scoring": lowest_scoring,
            "consistency": consistency,
            "discriminative_power": discriminative_power,
            "calibration_quality": max(0.0, calibration_quality),
        }

    def _get_expected_score_for_category(self, category: ImportanceCategory) -> float:
        """Get expected score for an importance category."""
        expected_scores = {
            ImportanceCategory.CRITICAL: 0.9,
            ImportanceCategory.HIGH: 0.7,
            ImportanceCategory.MEDIUM: 0.5,
            ImportanceCategory.LOW: 0.3,
            ImportanceCategory.MINIMAL: 0.1,
        }
        return expected_scores.get(category, 0.5)

    async def _add_scoring_insights(self, result: ScoringResult, collection: RelationalPathCollection):
        """Add insights and warnings to scoring result."""
        insights = []
        warnings = []

        # Distribution analysis
        total_paths = sum(result.score_distribution.values())
        critical_count = result.score_distribution.get(ImportanceCategory.CRITICAL, 0)
        high_count = result.score_distribution.get(ImportanceCategory.HIGH, 0)

        if critical_count + high_count > total_paths * 0.5:
            insights.append("High proportion of important paths - rich codebase")
        elif critical_count + high_count < total_paths * 0.1:
            warnings.append("Few high-importance paths detected - review scoring criteria")

        # Score variance analysis
        if result.score_variance < 0.05:
            warnings.append("Low score variance - paths may be too similar or scoring needs adjustment")
        elif result.score_variance > 0.3:
            insights.append("High score variance indicates good discriminative power")

        # Consistency analysis
        if result.scoring_consistency > 0.8:
            insights.append("High scoring consistency - reliable importance scores")
        elif result.scoring_consistency < 0.5:
            warnings.append("Low scoring consistency - some scores may be unreliable")

        # Performance analysis
        if result.processing_time_ms < 1000:
            insights.append("Fast scoring operation - good performance")
        elif result.processing_time_ms > 10000:
            insights.append("Slow scoring operation - consider optimization")

        result.scoring_insights = insights
        result.quality_warnings = warnings

    def _generate_cache_key(self, path: AnyPath, context_info: dict[str, Any]) -> str:
        """Generate cache key for a path and context."""
        context_hash = hash(str(sorted(context_info.items())))
        return f"{path.path_id}_{context_hash}"

    def _create_empty_result(self, collection: RelationalPathCollection, error_message: str) -> ScoringResult:
        """Create empty scoring result for error cases."""
        return ScoringResult(
            path_scores=[],
            score_distribution={},
            average_score=0.0,
            score_variance=0.0,
            highest_scoring_paths=[],
            lowest_scoring_paths=[],
            scoring_consistency=0.0,
            discriminative_power=0.0,
            calibration_quality=0.0,
            processing_time_ms=0.0,
            paths_processed=0,
            scoring_method=self.config.scoring_method,
            quality_warnings=[error_message],
        )

    def _update_performance_stats(self, processing_time_ms: float, paths_processed: int, success: bool):
        """Update internal performance statistics."""
        self._scoring_stats["total_scoring_operations"] += 1

        if success:
            self._scoring_stats["successful_operations"] += 1

            # Update averages
            operations = self._scoring_stats["successful_operations"]

            # Average processing time
            current_avg_time = self._scoring_stats["average_processing_time_ms"]
            self._scoring_stats["average_processing_time_ms"] = (current_avg_time * (operations - 1) + processing_time_ms) / operations

            # Average paths per operation
            current_avg_paths = self._scoring_stats["average_paths_per_operation"]
            self._scoring_stats["average_paths_per_operation"] = (current_avg_paths * (operations - 1) + paths_processed) / operations

            self._scoring_stats["total_paths_scored"] += paths_processed
        else:
            self._scoring_stats["failed_operations"] += 1

        # Update cache hit rate
        total_cache_requests = self._cache_hits + self._cache_misses
        if total_cache_requests > 0:
            self._scoring_stats["cache_hit_rate"] = self._cache_hits / total_cache_requests

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        stats = dict(self._scoring_stats)

        # Add derived metrics
        total_ops = stats["total_scoring_operations"]
        if total_ops > 0:
            stats["success_rate"] = stats["successful_operations"] / total_ops
            stats["failure_rate"] = stats["failed_operations"] / total_ops
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        # Add cache statistics
        stats["factor_cache_size"] = len(self._factor_cache)
        stats["score_cache_size"] = len(self._score_cache)
        stats["cache_hits"] = self._cache_hits
        stats["cache_misses"] = self._cache_misses

        return stats

    def clear_caches(self):
        """Clear internal caches to free memory."""
        self._factor_cache.clear()
        self._score_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info("Path importance scorer caches cleared")


# Factory function
def create_path_importance_scorer(config: ScoringConfig | None = None) -> PathImportanceScorer:
    """
    Factory function to create a PathImportanceScorer instance.

    Args:
        config: Optional scoring configuration

    Returns:
        Configured PathImportanceScorer instance
    """
    return PathImportanceScorer(config)
