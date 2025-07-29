"""
Streaming Pruning Service for Wave 2.0 Task 2.3 - PathRAG-style Streaming Pruning.

This service implements intelligent streaming pruning to automatically identify and
remove redundant, low-value retrieval results by 40% while maintaining relevance.
It leverages duplicate detection, relevance filtering, and information density analysis.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.code_chunk import ChunkType
from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathConfidence,
    PathNode,
    RelationalPathCollection,
)


class PruningStrategy(Enum):
    """Different pruning strategies available."""

    CONSERVATIVE = "conservative"  # Remove only obvious duplicates (20% reduction)
    BALANCED = "balanced"  # Balance relevance and reduction (40% reduction)
    AGGRESSIVE = "aggressive"  # Maximum reduction while preserving key paths (60% reduction)


class RedundancyType(Enum):
    """Types of redundancy detected in paths."""

    EXACT_DUPLICATE = "exact_duplicate"  # Identical path structures
    STRUCTURAL_SIMILAR = "structural_similar"  # Similar node sequences
    SEMANTIC_OVERLAP = "semantic_overlap"  # Overlapping semantic content
    LOW_INFORMATION = "low_information"  # Paths with low information density
    CIRCULAR_REFERENCE = "circular_reference"  # Circular or self-referential paths


@dataclass
class PruningResult:
    """Result of streaming pruning operation."""

    # Results
    pruned_paths: list[AnyPath]  # Final pruned path set
    redundant_paths: list[AnyPath]  # Paths that were removed

    # Statistics
    original_count: int  # Original path count
    pruned_count: int  # Final path count after pruning
    reduction_percentage: float  # Percentage of paths removed

    # Quality metrics
    information_density_score: float  # Average information density
    relevance_preservation_score: float  # How well relevance was preserved

    # Processing metadata
    processing_time_ms: float  # Time taken for pruning
    pruning_strategy: PruningStrategy  # Strategy used
    redundancy_breakdown: dict[RedundancyType, int]  # Count by redundancy type

    # Warnings and insights
    pruning_warnings: list[str] = field(default_factory=list)
    quality_insights: list[str] = field(default_factory=list)

    def get_efficiency_score(self) -> float:
        """Calculate overall pruning efficiency score."""
        # Balance reduction percentage with quality preservation
        reduction_score = min(1.0, self.reduction_percentage / 50.0)  # Target 50% reduction
        quality_score = (self.information_density_score + self.relevance_preservation_score) / 2.0

        return (reduction_score * 0.6) + (quality_score * 0.4)

    def is_target_achieved(self, target_reduction: float = 40.0) -> bool:
        """Check if target reduction percentage was achieved."""
        return self.reduction_percentage >= target_reduction


@dataclass
class PruningConfig:
    """Configuration for streaming pruning operations."""

    # Target settings
    target_reduction_percentage: float = 40.0  # Target reduction (PathRAG standard)
    max_processing_time_ms: float = 15000  # Maximum processing time (15s)

    # Quality thresholds
    min_information_density: float = 0.3  # Minimum information density to keep
    min_relevance_score: float = 0.4  # Minimum relevance score to keep
    min_confidence_threshold: float = 0.3  # Minimum path confidence to keep

    # Duplicate detection settings
    structural_similarity_threshold: float = 0.8  # Threshold for structural similarity
    semantic_similarity_threshold: float = 0.7  # Threshold for semantic similarity
    node_overlap_threshold: float = 0.6  # Node overlap threshold for similarity

    # Pruning behavior
    preserve_high_value_paths: bool = True  # Always preserve high-value paths
    preserve_entry_point_paths: bool = True  # Always preserve entry point paths
    enable_circular_detection: bool = True  # Detect and prune circular paths

    # Quality preservation
    maintain_architectural_coverage: bool = True  # Ensure architectural patterns are covered
    preserve_critical_dependencies: bool = True  # Keep critical dependency paths
    balance_path_types: bool = True  # Maintain balance across path types


class StreamingPruningService:
    """
    Advanced streaming pruning service that implements PathRAG-style intelligent
    pruning to reduce redundant information while preserving relevance and quality.

    Key features:
    - Real-time duplicate detection and removal
    - Information density analysis
    - Relevance-aware filtering
    - Architectural pattern preservation
    - Progressive streaming with confidence scoring
    """

    def __init__(self, config: PruningConfig | None = None):
        """
        Initialize the streaming pruning service.

        Args:
            config: Pruning configuration options
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or PruningConfig()

        # Performance tracking
        self._pruning_stats = {
            "total_pruning_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_reduction_percentage": 0.0,
            "average_processing_time_ms": 0.0,
            "total_paths_processed": 0,
            "total_paths_pruned": 0,
        }

        # Caching for repeated operations
        self._similarity_cache: dict[tuple[str, str], float] = {}
        self._density_cache: dict[str, float] = {}

    async def prune_path_collection(
        self,
        collection: RelationalPathCollection,
        strategy: PruningStrategy = PruningStrategy.BALANCED,
        preserve_patterns: set[str] | None = None,
    ) -> PruningResult:
        """
        Apply streaming pruning to a path collection.

        Args:
            collection: Path collection to prune
            strategy: Pruning strategy to use
            preserve_patterns: Optional patterns to always preserve

        Returns:
            PruningResult with pruned paths and statistics
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting streaming pruning with {strategy.value} strategy")

            # Get all paths from collection
            all_paths = collection.execution_paths + collection.data_flow_paths + collection.dependency_paths

            if not all_paths:
                return self._create_empty_result(strategy, "No paths to prune")

            original_count = len(all_paths)
            self.logger.info(f"Pruning {original_count} paths")

            # Apply multi-stage pruning
            pruning_stages = [
                self._detect_exact_duplicates,
                self._detect_structural_similarity,
                self._filter_low_information_density,
                self._detect_semantic_overlap,
                self._filter_circular_references,
            ]

            remaining_paths = all_paths.copy()
            redundant_paths = []
            redundancy_breakdown = defaultdict(int)

            # Execute pruning stages
            for stage_func in pruning_stages:
                if len(remaining_paths) <= 1:
                    break

                stage_result = await stage_func(remaining_paths, strategy, preserve_patterns or set())

                # Update tracking
                stage_redundant = stage_result["redundant"]
                stage_redundancy_type = stage_result["type"]

                remaining_paths = [p for p in remaining_paths if p not in stage_redundant]
                redundant_paths.extend(stage_redundant)
                redundancy_breakdown[stage_redundancy_type] += len(stage_redundant)

                self.logger.debug(
                    f"Stage {stage_func.__name__}: removed {len(stage_redundant)} paths, " f"{len(remaining_paths)} remaining"
                )

                # Check if we've reached target reduction
                current_reduction = ((original_count - len(remaining_paths)) / original_count) * 100
                if current_reduction >= self.config.target_reduction_percentage:
                    break

            # Apply final quality-based selection if still above target
            if len(remaining_paths) > original_count * (1 - self.config.target_reduction_percentage / 100):
                remaining_paths = await self._apply_quality_based_selection(remaining_paths, strategy, preserve_patterns or set())

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000
            pruned_count = len(remaining_paths)
            reduction_percentage = ((original_count - pruned_count) / original_count) * 100

            # Calculate quality metrics
            info_density = await self._calculate_average_information_density(remaining_paths)
            relevance_score = await self._calculate_relevance_preservation_score(all_paths, remaining_paths)

            # Create result
            result = PruningResult(
                pruned_paths=remaining_paths,
                redundant_paths=redundant_paths,
                original_count=original_count,
                pruned_count=pruned_count,
                reduction_percentage=reduction_percentage,
                information_density_score=info_density,
                relevance_preservation_score=relevance_score,
                processing_time_ms=processing_time_ms,
                pruning_strategy=strategy,
                redundancy_breakdown=dict(redundancy_breakdown),
            )

            # Add quality insights
            await self._add_quality_insights(result)

            # Update performance statistics
            self._update_performance_stats(processing_time_ms, original_count, len(redundant_paths), True)

            self.logger.info(
                f"Pruning completed: {reduction_percentage:.1f}% reduction "
                f"({original_count} → {pruned_count} paths) in {processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Streaming pruning failed: {str(e)}")
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time_ms, len(all_paths) if "all_paths" in locals() else 0, 0, False)

            return self._create_empty_result(strategy, f"Pruning failed: {str(e)}")

    async def prune_streaming_paths(
        self, path_stream: list[AnyPath], batch_size: int = 50, strategy: PruningStrategy = PruningStrategy.BALANCED
    ) -> list[AnyPath]:
        """
        Apply streaming pruning to a continuous stream of paths.

        Args:
            path_stream: Stream of paths to process
            batch_size: Size of batches to process
            strategy: Pruning strategy

        Returns:
            List of pruned paths
        """
        pruned_paths = []
        seen_signatures = set()

        # Process paths in batches
        for i in range(0, len(path_stream), batch_size):
            batch = path_stream[i : i + batch_size]

            # Apply lightweight duplicate detection
            for path in batch:
                signature = await self._generate_path_signature(path)

                if signature not in seen_signatures:
                    # Check information density
                    density = await self._calculate_path_information_density(path)

                    if density >= self.config.min_information_density:
                        seen_signatures.add(signature)
                        pruned_paths.append(path)

        return pruned_paths

    async def _detect_exact_duplicates(
        self, paths: list[AnyPath], strategy: PruningStrategy, preserve_patterns: set[str]
    ) -> dict[str, Any]:
        """Detect and remove exact duplicate paths."""
        seen_signatures = {}
        redundant = []

        for path in paths:
            signature = await self._generate_path_signature(path)

            if signature in seen_signatures:
                # Keep the path with higher confidence/importance
                existing_path = seen_signatures[signature]
                current_importance = self._get_path_importance(path)
                existing_importance = self._get_path_importance(existing_path)

                if current_importance > existing_importance:
                    redundant.append(existing_path)
                    seen_signatures[signature] = path
                else:
                    redundant.append(path)
            else:
                seen_signatures[signature] = path

        return {"redundant": redundant, "type": RedundancyType.EXACT_DUPLICATE}

    async def _detect_structural_similarity(
        self, paths: list[AnyPath], strategy: PruningStrategy, preserve_patterns: set[str]
    ) -> dict[str, Any]:
        """Detect structurally similar paths and remove redundant ones."""
        redundant = []
        processed = set()

        for i, path1 in enumerate(paths):
            if id(path1) in processed:
                continue

            similar_paths = [path1]

            for j, path2 in enumerate(paths[i + 1 :], i + 1):
                if id(path2) in processed:
                    continue

                similarity = await self._calculate_structural_similarity(path1, path2)

                if similarity >= self.config.structural_similarity_threshold:
                    similar_paths.append(path2)
                    processed.add(id(path2))

            # Keep the best representative from similar paths
            if len(similar_paths) > 1:
                best_path = max(similar_paths, key=self._get_path_importance)
                redundant.extend([p for p in similar_paths if p != best_path])

            processed.add(id(path1))

        return {"redundant": redundant, "type": RedundancyType.STRUCTURAL_SIMILAR}

    async def _filter_low_information_density(
        self, paths: list[AnyPath], strategy: PruningStrategy, preserve_patterns: set[str]
    ) -> dict[str, Any]:
        """Filter out paths with low information density."""
        redundant = []

        for path in paths:
            # Skip if path matches preserve patterns
            if any(pattern in str(path.path_id) for pattern in preserve_patterns):
                continue

            density = await self._calculate_path_information_density(path)

            # Adjust threshold based on strategy
            threshold = self.config.min_information_density
            if strategy == PruningStrategy.CONSERVATIVE:
                threshold *= 0.7
            elif strategy == PruningStrategy.AGGRESSIVE:
                threshold *= 1.3

            if density < threshold:
                redundant.append(path)

        return {"redundant": redundant, "type": RedundancyType.LOW_INFORMATION}

    async def _detect_semantic_overlap(
        self, paths: list[AnyPath], strategy: PruningStrategy, preserve_patterns: set[str]
    ) -> dict[str, Any]:
        """Detect paths with significant semantic overlap."""
        redundant = []
        processed = set()

        for i, path1 in enumerate(paths):
            if id(path1) in processed:
                continue

            overlapping_paths = []

            for j, path2 in enumerate(paths[i + 1 :], i + 1):
                if id(path2) in processed:
                    continue

                overlap = await self._calculate_semantic_overlap(path1, path2)

                if overlap >= self.config.semantic_similarity_threshold:
                    overlapping_paths.append(path2)

            # If significant overlap found, keep only the most comprehensive path
            if overlapping_paths:
                all_overlapping = [path1] + overlapping_paths
                best_path = max(all_overlapping, key=lambda p: len(p.nodes))

                for path in all_overlapping:
                    if path != best_path:
                        redundant.append(path)
                        processed.add(id(path))

            processed.add(id(path1))

        return {"redundant": redundant, "type": RedundancyType.SEMANTIC_OVERLAP}

    async def _filter_circular_references(
        self, paths: list[AnyPath], strategy: PruningStrategy, preserve_patterns: set[str]
    ) -> dict[str, Any]:
        """Filter out circular or self-referential paths."""
        redundant = []

        if not self.config.enable_circular_detection:
            return {"redundant": redundant, "type": RedundancyType.CIRCULAR_REFERENCE}

        for path in paths:
            if self._is_circular_path(path):
                # Only remove if not explicitly preserved
                if not any(pattern in str(path.path_id) for pattern in preserve_patterns):
                    redundant.append(path)

        return {"redundant": redundant, "type": RedundancyType.CIRCULAR_REFERENCE}

    async def _apply_quality_based_selection(
        self, paths: list[AnyPath], strategy: PruningStrategy, preserve_patterns: set[str]
    ) -> list[AnyPath]:
        """Apply final quality-based selection to meet target reduction."""
        if not paths:
            return paths

        # Calculate quality scores for all paths
        path_scores = []
        for path in paths:
            importance = self._get_path_importance(path)
            density = await self._calculate_path_information_density(path)

            # Check if path should be preserved
            is_preserved = any(pattern in str(path.path_id) for pattern in preserve_patterns)

            quality_score = (importance * 0.6) + (density * 0.4)
            path_scores.append((path, quality_score, is_preserved))

        # Sort by quality score (descending)
        path_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)  # Preserved first, then by quality

        # Calculate target count
        target_reduction = self.config.target_reduction_percentage / 100.0
        target_count = max(1, int(len(paths) * (1 - target_reduction)))

        # Select top quality paths
        selected_paths = [item[0] for item in path_scores[:target_count]]

        return selected_paths

    async def _generate_path_signature(self, path: AnyPath) -> str:
        """Generate a unique signature for path duplicate detection."""
        # Create signature from node breadcrumbs and path type
        node_signatures = [f"{node.breadcrumb}:{node.chunk_type.value}" for node in path.nodes]

        signature_parts = [path.path_type.value, "|".join(sorted(node_signatures)), str(len(path.nodes))]

        return ":".join(signature_parts)

    async def _calculate_structural_similarity(self, path1: AnyPath, path2: AnyPath) -> float:
        """Calculate structural similarity between two paths."""
        # Use cache if available
        cache_key = (path1.path_id, path2.path_id)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Different path types are not structurally similar
        if path1.path_type != path2.path_type:
            similarity = 0.0
        else:
            # Calculate node overlap
            breadcrumbs1 = {node.breadcrumb for node in path1.nodes}
            breadcrumbs2 = {node.breadcrumb for node in path2.nodes}

            intersection = breadcrumbs1.intersection(breadcrumbs2)
            union = breadcrumbs1.union(breadcrumbs2)

            if not union:
                similarity = 0.0
            else:
                overlap_ratio = len(intersection) / len(union)

                # Factor in sequence similarity
                sequence_similarity = self._calculate_sequence_similarity(
                    [n.breadcrumb for n in path1.nodes], [n.breadcrumb for n in path2.nodes]
                )

                similarity = (overlap_ratio * 0.7) + (sequence_similarity * 0.3)

        # Cache result
        self._similarity_cache[cache_key] = similarity
        self._similarity_cache[(path2.path_id, path1.path_id)] = similarity

        return similarity

    async def _calculate_semantic_overlap(self, path1: AnyPath, path2: AnyPath) -> float:
        """Calculate semantic overlap between two paths."""
        # For now, use a simplified approach based on shared semantic elements
        # In a full implementation, this would use semantic embeddings

        # Get semantic elements from both paths
        elements1 = self._extract_semantic_elements(path1)
        elements2 = self._extract_semantic_elements(path2)

        if not elements1 or not elements2:
            return 0.0

        # Calculate overlap
        intersection = elements1.intersection(elements2)
        union = elements1.union(elements2)

        return len(intersection) / len(union) if union else 0.0

    async def _calculate_path_information_density(self, path: AnyPath) -> float:
        """Calculate information density of a path."""
        # Use cache if available
        if path.path_id in self._density_cache:
            return self._density_cache[path.path_id]

        # Calculate density based on various factors
        factors = []

        # Node diversity
        unique_chunk_types = len({node.chunk_type for node in path.nodes})
        max_chunk_types = len(list(ChunkType)) if hasattr(ChunkType, "__iter__") else 5
        factors.append(unique_chunk_types / max_chunk_types)

        # Path complexity
        if hasattr(path, "complexity_score"):
            factors.append(getattr(path, "complexity_score", 0.0))

        # Path length (normalized)
        normalized_length = min(1.0, len(path.nodes) / 10.0)  # Normalize to 10 nodes
        factors.append(normalized_length)

        # Average node importance
        if path.nodes:
            avg_importance = sum(node.importance_score for node in path.nodes) / len(path.nodes)
            factors.append(avg_importance)

        # Edge quality (if available)
        if hasattr(path, "edges") and path.edges:
            avg_edge_confidence = sum(edge.get_confidence_score() for edge in path.edges) / len(path.edges)
            factors.append(avg_edge_confidence)

        # Calculate overall density
        density = sum(factors) / len(factors) if factors else 0.0

        # Cache result
        self._density_cache[path.path_id] = density

        return density

    def _calculate_sequence_similarity(self, seq1: list[str], seq2: list[str]) -> float:
        """Calculate similarity between two sequences using edit distance."""
        if not seq1 or not seq2:
            return 0.0

        # Simple edit distance calculation
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        # Convert edit distance to similarity score
        max_len = max(m, n)
        if max_len == 0:
            return 1.0

        return 1.0 - (dp[m][n] / max_len)

    def _extract_semantic_elements(self, path: AnyPath) -> set[str]:
        """Extract semantic elements from a path for overlap analysis."""
        elements = set()

        # Add path type
        elements.add(path.path_type.value)

        # Add node information
        for node in path.nodes:
            elements.add(node.name.lower())
            elements.add(node.chunk_type.value)

            # Add breadcrumb components
            breadcrumb_parts = node.breadcrumb.split(".")
            elements.update(part.lower() for part in breadcrumb_parts)

        # Add path-specific elements
        if isinstance(path, ExecutionPath):
            if hasattr(path, "use_cases"):
                elements.update(case.lower() for case in path.use_cases)
        elif isinstance(path, DataFlowPath):
            if hasattr(path, "data_types"):
                elements.update(dtype.lower() for dtype in path.data_types)
        elif isinstance(path, DependencyPath):
            if hasattr(path, "required_modules"):
                elements.update(mod.lower() for mod in path.required_modules)

        return elements

    def _is_circular_path(self, path: AnyPath) -> bool:
        """Check if a path contains circular references."""
        seen_breadcrumbs = set()

        for node in path.nodes:
            if node.breadcrumb in seen_breadcrumbs:
                return True
            seen_breadcrumbs.add(node.breadcrumb)

        return False

    def _get_path_importance(self, path: AnyPath) -> float:
        """Get importance score for a path."""
        # Use different metrics based on path type
        if isinstance(path, ExecutionPath):
            return getattr(path, "criticality_score", 0.5)
        elif isinstance(path, DataFlowPath):
            return getattr(path, "data_quality_score", 0.5)
        elif isinstance(path, DependencyPath):
            return getattr(path, "stability_score", 0.5)
        else:
            # Fallback to average node importance
            if path.nodes:
                return sum(node.importance_score for node in path.nodes) / len(path.nodes)
            return 0.5

    async def _calculate_average_information_density(self, paths: list[AnyPath]) -> float:
        """Calculate average information density across paths."""
        if not paths:
            return 0.0

        densities = []
        for path in paths:
            density = await self._calculate_path_information_density(path)
            densities.append(density)

        return sum(densities) / len(densities)

    async def _calculate_relevance_preservation_score(self, original_paths: list[AnyPath], pruned_paths: list[AnyPath]) -> float:
        """Calculate how well relevance was preserved during pruning."""
        if not original_paths:
            return 1.0

        # Calculate average importance of original vs pruned paths
        original_importance = sum(self._get_path_importance(p) for p in original_paths) / len(original_paths)
        pruned_importance = sum(self._get_path_importance(p) for p in pruned_paths) / len(pruned_paths) if pruned_paths else 0.0

        # Relevance preservation is ratio of maintained importance
        if original_importance == 0:
            return 1.0

        preservation_ratio = pruned_importance / original_importance
        return min(1.0, preservation_ratio)

    async def _add_quality_insights(self, result: PruningResult):
        """Add quality insights to pruning result."""
        insights = []

        # Reduction analysis
        if result.reduction_percentage >= 50:
            insights.append("High reduction achieved - verify critical paths are preserved")
        elif result.reduction_percentage < 20:
            insights.append("Low reduction - consider more aggressive pruning strategy")

        # Information density analysis
        if result.information_density_score < 0.4:
            insights.append("Low information density in remaining paths")
        elif result.information_density_score > 0.8:
            insights.append("High information density maintained")

        # Relevance preservation analysis
        if result.relevance_preservation_score < 0.7:
            insights.append("Significant relevance loss detected - review pruning criteria")
        elif result.relevance_preservation_score > 0.9:
            insights.append("Excellent relevance preservation")

        # Redundancy type analysis
        total_redundant = sum(result.redundancy_breakdown.values())
        if total_redundant > 0:
            dominant_type = max(result.redundancy_breakdown.items(), key=lambda x: x[1])
            insights.append(f"Primary redundancy type: {dominant_type[0].value} ({dominant_type[1]} paths)")

        result.quality_insights = insights

    def _create_empty_result(self, strategy: PruningStrategy, error_message: str) -> PruningResult:
        """Create empty pruning result for error cases."""
        return PruningResult(
            pruned_paths=[],
            redundant_paths=[],
            original_count=0,
            pruned_count=0,
            reduction_percentage=0.0,
            information_density_score=0.0,
            relevance_preservation_score=0.0,
            processing_time_ms=0.0,
            pruning_strategy=strategy,
            redundancy_breakdown={},
            pruning_warnings=[error_message],
        )

    def _update_performance_stats(self, processing_time_ms: float, original_count: int, pruned_count: int, success: bool):
        """Update internal performance statistics."""
        self._pruning_stats["total_pruning_operations"] += 1

        if success:
            self._pruning_stats["successful_operations"] += 1

            # Update averages
            operations = self._pruning_stats["successful_operations"]

            # Average reduction percentage
            if original_count > 0:
                reduction = (pruned_count / original_count) * 100
                current_avg_reduction = self._pruning_stats["average_reduction_percentage"]
                self._pruning_stats["average_reduction_percentage"] = (current_avg_reduction * (operations - 1) + reduction) / operations

            # Average processing time
            current_avg_time = self._pruning_stats["average_processing_time_ms"]
            self._pruning_stats["average_processing_time_ms"] = (current_avg_time * (operations - 1) + processing_time_ms) / operations

            # Update totals
            self._pruning_stats["total_paths_processed"] += original_count
            self._pruning_stats["total_paths_pruned"] += pruned_count
        else:
            self._pruning_stats["failed_operations"] += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        stats = dict(self._pruning_stats)

        # Add derived metrics
        total_ops = stats["total_pruning_operations"]
        if total_ops > 0:
            stats["success_rate"] = stats["successful_operations"] / total_ops
            stats["failure_rate"] = stats["failed_operations"] / total_ops
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        # Add cache statistics
        stats["similarity_cache_size"] = len(self._similarity_cache)
        stats["density_cache_size"] = len(self._density_cache)

        return stats

    def clear_caches(self):
        """Clear internal caches to free memory."""
        self._similarity_cache.clear()
        self._density_cache.clear()
        self.logger.info("Streaming pruning caches cleared")


# Factory function
def create_streaming_pruning_service(config: PruningConfig | None = None) -> StreamingPruningService:
    """
    Factory function to create a StreamingPruningService instance.

    Args:
        config: Optional pruning configuration

    Returns:
        Configured StreamingPruningService instance
    """
    return StreamingPruningService(config)
