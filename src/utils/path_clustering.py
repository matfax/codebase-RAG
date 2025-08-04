"""
Path Clustering Service for Wave 2.0 Task 2.4 - Path Clustering and Representative Selection.

This service implements sophisticated clustering algorithms to group similar paths
and select representative paths from each cluster, enabling efficient path organization
and reduced redundancy while maintaining comprehensive coverage.
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

from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathNode,
    RelationalPathCollection,
)


class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""

    HIERARCHICAL = "hierarchical"  # Hierarchical clustering with linkage
    KMEANS = "kmeans"  # K-means clustering for path grouping
    DBSCAN = "dbscan"  # Density-based clustering
    SEMANTIC = "semantic"  # Semantic similarity clustering
    HYBRID = "hybrid"  # Combination of multiple algorithms


class ClusteringStrategy(Enum):
    """Different clustering strategies."""

    CONSERVATIVE = "conservative"  # Fewer, larger clusters
    BALANCED = "balanced"  # Balanced cluster sizes
    AGGRESSIVE = "aggressive"  # Many smaller, specific clusters


class RepresentativeSelection(Enum):
    """Methods for selecting representative paths from clusters."""

    CENTROID = "centroid"  # Path closest to cluster centroid
    HIGHEST_IMPORTANCE = "highest_importance"  # Path with highest importance score
    MOST_CONNECTED = "most_connected"  # Path with most connections
    COMPOSITE_SCORE = "composite_score"  # Combination of multiple factors


@dataclass
class PathCluster:
    """Represents a cluster of similar paths."""

    cluster_id: str
    cluster_name: str
    paths: list[AnyPath]
    representative_path: AnyPath | None = None

    # Cluster characteristics
    cluster_center: dict[str, float] = field(default_factory=dict)
    cohesion_score: float = 0.0  # How similar paths are within cluster
    separation_score: float = 0.0  # How different this cluster is from others
    coverage_score: float = 0.0  # How much of the search space this cluster covers

    # Quality metrics
    intra_cluster_similarity: float = 0.0  # Average similarity within cluster
    silhouette_score: float = 0.0  # Cluster quality metric
    cluster_stability: float = 0.0  # How stable the cluster is

    # Metadata
    dominant_path_types: list[str] = field(default_factory=list)
    common_breadcrumbs: set[str] = field(default_factory=set)
    cluster_tags: set[str] = field(default_factory=set)
    creation_time: float = field(default_factory=time.time)

    def get_cluster_size(self) -> int:
        """Get the number of paths in this cluster."""
        return len(self.paths)

    def get_average_importance(self) -> float:
        """Calculate average importance score of paths in cluster."""
        if not self.paths:
            return 0.0

        total_importance = 0.0
        for path in self.paths:
            if isinstance(path, ExecutionPath):
                total_importance += getattr(path, "criticality_score", 0.5)
            elif isinstance(path, DataFlowPath):
                total_importance += getattr(path, "data_quality_score", 0.5)
            elif isinstance(path, DependencyPath):
                total_importance += getattr(path, "stability_score", 0.5)
            else:
                total_importance += 0.5

        return total_importance / len(self.paths)

    def get_path_type_distribution(self) -> dict[str, int]:
        """Get distribution of path types in cluster."""
        distribution = defaultdict(int)
        for path in self.paths:
            distribution[path.path_type.value] += 1
        return dict(distribution)

    def is_homogeneous(self, threshold: float = 0.8) -> bool:
        """Check if cluster is homogeneous (single dominant path type)."""
        distribution = self.get_path_type_distribution()
        if not distribution:
            return False

        max_count = max(distribution.values())
        return (max_count / len(self.paths)) >= threshold


@dataclass
class ClusteringResult:
    """Result of path clustering operation."""

    # Core results
    clusters: list[PathCluster]
    representative_paths: list[AnyPath]

    # Statistics
    original_path_count: int
    clustered_path_count: int
    cluster_count: int
    reduction_percentage: float

    # Quality metrics
    average_silhouette_score: float
    overall_cohesion_score: float
    cluster_separation_score: float
    coverage_completeness: float

    # Processing metadata
    processing_time_ms: float
    clustering_algorithm: ClusteringAlgorithm
    clustering_strategy: ClusteringStrategy
    representative_selection: RepresentativeSelection

    # Insights and warnings
    clustering_warnings: list[str] = field(default_factory=list)
    quality_insights: list[str] = field(default_factory=list)

    def get_effectiveness_score(self) -> float:
        """Calculate overall clustering effectiveness."""
        # Balance reduction with quality preservation
        reduction_score = min(1.0, self.reduction_percentage / 50.0)  # Target 50% reduction
        quality_score = (self.average_silhouette_score + self.overall_cohesion_score) / 2.0
        coverage_score = self.coverage_completeness

        return (reduction_score * 0.4) + (quality_score * 0.4) + (coverage_score * 0.2)

    def get_cluster_by_id(self, cluster_id: str) -> PathCluster | None:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None


@dataclass
class ClusteringConfig:
    """Configuration for path clustering operations."""

    # Algorithm selection
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HYBRID
    strategy: ClusteringStrategy = ClusteringStrategy.BALANCED
    representative_selection: RepresentativeSelection = RepresentativeSelection.COMPOSITE_SCORE

    # Clustering parameters
    target_cluster_count: int | None = None  # Auto-determine if None
    min_cluster_size: int = 2  # Minimum paths per cluster
    max_cluster_size: int = 20  # Maximum paths per cluster
    similarity_threshold: float = 0.7  # Minimum similarity for clustering

    # Quality thresholds
    min_silhouette_score: float = 0.3  # Minimum acceptable silhouette score
    min_cohesion_score: float = 0.5  # Minimum cluster cohesion
    min_separation_score: float = 0.4  # Minimum cluster separation

    # Performance settings
    max_processing_time_ms: float = 30000  # 30 second timeout
    enable_parallel_clustering: bool = True  # Parallel processing
    cache_similarity_matrix: bool = True  # Cache similarity calculations

    # Advanced options
    enable_hierarchical_merging: bool = True  # Merge similar clusters
    enable_outlier_detection: bool = True  # Detect and handle outliers
    preserve_path_diversity: bool = True  # Maintain diverse path types
    adaptive_cluster_sizing: bool = True  # Adapt cluster sizes based on data


class PathClusteringService:
    """
    Advanced path clustering service that groups similar paths and selects
    representative paths to reduce redundancy while maintaining coverage.

    Key features:
    - Multiple clustering algorithms (hierarchical, k-means, DBSCAN, semantic)
    - Intelligent representative selection
    - Quality-aware cluster evaluation
    - Adaptive cluster sizing
    - Performance optimization with caching
    """

    def __init__(self, config: ClusteringConfig = None):
        """
        Initialize the path clustering service.

        Args:
            config: Clustering configuration options
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ClusteringConfig()

        # Performance tracking
        self._clustering_stats = {
            "total_clustering_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time_ms": 0.0,
            "average_reduction_percentage": 0.0,
            "total_paths_clustered": 0,
            "total_clusters_created": 0,
        }

        # Caching for performance
        self._similarity_cache: dict[tuple[str, str], float] = {}
        self._feature_cache: dict[str, dict[str, float]] = {}

    async def cluster_path_collection(
        self, collection: RelationalPathCollection, custom_config: ClusteringConfig | None = None
    ) -> ClusteringResult:
        """
        Cluster paths in a path collection and select representatives.

        Args:
            collection: Path collection to cluster
            custom_config: Optional custom configuration

        Returns:
            ClusteringResult with clusters and representatives
        """
        start_time = time.time()
        config = custom_config or self.config

        try:
            self.logger.info("Starting path clustering operation")

            # Get all paths from collection
            all_paths = collection.execution_paths + collection.data_flow_paths + collection.dependency_paths

            if not all_paths:
                return self._create_empty_result(config, "No paths to cluster")

            original_count = len(all_paths)
            self.logger.info(f"Clustering {original_count} paths")

            # Extract features for clustering
            self.logger.debug("Extracting path features")
            path_features = await self._extract_path_features(all_paths)

            # Determine optimal cluster count
            if config.target_cluster_count is None:
                target_clusters = await self._determine_optimal_cluster_count(all_paths, path_features, config)
            else:
                target_clusters = config.target_cluster_count

            self.logger.info(f"Target cluster count: {target_clusters}")

            # Apply clustering algorithm
            clusters = await self._apply_clustering_algorithm(all_paths, path_features, target_clusters, config)

            # Select representative paths
            representative_paths = await self._select_representative_paths(clusters, config)

            # Calculate quality metrics
            quality_metrics = await self._calculate_clustering_quality(clusters, path_features, config)

            # Create clustering result
            processing_time_ms = (time.time() - start_time) * 1000
            reduction_percentage = ((original_count - len(representative_paths)) / original_count) * 100

            result = ClusteringResult(
                clusters=clusters,
                representative_paths=representative_paths,
                original_path_count=original_count,
                clustered_path_count=len(representative_paths),
                cluster_count=len(clusters),
                reduction_percentage=reduction_percentage,
                average_silhouette_score=quality_metrics["silhouette_score"],
                overall_cohesion_score=quality_metrics["cohesion_score"],
                cluster_separation_score=quality_metrics["separation_score"],
                coverage_completeness=quality_metrics["coverage_score"],
                processing_time_ms=processing_time_ms,
                clustering_algorithm=config.algorithm,
                clustering_strategy=config.strategy,
                representative_selection=config.representative_selection,
            )

            # Add quality insights
            await self._add_clustering_insights(result, config)

            # Update performance statistics
            self._update_performance_stats(processing_time_ms, original_count, len(clusters), True)

            self.logger.info(
                f"Clustering completed: {len(clusters)} clusters, " f"{reduction_percentage:.1f}% reduction in {processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Path clustering failed: {str(e)}")
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time_ms, len(all_paths) if "all_paths" in locals() else 0, 0, False)
            return self._create_empty_result(config, f"Clustering failed: {str(e)}")

    async def _extract_path_features(self, paths: list[AnyPath]) -> dict[str, dict[str, float]]:
        """Extract numerical features from paths for clustering."""
        features = {}

        for path in paths:
            if path.path_id in self._feature_cache:
                features[path.path_id] = self._feature_cache[path.path_id]
                continue

            path_features = {}

            # Basic path characteristics
            path_features["path_length"] = len(path.nodes)
            path_features["edge_count"] = len(getattr(path, "edges", []))
            path_features["path_type_numeric"] = self._encode_path_type(path)

            # Node characteristics
            if path.nodes:
                path_features["avg_node_importance"] = sum(node.importance_score for node in path.nodes) / len(path.nodes)
                path_features["node_type_diversity"] = len(set(node.chunk_type for node in path.nodes))
            else:
                path_features["avg_node_importance"] = 0.0
                path_features["node_type_diversity"] = 0.0

            # Path-specific features
            if isinstance(path, ExecutionPath):
                path_features["complexity_score"] = getattr(path, "complexity_score", 0.0)
                path_features["criticality_score"] = getattr(path, "criticality_score", 0.0)
                path_features["max_depth"] = getattr(path, "max_depth", 0)
            elif isinstance(path, DataFlowPath):
                path_features["data_quality_score"] = getattr(path, "data_quality_score", 0.0)
                path_features["transformation_count"] = len(getattr(path, "transformations", []))
            elif isinstance(path, DependencyPath):
                path_features["stability_score"] = getattr(path, "stability_score", 0.0)
                path_features["coupling_strength"] = getattr(path, "coupling_strength", 0.0)
                path_features["module_count"] = len(getattr(path, "required_modules", []))

            # Breadcrumb-based features
            breadcrumbs = [node.breadcrumb for node in path.nodes]
            path_features["avg_breadcrumb_depth"] = (
                sum(breadcrumb.count(".") for breadcrumb in breadcrumbs) / len(breadcrumbs) if breadcrumbs else 0.0
            )

            # File-based features
            file_paths = list(set(node.file_path for node in path.nodes))
            path_features["file_diversity"] = len(file_paths)

            features[path.path_id] = path_features

            # Cache for future use
            if self.config.cache_similarity_matrix:
                self._feature_cache[path.path_id] = path_features

        return features

    def _encode_path_type(self, path: AnyPath) -> float:
        """Encode path type as numerical value."""
        type_mapping = {
            "execution_path": 1.0,
            "data_flow": 2.0,
            "dependency_path": 3.0,
            "control_flow": 4.0,
            "async_execution": 5.0,
            "data_dependency": 6.0,
            "state_transition": 7.0,
            "inheritance_path": 8.0,
            "composition_path": 9.0,
        }
        return type_mapping.get(path.path_type.value, 0.0)

    async def _determine_optimal_cluster_count(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]], config: ClusteringConfig
    ) -> int:
        """Determine optimal number of clusters using elbow method and silhouette analysis."""
        path_count = len(paths)

        # Basic heuristics
        if path_count <= 5:
            return min(2, path_count)
        elif path_count <= 20:
            return max(2, path_count // 4)
        elif path_count <= 100:
            return max(3, path_count // 8)
        else:
            return max(5, min(20, path_count // 15))

    async def _apply_clustering_algorithm(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]], target_clusters: int, config: ClusteringConfig
    ) -> list[PathCluster]:
        """Apply the selected clustering algorithm."""
        if config.algorithm == ClusteringAlgorithm.HIERARCHICAL:
            return await self._hierarchical_clustering(paths, features, target_clusters, config)
        elif config.algorithm == ClusteringAlgorithm.SEMANTIC:
            return await self._semantic_clustering(paths, features, target_clusters, config)
        elif config.algorithm == ClusteringAlgorithm.HYBRID:
            return await self._hybrid_clustering(paths, features, target_clusters, config)
        else:
            # Default to semantic clustering
            return await self._semantic_clustering(paths, features, target_clusters, config)

    async def _hierarchical_clustering(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]], target_clusters: int, config: ClusteringConfig
    ) -> list[PathCluster]:
        """Perform hierarchical clustering on paths."""
        self.logger.debug("Applying hierarchical clustering")

        # Calculate similarity matrix
        similarity_matrix = await self._calculate_similarity_matrix(paths, features)

        # Simple agglomerative clustering implementation
        clusters = []
        path_to_cluster = {}

        # Initialize each path as its own cluster
        for i, path in enumerate(paths):
            cluster = PathCluster(cluster_id=f"cluster_{i}", cluster_name=f"Cluster {i}", paths=[path])
            clusters.append(cluster)
            path_to_cluster[path.path_id] = i

        # Merge clusters until we reach target count
        while len(clusters) > target_clusters and len(clusters) > 1:
            # Find most similar clusters to merge
            max_similarity = -1.0
            merge_indices = (0, 1)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = await self._calculate_cluster_similarity(clusters[i], clusters[j], similarity_matrix)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        merge_indices = (i, j)

            # Merge the most similar clusters
            if max_similarity >= config.similarity_threshold:
                cluster_i, cluster_j = merge_indices
                merged_cluster = PathCluster(
                    cluster_id=f"merged_{uuid.uuid4().hex[:8]}",
                    cluster_name=f"Merged Cluster {len(clusters)}",
                    paths=clusters[cluster_i].paths + clusters[cluster_j].paths,
                )

                # Remove old clusters and add merged one
                clusters = [c for idx, c in enumerate(clusters) if idx not in merge_indices]
                clusters.append(merged_cluster)
            else:
                break  # No more similar clusters to merge

        # Update cluster metadata
        for cluster in clusters:
            await self._update_cluster_metadata(cluster, features)

        return clusters

    async def _semantic_clustering(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]], target_clusters: int, config: ClusteringConfig
    ) -> list[PathCluster]:
        """Perform semantic clustering based on path content and structure."""
        self.logger.debug("Applying semantic clustering")

        # Group paths by type first
        type_groups = defaultdict(list)
        for path in paths:
            type_groups[path.path_type.value].append(path)

        clusters = []
        cluster_id_counter = 0

        # Cluster within each type group
        for path_type, type_paths in type_groups.items():
            if len(type_paths) <= config.min_cluster_size:
                # Small group becomes single cluster
                cluster = PathCluster(
                    cluster_id=f"semantic_{cluster_id_counter}", cluster_name=f"{path_type.title()} Cluster", paths=type_paths
                )
                await self._update_cluster_metadata(cluster, features)
                clusters.append(cluster)
                cluster_id_counter += 1
            else:
                # Apply similarity-based clustering within type
                type_clusters = await self._cluster_by_similarity(type_paths, features, max(1, len(type_paths) // 3), config)

                for i, type_cluster in enumerate(type_clusters):
                    type_cluster.cluster_id = f"semantic_{cluster_id_counter}"
                    type_cluster.cluster_name = f"{path_type.title()} Cluster {i + 1}"
                    await self._update_cluster_metadata(type_cluster, features)
                    clusters.append(type_cluster)
                    cluster_id_counter += 1

        return clusters

    async def _hybrid_clustering(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]], target_clusters: int, config: ClusteringConfig
    ) -> list[PathCluster]:
        """Apply hybrid clustering combining multiple approaches."""
        self.logger.debug("Applying hybrid clustering")

        # Start with semantic clustering
        semantic_clusters = await self._semantic_clustering(paths, features, target_clusters * 2, config)

        # If we have too many clusters, merge similar ones using hierarchical approach
        if len(semantic_clusters) > target_clusters:
            # Calculate inter-cluster similarities
            similarity_matrix = {}
            for i, cluster_i in enumerate(semantic_clusters):
                for j, cluster_j in enumerate(semantic_clusters[i + 1 :], i + 1):
                    similarity = await self._calculate_inter_cluster_similarity(cluster_i, cluster_j)
                    similarity_matrix[(i, j)] = similarity

            # Merge most similar clusters until target is reached
            clusters = semantic_clusters.copy()
            while len(clusters) > target_clusters and len(clusters) > 1:
                # Find most similar pair
                max_similarity = -1.0
                merge_pair = None

                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        similarity = await self._calculate_inter_cluster_similarity(clusters[i], clusters[j])
                        if similarity > max_similarity:
                            max_similarity = similarity
                            merge_pair = (i, j)

                if merge_pair and max_similarity >= config.similarity_threshold:
                    i, j = merge_pair
                    merged_cluster = PathCluster(
                        cluster_id=f"hybrid_{uuid.uuid4().hex[:8]}",
                        cluster_name=f"Hybrid Cluster {len(clusters)}",
                        paths=clusters[i].paths + clusters[j].paths,
                    )
                    await self._update_cluster_metadata(merged_cluster, features)

                    clusters = [c for idx, c in enumerate(clusters) if idx not in merge_pair]
                    clusters.append(merged_cluster)
                else:
                    break

            return clusters
        else:
            return semantic_clusters

    async def _cluster_by_similarity(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]], target_clusters: int, config: ClusteringConfig
    ) -> list[PathCluster]:
        """Cluster paths by feature similarity."""
        if len(paths) <= target_clusters:
            # Each path becomes its own cluster
            return [
                PathCluster(cluster_id=f"sim_{i}", cluster_name=f"Similarity Cluster {i}", paths=[path]) for i, path in enumerate(paths)
            ]

        # Simple k-means style clustering
        clusters = []

        # Randomly initialize cluster centers
        import random

        random.shuffle(paths)
        initial_paths = paths[:target_clusters]

        for i, center_path in enumerate(initial_paths):
            cluster = PathCluster(cluster_id=f"sim_{i}", cluster_name=f"Similarity Cluster {i}", paths=[center_path])
            clusters.append(cluster)

        # Assign remaining paths to closest clusters
        for path in paths[target_clusters:]:
            best_cluster = None
            best_similarity = -1.0

            for cluster in clusters:
                # Calculate similarity to cluster center (first path)
                center_path = cluster.paths[0]
                similarity = await self._calculate_path_similarity(path, center_path, features)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if best_cluster and best_similarity >= config.similarity_threshold:
                best_cluster.paths.append(path)

        # Remove empty clusters
        clusters = [c for c in clusters if c.paths]

        return clusters

    async def _calculate_similarity_matrix(
        self, paths: list[AnyPath], features: dict[str, dict[str, float]]
    ) -> dict[tuple[str, str], float]:
        """Calculate pairwise similarity matrix for paths."""
        similarity_matrix = {}

        for i, path_i in enumerate(paths):
            for j, path_j in enumerate(paths[i + 1 :], i + 1):
                cache_key = (path_i.path_id, path_j.path_id)

                if cache_key in self._similarity_cache:
                    similarity = self._similarity_cache[cache_key]
                else:
                    similarity = await self._calculate_path_similarity(path_i, path_j, features)
                    if self.config.cache_similarity_matrix:
                        self._similarity_cache[cache_key] = similarity
                        self._similarity_cache[(path_j.path_id, path_i.path_id)] = similarity

                similarity_matrix[(i, j)] = similarity

        return similarity_matrix

    async def _calculate_path_similarity(self, path1: AnyPath, path2: AnyPath, features: dict[str, dict[str, float]]) -> float:
        """Calculate similarity between two paths."""
        # Different path types have lower similarity
        if path1.path_type != path2.path_type:
            return 0.2  # Some base similarity for different types

        features1 = features.get(path1.path_id, {})
        features2 = features.get(path2.path_id, {})

        if not features1 or not features2:
            return 0.0

        # Calculate feature-based similarity
        feature_similarities = []

        common_features = set(features1.keys()).intersection(set(features2.keys()))
        for feature in common_features:
            val1 = features1[feature]
            val2 = features2[feature]

            # Normalize and calculate similarity
            if val1 == 0 and val2 == 0:
                feature_sim = 1.0
            elif val1 == 0 or val2 == 0:
                feature_sim = 0.0
            else:
                max_val = max(abs(val1), abs(val2))
                min_val = min(abs(val1), abs(val2))
                feature_sim = min_val / max_val if max_val > 0 else 1.0

            feature_similarities.append(feature_sim)

        # Calculate structural similarity (shared nodes/breadcrumbs)
        breadcrumbs1 = {node.breadcrumb for node in path1.nodes}
        breadcrumbs2 = {node.breadcrumb for node in path2.nodes}

        intersection = breadcrumbs1.intersection(breadcrumbs2)
        union = breadcrumbs1.union(breadcrumbs2)

        structural_similarity = len(intersection) / len(union) if union else 0.0

        # Combine feature and structural similarities
        if feature_similarities:
            feature_avg = sum(feature_similarities) / len(feature_similarities)
            return (feature_avg * 0.7) + (structural_similarity * 0.3)
        else:
            return structural_similarity

    async def _calculate_cluster_similarity(
        self, cluster1: PathCluster, cluster2: PathCluster, similarity_matrix: dict[tuple[int, int], float]
    ) -> float:
        """Calculate similarity between two clusters."""
        return await self._calculate_inter_cluster_similarity(cluster1, cluster2)

    async def _calculate_inter_cluster_similarity(self, cluster1: PathCluster, cluster2: PathCluster) -> float:
        """Calculate similarity between two clusters using average linkage."""
        if not cluster1.paths or not cluster2.paths:
            return 0.0

        total_similarity = 0.0
        comparison_count = 0

        # Calculate average similarity between all pairs
        for path1 in cluster1.paths:
            for path2 in cluster2.paths:
                cache_key = (path1.path_id, path2.path_id)
                if cache_key in self._similarity_cache:
                    similarity = self._similarity_cache[cache_key]
                else:
                    # Simple similarity based on shared characteristics
                    similarity = 0.5 if path1.path_type == path2.path_type else 0.2

                total_similarity += similarity
                comparison_count += 1

        return total_similarity / comparison_count if comparison_count > 0 else 0.0

    async def _update_cluster_metadata(self, cluster: PathCluster, features: dict[str, dict[str, float]]):
        """Update cluster metadata and characteristics."""
        if not cluster.paths:
            return

        # Calculate common breadcrumbs
        all_breadcrumbs = []
        for path in cluster.paths:
            for node in path.nodes:
                all_breadcrumbs.append(node.breadcrumb)

        breadcrumb_counts = defaultdict(int)
        for breadcrumb in all_breadcrumbs:
            breadcrumb_counts[breadcrumb] += 1

        # Common breadcrumbs appear in at least 50% of paths
        threshold = len(cluster.paths) * 0.5
        cluster.common_breadcrumbs = {breadcrumb for breadcrumb, count in breadcrumb_counts.items() if count >= threshold}

        # Dominant path types
        type_counts = defaultdict(int)
        for path in cluster.paths:
            type_counts[path.path_type.value] += 1

        cluster.dominant_path_types = [path_type for path_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)]

        # Calculate intra-cluster similarity
        if len(cluster.paths) > 1:
            total_similarity = 0.0
            comparison_count = 0

            for i, path1 in enumerate(cluster.paths):
                for path2 in cluster.paths[i + 1 :]:
                    similarity = await self._calculate_path_similarity(path1, path2, features)
                    total_similarity += similarity
                    comparison_count += 1

            cluster.intra_cluster_similarity = total_similarity / comparison_count if comparison_count > 0 else 1.0
        else:
            cluster.intra_cluster_similarity = 1.0

        # Calculate cohesion score
        cluster.cohesion_score = cluster.intra_cluster_similarity

    async def _select_representative_paths(self, clusters: list[PathCluster], config: ClusteringConfig) -> list[AnyPath]:
        """Select representative paths from each cluster."""
        representative_paths = []

        for cluster in clusters:
            if not cluster.paths:
                continue

            if len(cluster.paths) == 1:
                representative = cluster.paths[0]
            else:
                representative = await self._select_cluster_representative(cluster, config)

            cluster.representative_path = representative
            representative_paths.append(representative)

        return representative_paths

    async def _select_cluster_representative(self, cluster: PathCluster, config: ClusteringConfig) -> AnyPath:
        """Select the best representative path from a cluster."""
        if config.representative_selection == RepresentativeSelection.HIGHEST_IMPORTANCE:
            return max(cluster.paths, key=lambda p: self._get_path_importance(p))

        elif config.representative_selection == RepresentativeSelection.MOST_CONNECTED:
            # Select path with most connections (highest node count)
            return max(cluster.paths, key=lambda p: len(p.nodes))

        elif config.representative_selection == RepresentativeSelection.COMPOSITE_SCORE:
            # Combine multiple factors
            best_path = None
            best_score = -1.0

            for path in cluster.paths:
                importance = self._get_path_importance(path)
                connectivity = len(path.nodes) / 10.0  # Normalize to 0-1
                complexity = getattr(path, "complexity_score", 0.5)

                composite_score = (importance * 0.5) + (connectivity * 0.3) + (complexity * 0.2)

                if composite_score > best_score:
                    best_score = composite_score
                    best_path = path

            return best_path or cluster.paths[0]

        else:  # CENTROID or default
            # Select path closest to cluster centroid (highest intra-cluster similarity)
            best_path = None
            best_avg_similarity = -1.0

            for candidate in cluster.paths:
                total_similarity = 0.0
                for other_path in cluster.paths:
                    if candidate != other_path:
                        # Simplified similarity calculation
                        similarity = 0.8 if candidate.path_type == other_path.path_type else 0.3
                        total_similarity += similarity

                avg_similarity = total_similarity / max(1, len(cluster.paths) - 1)

                if avg_similarity > best_avg_similarity:
                    best_avg_similarity = avg_similarity
                    best_path = candidate

            return best_path or cluster.paths[0]

    def _get_path_importance(self, path: AnyPath) -> float:
        """Get importance score for a path."""
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

    async def _calculate_clustering_quality(
        self, clusters: list[PathCluster], features: dict[str, dict[str, float]], config: ClusteringConfig
    ) -> dict[str, float]:
        """Calculate overall clustering quality metrics."""
        if not clusters:
            return {"silhouette_score": 0.0, "cohesion_score": 0.0, "separation_score": 0.0, "coverage_score": 0.0}

        # Calculate average silhouette score
        silhouette_scores = []
        for cluster in clusters:
            if len(cluster.paths) > 1:
                silhouette_scores.append(cluster.intra_cluster_similarity)

        avg_silhouette = sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0

        # Calculate overall cohesion
        cohesion_scores = [cluster.cohesion_score for cluster in clusters if cluster.cohesion_score > 0]
        avg_cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.0

        # Calculate separation (simplified)
        separation_score = 0.8 if len(clusters) > 1 else 1.0

        # Calculate coverage completeness
        total_paths = sum(len(cluster.paths) for cluster in clusters)
        coverage_score = min(1.0, total_paths / max(1, len(features)))

        return {
            "silhouette_score": avg_silhouette,
            "cohesion_score": avg_cohesion,
            "separation_score": separation_score,
            "coverage_score": coverage_score,
        }

    async def _add_clustering_insights(self, result: ClusteringResult, config: ClusteringConfig):
        """Add quality insights to clustering result."""
        insights = []

        # Cluster count analysis
        if result.cluster_count == 1:
            insights.append("Single cluster created - paths may be too similar")
        elif result.cluster_count > result.original_path_count * 0.8:
            insights.append("Many small clusters - consider more aggressive clustering")

        # Quality analysis
        if result.average_silhouette_score < 0.3:
            insights.append("Low silhouette score - clusters may not be well-separated")
        elif result.average_silhouette_score > 0.7:
            insights.append("High-quality clusters with good separation")

        # Reduction analysis
        if result.reduction_percentage < 20:
            insights.append("Low reduction achieved - paths may be too diverse")
        elif result.reduction_percentage > 80:
            insights.append("High reduction achieved - verify important paths are preserved")

        # Coverage analysis
        if result.coverage_completeness < 0.8:
            insights.append("Low coverage completeness - some paths may be missing")

        result.quality_insights = insights

    def _create_empty_result(self, config: ClusteringConfig, error_message: str) -> ClusteringResult:
        """Create empty clustering result for error cases."""
        return ClusteringResult(
            clusters=[],
            representative_paths=[],
            original_path_count=0,
            clustered_path_count=0,
            cluster_count=0,
            reduction_percentage=0.0,
            average_silhouette_score=0.0,
            overall_cohesion_score=0.0,
            cluster_separation_score=0.0,
            coverage_completeness=0.0,
            processing_time_ms=0.0,
            clustering_algorithm=config.algorithm,
            clustering_strategy=config.strategy,
            representative_selection=config.representative_selection,
            clustering_warnings=[error_message],
        )

    def _update_performance_stats(self, processing_time_ms: float, path_count: int, cluster_count: int, success: bool):
        """Update internal performance statistics."""
        self._clustering_stats["total_clustering_operations"] += 1

        if success:
            self._clustering_stats["successful_operations"] += 1

            # Update averages
            operations = self._clustering_stats["successful_operations"]

            # Average processing time
            current_avg_time = self._clustering_stats["average_processing_time_ms"]
            self._clustering_stats["average_processing_time_ms"] = (current_avg_time * (operations - 1) + processing_time_ms) / operations

            # Average reduction percentage
            if path_count > 0:
                reduction = ((path_count - cluster_count) / path_count) * 100
                current_avg_reduction = self._clustering_stats["average_reduction_percentage"]
                self._clustering_stats["average_reduction_percentage"] = (current_avg_reduction * (operations - 1) + reduction) / operations

            # Update totals
            self._clustering_stats["total_paths_clustered"] += path_count
            self._clustering_stats["total_clusters_created"] += cluster_count
        else:
            self._clustering_stats["failed_operations"] += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        stats = dict(self._clustering_stats)

        # Add derived metrics
        total_ops = stats["total_clustering_operations"]
        if total_ops > 0:
            stats["success_rate"] = stats["successful_operations"] / total_ops
            stats["failure_rate"] = stats["failed_operations"] / total_ops
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        # Add cache statistics
        stats["similarity_cache_size"] = len(self._similarity_cache)
        stats["feature_cache_size"] = len(self._feature_cache)

        return stats

    def clear_caches(self):
        """Clear internal caches to free memory."""
        self._similarity_cache.clear()
        self._feature_cache.clear()
        self.logger.info("Path clustering caches cleared")


# Factory function
def create_path_clustering_service(config: ClusteringConfig = None) -> PathClusteringService:
    """
    Factory function to create a PathClusteringService instance.

    Args:
        config: Optional clustering configuration

    Returns:
        Configured PathClusteringService instance
    """
    return PathClusteringService(config)
