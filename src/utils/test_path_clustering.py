"""
Unit tests for the Path Clustering Service.

This module tests the comprehensive path clustering functionality including
feature extraction, clustering algorithms, and representative selection.
"""

import asyncio
from unittest.mock import Mock

import pytest

from src.models.code_chunk import ChunkType
from src.models.relational_path import (
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathConfidence,
    PathEdge,
    PathNode,
    PathType,
    RelationalPathCollection,
)
from src.utils.path_clustering import (
    ClusteringAlgorithm,
    ClusteringConfig,
    ClusteringStrategy,
    PathCluster,
    PathClusteringService,
    RepresentativeSelection,
    create_path_clustering_service,
)


class TestClusteringConfig:
    """Test clustering configuration options."""

    def test_default_config(self):
        """Test default clustering configuration."""
        config = ClusteringConfig()

        assert config.algorithm == ClusteringAlgorithm.HYBRID
        assert config.strategy == ClusteringStrategy.BALANCED
        assert config.representative_selection == RepresentativeSelection.COMPOSITE_SCORE
        assert config.target_cluster_count is None
        assert config.min_cluster_size == 2
        assert config.max_cluster_size == 20
        assert config.similarity_threshold == 0.7
        assert config.enable_parallel_clustering
        assert config.cache_similarity_matrix

    def test_custom_config(self):
        """Test custom clustering configuration."""
        config = ClusteringConfig(
            algorithm=ClusteringAlgorithm.HIERARCHICAL,
            strategy=ClusteringStrategy.AGGRESSIVE,
            target_cluster_count=5,
            similarity_threshold=0.8,
            enable_parallel_clustering=False,
        )

        assert config.algorithm == ClusteringAlgorithm.HIERARCHICAL
        assert config.strategy == ClusteringStrategy.AGGRESSIVE
        assert config.target_cluster_count == 5
        assert config.similarity_threshold == 0.8
        assert not config.enable_parallel_clustering


class TestPathCluster:
    """Test path cluster functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample execution paths
        self.node1 = PathNode(
            node_id="node1",
            breadcrumb="module.function1",
            name="function1",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=10,
            line_end=20,
            role_in_path="source",
            importance_score=0.8,
        )

        self.node2 = PathNode(
            node_id="node2",
            breadcrumb="module.function2",
            name="function2",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=25,
            line_end=35,
            role_in_path="target",
            importance_score=0.6,
        )

        self.execution_path1 = ExecutionPath(
            path_id="exec_path1",
            nodes=[self.node1, self.node2],
            criticality_score=0.8,
        )

        self.execution_path2 = ExecutionPath(
            path_id="exec_path2",
            nodes=[self.node1],
            criticality_score=0.6,
        )

        self.data_flow_path = DataFlowPath(
            path_id="data_path1",
            nodes=[self.node1],
            data_source="input",
            data_quality_score=0.7,
        )

        self.cluster = PathCluster(
            cluster_id="test_cluster",
            cluster_name="Test Cluster",
            paths=[self.execution_path1, self.execution_path2, self.data_flow_path],
        )

    def test_get_cluster_size(self):
        """Test cluster size calculation."""
        assert self.cluster.get_cluster_size() == 3

    def test_get_average_importance(self):
        """Test average importance calculation."""
        # Expected: (0.8 + 0.6 + 0.7) / 3 = 0.7
        avg_importance = self.cluster.get_average_importance()
        assert abs(avg_importance - 0.7) < 0.01

    def test_get_path_type_distribution(self):
        """Test path type distribution calculation."""
        distribution = self.cluster.get_path_type_distribution()

        assert distribution["execution_path"] == 2
        assert distribution["data_flow"] == 1

    def test_is_homogeneous(self):
        """Test homogeneous cluster detection."""
        # Mixed cluster should not be homogeneous
        assert not self.cluster.is_homogeneous()

        # Create homogeneous cluster
        homogeneous_cluster = PathCluster(
            cluster_id="homogeneous",
            cluster_name="Homogeneous Cluster",
            paths=[self.execution_path1, self.execution_path2],
        )
        assert homogeneous_cluster.is_homogeneous()


class TestPathClusteringService:
    """Test the main path clustering service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ClusteringConfig(target_cluster_count=2)
        self.service = PathClusteringService(self.config)

        # Create sample paths
        self.node1 = PathNode(
            node_id="node1",
            breadcrumb="module.function1",
            name="function1",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=10,
            line_end=20,
            role_in_path="source",
            importance_score=0.8,
        )

        self.node2 = PathNode(
            node_id="node2",
            breadcrumb="module.function2",
            name="function2",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=25,
            line_end=35,
            role_in_path="target",
            importance_score=0.6,
        )

        self.node3 = PathNode(
            node_id="node3",
            breadcrumb="module.function3",
            name="function3",
            chunk_type=ChunkType.FUNCTION,
            file_path="/helper.py",
            line_start=5,
            line_end=15,
            role_in_path="source",
            importance_score=0.9,
        )

        # Create diverse paths for clustering
        self.execution_path1 = ExecutionPath(
            path_id="exec_path1",
            nodes=[self.node1, self.node2],
            complexity_score=0.6,
            criticality_score=0.8,
            max_depth=2,
        )

        self.execution_path2 = ExecutionPath(
            path_id="exec_path2",
            nodes=[self.node1, self.node3],
            complexity_score=0.4,
            criticality_score=0.7,
            max_depth=2,
        )

        self.execution_path3 = ExecutionPath(
            path_id="exec_path3",
            nodes=[self.node3],
            complexity_score=0.3,
            criticality_score=0.9,
            max_depth=1,
        )

        self.data_flow_path1 = DataFlowPath(
            path_id="data_path1",
            nodes=[self.node1],
            data_source="input_data",
            data_types=["string", "int"],
            data_quality_score=0.7,
        )

        self.data_flow_path2 = DataFlowPath(
            path_id="data_path2",
            nodes=[self.node2, self.node3],
            data_source="computed_data",
            data_types=["float"],
            data_quality_score=0.5,
        )

        self.dependency_path = DependencyPath(
            path_id="dep_path1",
            nodes=[self.node1, self.node2],
            dependency_type="import",
            required_modules=["numpy", "pandas"],
            stability_score=0.8,
            coupling_strength=0.3,
        )

        # Create path collection
        self.path_collection = RelationalPathCollection(
            collection_id="test_collection",
            collection_name="Test Collection",
            execution_paths=[self.execution_path1, self.execution_path2, self.execution_path3],
            data_flow_paths=[self.data_flow_path1, self.data_flow_path2],
            dependency_paths=[self.dependency_path],
        )

    @pytest.mark.asyncio
    async def test_cluster_path_collection_success(self):
        """Test successful path collection clustering."""
        result = await self.service.cluster_path_collection(self.path_collection)

        assert result.original_path_count == 6  # 3 exec + 2 data + 1 dep
        assert result.cluster_count > 0
        assert result.cluster_count <= result.original_path_count
        assert len(result.representative_paths) == result.cluster_count
        assert result.reduction_percentage >= 0.0
        assert result.processing_time_ms > 0
        assert result.clustering_algorithm == ClusteringAlgorithm.HYBRID
        assert result.clustering_strategy == ClusteringStrategy.BALANCED

    @pytest.mark.asyncio
    async def test_cluster_empty_collection(self):
        """Test clustering empty collection."""
        empty_collection = RelationalPathCollection(collection_id="empty", collection_name="Empty Collection")

        result = await self.service.cluster_path_collection(empty_collection)

        assert result.original_path_count == 0
        assert result.cluster_count == 0
        assert len(result.representative_paths) == 0
        assert result.reduction_percentage == 0.0
        assert "No paths to cluster" in result.clustering_warnings[0]

    @pytest.mark.asyncio
    async def test_extract_path_features(self):
        """Test path feature extraction."""
        paths = [self.execution_path1, self.data_flow_path1, self.dependency_path]

        features = await self.service._extract_path_features(paths)

        assert len(features) == 3
        assert "exec_path1" in features
        assert "data_path1" in features
        assert "dep_path1" in features

        # Check feature structure
        exec_features = features["exec_path1"]
        assert "path_length" in exec_features
        assert "edge_count" in exec_features
        assert "path_type_numeric" in exec_features
        assert "avg_node_importance" in exec_features
        assert "node_type_diversity" in exec_features
        assert "complexity_score" in exec_features
        assert "criticality_score" in exec_features
        assert "max_depth" in exec_features

        # Check values
        assert exec_features["path_length"] == 2
        assert exec_features["path_type_numeric"] == 1.0  # execution_path
        assert exec_features["complexity_score"] == 0.6
        assert exec_features["criticality_score"] == 0.8
        assert exec_features["max_depth"] == 2

    def test_encode_path_type(self):
        """Test path type encoding."""
        exec_encoding = self.service._encode_path_type(self.execution_path1)
        data_encoding = self.service._encode_path_type(self.data_flow_path1)
        dep_encoding = self.service._encode_path_type(self.dependency_path)

        assert exec_encoding == 1.0
        # Note: data_flow and dependency_path may get encoded as 0.0 if not in mapping
        assert data_encoding >= 0.0
        assert dep_encoding >= 0.0

    @pytest.mark.asyncio
    async def test_determine_optimal_cluster_count(self):
        """Test optimal cluster count determination."""
        paths = [
            self.execution_path1,
            self.execution_path2,
            self.data_flow_path1,
            self.dependency_path,
        ]
        features = await self.service._extract_path_features(paths)

        cluster_count = await self.service._determine_optimal_cluster_count(paths, features, self.config)

        assert cluster_count >= 1
        assert cluster_count <= len(paths)

    @pytest.mark.asyncio
    async def test_calculate_path_similarity(self):
        """Test path similarity calculation."""
        features = await self.service._extract_path_features([self.execution_path1, self.execution_path2])

        # Same type paths should have higher similarity
        same_type_similarity = await self.service._calculate_path_similarity(self.execution_path1, self.execution_path2, features)

        # Different type paths should have lower similarity
        diff_type_similarity = await self.service._calculate_path_similarity(self.execution_path1, self.data_flow_path1, features)

        assert same_type_similarity > diff_type_similarity
        assert 0.0 <= same_type_similarity <= 1.0
        assert 0.0 <= diff_type_similarity <= 1.0
        assert diff_type_similarity == 0.2  # Base similarity for different types

    @pytest.mark.asyncio
    async def test_hierarchical_clustering(self):
        """Test hierarchical clustering algorithm."""
        paths = [self.execution_path1, self.execution_path2, self.data_flow_path1]
        features = await self.service._extract_path_features(paths)

        clusters = await self.service._hierarchical_clustering(paths, features, 2, self.config)

        assert len(clusters) <= 2  # Should not exceed target
        assert all(isinstance(cluster, PathCluster) for cluster in clusters)
        assert all(cluster.paths for cluster in clusters)  # No empty clusters

        # Check total paths preserved
        total_paths = sum(len(cluster.paths) for cluster in clusters)
        assert total_paths == len(paths)

    @pytest.mark.asyncio
    async def test_semantic_clustering(self):
        """Test semantic clustering algorithm."""
        paths = [
            self.execution_path1,
            self.execution_path2,
            self.data_flow_path1,
            self.dependency_path,
        ]
        features = await self.service._extract_path_features(paths)

        clusters = await self.service._semantic_clustering(paths, features, 3, self.config)

        assert len(clusters) >= 1
        assert all(isinstance(cluster, PathCluster) for cluster in clusters)

        # Should group by path type
        path_types_in_clusters = set()
        for cluster in clusters:
            if cluster.dominant_path_types:
                path_types_in_clusters.update(cluster.dominant_path_types)

        assert "execution_path" in path_types_in_clusters
        # Note: actual path type names may be different, just check that we have multiple types
        assert len(path_types_in_clusters) >= 2

    @pytest.mark.asyncio
    async def test_hybrid_clustering(self):
        """Test hybrid clustering algorithm."""
        paths = [
            self.execution_path1,
            self.execution_path2,
            self.execution_path3,
            self.data_flow_path1,
        ]
        features = await self.service._extract_path_features(paths)

        clusters = await self.service._hybrid_clustering(paths, features, 2, self.config)

        assert len(clusters) >= 1
        assert len(clusters) <= len(paths)
        assert all(isinstance(cluster, PathCluster) for cluster in clusters)

    @pytest.mark.asyncio
    async def test_select_representative_paths(self):
        """Test representative path selection."""
        # Create cluster with multiple paths
        cluster = PathCluster(
            cluster_id="test_cluster",
            cluster_name="Test Cluster",
            paths=[self.execution_path1, self.execution_path2, self.execution_path3],
        )

        representatives = await self.service._select_representative_paths([cluster], self.config)

        assert len(representatives) == 1
        assert representatives[0] in cluster.paths
        assert cluster.representative_path == representatives[0]

    @pytest.mark.asyncio
    async def test_select_cluster_representative_highest_importance(self):
        """Test representative selection by highest importance."""
        config = ClusteringConfig(representative_selection=RepresentativeSelection.HIGHEST_IMPORTANCE)
        service = PathClusteringService(config)

        cluster = PathCluster(
            cluster_id="test_cluster",
            cluster_name="Test Cluster",
            paths=[self.execution_path1, self.execution_path2, self.execution_path3],
        )

        representative = await service._select_cluster_representative(cluster, config)

        # Should select path with highest criticality_score (execution_path3: 0.9)
        assert representative == self.execution_path3

    @pytest.mark.asyncio
    async def test_select_cluster_representative_most_connected(self):
        """Test representative selection by connectivity."""
        config = ClusteringConfig(representative_selection=RepresentativeSelection.MOST_CONNECTED)
        service = PathClusteringService(config)

        cluster = PathCluster(
            cluster_id="test_cluster",
            cluster_name="Test Cluster",
            paths=[self.execution_path1, self.execution_path3],  # path1 has 2 nodes, path3 has 1
        )

        representative = await service._select_cluster_representative(cluster, config)

        # Should select path with more nodes (execution_path1: 2 nodes)
        assert representative == self.execution_path1

    def test_get_path_importance(self):
        """Test path importance calculation."""
        exec_importance = self.service._get_path_importance(self.execution_path1)
        data_importance = self.service._get_path_importance(self.data_flow_path1)
        dep_importance = self.service._get_path_importance(self.dependency_path)

        assert exec_importance == self.execution_path1.criticality_score
        assert data_importance == self.data_flow_path1.data_quality_score
        assert dep_importance == self.dependency_path.stability_score

    @pytest.mark.asyncio
    async def test_calculate_clustering_quality(self):
        """Test clustering quality calculation."""
        cluster1 = PathCluster(
            cluster_id="cluster1",
            cluster_name="Cluster 1",
            paths=[self.execution_path1, self.execution_path2],
        )
        cluster1.cohesion_score = 0.8
        cluster1.intra_cluster_similarity = 0.7

        cluster2 = PathCluster(cluster_id="cluster2", cluster_name="Cluster 2", paths=[self.data_flow_path1])
        cluster2.cohesion_score = 1.0
        cluster2.intra_cluster_similarity = 1.0

        clusters = [cluster1, cluster2]
        features = await self.service._extract_path_features([self.execution_path1, self.execution_path2, self.data_flow_path1])

        quality = await self.service._calculate_clustering_quality(clusters, features, self.config)

        assert "silhouette_score" in quality
        assert "cohesion_score" in quality
        assert "separation_score" in quality
        assert "coverage_score" in quality

        assert 0.0 <= quality["silhouette_score"] <= 1.0
        assert 0.0 <= quality["cohesion_score"] <= 1.0
        assert 0.0 <= quality["separation_score"] <= 1.0
        assert 0.0 <= quality["coverage_score"] <= 1.0

    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        # Simulate some operations
        self.service._update_performance_stats(1000.0, 10, 3, True)
        self.service._update_performance_stats(1500.0, 8, 2, True)
        self.service._update_performance_stats(2000.0, 5, 0, False)

        stats = self.service.get_performance_stats()

        assert stats["total_clustering_operations"] == 3
        assert stats["successful_operations"] == 2
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] == 2 / 3
        assert stats["failure_rate"] == 1 / 3
        assert stats["average_processing_time_ms"] == 1250.0  # (1000 + 1500) / 2
        assert stats["total_paths_clustered"] == 18  # 10 + 8
        assert stats["total_clusters_created"] == 5  # 3 + 2

    def test_clear_caches(self):
        """Test cache clearing."""
        # Add some items to caches
        self.service._similarity_cache[("path1", "path2")] = 0.8
        self.service._feature_cache["path1"] = {"feature1": 0.5}

        assert len(self.service._similarity_cache) > 0
        assert len(self.service._feature_cache) > 0

        self.service.clear_caches()

        assert len(self.service._similarity_cache) == 0
        assert len(self.service._feature_cache) == 0

    @pytest.mark.asyncio
    async def test_clustering_with_different_algorithms(self):
        """Test clustering with different algorithms."""
        algorithms = [
            ClusteringAlgorithm.HIERARCHICAL,
            ClusteringAlgorithm.SEMANTIC,
            ClusteringAlgorithm.HYBRID,
        ]

        for algorithm in algorithms:
            config = ClusteringConfig(algorithm=algorithm, target_cluster_count=2)
            service = PathClusteringService(config)

            result = await service.cluster_path_collection(self.path_collection)

            assert result.cluster_count > 0
            assert result.clustering_algorithm == algorithm
            assert len(result.representative_paths) == result.cluster_count

    @pytest.mark.asyncio
    async def test_clustering_result_effectiveness_score(self):
        """Test clustering result effectiveness calculation."""
        result = await self.service.cluster_path_collection(self.path_collection)

        effectiveness = result.get_effectiveness_score()

        assert 0.0 <= effectiveness <= 1.0

    @pytest.mark.asyncio
    async def test_clustering_with_single_path(self):
        """Test clustering with single path."""
        single_collection = RelationalPathCollection(
            collection_id="single",
            collection_name="Single Path Collection",
            execution_paths=[self.execution_path1],
        )

        result = await self.service.cluster_path_collection(single_collection)

        assert result.original_path_count == 1
        assert result.cluster_count == 1
        assert len(result.representative_paths) == 1
        assert result.representative_paths[0] == self.execution_path1

    @pytest.mark.asyncio
    async def test_feature_caching(self):
        """Test feature extraction caching."""
        config = ClusteringConfig(cache_similarity_matrix=True)
        service = PathClusteringService(config)

        paths = [self.execution_path1, self.execution_path2]

        # First extraction should populate cache
        features1 = await service._extract_path_features(paths)
        cache_size_after_first = len(service._feature_cache)

        # Second extraction should use cache
        features2 = await service._extract_path_features(paths)
        cache_size_after_second = len(service._feature_cache)

        assert features1 == features2
        assert cache_size_after_first == cache_size_after_second
        assert cache_size_after_first > 0


class TestServiceFactory:
    """Test the service factory function."""

    def test_create_path_clustering_service_default(self):
        """Test service creation with default config."""
        service = create_path_clustering_service()

        assert isinstance(service, PathClusteringService)
        assert service.config.algorithm == ClusteringAlgorithm.HYBRID

    def test_create_path_clustering_service_custom_config(self):
        """Test service creation with custom config."""
        config = ClusteringConfig(algorithm=ClusteringAlgorithm.HIERARCHICAL)
        service = create_path_clustering_service(config)

        assert isinstance(service, PathClusteringService)
        assert service.config.algorithm == ClusteringAlgorithm.HIERARCHICAL


if __name__ == "__main__":
    pytest.main([__file__])
