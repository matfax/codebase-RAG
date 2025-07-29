"""
Integration tests for the Streaming Pruning Service.

This module tests the comprehensive streaming pruning functionality including
duplicate detection, information density filtering, and quality preservation.
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
from src.services.streaming_pruning_service import (
    PruningConfig,
    PruningStrategy,
    RedundancyType,
    StreamingPruningService,
    create_streaming_pruning_service,
)


class TestPruningConfig:
    """Test pruning configuration options."""

    def test_default_config(self):
        """Test default pruning configuration."""
        config = PruningConfig()

        assert config.target_reduction_percentage == 40.0
        assert config.max_processing_time_ms == 15000
        assert config.min_information_density == 0.3
        assert config.min_relevance_score == 0.4
        assert config.min_confidence_threshold == 0.3
        assert config.structural_similarity_threshold == 0.8
        assert config.semantic_similarity_threshold == 0.7
        assert config.node_overlap_threshold == 0.6
        assert config.preserve_high_value_paths
        assert config.preserve_entry_point_paths
        assert config.enable_circular_detection

    def test_custom_config(self):
        """Test custom pruning configuration."""
        config = PruningConfig(
            target_reduction_percentage=60.0,
            min_information_density=0.5,
            structural_similarity_threshold=0.9,
            preserve_high_value_paths=False,
        )

        assert config.target_reduction_percentage == 60.0
        assert config.min_information_density == 0.5
        assert config.structural_similarity_threshold == 0.9
        assert not config.preserve_high_value_paths


class TestStreamingPruningService:
    """Test the main streaming pruning service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = PruningConfig(target_reduction_percentage=40.0)
        self.service = StreamingPruningService(self.config)

        # Create sample path nodes
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
            file_path="/test.py",
            line_start=40,
            line_end=50,
            role_in_path="intermediate",
            importance_score=0.7,
        )

        # Create sample path edges
        self.edge1 = PathEdge(
            source_node_id="node1",
            target_node_id="node2",
            relationship_type="function_call",
            weight=0.9,
            confidence=PathConfidence.HIGH,
            call_expression="function2()",
            line_number=15,
        )

        # Create sample execution paths
        self.execution_path1 = ExecutionPath(
            path_id="exec_path1",
            nodes=[self.node1, self.node2],
            edges=[self.edge1],
            entry_points=["node1"],
            exit_points=["node2"],
            max_depth=1,
            complexity_score=0.5,
            criticality_score=0.8,
        )

        self.execution_path2 = ExecutionPath(
            path_id="exec_path2",
            nodes=[self.node1, self.node3],
            edges=[
                PathEdge(
                    source_node_id="node1",
                    target_node_id="node3",
                    relationship_type="function_call",
                    weight=0.7,
                    confidence=PathConfidence.MEDIUM,
                )
            ],
            entry_points=["node1"],
            exit_points=["node3"],
            max_depth=1,
            complexity_score=0.4,
            criticality_score=0.7,
        )

        # Create duplicate execution path (same as path1)
        self.execution_path_duplicate = ExecutionPath(
            path_id="exec_path_dup",
            nodes=[self.node1, self.node2],
            edges=[self.edge1],
            entry_points=["node1"],
            exit_points=["node2"],
            max_depth=1,
            complexity_score=0.5,
            criticality_score=0.6,  # Lower criticality than original
        )

        # Create data flow path
        self.data_flow_path = DataFlowPath(
            path_id="data_path1",
            nodes=[self.node1],
            data_source="input_data",
            data_types=["string"],
            creation_point="module.function1",
            data_quality_score=0.6,
        )

        # Create dependency path
        self.dependency_path = DependencyPath(
            path_id="dep_path1",
            nodes=[self.node1, self.node2],
            dependency_type="import",
            required_modules=["requests", "json"],
            stability_score=0.8,
            coupling_strength=0.3,
        )

        # Create path collection
        self.path_collection = RelationalPathCollection(
            collection_id="test_collection",
            collection_name="Test Collection",
            execution_paths=[self.execution_path1, self.execution_path2, self.execution_path_duplicate],
            data_flow_paths=[self.data_flow_path],
            dependency_paths=[self.dependency_path],
        )

    @pytest.mark.asyncio
    async def test_prune_path_collection_balanced_strategy(self):
        """Test path collection pruning with balanced strategy."""
        result = await self.service.prune_path_collection(self.path_collection, strategy=PruningStrategy.BALANCED)

        assert result.original_count == 5  # 3 exec + 1 data + 1 dep
        assert result.pruned_count < result.original_count
        assert result.reduction_percentage > 0
        assert result.pruning_strategy == PruningStrategy.BALANCED
        assert result.information_density_score >= 0.0
        assert result.relevance_preservation_score >= 0.0
        assert result.processing_time_ms > 0

        # Should have removed at least the duplicate
        assert len(result.redundant_paths) >= 1
        assert RedundancyType.EXACT_DUPLICATE in result.redundancy_breakdown

    @pytest.mark.asyncio
    async def test_prune_path_collection_conservative_strategy(self):
        """Test path collection pruning with conservative strategy."""
        result = await self.service.prune_path_collection(self.path_collection, strategy=PruningStrategy.CONSERVATIVE)

        assert result.original_count == 5
        assert result.pruning_strategy == PruningStrategy.CONSERVATIVE
        # Conservative should still achieve some reduction due to duplicates
        assert result.reduction_percentage >= 0.0

    @pytest.mark.asyncio
    async def test_prune_path_collection_aggressive_strategy(self):
        """Test path collection pruning with aggressive strategy."""
        result = await self.service.prune_path_collection(self.path_collection, strategy=PruningStrategy.AGGRESSIVE)

        assert result.original_count == 5
        assert result.pruning_strategy == PruningStrategy.AGGRESSIVE
        # Aggressive might have higher reduction

    @pytest.mark.asyncio
    async def test_prune_path_collection_with_preserve_patterns(self):
        """Test path collection pruning with preserve patterns."""
        preserve_patterns = {"exec_path1"}  # Preserve specific path

        result = await self.service.prune_path_collection(
            self.path_collection, strategy=PruningStrategy.BALANCED, preserve_patterns=preserve_patterns
        )

        # Check that preserved path is still in results
        preserved_path_found = any(path.path_id == "exec_path1" for path in result.pruned_paths)
        assert preserved_path_found

    @pytest.mark.asyncio
    async def test_prune_empty_collection(self):
        """Test pruning empty path collection."""
        empty_collection = RelationalPathCollection(collection_id="empty", collection_name="Empty Collection")

        result = await self.service.prune_path_collection(empty_collection)

        assert result.original_count == 0
        assert result.pruned_count == 0
        assert result.reduction_percentage == 0.0
        assert len(result.redundant_paths) == 0
        assert "No paths to prune" in result.pruning_warnings[0]

    @pytest.mark.asyncio
    async def test_prune_streaming_paths(self):
        """Test streaming path pruning."""
        path_stream = [
            self.execution_path1,
            self.execution_path_duplicate,  # Should be filtered as duplicate
            self.execution_path2,
            self.data_flow_path,
        ]

        pruned_paths = await self.service.prune_streaming_paths(path_stream, batch_size=2, strategy=PruningStrategy.BALANCED)

        # Should remove duplicates and low-density paths
        assert len(pruned_paths) <= len(path_stream)

        # Check that paths meet information density threshold
        for path in pruned_paths:
            density = await self.service._calculate_path_information_density(path)
            assert density >= self.config.min_information_density

    @pytest.mark.asyncio
    async def test_detect_exact_duplicates(self):
        """Test exact duplicate detection."""
        paths = [self.execution_path1, self.execution_path_duplicate, self.execution_path2]

        result = await self.service._detect_exact_duplicates(paths, PruningStrategy.BALANCED, set())

        assert result["type"] == RedundancyType.EXACT_DUPLICATE
        assert len(result["redundant"]) == 1  # One duplicate should be removed
        # Higher criticality path should be kept
        redundant_path = result["redundant"][0]
        assert redundant_path.criticality_score <= self.execution_path1.criticality_score

    @pytest.mark.asyncio
    async def test_detect_structural_similarity(self):
        """Test structural similarity detection."""
        # Create structurally similar paths
        similar_path = ExecutionPath(
            path_id="similar_path",
            nodes=[self.node1, self.node2],  # Same nodes as execution_path1
            edges=[self.edge1],
            entry_points=["node1"],
            exit_points=["node2"],
            max_depth=1,
            complexity_score=0.3,
            criticality_score=0.5,
        )

        paths = [self.execution_path1, similar_path, self.execution_path2]

        result = await self.service._detect_structural_similarity(paths, PruningStrategy.BALANCED, set())

        assert result["type"] == RedundancyType.STRUCTURAL_SIMILAR
        # Should detect similarity between execution_path1 and similar_path

    @pytest.mark.asyncio
    async def test_filter_low_information_density(self):
        """Test low information density filtering."""
        # Create low-density path
        low_density_path = ExecutionPath(
            path_id="low_density",
            nodes=[
                PathNode(
                    node_id="simple_node",
                    breadcrumb="simple",
                    name="simple",
                    chunk_type=ChunkType.VARIABLE,
                    file_path="/test.py",
                    line_start=1,
                    line_end=1,
                    role_in_path="source",
                    importance_score=0.1,  # Very low importance
                )
            ],
            entry_points=["simple_node"],
            exit_points=["simple_node"],
            max_depth=0,
            complexity_score=0.0,
            criticality_score=0.1,
        )

        paths = [self.execution_path1, low_density_path]

        result = await self.service._filter_low_information_density(paths, PruningStrategy.BALANCED, set())

        assert result["type"] == RedundancyType.LOW_INFORMATION
        # Low density path should be in redundant list
        assert low_density_path in result["redundant"]

    @pytest.mark.asyncio
    async def test_detect_semantic_overlap(self):
        """Test semantic overlap detection."""
        # Create paths with semantic overlap
        overlap_path = ExecutionPath(
            path_id="overlap_path",
            nodes=[self.node1, self.node3],  # Shares node1 with execution_path1
            entry_points=["node1"],
            exit_points=["node3"],
            max_depth=1,
            complexity_score=0.4,
            criticality_score=0.6,
        )

        paths = [self.execution_path1, overlap_path]

        result = await self.service._detect_semantic_overlap(paths, PruningStrategy.BALANCED, set())

        assert result["type"] == RedundancyType.SEMANTIC_OVERLAP

    @pytest.mark.asyncio
    async def test_filter_circular_references(self):
        """Test circular reference filtering."""
        # Create circular path
        circular_node = PathNode(
            node_id="circular",
            breadcrumb="module.circular",
            name="circular",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="source",
            importance_score=0.5,
        )

        circular_path = ExecutionPath(
            path_id="circular_path",
            nodes=[circular_node, self.node1, circular_node],  # Circular reference
            entry_points=["circular"],
            exit_points=["circular"],
            max_depth=2,
            complexity_score=0.3,
            criticality_score=0.4,
        )

        paths = [self.execution_path1, circular_path]

        result = await self.service._filter_circular_references(paths, PruningStrategy.BALANCED, set())

        assert result["type"] == RedundancyType.CIRCULAR_REFERENCE
        assert circular_path in result["redundant"]

    @pytest.mark.asyncio
    async def test_generate_path_signature(self):
        """Test path signature generation."""
        signature1 = await self.service._generate_path_signature(self.execution_path1)
        signature2 = await self.service._generate_path_signature(self.execution_path_duplicate)
        signature3 = await self.service._generate_path_signature(self.execution_path2)

        # Same structure should have same signature
        assert signature1 == signature2
        # Different structure should have different signature
        assert signature1 != signature3

    @pytest.mark.asyncio
    async def test_calculate_structural_similarity(self):
        """Test structural similarity calculation."""
        # Same paths should have high similarity
        similarity1 = await self.service._calculate_structural_similarity(self.execution_path1, self.execution_path_duplicate)
        assert similarity1 > 0.8

        # Different path types should have zero similarity
        similarity2 = await self.service._calculate_structural_similarity(self.execution_path1, self.data_flow_path)
        assert similarity2 == 0.0

        # Different structures should have lower similarity
        similarity3 = await self.service._calculate_structural_similarity(self.execution_path1, self.execution_path2)
        assert similarity3 < similarity1

    @pytest.mark.asyncio
    async def test_calculate_semantic_overlap(self):
        """Test semantic overlap calculation."""
        overlap = await self.service._calculate_semantic_overlap(self.execution_path1, self.execution_path2)

        assert 0.0 <= overlap <= 1.0
        # Should have some overlap due to shared node1
        assert overlap > 0.0

    @pytest.mark.asyncio
    async def test_calculate_path_information_density(self):
        """Test path information density calculation."""
        density1 = await self.service._calculate_path_information_density(self.execution_path1)
        density2 = await self.service._calculate_path_information_density(self.data_flow_path)

        assert 0.0 <= density1 <= 1.0
        assert 0.0 <= density2 <= 1.0

        # Should use cache for second call
        density1_cached = await self.service._calculate_path_information_density(self.execution_path1)
        assert density1 == density1_cached

    def test_calculate_sequence_similarity(self):
        """Test sequence similarity calculation."""
        seq1 = ["a", "b", "c"]
        seq2 = ["a", "b", "d"]
        seq3 = ["x", "y", "z"]

        similarity1 = self.service._calculate_sequence_similarity(seq1, seq2)
        similarity2 = self.service._calculate_sequence_similarity(seq1, seq3)

        assert 0.0 <= similarity1 <= 1.0
        assert 0.0 <= similarity2 <= 1.0
        assert similarity1 > similarity2  # seq1 and seq2 are more similar

    def test_extract_semantic_elements(self):
        """Test semantic elements extraction."""
        elements = self.service._extract_semantic_elements(self.execution_path1)

        assert isinstance(elements, set)
        assert len(elements) > 0
        assert "execution_path" in elements  # Path type
        assert "function1" in elements  # Node name
        assert "function" in elements  # Chunk type

    def test_is_circular_path(self):
        """Test circular path detection."""
        # Non-circular path
        assert not self.service._is_circular_path(self.execution_path1)

        # Create circular path
        circular_node = PathNode(
            node_id="circular",
            breadcrumb="module.circular",
            name="circular",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="source",
            importance_score=0.5,
        )

        circular_path = ExecutionPath(
            path_id="circular",
            nodes=[circular_node, self.node1, circular_node],  # Repeated breadcrumb
        )

        assert self.service._is_circular_path(circular_path)

    def test_get_path_importance(self):
        """Test path importance calculation."""
        importance1 = self.service._get_path_importance(self.execution_path1)
        importance2 = self.service._get_path_importance(self.data_flow_path)
        importance3 = self.service._get_path_importance(self.dependency_path)

        assert importance1 == self.execution_path1.criticality_score
        assert importance2 == self.data_flow_path.data_quality_score
        assert importance3 == self.dependency_path.stability_score

    @pytest.mark.asyncio
    async def test_calculate_average_information_density(self):
        """Test average information density calculation."""
        paths = [self.execution_path1, self.execution_path2, self.data_flow_path]

        avg_density = await self.service._calculate_average_information_density(paths)

        assert 0.0 <= avg_density <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_relevance_preservation_score(self):
        """Test relevance preservation score calculation."""
        original_paths = [self.execution_path1, self.execution_path2, self.data_flow_path]
        pruned_paths = [self.execution_path1, self.data_flow_path]  # One path removed

        score = await self.service._calculate_relevance_preservation_score(original_paths, pruned_paths)

        assert 0.0 <= score <= 1.0

    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        # Update some stats
        self.service._update_performance_stats(1000.0, 10, 3, True)
        self.service._update_performance_stats(2000.0, 8, 2, False)

        stats = self.service.get_performance_stats()

        assert stats["total_pruning_operations"] == 2
        assert stats["successful_operations"] == 1
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["failure_rate"] == 0.5
        assert "similarity_cache_size" in stats
        assert "density_cache_size" in stats

    def test_clear_caches(self):
        """Test cache clearing."""
        # Add some items to caches
        self.service._similarity_cache[("path1", "path2")] = 0.8
        self.service._density_cache["path1"] = 0.6

        assert len(self.service._similarity_cache) > 0
        assert len(self.service._density_cache) > 0

        self.service.clear_caches()

        assert len(self.service._similarity_cache) == 0
        assert len(self.service._density_cache) == 0

    @pytest.mark.asyncio
    async def test_target_reduction_achievement(self):
        """Test that target reduction is achieved."""
        # Create collection with many similar paths to enable pruning
        similar_paths = []
        for i in range(20):
            path = ExecutionPath(
                path_id=f"similar_path_{i}",
                nodes=[self.node1, self.node2],
                edges=[self.edge1],
                entry_points=["node1"],
                exit_points=["node2"],
                max_depth=1,
                complexity_score=0.3,
                criticality_score=0.4,
            )
            similar_paths.append(path)

        large_collection = RelationalPathCollection(
            collection_id="large_collection", collection_name="Large Collection", execution_paths=similar_paths
        )

        result = await self.service.prune_path_collection(large_collection, strategy=PruningStrategy.BALANCED)

        # Should achieve significant reduction due to duplicates
        assert result.reduction_percentage > 0
        assert result.is_target_achieved(target_reduction=20.0)  # Lower target for test


class TestServiceFactory:
    """Test the service factory function."""

    def test_create_streaming_pruning_service_default(self):
        """Test service creation with default config."""
        service = create_streaming_pruning_service()

        assert isinstance(service, StreamingPruningService)
        assert service.config.target_reduction_percentage == 40.0

    def test_create_streaming_pruning_service_custom_config(self):
        """Test service creation with custom config."""
        config = PruningConfig(target_reduction_percentage=60.0)
        service = create_streaming_pruning_service(config)

        assert isinstance(service, StreamingPruningService)
        assert service.config.target_reduction_percentage == 60.0


if __name__ == "__main__":
    pytest.main([__file__])
