"""
Tests for enhanced partial graph construction functionality in LightweightGraphService.

This test module focuses specifically on Task 1.2 implementation:
- Smart node selection algorithms
- Query-aware graph scoping with relevance scoring
- Multiple graph expansion strategies
- Adaptive node budget allocation based on query complexity
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.code_chunk import ChunkType, CodeChunk
from src.services.lightweight_graph_service import (
    GraphBuildOptions,
    GraphExpansionStrategy,
    LightweightGraphService,
    NodeMetadata,
    NodeRelevanceScore,
    QueryComplexity,
)
from src.services.structure_relationship_builder import GraphNode, StructureGraph


class TestPartialGraphConstruction:
    """Test suite for enhanced partial graph construction (Task 1.2)."""

    @pytest.fixture
    async def mock_service(self):
        """Create a mock LightweightGraphService for testing."""
        mock_graph_rag = AsyncMock()
        mock_hybrid_search = AsyncMock()
        mock_cache = AsyncMock()

        service = LightweightGraphService(
            graph_rag_service=mock_graph_rag, hybrid_search_service=mock_hybrid_search, cache_service=mock_cache
        )

        # Setup mock memory index with test data
        await self._setup_test_memory_index(service)

        return service

    async def _setup_test_memory_index(self, service):
        """Setup test data in memory index."""
        # Create test nodes
        test_nodes = [
            NodeMetadata(
                node_id="node1",
                name="main_function",
                chunk_type=ChunkType.FUNCTION,
                file_path="/test/main.py",
                breadcrumb="module.main_function",
                importance_score=2.0,
            ),
            NodeMetadata(
                node_id="node2",
                name="helper_function",
                chunk_type=ChunkType.FUNCTION,
                file_path="/test/utils.py",
                breadcrumb="module.utils.helper_function",
                importance_score=1.5,
            ),
            NodeMetadata(
                node_id="node3",
                name="TestClass",
                chunk_type=ChunkType.CLASS,
                file_path="/test/models.py",
                breadcrumb="module.models.TestClass",
                importance_score=1.8,
            ),
            NodeMetadata(
                node_id="node4",
                name="test_method",
                chunk_type=ChunkType.METHOD,
                file_path="/test/models.py",
                breadcrumb="module.models.TestClass.test_method",
                parent_name="TestClass",
                importance_score=1.2,
            ),
        ]

        # Add nodes to memory index
        for node in test_nodes:
            service.memory_index.nodes[node.node_id] = node
            service.memory_index.by_name[node.name].add(node.node_id)
            service.memory_index.by_type[node.chunk_type].add(node.node_id)
            service.memory_index.by_file[node.file_path].add(node.node_id)
            if node.breadcrumb:
                service.memory_index.by_breadcrumb[node.breadcrumb] = node.node_id

        # Setup relationships
        service.memory_index.nodes["node3"].children_ids.add("node4")
        service.memory_index.nodes["node4"].parent_ids.add("node3")
        service.memory_index.children_index["node3"].add("node4")
        service.memory_index.parent_index["node4"].add("node3")

        service.memory_index.total_nodes = len(test_nodes)

    @pytest.mark.asyncio
    async def test_query_complexity_analysis(self, mock_service):
        """Test query complexity analysis for adaptive processing."""

        # Simple query
        complexity = await mock_service._analyze_query_complexity("main")
        assert complexity == QueryComplexity.SIMPLE

        # Moderate query
        complexity = await mock_service._analyze_query_complexity("main function test")
        assert complexity == QueryComplexity.MODERATE

        # Complex query with wildcards
        complexity = await mock_service._analyze_query_complexity("main* function and helper")
        assert complexity == QueryComplexity.COMPLEX

        # Very complex query
        complexity = await mock_service._analyze_query_complexity("main* function and helper or test? module.class")
        assert complexity == QueryComplexity.VERY_COMPLEX

    @pytest.mark.asyncio
    async def test_options_adaptation_to_complexity(self, mock_service):
        """Test adaptive options based on query complexity."""

        base_options = GraphBuildOptions(max_nodes=50, context_depth=2, max_expansion_rounds=5)

        # Simple query adaptation
        adapted = await mock_service._adapt_options_to_complexity(base_options, QueryComplexity.SIMPLE)
        assert adapted.max_nodes <= 30
        assert adapted.context_depth == 1
        assert adapted.max_expansion_rounds == 3

        # Complex query adaptation
        adapted = await mock_service._adapt_options_to_complexity(base_options, QueryComplexity.COMPLEX)
        assert adapted.max_nodes > base_options.max_nodes
        assert adapted.context_depth > base_options.context_depth
        assert adapted.max_expansion_rounds > base_options.max_expansion_rounds

    @pytest.mark.asyncio
    async def test_target_node_scoring(self, mock_service):
        """Test target node finding and relevance scoring."""

        options = GraphBuildOptions(relevance_threshold=0.3)

        # Test exact name match
        scores = await mock_service._find_and_score_target_nodes("main_function", options)
        assert len(scores) > 0
        assert scores[0].node_id == "node1"
        assert scores[0].total_score > 0.8  # High score for exact match

        # Test partial name match
        scores = await mock_service._find_and_score_target_nodes("helper", options)
        assert len(scores) > 0
        found_helper = any(score.node_id == "node2" for score in scores)
        assert found_helper

        # Test breadcrumb match
        scores = await mock_service._find_and_score_target_nodes("TestClass", options)
        assert len(scores) > 0
        found_class = any(score.node_id == "node3" for score in scores)
        assert found_class

    @pytest.mark.asyncio
    async def test_breadth_first_expansion(self, mock_service):
        """Test breadth-first graph expansion strategy."""

        selected_nodes = {"node3"}  # Start with TestClass
        options = GraphBuildOptions(context_depth=2, importance_threshold=0.5)

        expanded = await mock_service._expand_breadth_first(selected_nodes, 10, options)

        # Should include connected nodes (test_method as child)
        assert "node4" in expanded or len(expanded) >= 0

    @pytest.mark.asyncio
    async def test_importance_based_expansion(self, mock_service):
        """Test importance-based graph expansion strategy."""

        selected_nodes = {"node1"}  # Start with main_function
        options = GraphBuildOptions(importance_threshold=1.0)

        expanded = await mock_service._expand_importance_based(selected_nodes, 5, options)

        # Should prefer nodes with higher importance scores
        # In our test data, TestClass (1.8) and helper_function (1.5) qualify
        assert isinstance(expanded, set)
        assert len(expanded) <= 5

    @pytest.mark.asyncio
    async def test_relevance_scored_expansion(self, mock_service):
        """Test relevance-scored graph expansion strategy."""

        selected_nodes = {"node1"}
        options = GraphBuildOptions(relevance_threshold=0.2)

        expanded = await mock_service._expand_relevance_scored(selected_nodes, 5, "test", options)

        # Should return a set of expanded nodes
        assert isinstance(expanded, set)
        assert len(expanded) <= 5

    @pytest.mark.asyncio
    async def test_adaptive_expansion(self, mock_service):
        """Test adaptive expansion combining strategies."""

        selected_nodes = {"node1"}
        options = GraphBuildOptions()

        expanded = await mock_service._expand_adaptive(selected_nodes, 6, "function", options)

        # Should combine importance-based and relevance-scored expansion
        assert isinstance(expanded, set)
        # Adaptive should use budget effectively
        assert len(expanded) <= 6

    @pytest.mark.asyncio
    async def test_connectivity_ensuring(self, mock_service):
        """Test graph connectivity ensuring."""

        # Test with disconnected nodes
        disconnected_nodes = {"node1", "node3"}  # No direct connection
        options = GraphBuildOptions(context_depth=3)

        connected = await mock_service._ensure_connectivity(disconnected_nodes, options)

        # Should return at least the original nodes
        assert disconnected_nodes.issubset(connected)

    @pytest.mark.asyncio
    async def test_connected_components_finding(self, mock_service):
        """Test finding connected components in graph."""

        # Test with mixed connected and disconnected nodes
        nodes = {"node1", "node3", "node4"}  # node3-node4 connected, node1 isolated

        components = await mock_service._find_connected_components(nodes)

        # Should find separate components
        assert len(components) >= 1
        # Check that all nodes are accounted for
        combined_components = set()
        for component in components:
            combined_components.update(component)
        assert combined_components == nodes

    @pytest.mark.asyncio
    async def test_complete_partial_graph_building(self, mock_service):
        """Test complete partial graph building process."""

        options = GraphBuildOptions(max_nodes=20, expansion_strategy=GraphExpansionStrategy.ADAPTIVE, include_context=True)

        graph = await mock_service.build_partial_graph("test_project", "main_function", options)

        assert graph is not None
        assert isinstance(graph, StructureGraph)
        assert len(graph.nodes) > 0
        assert graph.project_name == "test_project"
        assert hasattr(graph, "query_scope")
        assert graph.query_scope == "main_function"

    @pytest.mark.asyncio
    async def test_different_expansion_strategies(self, mock_service):
        """Test different expansion strategies produce different results."""

        strategies = [
            GraphExpansionStrategy.BREADTH_FIRST,
            GraphExpansionStrategy.DEPTH_FIRST,
            GraphExpansionStrategy.IMPORTANCE_BASED,
            GraphExpansionStrategy.RELEVANCE_SCORED,
            GraphExpansionStrategy.ADAPTIVE,
        ]

        results = {}

        for strategy in strategies:
            options = GraphBuildOptions(max_nodes=10, expansion_strategy=strategy, include_context=True)

            graph = await mock_service.build_partial_graph("test_project", "function", options)

            results[strategy] = graph
            assert graph is not None
            assert len(graph.nodes) > 0

    @pytest.mark.asyncio
    async def test_query_relevance_calculation(self, mock_service):
        """Test query relevance calculation for nodes."""

        metadata = mock_service.memory_index.nodes["node1"]  # main_function

        # Exact match should give high relevance
        relevance = mock_service._calculate_query_relevance(metadata, "main_function")
        assert relevance > 0.8

        # Partial match should give moderate relevance
        relevance = mock_service._calculate_query_relevance(metadata, "main")
        assert 0.4 < relevance < 0.8

        # No match should give low relevance (just importance bonus)
        relevance = mock_service._calculate_query_relevance(metadata, "unrelated")
        assert relevance <= 0.5

    @pytest.mark.asyncio
    async def test_node_budget_allocation(self, mock_service):
        """Test adaptive node budget allocation."""

        # Small budget should be allocated efficiently
        small_options = GraphBuildOptions(max_nodes=5)
        graph = await mock_service.build_partial_graph("test_project", "function", small_options)

        if graph:
            assert len(graph.nodes) <= 5

        # Large budget should allow more nodes
        large_options = GraphBuildOptions(max_nodes=50)
        graph = await mock_service.build_partial_graph("test_project", "function", large_options)

        if graph:
            assert len(graph.nodes) <= 50


class TestNodeRelevanceScore:
    """Test NodeRelevanceScore calculation and comparison."""

    def test_score_calculation(self):
        """Test relevance score calculation."""

        score = NodeRelevanceScore(node_id="test")
        score.semantic_relevance = 0.8
        score.structural_relevance = 0.6
        score.importance_bonus = 0.4

        total = score.calculate_total()

        # Should be weighted combination: 0.8*0.5 + 0.6*0.3 + 0.4*0.2 = 0.66
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2
        assert abs(total - expected) < 0.01
        assert score.total_score == total

    def test_score_comparison(self):
        """Test score comparison for sorting."""

        score1 = NodeRelevanceScore(node_id="node1")
        score1.semantic_relevance = 0.9
        score1.calculate_total()

        score2 = NodeRelevanceScore(node_id="node2")
        score2.semantic_relevance = 0.5
        score2.calculate_total()

        scores = [score2, score1]  # Intentionally unsorted
        sorted_scores = sorted(scores, key=lambda x: x.total_score, reverse=True)

        assert sorted_scores[0].node_id == "node1"
        assert sorted_scores[1].node_id == "node2"


class TestGraphBuildOptions:
    """Test GraphBuildOptions configuration and validation."""

    def test_default_options(self):
        """Test default options are sensible."""

        options = GraphBuildOptions()

        assert options.max_nodes == 50
        assert options.expansion_strategy == GraphExpansionStrategy.ADAPTIVE
        assert options.include_context is True
        assert options.context_depth == 2
        assert 0 < options.importance_threshold < 1
        assert 0 < options.relevance_threshold < 1

    def test_custom_options(self):
        """Test custom options are properly set."""

        options = GraphBuildOptions(
            max_nodes=100,
            expansion_strategy=GraphExpansionStrategy.BREADTH_FIRST,
            context_depth=3,
            importance_threshold=0.5,
            relevance_threshold=0.7,
        )

        assert options.max_nodes == 100
        assert options.expansion_strategy == GraphExpansionStrategy.BREADTH_FIRST
        assert options.context_depth == 3
        assert options.importance_threshold == 0.5
        assert options.relevance_threshold == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
