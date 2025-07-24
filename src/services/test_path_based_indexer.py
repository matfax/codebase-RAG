"""
Unit tests for the Path-Based Indexer service.

This module tests the comprehensive path extraction algorithms including
execution path extraction, data flow path extraction, and dependency path extraction.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

from src.models.code_chunk import ChunkType
from src.models.relational_path import (
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathConfidence,
    PathType,
)
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph
from src.services.path_based_indexer import (
    DependencyPathExtractor,
    ExtractionContext,
    ExtractionOptions,
    ExecutionPathExtractor,
    PathBasedIndexer,
    create_path_based_indexer,
)


class TestExtractionOptions:
    """Test extraction options configuration."""
    
    def test_default_options(self):
        """Test default extraction options."""
        options = ExtractionOptions()
        
        assert options.max_path_length == 50
        assert options.max_paths_per_type == 100
        assert options.min_confidence_threshold == 0.3
        assert options.enable_execution_paths
        assert options.enable_data_flow_paths
        assert options.enable_dependency_paths
        assert options.filter_trivial_paths
        assert options.filter_duplicate_paths
        assert options.prioritize_critical_paths
    
    def test_custom_options(self):
        """Test custom extraction options."""
        options = ExtractionOptions(
            max_path_length=20,
            max_paths_per_type=50,
            min_confidence_threshold=0.5,
            enable_execution_paths=False,
            filter_trivial_paths=False
        )
        
        assert options.max_path_length == 20
        assert options.max_paths_per_type == 50
        assert options.min_confidence_threshold == 0.5
        assert not options.enable_execution_paths
        assert not options.filter_trivial_paths


class TestExtractionContext:
    """Test extraction context management."""
    
    def test_context_initialization(self):
        """Test extraction context initialization."""
        context = ExtractionContext(project_name="test_project")
        
        assert context.project_name == "test_project"
        assert len(context.processed_nodes) == 0
        assert len(context.processed_edges) == 0
        assert len(context.path_cache) == 0
        assert len(context.extraction_warnings) == 0
        assert context.statistics["nodes_processed"] == 0
        assert context.statistics["edges_processed"] == 0
        assert context.statistics["paths_extracted"] == 0
        assert context.statistics["paths_filtered"] == 0
    
    def test_context_state_tracking(self):
        """Test context state tracking."""
        context = ExtractionContext(project_name="test_project")
        
        # Add processed items
        context.processed_nodes.add("node1")
        context.processed_edges.add(("node1", "node2"))
        context.extraction_warnings.append("Warning message")
        context.statistics["nodes_processed"] = 5
        
        assert "node1" in context.processed_nodes
        assert ("node1", "node2") in context.processed_edges
        assert "Warning message" in context.extraction_warnings
        assert context.statistics["nodes_processed"] == 5


class TestExecutionPathExtractor:
    """Test execution path extraction logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ExecutionPathExtractor()
        
        # Create sample graph nodes
        self.node1 = GraphNode(
            chunk_id="chunk1",
            breadcrumb="module.function1",
            name="function1",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            semantic_weight=0.8
        )
        
        self.node2 = GraphNode(
            chunk_id="chunk2",
            breadcrumb="module.function2",
            name="function2",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            semantic_weight=0.6
        )
        
        # Create sample graph with function call edge
        self.graph = StructureGraph(
            nodes={
                "module.function1": self.node1,
                "module.function2": self.node2
            },
            edges=[
                GraphEdge(
                    source_breadcrumb="module.function1",
                    target_breadcrumb="module.function2",
                    relationship_type="function_call",
                    weight=0.9,
                    confidence=0.8,
                    metadata={"call_type": "direct", "line_number": 15}
                )
            ],
            project_name="test_project"
        )
    
    @pytest.mark.asyncio
    async def test_extract_from_node_success(self):
        """Test successful execution path extraction."""
        options = ExtractionOptions()
        context = ExtractionContext(project_name="test_project")
        
        paths = await self.extractor.extract_from_node(self.node1, self.graph, options, context)
        
        assert len(paths) == 1
        path = paths[0]
        assert isinstance(path, ExecutionPath)
        assert path.path_type == PathType.EXECUTION_PATH
        assert len(path.nodes) == 2
        assert len(path.edges) == 1
        assert path.nodes[0].breadcrumb == "module.function1"
        assert path.nodes[1].breadcrumb == "module.function2"
        assert path.nodes[0].role_in_path == "source"
        assert path.nodes[1].role_in_path == "intermediate"
    
    @pytest.mark.asyncio
    async def test_extract_from_node_no_calls(self):
        """Test extraction from node with no function calls."""
        # Create graph with no function call edges
        empty_graph = StructureGraph(
            nodes={"module.function1": self.node1},
            edges=[],
            project_name="test_project"
        )
        
        options = ExtractionOptions()
        context = ExtractionContext(project_name="test_project")
        
        paths = await self.extractor.extract_from_node(self.node1, empty_graph, options, context)
        
        assert len(paths) == 0
    
    @pytest.mark.asyncio
    async def test_extract_with_max_depth_limit(self):
        """Test extraction respects max path length limit."""
        # Create a longer chain: function1 -> function2 -> function3
        node3 = GraphNode(
            chunk_id="chunk3",
            breadcrumb="module.function3",
            name="function3",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            semantic_weight=0.4
        )
        
        extended_graph = StructureGraph(
            nodes={
                "module.function1": self.node1,
                "module.function2": self.node2,
                "module.function3": node3
            },
            edges=[
                GraphEdge(
                    source_breadcrumb="module.function1",
                    target_breadcrumb="module.function2",
                    relationship_type="function_call",
                    weight=0.9,
                    confidence=0.8
                ),
                GraphEdge(
                    source_breadcrumb="module.function2",
                    target_breadcrumb="module.function3",
                    relationship_type="function_call",
                    weight=0.7,
                    confidence=0.6
                )
            ],
            project_name="test_project"
        )
        
        options = ExtractionOptions(max_path_length=2)
        context = ExtractionContext(project_name="test_project")
        
        paths = await self.extractor.extract_from_node(self.node1, extended_graph, options, context)
        
        assert len(paths) == 1
        path = paths[0]
        assert len(path.nodes) <= 3  # Should respect max_path_length + 1 (start node)
    
    def test_convert_to_path_node(self):
        """Test GraphNode to PathNode conversion."""
        path_node = self.extractor._convert_to_path_node(self.node1, "source")
        
        assert path_node.breadcrumb == "module.function1"
        assert path_node.name == "function1"
        assert path_node.chunk_type == ChunkType.FUNCTION
        assert path_node.file_path == "/test.py"
        assert path_node.role_in_path == "source"
        assert path_node.importance_score == 0.8
    
    def test_convert_to_path_edge(self):
        """Test GraphEdge to PathEdge conversion."""
        graph_edge = GraphEdge(
            source_breadcrumb="source",
            target_breadcrumb="target",
            relationship_type="function_call",
            weight=0.9,
            confidence=0.8,
            metadata={"call_expression": "target()", "line_number": 10}
        )
        
        path_edge = self.extractor._convert_to_path_edge(graph_edge)
        
        assert path_edge.source_node_id == "source"
        assert path_edge.target_node_id == "target"
        assert path_edge.relationship_type == "function_call"
        assert path_edge.weight == 0.9
        assert path_edge.confidence == PathConfidence.VERY_HIGH  # 0.8 confidence maps to VERY_HIGH
        assert path_edge.call_expression == "target()"
        assert path_edge.line_number == 10


class TestDependencyPathExtractor:
    """Test dependency path extraction logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = DependencyPathExtractor()
        
        # Create sample import node
        self.import_node = GraphNode(
            chunk_id="import1",
            breadcrumb="module.imports.requests",
            name="requests",
            chunk_type=ChunkType.IMPORT,
            file_path="/main.py",
            semantic_weight=0.7
        )
        
        # Create target module node
        self.module_node = GraphNode(
            chunk_id="module1",
            breadcrumb="external.requests",
            name="requests",
            chunk_type=ChunkType.MODULE_DOCSTRING,
            file_path="/external/requests.py",
            semantic_weight=0.9
        )
        
        # Create graph with dependency edge
        self.graph = StructureGraph(
            nodes={
                "module.imports.requests": self.import_node,
                "external.requests": self.module_node
            },
            edges=[
                GraphEdge(
                    source_breadcrumb="module.imports.requests",
                    target_breadcrumb="external.requests",
                    relationship_type="dependency",
                    weight=0.8,
                    confidence=0.9
                )
            ],
            project_name="test_project"
        )
    
    @pytest.mark.asyncio
    async def test_extract_dependency_paths(self):
        """Test dependency path extraction."""
        options = ExtractionOptions()
        context = ExtractionContext(project_name="test_project")
        
        paths = await self.extractor.extract_from_node(self.import_node, self.graph, options, context)
        
        assert len(paths) == 1
        path = paths[0]
        assert isinstance(path, DependencyPath)
        assert path.path_type == PathType.DEPENDENCY_PATH
        assert len(path.nodes) == 2
        assert path.dependency_type == "import"
        assert "requests" in path.required_modules
        assert path.stability_score == 0.7  # Same as import node semantic weight
    
    @pytest.mark.asyncio
    async def test_extract_no_dependencies(self):
        """Test extraction from import node with no dependencies."""
        empty_graph = StructureGraph(
            nodes={"module.imports.requests": self.import_node},
            edges=[],
            project_name="test_project"
        )
        
        options = ExtractionOptions()
        context = ExtractionContext(project_name="test_project")
        
        paths = await self.extractor.extract_from_node(self.import_node, empty_graph, options, context)
        
        assert len(paths) == 0


class TestPathBasedIndexer:
    """Test the main PathBasedIndexer service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock lightweight graph service
        self.mock_lightweight_graph = Mock()
        self.mock_graph_rag_service = Mock()
        
        # Create indexer
        self.indexer = PathBasedIndexer(
            self.mock_lightweight_graph,
            self.mock_graph_rag_service
        )
        
        # Create sample structure graph
        self.sample_graph = StructureGraph(
            nodes={
                "main.function1": GraphNode(
                    chunk_id="f1",
                    breadcrumb="main.function1",
                    name="function1",
                    chunk_type=ChunkType.FUNCTION,
                    file_path="/main.py",
                    semantic_weight=0.8
                ),
                "main.function2": GraphNode(
                    chunk_id="f2",
                    breadcrumb="main.function2",
                    name="function2",
                    chunk_type=ChunkType.FUNCTION,
                    file_path="/main.py",
                    semantic_weight=0.6
                ),
                "main.variable1": GraphNode(
                    chunk_id="v1",
                    breadcrumb="main.variable1",
                    name="variable1",
                    chunk_type=ChunkType.VARIABLE,
                    file_path="/main.py",
                    semantic_weight=0.5
                )
            },
            edges=[
                GraphEdge(
                    source_breadcrumb="main.function1",
                    target_breadcrumb="main.function2",
                    relationship_type="function_call",
                    weight=0.9,
                    confidence=0.8
                )
            ],
            project_name="test_project"
        )
    
    @pytest.mark.asyncio
    async def test_extract_relational_paths_success(self):
        """Test successful path extraction."""
        # Mock the structure graph retrieval
        self.indexer._get_project_structure_graph = AsyncMock(return_value=self.sample_graph)
        
        result = await self.indexer.extract_relational_paths("test_project")
        
        assert result.is_successful()
        assert result.path_collection.get_total_path_count() >= 0
        assert result.processing_time_ms > 0
        assert result.source_chunks_count == 3  # Number of nodes in sample graph
    
    @pytest.mark.asyncio
    async def test_extract_relational_paths_with_entry_points(self):
        """Test path extraction with specific entry points."""
        self.indexer._get_project_structure_graph = AsyncMock(return_value=self.sample_graph)
        
        result = await self.indexer.extract_relational_paths(
            "test_project",
            entry_points=["main.function1"]
        )
        
        assert result.is_successful()
        assert result.path_collection.primary_entry_points == ["main.function1"]
    
    @pytest.mark.asyncio
    async def test_extract_relational_paths_with_options(self):
        """Test path extraction with custom options."""
        self.indexer._get_project_structure_graph = AsyncMock(return_value=self.sample_graph)
        
        options = ExtractionOptions(
            enable_execution_paths=True,
            enable_data_flow_paths=False,
            enable_dependency_paths=False,
            max_paths_per_type=10
        )
        
        result = await self.indexer.extract_relational_paths(
            "test_project",
            options=options
        )
        
        assert result.is_successful()
        assert len(result.path_collection.data_flow_paths) == 0
        assert len(result.path_collection.dependency_paths) == 0
    
    @pytest.mark.asyncio
    async def test_extract_relational_paths_no_graph(self):
        """Test path extraction when no structure graph is available."""
        self.indexer._get_project_structure_graph = AsyncMock(return_value=None)
        
        result = await self.indexer.extract_relational_paths("test_project")
        
        assert not result.is_successful()
        assert result.path_collection.get_total_path_count() == 0
        assert "No structure graph available" in result.extraction_warnings[0]
    
    @pytest.mark.asyncio
    async def test_extract_relational_paths_with_error(self):
        """Test path extraction with errors."""
        # Mock to raise an exception
        self.indexer._get_project_structure_graph = AsyncMock(side_effect=Exception("Test error"))
        
        result = await self.indexer.extract_relational_paths("test_project")
        
        assert not result.is_successful()
        assert result.path_collection.get_total_path_count() == 0
        assert "Extraction failed: Test error" in result.extraction_warnings[0]
    
    @pytest.mark.asyncio
    async def test_determine_extraction_scope_with_entry_points(self):
        """Test extraction scope determination with entry points."""
        entry_points = ["main.function1"]
        options = ExtractionOptions()
        
        target_nodes = await self.indexer._determine_extraction_scope(
            self.sample_graph, entry_points, options
        )
        
        # Should include the entry point and its neighbors
        breadcrumbs = {node.breadcrumb for node in target_nodes}
        assert "main.function1" in breadcrumbs
        assert "main.function2" in breadcrumbs  # Connected via function call
    
    @pytest.mark.asyncio
    async def test_determine_extraction_scope_no_entry_points(self):
        """Test extraction scope determination without entry points."""
        options = ExtractionOptions(prioritize_critical_paths=True)
        
        target_nodes = await self.indexer._determine_extraction_scope(
            self.sample_graph, None, options
        )
        
        # Should include all nodes, sorted by importance
        assert len(target_nodes) == 3
        # Should be sorted by semantic weight (descending)
        assert target_nodes[0].semantic_weight >= target_nodes[1].semantic_weight
    
    def test_filter_duplicate_execution_paths(self):
        """Test duplicate execution path filtering."""
        # Create paths with same node sequence
        path1 = ExecutionPath(
            path_id="path1",
            nodes=[
                Mock(breadcrumb="a"),
                Mock(breadcrumb="b")
            ]
        )
        path2 = ExecutionPath(
            path_id="path2",
            nodes=[
                Mock(breadcrumb="a"),
                Mock(breadcrumb="b")
            ]
        )
        path3 = ExecutionPath(
            path_id="path3",
            nodes=[
                Mock(breadcrumb="c"),
                Mock(breadcrumb="d")
            ]
        )
        
        filtered = self.indexer._filter_duplicate_execution_paths([path1, path2, path3])
        
        assert len(filtered) == 2  # path1 and path3 (path2 is duplicate of path1)
    
    def test_calculate_coverage_score(self):
        """Test coverage score calculation."""
        target_nodes = [
            Mock(breadcrumb="a"),
            Mock(breadcrumb="b"),
            Mock(breadcrumb="c")
        ]
        
        paths = [
            Mock(nodes=[Mock(breadcrumb="a"), Mock(breadcrumb="b")]),
            Mock(nodes=[Mock(breadcrumb="c")])
        ]
        
        coverage = self.indexer._calculate_coverage_score(target_nodes, paths)
        
        assert coverage == 1.0  # All target nodes are covered
    
    def test_calculate_coherence_score(self):
        """Test coherence score calculation."""
        paths = [
            Mock(nodes=[Mock(breadcrumb="a"), Mock(breadcrumb="b")]),
            Mock(nodes=[Mock(breadcrumb="b"), Mock(breadcrumb="c")])
        ]
        
        coherence = self.indexer._calculate_coherence_score(paths)
        
        assert 0.0 <= coherence <= 1.0
        assert coherence > 0.5  # Should have some coherence due to shared node "b"
    
    def test_is_high_confidence_path(self):
        """Test high confidence path detection."""
        # High confidence path
        high_confidence_path = Mock(
            edges=[
                Mock(confidence=PathConfidence.HIGH),
                Mock(confidence=PathConfidence.VERY_HIGH)
            ]
        )
        
        assert self.indexer._is_high_confidence_path(high_confidence_path)
        
        # Low confidence path
        low_confidence_path = Mock(
            edges=[
                Mock(confidence=PathConfidence.LOW),
                Mock(confidence=PathConfidence.MEDIUM)
            ]
        )
        
        assert not self.indexer._is_high_confidence_path(low_confidence_path)
    
    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        # Update some stats
        self.indexer._update_performance_stats(1000.0, 5, True)
        self.indexer._update_performance_stats(2000.0, 3, False)
        
        stats = self.indexer.get_performance_stats()
        
        assert stats["total_extractions"] == 2
        assert stats["successful_extractions"] == 1
        assert stats["failed_extractions"] == 1
        assert stats["total_paths_extracted"] == 8
        assert stats["average_extraction_time_ms"] == 1500.0  # (1000 + 2000) / 2


class TestServiceFactory:
    """Test the service factory function."""
    
    def test_create_path_based_indexer(self):
        """Test path-based indexer creation."""
        mock_lightweight_graph = Mock()
        mock_graph_rag = Mock()
        
        indexer = create_path_based_indexer(mock_lightweight_graph, mock_graph_rag)
        
        assert isinstance(indexer, PathBasedIndexer)
        assert indexer.lightweight_graph == mock_lightweight_graph
        assert indexer.graph_rag_service == mock_graph_rag
    
    def test_create_path_based_indexer_minimal(self):
        """Test path-based indexer creation with minimal parameters."""
        mock_lightweight_graph = Mock()
        
        indexer = create_path_based_indexer(mock_lightweight_graph)
        
        assert isinstance(indexer, PathBasedIndexer)
        assert indexer.lightweight_graph == mock_lightweight_graph
        assert indexer.graph_rag_service is None


if __name__ == "__main__":
    pytest.main([__file__])