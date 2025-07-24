"""
Unit tests for the Lightweight Graph Service.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.append("/Users/jeff/Documents/personal/Agentic-RAG/trees/agentic-rag-performance-enhancement-wave")

from src.models.code_chunk import ChunkType, CodeChunk
from src.services.lightweight_graph_service import LightweightGraphService, MemoryIndex, NodeMetadata
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing."""
    return [
        CodeChunk(
            chunk_id="chunk_1",
            file_path="/test/main.py",
            content="def main():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            language="python",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
            name="main",
            breadcrumb="main",
            signature="def main():",
            docstring="Main entry point",
        ),
        CodeChunk(
            chunk_id="chunk_2",
            file_path="/test/utils.py",
            content="class Utils:\n    def helper(self):\n        pass",
            chunk_type=ChunkType.CLASS,
            language="python",
            start_line=1,
            end_line=3,
            start_byte=0,
            end_byte=40,
            name="Utils",
            breadcrumb="utils.Utils",
            signature="class Utils:",
        ),
        CodeChunk(
            chunk_id="chunk_3",
            file_path="/test/utils.py",
            content="    def helper(self):\n        pass",
            chunk_type=ChunkType.METHOD,
            language="python",
            start_line=2,
            end_line=3,
            start_byte=15,
            end_byte=40,
            name="helper",
            parent_name="Utils",
            breadcrumb="utils.Utils.helper",
            signature="def helper(self):",
        ),
    ]


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    graph_rag_service = AsyncMock()
    hybrid_search_service = AsyncMock()
    cache_service = AsyncMock()

    return graph_rag_service, hybrid_search_service, cache_service


@pytest.fixture
def lightweight_service(mock_services):
    """Create LightweightGraphService instance for testing."""
    graph_rag_service, hybrid_search_service, cache_service = mock_services

    service = LightweightGraphService(
        graph_rag_service=graph_rag_service,
        hybrid_search_service=hybrid_search_service,
        cache_service=cache_service,
        enable_timeout=True,
        default_timeout=15,
        enable_progressive_results=True,
        confidence_threshold=0.7,
    )

    return service


class TestLightweightGraphService:
    """Test cases for LightweightGraphService."""

    @pytest.mark.asyncio
    async def test_initialize_memory_index(self, lightweight_service, sample_chunks, mock_services):
        """Test memory index initialization."""
        _, hybrid_search_service, _ = mock_services
        hybrid_search_service.get_all_chunks.return_value = sample_chunks

        # Test successful initialization
        result = await lightweight_service.initialize_memory_index("test_project")

        assert result is True
        assert lightweight_service.memory_index.total_nodes == 3
        assert len(lightweight_service.memory_index.nodes) == 3

        # Verify nodes are properly indexed
        assert "chunk_1" in lightweight_service.memory_index.nodes
        assert "chunk_2" in lightweight_service.memory_index.nodes
        assert "chunk_3" in lightweight_service.memory_index.nodes

        # Verify secondary indices
        assert "main" in lightweight_service.memory_index.by_name
        assert "Utils" in lightweight_service.memory_index.by_name
        assert "helper" in lightweight_service.memory_index.by_name

        # Verify breadcrumb index
        assert "main" in lightweight_service.memory_index.by_breadcrumb
        assert "utils.Utils" in lightweight_service.memory_index.by_breadcrumb
        assert "utils.Utils.helper" in lightweight_service.memory_index.by_breadcrumb

    @pytest.mark.asyncio
    async def test_memory_index_with_empty_chunks(self, lightweight_service, mock_services):
        """Test memory index initialization with empty chunks."""
        _, hybrid_search_service, _ = mock_services
        hybrid_search_service.get_all_chunks.return_value = []

        result = await lightweight_service.initialize_memory_index("empty_project")

        assert result is False
        assert lightweight_service.memory_index.total_nodes == 0

    @pytest.mark.asyncio
    async def test_add_chunk_to_index(self, lightweight_service, sample_chunks):
        """Test adding individual chunks to memory index."""
        chunk = sample_chunks[0]

        await lightweight_service._add_chunk_to_index(chunk)

        # Verify node was added
        assert chunk.chunk_id in lightweight_service.memory_index.nodes

        node_metadata = lightweight_service.memory_index.nodes[chunk.chunk_id]
        assert node_metadata.name == "main"
        assert node_metadata.chunk_type == ChunkType.FUNCTION
        assert node_metadata.file_path == "/test/main.py"
        assert node_metadata.breadcrumb == "main"
        assert node_metadata.importance_score > 0

    @pytest.mark.asyncio
    async def test_build_relationship_indices(self, lightweight_service, sample_chunks):
        """Test building relationship indices."""
        # Add chunks to index first
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)

        # Build relationships
        await lightweight_service._build_relationship_indices(sample_chunks)

        # Verify parent-child relationships
        utils_id = "chunk_2"  # Utils class
        helper_id = "chunk_3"  # helper method

        assert helper_id in lightweight_service.memory_index.children_index[utils_id]
        assert utils_id in lightweight_service.memory_index.parent_index[helper_id]

        # Verify node metadata relationships
        helper_metadata = lightweight_service.memory_index.nodes[helper_id]
        assert utils_id in helper_metadata.parent_ids

    @pytest.mark.asyncio
    async def test_precompute_common_queries(self, lightweight_service, sample_chunks):
        """Test pre-computing common queries."""
        # Initialize index with sample chunks
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)

        await lightweight_service._precompute_common_queries("test_project")

        # Verify entry points were identified
        entry_points = lightweight_service.precomputed_queries["entry_points"]["test_project"]
        assert "chunk_1" in entry_points  # main function should be identified as entry point

        # Verify main functions were identified
        main_functions = lightweight_service.precomputed_queries["main_functions"]["test_project"]
        assert len(main_functions) > 0

        # Verify public APIs were identified
        public_apis = lightweight_service.precomputed_queries["public_apis"]["test_project"]
        assert len(public_apis) > 0

    def test_calculate_importance_score(self, lightweight_service, sample_chunks):
        """Test importance score calculation."""
        # Test main function (should have high score)
        main_chunk = sample_chunks[0]
        main_score = lightweight_service._calculate_importance_score(main_chunk)
        assert main_score > 1.5  # Should be high due to name "main" and docstring

        # Test class (should have high score)
        class_chunk = sample_chunks[1]
        class_score = lightweight_service._calculate_importance_score(class_chunk)
        assert class_score > 1.0  # Classes have high base score

        # Test method (should have moderate score)
        method_chunk = sample_chunks[2]
        method_score = lightweight_service._calculate_importance_score(method_chunk)
        assert method_score > 0.5

    def test_is_index_fresh(self, lightweight_service):
        """Test index freshness check."""
        # Empty index should not be fresh
        assert not lightweight_service._is_index_fresh("test_project")

        # Add some data and update timestamp
        lightweight_service.memory_index.nodes["test"] = NodeMetadata("test", "test", ChunkType.FUNCTION, "/test")
        lightweight_service.memory_index.last_updated = datetime.now()

        # Should be fresh now
        assert lightweight_service._is_index_fresh("test_project")

        # Make it old
        lightweight_service.memory_index.last_updated = datetime.now() - timedelta(hours=1)
        assert not lightweight_service._is_index_fresh("test_project")

    @pytest.mark.asyncio
    async def test_build_partial_graph(self, lightweight_service, sample_chunks, mock_services):
        """Test building partial graph for specific query scope."""
        # Initialize memory index
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)
        await lightweight_service._build_relationship_indices(sample_chunks)

        # Test building partial graph for "Utils" class
        graph = await lightweight_service.build_partial_graph(
            project_name="test_project", query_scope="Utils", max_nodes=10, include_context=True
        )

        assert graph is not None
        assert isinstance(graph, StructureGraph)
        assert len(graph.nodes) <= 10
        assert "chunk_2" in graph.nodes  # Utils class should be included

    @pytest.mark.asyncio
    async def test_find_target_nodes(self, lightweight_service, sample_chunks):
        """Test finding target nodes by query scope."""
        # Initialize memory index
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)

        # Test exact name match
        target_nodes = await lightweight_service._find_target_nodes("main")
        assert "chunk_1" in target_nodes

        # Test partial name match
        target_nodes = await lightweight_service._find_target_nodes("util")
        assert "chunk_2" in target_nodes  # Should match "Utils"

        # Test breadcrumb search
        target_nodes = await lightweight_service._find_target_nodes("utils.Utils")
        assert "chunk_2" in target_nodes

    @pytest.mark.asyncio
    async def test_get_context_nodes(self, lightweight_service, sample_chunks):
        """Test getting context nodes for target nodes."""
        # Initialize memory index with relationships
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)
        await lightweight_service._build_relationship_indices(sample_chunks)

        target_nodes = {"chunk_2"}  # Utils class
        context_nodes = await lightweight_service._get_context_nodes(target_nodes, 5)

        # Should include helper method as child context
        assert "chunk_3" in context_nodes
        assert "chunk_2" not in context_nodes  # Should exclude target nodes

    @pytest.mark.asyncio
    async def test_get_precomputed_query(self, lightweight_service):
        """Test getting pre-computed query results."""
        # Set up some pre-computed data
        lightweight_service.precomputed_queries["entry_points"]["test_project"] = ["chunk_1", "chunk_2"]

        result = await lightweight_service.get_precomputed_query("test_project", "entry_points")
        assert result == ["chunk_1", "chunk_2"]

        # Test non-existent query type
        result = await lightweight_service.get_precomputed_query("test_project", "non_existent")
        assert result == []

        # Test non-existent project
        result = await lightweight_service.get_precomputed_query("non_existent", "entry_points")
        assert result == []

    @pytest.mark.asyncio
    async def test_find_intelligent_path(self, lightweight_service, sample_chunks):
        """Test intelligent path finding."""
        # Initialize memory index with relationships
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)
        await lightweight_service._build_relationship_indices(sample_chunks)

        # Test path finding between Utils class and helper method
        path = await lightweight_service.find_intelligent_path("Utils", "helper", max_depth=5)

        assert path is not None
        assert len(path) >= 2
        assert "chunk_2" in path  # Utils class
        assert "chunk_3" in path  # helper method

        # Test caching - second call should hit cache
        path2 = await lightweight_service.find_intelligent_path("Utils", "helper", max_depth=5)
        assert path2 == path
        assert lightweight_service.performance_metrics["cache_hits"] > 0

    @pytest.mark.asyncio
    async def test_resolve_node_id(self, lightweight_service, sample_chunks):
        """Test resolving node names/breadcrumbs to node IDs."""
        # Initialize memory index
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)

        # Test resolving by existing node ID
        node_id = await lightweight_service._resolve_node_id("chunk_1")
        assert node_id == "chunk_1"

        # Test resolving by name
        node_id = await lightweight_service._resolve_node_id("main")
        assert node_id == "chunk_1"

        # Test resolving by breadcrumb
        node_id = await lightweight_service._resolve_node_id("utils.Utils")
        assert node_id == "chunk_2"

        # Test non-existent identifier
        node_id = await lightweight_service._resolve_node_id("non_existent")
        assert node_id is None

    @pytest.mark.asyncio
    async def test_bfs_path_search(self, lightweight_service, sample_chunks):
        """Test BFS path search algorithm."""
        # Initialize memory index with relationships
        for chunk in sample_chunks:
            await lightweight_service._add_chunk_to_index(chunk)
        await lightweight_service._build_relationship_indices(sample_chunks)

        # Test direct path (same node)
        path = await lightweight_service._bfs_path_search("chunk_1", "chunk_1", 5)
        assert path == ["chunk_1"]

        # Test path between related nodes
        path = await lightweight_service._bfs_path_search("chunk_2", "chunk_3", 5)
        assert path is not None
        assert path[0] == "chunk_2"
        assert path[-1] == "chunk_3"

        # Test no path found
        lightweight_service.memory_index.nodes["isolated"] = NodeMetadata("isolated", "isolated", ChunkType.FUNCTION, "/test")
        path = await lightweight_service._bfs_path_search("chunk_1", "isolated", 2)
        assert path is None

    def test_get_memory_index_stats(self, lightweight_service, sample_chunks):
        """Test getting memory index statistics."""
        # Initialize with sample data
        for chunk in sample_chunks:
            lightweight_service.memory_index.nodes[chunk.chunk_id] = NodeMetadata(
                chunk.chunk_id, chunk.name or "unnamed", chunk.chunk_type, chunk.file_path
            )
            lightweight_service.memory_index.by_type[chunk.chunk_type].add(chunk.chunk_id)
            lightweight_service.memory_index.by_language[chunk.language].add(chunk.chunk_id)

        lightweight_service.memory_index.total_nodes = 3
        lightweight_service.memory_index.hit_count = 10
        lightweight_service.memory_index.miss_count = 5

        stats = lightweight_service.get_memory_index_stats()

        assert stats["total_nodes"] == 3
        assert "nodes_by_type" in stats
        assert "nodes_by_language" in stats
        assert stats["hit_rate"] == 10 / 15  # 10 hits out of 15 total
        assert "last_updated" in stats
        assert "performance_metrics" in stats

    @pytest.mark.asyncio
    async def test_force_rebuild_index(self, lightweight_service, sample_chunks, mock_services):
        """Test force rebuilding memory index."""
        _, hybrid_search_service, _ = mock_services
        hybrid_search_service.get_all_chunks.return_value = sample_chunks

        # Initialize index first time
        result1 = await lightweight_service.initialize_memory_index("test_project")
        assert result1 is True
        original_timestamp = lightweight_service.memory_index.last_updated

        # Wait a bit and force rebuild
        await asyncio.sleep(0.1)
        result2 = await lightweight_service.initialize_memory_index("test_project", force_rebuild=True)
        assert result2 is True

        # Timestamp should be updated
        assert lightweight_service.memory_index.last_updated > original_timestamp

    @pytest.mark.asyncio
    async def test_concurrent_access(self, lightweight_service, sample_chunks, mock_services):
        """Test concurrent access to memory index."""
        _, hybrid_search_service, _ = mock_services
        hybrid_search_service.get_all_chunks.return_value = sample_chunks

        # Test concurrent initialization (should be thread-safe)
        tasks = [
            lightweight_service.initialize_memory_index("test_project"),
            lightweight_service.initialize_memory_index("test_project"),
            lightweight_service.initialize_memory_index("test_project"),
        ]

        results = await asyncio.gather(*tasks)
        assert all(results)  # All should succeed
        assert lightweight_service.memory_index.total_nodes == 3  # Should not be corrupted


class TestNodeMetadata:
    """Test cases for NodeMetadata dataclass."""

    def test_node_metadata_creation(self):
        """Test creating NodeMetadata instance."""
        metadata = NodeMetadata(
            node_id="test_id",
            name="test_name",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/file.py",
            breadcrumb="test.breadcrumb",
            importance_score=1.5,
        )

        assert metadata.node_id == "test_id"
        assert metadata.name == "test_name"
        assert metadata.chunk_type == ChunkType.FUNCTION
        assert metadata.file_path == "/test/file.py"
        assert metadata.breadcrumb == "test.breadcrumb"
        assert metadata.importance_score == 1.5
        assert metadata.access_count == 0
        assert isinstance(metadata.children_ids, set)
        assert isinstance(metadata.parent_ids, set)
        assert isinstance(metadata.dependency_ids, set)


class TestMemoryIndex:
    """Test cases for MemoryIndex dataclass."""

    def test_memory_index_creation(self):
        """Test creating MemoryIndex instance."""
        index = MemoryIndex()

        assert isinstance(index.nodes, dict)
        assert isinstance(index.by_name, dict)
        assert isinstance(index.by_type, dict)
        assert isinstance(index.by_file, dict)
        assert isinstance(index.by_breadcrumb, dict)
        assert isinstance(index.by_language, dict)
        assert index.total_nodes == 0
        assert index.hit_count == 0
        assert index.miss_count == 0

    def test_memory_index_operations(self):
        """Test basic memory index operations."""
        index = MemoryIndex()

        # Add a node
        metadata = NodeMetadata("test_id", "test_name", ChunkType.FUNCTION, "/test/file.py")
        index.nodes["test_id"] = metadata
        index.by_name["test_name"].add("test_id")
        index.by_type[ChunkType.FUNCTION].add("test_id")
        index.total_nodes = 1

        assert "test_id" in index.nodes
        assert "test_id" in index.by_name["test_name"]
        assert "test_id" in index.by_type[ChunkType.FUNCTION]
        assert index.total_nodes == 1


if __name__ == "__main__":
    pytest.main([__file__])
