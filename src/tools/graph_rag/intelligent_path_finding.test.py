"""
Task 1.4 Test: Intelligent Path Finding with Cache and Index Optimization

This test validates the implementation of intelligent path finding functionality
that prioritizes cache and index usage for fast path finding, building on the
existing memory indexing and caching infrastructure from Tasks 1.1-1.3.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.code_chunk import ChunkType, CodeChunk
from src.services.lightweight_graph_service import LightweightGraphService, MemoryIndex, NodeMetadata


class TestIntelligentPathFinding:
    """Test suite for Task 1.4: Intelligent Path Finding functionality."""

    @pytest.fixture
    async def mock_services(self):
        """Create mock services."""
        mock_graph_rag_service = AsyncMock()
        mock_hybrid_search_service = AsyncMock()

        # Mock get_all_chunks to return test data
        test_chunks = [
            CodeChunk(
                chunk_id="func1",
                chunk_type=ChunkType.FUNCTION,
                name="main_function",
                breadcrumb="main.main_function",
                file_path="/test/main.py",
                start_line=1,
                end_line=10,
                language="python",
            ),
            CodeChunk(
                chunk_id="func2",
                chunk_type=ChunkType.FUNCTION,
                name="helper_function",
                breadcrumb="main.helper_function",
                file_path="/test/main.py",
                start_line=12,
                end_line=20,
                language="python",
            ),
            CodeChunk(
                chunk_id="func3",
                chunk_type=ChunkType.FUNCTION,
                name="api_endpoint",
                breadcrumb="api.api_endpoint",
                file_path="/test/api.py",
                start_line=1,
                end_line=15,
                language="python",
            ),
        ]

        mock_hybrid_search_service.get_all_chunks.return_value = test_chunks

        return mock_graph_rag_service, mock_hybrid_search_service

    @pytest.fixture
    async def lightweight_service(self, mock_services):
        """Create LightweightGraphService with mocked dependencies."""
        mock_graph_rag_service, mock_hybrid_search_service = mock_services

        service = LightweightGraphService(
            graph_rag_service=mock_graph_rag_service,
            hybrid_search_service=mock_hybrid_search_service,
            enable_timeout=True,
            default_timeout=15,
            enable_progressive_results=True,
            confidence_threshold=0.7,
        )

        # Initialize memory index with test data
        await service.initialize_memory_index("test_project", force_rebuild=True)

        return service

    @pytest.mark.asyncio
    async def test_memory_index_initialization(self, lightweight_service):
        """Test that memory index is properly initialized with fast lookups."""

        # Validate memory index contains expected nodes
        assert lightweight_service.memory_index.total_nodes == 3

        # Test fast lookups by name
        assert "main_function" in lightweight_service.memory_index.by_name
        assert "helper_function" in lightweight_service.memory_index.by_name
        assert "api_endpoint" in lightweight_service.memory_index.by_name

        # Test fast lookups by breadcrumb
        assert "main.main_function" in lightweight_service.memory_index.by_breadcrumb
        assert "main.helper_function" in lightweight_service.memory_index.by_breadcrumb
        assert "api.api_endpoint" in lightweight_service.memory_index.by_breadcrumb

        # Test fast lookups by type
        assert ChunkType.FUNCTION in lightweight_service.memory_index.by_type
        assert len(lightweight_service.memory_index.by_type[ChunkType.FUNCTION]) == 3

    @pytest.mark.asyncio
    async def test_intelligent_path_finding_direct_connection(self, lightweight_service):
        """Test intelligent path finding for directly connected nodes."""

        # Create parent-child relationship
        lightweight_service.memory_index.nodes["func1"].children_ids.add("func2")
        lightweight_service.memory_index.nodes["func2"].parent_ids.add("func1")

        start_time = time.time()

        result = await lightweight_service.find_intelligent_path(start_node="main_function", end_node="helper_function", max_depth=5)

        execution_time = time.time() - start_time

        # Validate successful path finding
        assert result["success"] is True
        assert result["path"] == ["func1", "func2"]
        assert result["path_length"] == 2
        assert result["quality_score"] > 0.5
        assert execution_time < 0.1  # Should be very fast with memory index

    @pytest.mark.asyncio
    async def test_multi_layer_caching(self, lightweight_service):
        """Test multi-layer caching (L1, L2, L3) for path finding results."""

        # Create test path
        lightweight_service.memory_index.nodes["func1"].children_ids.add("func2")
        lightweight_service.memory_index.nodes["func2"].parent_ids.add("func1")

        # First call - should compute and cache
        result1 = await lightweight_service.find_intelligent_path(start_node="main_function", end_node="helper_function", max_depth=5)

        assert result1["success"] is True
        assert result1["cache_hit"] is None  # No cache hit on first call

        # Second call - should hit L1 cache
        result2 = await lightweight_service.find_intelligent_path(start_node="main_function", end_node="helper_function", max_depth=5)

        assert result2["success"] is True
        assert result2["cache_hit"] == "L1"
        assert result2["response_time_ms"] < result1["response_time_ms"]

    @pytest.mark.asyncio
    async def test_strategy_selection_intelligence(self, lightweight_service):
        """Test intelligent strategy selection based on graph characteristics."""

        # Test different node characteristics
        lightweight_service.memory_index.nodes["func1"].importance_score = 2.0  # High importance
        lightweight_service.memory_index.nodes["func1"].children_ids = {"func2", "func3", "func4", "func5"}  # High degree

        # Should select weighted strategies for high-importance, high-degree nodes
        strategies = await lightweight_service._select_optimal_strategies("func1", "func3", max_depth=10)

        assert "dijkstra" in strategies or "astar" in strategies
        assert len(strategies) >= 2  # Should try multiple strategies

    @pytest.mark.asyncio
    async def test_path_quality_scoring(self, lightweight_service):
        """Test path quality scoring based on multiple factors."""

        # Test short, high-quality path
        short_path = ["func1", "func2"]
        quality1 = await lightweight_service._calculate_path_quality_score(short_path)

        # Test longer path
        long_path = ["func1", "func2", "func3", "func4", "func5"]
        quality2 = await lightweight_service._calculate_path_quality_score(long_path)

        # Short path should generally have higher quality
        assert quality1 >= 0.0
        assert quality2 >= 0.0

        # Single node path should have perfect quality
        single_path = ["func1"]
        quality3 = await lightweight_service._calculate_path_quality_score(single_path)
        assert quality3 == 1.0

    @pytest.mark.asyncio
    async def test_optimized_bfs_with_memory_index(self, lightweight_service):
        """Test optimized BFS using memory index for O(1) neighbor lookups."""

        # Create path: func1 -> func2 -> func3
        lightweight_service.memory_index.nodes["func1"].children_ids.add("func2")
        lightweight_service.memory_index.nodes["func2"].parent_ids.add("func1")
        lightweight_service.memory_index.nodes["func2"].children_ids.add("func3")
        lightweight_service.memory_index.nodes["func3"].parent_ids.add("func2")

        start_time = time.time()

        path = await lightweight_service._optimized_bfs_path_search("func1", "func3", max_depth=5)

        execution_time = time.time() - start_time

        assert path == ["func1", "func2", "func3"]
        assert execution_time < 0.05  # Should be very fast with memory index

    @pytest.mark.asyncio
    async def test_dijkstra_with_importance_weights(self, lightweight_service):
        """Test Dijkstra algorithm using importance scores as weights."""

        # Set up importance scores
        lightweight_service.memory_index.nodes["func1"].importance_score = 1.0
        lightweight_service.memory_index.nodes["func2"].importance_score = 2.0  # Higher importance
        lightweight_service.memory_index.nodes["func3"].importance_score = 0.5

        # Create connections
        lightweight_service.memory_index.nodes["func1"].children_ids = {"func2", "func3"}
        lightweight_service.memory_index.nodes["func2"].parent_ids.add("func1")
        lightweight_service.memory_index.nodes["func3"].parent_ids.add("func1")

        path = await lightweight_service._dijkstra_path_search("func1", "func2", max_depth=5)

        assert path is not None
        assert "func1" in path
        assert "func2" in path

    @pytest.mark.asyncio
    async def test_bidirectional_search(self, lightweight_service):
        """Test bidirectional BFS for faster pathfinding."""

        # Create chain: func1 -> func2 -> func3
        lightweight_service.memory_index.nodes["func1"].children_ids.add("func2")
        lightweight_service.memory_index.nodes["func2"].parent_ids.add("func1")
        lightweight_service.memory_index.nodes["func2"].children_ids.add("func3")
        lightweight_service.memory_index.nodes["func3"].parent_ids.add("func2")

        path = await lightweight_service._bidirectional_path_search("func1", "func3", max_depth=10)

        assert path == ["func1", "func2", "func3"]

    @pytest.mark.asyncio
    async def test_cache_management_and_ttl(self, lightweight_service):
        """Test cache size management and TTL expiration."""

        # Test L1 cache size management
        for i in range(150):  # Exceed L1 cache limit
            cache_key = f"test_key_{i}"
            lightweight_service._store_l1_cache(cache_key, {"test": "data", "cached_at": time.time()})

        # Should not exceed size limit
        assert len(lightweight_service.l1_cache) <= 100

        # Test TTL expiration
        old_result = {"test": "old", "cached_at": time.time() - 400}  # 6+ minutes old
        lightweight_service.l1_cache["old_key"] = old_result

        result = lightweight_service._check_l1_cache("old_key")
        assert result is None  # Should be expired and removed

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, lightweight_service):
        """Test performance metrics collection during path finding."""

        # Create simple path
        lightweight_service.memory_index.nodes["func1"].children_ids.add("func2")
        lightweight_service.memory_index.nodes["func2"].parent_ids.add("func1")

        result = await lightweight_service.find_intelligent_path(start_node="main_function", end_node="helper_function", max_depth=5)

        # Validate performance metrics
        assert "response_time_ms" in result
        assert isinstance(result["response_time_ms"], (int, float))
        assert result["response_time_ms"] >= 0

        # Check internal performance metrics
        assert lightweight_service.performance_metrics["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_precomputed_routes_l3_cache(self, lightweight_service):
        """Test L3 cache for pre-computed common routes."""

        # Add nodes to precomputed query categories
        lightweight_service.precomputed_queries["entry_points"]["test_project"] = ["func1"]
        lightweight_service.precomputed_queries["main_functions"]["test_project"] = ["func2"]

        # Store a precomputed route
        route_key = "l3_route_entry_points_main_functions_func1_func2"
        route_result = {"success": True, "path": ["func1", "func2"], "quality_score": 0.9, "cached_at": time.time()}
        lightweight_service.l3_cache[route_key] = route_result

        # Should find the precomputed route
        l3_result = await lightweight_service._check_l3_precomputed_routes("func1", "func2")
        assert l3_result is not None
        assert l3_result["path"] == ["func1", "func2"]

    @pytest.mark.asyncio
    async def test_fallback_to_traditional_method(self, lightweight_service):
        """Test fallback to traditional path finding when intelligent method fails."""

        # Test with nodes that don't exist in memory index
        result = await lightweight_service.find_intelligent_path(start_node="nonexistent_start", end_node="nonexistent_end", max_depth=5)

        # Should fail gracefully
        assert result["success"] is False
        assert "error" in result
        assert "Node resolution failed" in result["error"]

    def test_memory_index_stats(self, lightweight_service):
        """Test memory index statistics reporting."""

        stats = lightweight_service.get_memory_index_stats()

        assert "total_nodes" in stats
        assert "nodes_by_type" in stats
        assert "nodes_by_language" in stats
        assert "hit_rate" in stats
        assert "last_updated" in stats
        assert "performance_metrics" in stats

        # Validate stats values
        assert stats["total_nodes"] == 3
        assert ChunkType.FUNCTION.value in stats["nodes_by_type"]
        assert stats["nodes_by_type"][ChunkType.FUNCTION.value] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
