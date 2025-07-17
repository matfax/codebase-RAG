"""Integration tests for Function Chain Analysis Tool

This module contains integration tests that verify the trace_function_chain tool
works correctly with the existing Graph RAG infrastructure, including real
service interactions and end-to-end workflows.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.services.embedding_service import EmbeddingService
from src.services.graph_rag_service import GraphRAGService
from src.services.hybrid_search_service import HybridSearchService
from src.services.implementation_chain_service import get_implementation_chain_service
from src.services.qdrant_service import QdrantService
from src.tools.graph_rag.function_chain_analysis import trace_function_chain


class TestGraphRAGIntegration:
    """Integration tests with Graph RAG infrastructure."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Mock Qdrant service for testing."""
        mock_service = MagicMock()
        mock_service.get_project_collections.return_value = ["test_project_code"]
        mock_service.search.return_value = []
        return mock_service

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing."""
        mock_service = MagicMock()
        mock_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        return mock_service

    @pytest.fixture
    def mock_search_service(self):
        """Mock search service for testing."""
        mock_service = MagicMock()
        mock_service.search.return_value = {
            "results": [
                {
                    "content": "def test_function():\n    return 'hello'",
                    "name": "test_function",
                    "breadcrumb": "test.module.test_function",
                    "chunk_type": "function",
                    "file_path": "/test/module.py",
                    "score": 0.9,
                    "line_start": 1,
                    "line_end": 2,
                    "language": "python",
                }
            ]
        }
        return mock_service

    @pytest.mark.asyncio
    async def test_integration_with_breadcrumb_resolver(self, mock_search_service):
        """Test integration with BreadcrumbResolver service."""
        # Mock the search function used by BreadcrumbResolver
        with patch("src.tools.indexing.search_tools.search_async_cached") as mock_search:
            mock_search.return_value = {
                "results": [
                    {
                        "content": "def test_function():\n    return 'hello'",
                        "name": "test_function",
                        "breadcrumb": "test.module.test_function",
                        "chunk_type": "function",
                        "file_path": "/test/module.py",
                        "score": 0.9,
                        "line_start": 1,
                        "line_end": 2,
                        "language": "python",
                    }
                ]
            }

            # Test BreadcrumbResolver integration
            resolver = BreadcrumbResolver()
            result = await resolver.resolve(query="test function that returns hello", target_projects=["test_project"])

            assert result.success is True
            assert result.primary_candidate.breadcrumb == "test.module.test_function"
            assert result.primary_candidate.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_integration_with_implementation_chain_service(self, mock_qdrant_service, mock_embedding_service):
        """Test integration with ImplementationChainService."""
        # Mock the Graph RAG service dependencies
        with (
            patch("src.services.graph_rag_service.GraphRAGService") as mock_graph_rag,
            patch("src.services.hybrid_search_service.HybridSearchService"),
        ):
            # Setup mock graph structure
            mock_graph_rag.return_value.get_project_structure_graph.return_value = MagicMock()
            mock_graph_rag.return_value.get_project_structure_graph.return_value.nodes = []
            mock_graph_rag.return_value.get_project_structure_graph.return_value.edges = []

            # Test ImplementationChainService integration
            implementation_service = get_implementation_chain_service()

            # Verify service is properly initialized
            assert implementation_service is not None
            assert hasattr(implementation_service, "trace_implementation_chain")
            assert hasattr(implementation_service, "analyze_project_chains")

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_mocked_services(self):
        """Test end-to-end workflow with mocked services."""
        # Mock all service dependencies
        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver") as mock_breadcrumb_resolver,
            patch("src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service") as mock_chain_service,
        ):
            # Setup successful breadcrumb resolution
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve.return_value = MagicMock()
            mock_resolver_instance.resolve.return_value.success = True
            mock_resolver_instance.resolve.return_value.primary_candidate = MagicMock()
            mock_resolver_instance.resolve.return_value.primary_candidate.breadcrumb = "test.module.function"
            mock_resolver_instance.resolve.return_value.primary_candidate.confidence_score = 0.9
            mock_breadcrumb_resolver.return_value = mock_resolver_instance

            # Setup successful chain tracing
            mock_chain_instance = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.chain_id = "test_chain"
            mock_chain_instance.trace_implementation_chain.return_value.links = []
            mock_chain_instance.trace_implementation_chain.return_value.depth = 1
            mock_chain_instance.trace_implementation_chain.return_value.branch_count = 0
            mock_chain_instance.trace_implementation_chain.return_value.complexity_score = 0.2
            mock_chain_instance.trace_implementation_chain.return_value.completeness_score = 0.8
            mock_chain_instance.trace_implementation_chain.return_value.reliability_score = 0.9
            mock_chain_instance.trace_implementation_chain.return_value.functional_purpose = "Test function"
            mock_chain_instance.trace_implementation_chain.return_value.total_components = 1
            mock_chain_instance.trace_implementation_chain.return_value.entry_point = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.entry_point.name = "test_function"
            mock_chain_instance.trace_implementation_chain.return_value.terminal_points = []
            mock_chain_instance.trace_implementation_chain.return_value.components_by_type = {}
            mock_chain_instance.trace_implementation_chain.return_value.avg_link_strength = 0.0
            mock_chain_instance.trace_implementation_chain.return_value.scope_breadcrumb = "test.module"
            mock_chain_service.return_value = mock_chain_instance

            # Execute end-to-end workflow
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                direction="forward",
                max_depth=5,
                output_format="both",
                performance_monitoring=True,
            )

            # Verify successful end-to-end execution
            assert result["success"] is True
            assert result["resolved_breadcrumb"] == "test.module.function"
            assert result["breadcrumb_confidence"] == 0.9
            assert "chain_info" in result
            assert "arrow_format" in result
            assert "mermaid_format" in result
            assert "performance" in result

            # Verify service interactions
            mock_resolver_instance.resolve.assert_called_once()
            mock_chain_instance.trace_implementation_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_compatibility_with_existing_graph_rag_tools(self):
        """Test compatibility with existing Graph RAG tools."""
        # Import other Graph RAG tools to ensure no conflicts
        from src.tools.graph_rag.pattern_identification import graph_identify_patterns
        from src.tools.graph_rag.similar_implementations import graph_find_similar_implementations
        from src.tools.graph_rag.structure_analysis import graph_analyze_structure

        # Verify all tools can be imported together
        assert graph_analyze_structure is not None
        assert graph_find_similar_implementations is not None
        assert graph_identify_patterns is not None
        assert trace_function_chain is not None

        # Test that they don't interfere with each other
        assert callable(graph_analyze_structure)
        assert callable(graph_find_similar_implementations)
        assert callable(graph_identify_patterns)
        assert callable(trace_function_chain)

    @pytest.mark.asyncio
    async def test_service_initialization_compatibility(self):
        """Test that service initialization works with existing infrastructure."""
        # Test that services can be initialized without conflicts
        with (
            patch("src.services.qdrant_service.QdrantService") as mock_qdrant,
            patch("src.services.embedding_service.EmbeddingService") as mock_embedding,
        ):
            # Mock service initialization
            mock_qdrant.return_value = MagicMock()
            mock_embedding.return_value = MagicMock()

            # Test BreadcrumbResolver initialization
            breadcrumb_resolver = BreadcrumbResolver()
            assert breadcrumb_resolver is not None
            assert hasattr(breadcrumb_resolver, "resolve")
            assert hasattr(breadcrumb_resolver, "is_valid_breadcrumb")

            # Test ImplementationChainService initialization
            implementation_service = get_implementation_chain_service()
            assert implementation_service is not None
            assert hasattr(implementation_service, "trace_implementation_chain")

    @pytest.mark.asyncio
    async def test_error_handling_with_graph_rag_infrastructure(self):
        """Test error handling when Graph RAG infrastructure has issues."""
        # Test with failing breadcrumb resolution
        with patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver") as mock_breadcrumb_resolver:
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve.return_value = MagicMock()
            mock_resolver_instance.resolve.return_value.success = False
            mock_resolver_instance.resolve.return_value.error_message = "Graph RAG service unavailable"
            mock_breadcrumb_resolver.return_value = mock_resolver_instance

            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
            )

            assert result["success"] is False
            assert "Failed to resolve entry point" in result["error"]
            assert "Graph RAG service unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_performance_with_graph_rag_infrastructure(self):
        """Test performance characteristics with Graph RAG infrastructure."""
        # Mock services for performance testing
        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver") as mock_breadcrumb_resolver,
            patch("src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service") as mock_chain_service,
        ):
            # Setup services with timing
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve.return_value = MagicMock()
            mock_resolver_instance.resolve.return_value.success = True
            mock_resolver_instance.resolve.return_value.primary_candidate = MagicMock()
            mock_resolver_instance.resolve.return_value.primary_candidate.breadcrumb = "test.module.function"
            mock_resolver_instance.resolve.return_value.primary_candidate.confidence_score = 0.9
            mock_breadcrumb_resolver.return_value = mock_resolver_instance

            mock_chain_instance = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.chain_id = "test_chain"
            mock_chain_instance.trace_implementation_chain.return_value.links = []
            mock_chain_instance.trace_implementation_chain.return_value.depth = 1
            mock_chain_instance.trace_implementation_chain.return_value.branch_count = 0
            mock_chain_instance.trace_implementation_chain.return_value.complexity_score = 0.2
            mock_chain_instance.trace_implementation_chain.return_value.completeness_score = 0.8
            mock_chain_instance.trace_implementation_chain.return_value.reliability_score = 0.9
            mock_chain_instance.trace_implementation_chain.return_value.functional_purpose = "Test function"
            mock_chain_instance.trace_implementation_chain.return_value.total_components = 1
            mock_chain_instance.trace_implementation_chain.return_value.entry_point = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.entry_point.name = "test_function"
            mock_chain_instance.trace_implementation_chain.return_value.terminal_points = []
            mock_chain_instance.trace_implementation_chain.return_value.components_by_type = {}
            mock_chain_instance.trace_implementation_chain.return_value.avg_link_strength = 0.0
            mock_chain_instance.trace_implementation_chain.return_value.scope_breadcrumb = "test.module"
            mock_chain_service.return_value = mock_chain_instance

            # Execute with performance monitoring
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                performance_monitoring=True,
            )

            # Verify performance metrics are captured
            assert result["success"] is True
            assert "performance" in result
            assert result["performance"]["total_time"] > 0
            assert result["performance"]["breadcrumb_resolution_time"] >= 0
            assert result["performance"]["chain_tracing_time"] >= 0
            assert result["performance"]["formatting_time"] >= 0

            # Verify reasonable performance (should complete within reasonable time)
            assert result["performance"]["total_time"] < 10000  # Less than 10 seconds

    @pytest.mark.asyncio
    async def test_concurrent_execution_with_graph_rag_infrastructure(self):
        """Test concurrent execution with Graph RAG infrastructure."""
        # Test multiple concurrent trace requests
        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver") as mock_breadcrumb_resolver,
            patch("src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service") as mock_chain_service,
        ):
            # Setup mock services
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve.return_value = MagicMock()
            mock_resolver_instance.resolve.return_value.success = True
            mock_resolver_instance.resolve.return_value.primary_candidate = MagicMock()
            mock_resolver_instance.resolve.return_value.primary_candidate.breadcrumb = "test.module.function"
            mock_resolver_instance.resolve.return_value.primary_candidate.confidence_score = 0.9
            mock_breadcrumb_resolver.return_value = mock_resolver_instance

            mock_chain_instance = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.chain_id = "test_chain"
            mock_chain_instance.trace_implementation_chain.return_value.links = []
            mock_chain_instance.trace_implementation_chain.return_value.depth = 1
            mock_chain_instance.trace_implementation_chain.return_value.branch_count = 0
            mock_chain_instance.trace_implementation_chain.return_value.complexity_score = 0.2
            mock_chain_instance.trace_implementation_chain.return_value.completeness_score = 0.8
            mock_chain_instance.trace_implementation_chain.return_value.reliability_score = 0.9
            mock_chain_instance.trace_implementation_chain.return_value.functional_purpose = "Test function"
            mock_chain_instance.trace_implementation_chain.return_value.total_components = 1
            mock_chain_instance.trace_implementation_chain.return_value.entry_point = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.entry_point.name = "test_function"
            mock_chain_instance.trace_implementation_chain.return_value.terminal_points = []
            mock_chain_instance.trace_implementation_chain.return_value.components_by_type = {}
            mock_chain_instance.trace_implementation_chain.return_value.avg_link_strength = 0.0
            mock_chain_instance.trace_implementation_chain.return_value.scope_breadcrumb = "test.module"
            mock_chain_service.return_value = mock_chain_instance

            # Execute multiple concurrent requests
            tasks = []
            for i in range(5):
                task = trace_function_chain(
                    entry_point=f"test function {i}",
                    project_name="test_project",
                    direction="forward",
                    max_depth=3,
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # Verify all requests succeeded
            assert len(results) == 5
            for result in results:
                assert result["success"] is True
                assert "chain_info" in result

    @pytest.mark.asyncio
    async def test_data_consistency_with_graph_rag_infrastructure(self):
        """Test data consistency when using Graph RAG infrastructure."""
        # Test that the same input produces consistent results
        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver") as mock_breadcrumb_resolver,
            patch("src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service") as mock_chain_service,
        ):
            # Setup consistent mock responses
            mock_resolver_instance = MagicMock()
            mock_resolver_instance.resolve.return_value = MagicMock()
            mock_resolver_instance.resolve.return_value.success = True
            mock_resolver_instance.resolve.return_value.primary_candidate = MagicMock()
            mock_resolver_instance.resolve.return_value.primary_candidate.breadcrumb = "test.module.function"
            mock_resolver_instance.resolve.return_value.primary_candidate.confidence_score = 0.9
            mock_breadcrumb_resolver.return_value = mock_resolver_instance

            mock_chain_instance = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.chain_id = "test_chain"
            mock_chain_instance.trace_implementation_chain.return_value.links = []
            mock_chain_instance.trace_implementation_chain.return_value.depth = 1
            mock_chain_instance.trace_implementation_chain.return_value.branch_count = 0
            mock_chain_instance.trace_implementation_chain.return_value.complexity_score = 0.2
            mock_chain_instance.trace_implementation_chain.return_value.completeness_score = 0.8
            mock_chain_instance.trace_implementation_chain.return_value.reliability_score = 0.9
            mock_chain_instance.trace_implementation_chain.return_value.functional_purpose = "Test function"
            mock_chain_instance.trace_implementation_chain.return_value.total_components = 1
            mock_chain_instance.trace_implementation_chain.return_value.entry_point = MagicMock()
            mock_chain_instance.trace_implementation_chain.return_value.entry_point.name = "test_function"
            mock_chain_instance.trace_implementation_chain.return_value.terminal_points = []
            mock_chain_instance.trace_implementation_chain.return_value.components_by_type = {}
            mock_chain_instance.trace_implementation_chain.return_value.avg_link_strength = 0.0
            mock_chain_instance.trace_implementation_chain.return_value.scope_breadcrumb = "test.module"
            mock_chain_service.return_value = mock_chain_instance

            # Execute same request multiple times
            results = []
            for _ in range(3):
                result = await trace_function_chain(
                    entry_point="test function",
                    project_name="test_project",
                    direction="forward",
                    max_depth=5,
                )
                results.append(result)

            # Verify consistent results
            assert len(results) == 3
            for result in results:
                assert result["success"] is True
                assert result["resolved_breadcrumb"] == "test.module.function"
                assert result["breadcrumb_confidence"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
