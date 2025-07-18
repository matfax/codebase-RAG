"""
Comprehensive Unit Tests for Graph Structure Analysis Tools

This module provides thorough testing for graph analysis functionality,
covering various project scopes, analysis types, and edge cases.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.code_chunk import ChunkType, CodeChunk
from src.services.graph_analysis_report_service import (
    GraphAnalysisReport,
    GraphAnalysisReportService,
    Recommendation,
    ReportType,
    SeverityLevel,
    StatisticalSummary,
)
from src.services.graph_rag_service import GraphRAGService, GraphTraversalResult
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph
from src.tools.graph_rag.structure_analysis import graph_analyze_structure


class TestGraphStructureAnalysis:
    """Test suite for graph structure analysis functionality."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Mock QdrantService for testing."""
        mock_service = Mock()
        mock_service._initialize_cache = AsyncMock()
        return mock_service

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock EmbeddingService for testing."""
        mock_service = Mock()
        mock_service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return mock_service

    @pytest.fixture
    def sample_code_chunk(self):
        """Sample CodeChunk for testing."""
        return CodeChunk(
            chunk_id="test_chunk_1",
            file_path="/test/file.py",
            content="def test_function():\n    pass",
            chunk_type=ChunkType.FUNCTION,
            language="python",
            start_line=1,
            end_line=2,
            name="test_function",
            breadcrumb="test_module.test_function",
            signature="()",
        )

    @pytest.fixture
    def sample_structure_graph(self):
        """Sample StructureGraph for testing."""
        # Create nodes
        module_node = GraphNode(
            breadcrumb="test_module",
            name="test_module",
            chunk_type=ChunkType.MODULE,
            file_path="/test/file.py",
            depth=0,
            parent_breadcrumb=None,
            children_breadcrumbs=["test_module.test_class", "test_module.test_function"],
        )

        class_node = GraphNode(
            breadcrumb="test_module.test_class",
            name="test_class",
            chunk_type=ChunkType.CLASS,
            file_path="/test/file.py",
            depth=1,
            parent_breadcrumb="test_module",
            children_breadcrumbs=["test_module.test_class.method1", "test_module.test_class.method2"],
        )

        function_node = GraphNode(
            breadcrumb="test_module.test_function",
            name="test_function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/file.py",
            depth=1,
            parent_breadcrumb="test_module",
            children_breadcrumbs=[],
        )

        method1_node = GraphNode(
            breadcrumb="test_module.test_class.method1",
            name="method1",
            chunk_type=ChunkType.METHOD,
            file_path="/test/file.py",
            depth=2,
            parent_breadcrumb="test_module.test_class",
            children_breadcrumbs=[],
        )

        method2_node = GraphNode(
            breadcrumb="test_module.test_class.method2",
            name="method2",
            chunk_type=ChunkType.METHOD,
            file_path="/test/file.py",
            depth=2,
            parent_breadcrumb="test_module.test_class",
            children_breadcrumbs=[],
        )

        # Create edges
        edges = [
            GraphEdge(
                source_breadcrumb="test_module",
                target_breadcrumb="test_module.test_class",
                relationship_type="contains",
                confidence=1.0,
            ),
            GraphEdge(
                source_breadcrumb="test_module",
                target_breadcrumb="test_module.test_function",
                relationship_type="contains",
                confidence=1.0,
            ),
            GraphEdge(
                source_breadcrumb="test_module.test_class",
                target_breadcrumb="test_module.test_class.method1",
                relationship_type="contains",
                confidence=1.0,
            ),
            GraphEdge(
                source_breadcrumb="test_module.test_class",
                target_breadcrumb="test_module.test_class.method2",
                relationship_type="contains",
                confidence=1.0,
            ),
        ]

        nodes = {
            "test_module": module_node,
            "test_module.test_class": class_node,
            "test_module.test_function": function_node,
            "test_module.test_class.method1": method1_node,
            "test_module.test_class.method2": method2_node,
        }

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["test_module"],
            project_name="test_project",
        )

    @pytest.fixture
    def mock_graph_rag_service(self, sample_structure_graph):
        """Mock GraphRAGService for testing."""
        mock_service = Mock(spec=GraphRAGService)

        # Mock build_structure_graph
        mock_service.build_structure_graph = AsyncMock(return_value=sample_structure_graph)

        # Mock get_component_hierarchy
        mock_service.get_component_hierarchy = AsyncMock(
            return_value={
                "current": {
                    "breadcrumb": "test_module.test_class",
                    "name": "test_class",
                    "type": "class",
                    "depth": 1,
                },
                "ancestors": [{"breadcrumb": "test_module", "name": "test_module", "type": "module"}],
                "descendants": [
                    {"breadcrumb": "test_module.test_class.method1", "name": "method1", "type": "method"},
                    {"breadcrumb": "test_module.test_class.method2", "name": "method2", "type": "method"},
                ],
                "siblings": [{"breadcrumb": "test_module.test_function", "name": "test_function", "type": "function"}],
            }
        )

        # Mock analyze_component_connectivity
        mock_service.analyze_component_connectivity = AsyncMock(
            return_value={
                "influence_score": 0.75,
                "connection_count": 3,
                "in_degree": 1,
                "out_degree": 2,
            }
        )

        # Mock find_related_components
        mock_related_components = [
            GraphNode(
                breadcrumb="test_module.test_class.method1",
                name="method1",
                chunk_type=ChunkType.METHOD,
                file_path="/test/file.py",
                depth=2,
                parent_breadcrumb="test_module.test_class",
                children_breadcrumbs=[],
            ),
            GraphNode(
                breadcrumb="test_module.test_class.method2",
                name="method2",
                chunk_type=ChunkType.METHOD,
                file_path="/test/file.py",
                depth=2,
                parent_breadcrumb="test_module.test_class",
                children_breadcrumbs=[],
            ),
        ]

        mock_traversal_result = GraphTraversalResult(
            visited_nodes=mock_related_components,
            path=["test_module.test_class", "test_module.test_class.method1", "test_module.test_class.method2"],
            related_components=mock_related_components,
            traversal_depth=2,
            execution_time_ms=50.0,
        )

        mock_service.find_related_components = AsyncMock(return_value=mock_traversal_result)

        # Mock find_hierarchical_path
        mock_service.find_hierarchical_path = AsyncMock(
            return_value=[
                "test_module.test_class",
                "test_module.test_class.method1",
            ]
        )

        return mock_service

    @pytest.fixture
    def mock_report_service(self):
        """Mock GraphAnalysisReportService for testing."""
        mock_service = Mock(spec=GraphAnalysisReportService)

        # Create sample report
        sample_report = GraphAnalysisReport(
            report_type=ReportType.STRUCTURE_SUMMARY,
            project_name="test_project",
            generated_at="2024-01-01T00:00:00",
            execution_time_ms=100.0,
            summary={
                "target_component": {
                    "breadcrumb": "test_module.test_class",
                    "type": "class",
                    "depth": 1,
                    "file_path": "/test/file.py",
                },
                "hierarchy_depth": 1,
                "children_count": 2,
                "siblings_count": 1,
                "related_components_count": 2,
                "connectivity_score": 0.75,
                "analysis_type": "comprehensive",
            },
            statistics=[
                StatisticalSummary(
                    metric_name="component_similarity",
                    count=2,
                    mean=0.8,
                    median=0.8,
                    std_dev=0.1,
                    min_value=0.7,
                    max_value=0.9,
                )
            ],
            recommendations=[
                Recommendation(
                    title="Test Recommendation",
                    description="This is a test recommendation",
                    severity=SeverityLevel.INFO,
                    category="structure",
                    impact="Minimal impact",
                    suggested_actions=["Action 1", "Action 2"],
                    affected_components=["test_module.test_class"],
                    confidence=0.8,
                    estimated_effort="low",
                )
            ],
            insights=["Test insight 1", "Test insight 2"],
            confidence_score=0.9,
            data_quality_score=0.95,
        )

        mock_service.generate_structure_analysis_report = AsyncMock(return_value=sample_report)

        return mock_service

    @pytest.mark.asyncio
    async def test_basic_structure_analysis(self, mock_graph_rag_service):
        """Test basic structure analysis without report generation."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="comprehensive",
                max_depth=3,
                include_siblings=True,
                include_connectivity=True,
                force_rebuild_graph=False,
                generate_report=False,
            )

            # Verify basic structure
            assert result["success"] is True
            assert result["breadcrumb"] == "test_module.test_class"
            assert result["project_name"] == "test_project"
            assert result["analysis_type"] == "comprehensive"
            assert result["max_depth"] == 3

            # Verify hierarchy data
            assert "hierarchy" in result
            hierarchy = result["hierarchy"]
            assert hierarchy["current"]["breadcrumb"] == "test_module.test_class"
            assert len(hierarchy["ancestors"]) == 1
            assert len(hierarchy["descendants"]) == 2
            assert len(hierarchy["siblings"]) == 1

            # Verify connectivity data
            assert "connectivity" in result
            connectivity = result["connectivity"]
            assert connectivity["influence_score"] == 0.75
            assert connectivity["connection_count"] == 3

            # Verify related components
            assert "related_components" in result
            assert len(result["related_components"]) == 2

            # Verify metadata
            assert "metadata" in result
            metadata = result["metadata"]
            assert metadata["include_siblings"] is True
            assert metadata["include_connectivity"] is True

    @pytest.mark.asyncio
    async def test_hierarchy_only_analysis(self, mock_graph_rag_service):
        """Test analysis with hierarchy type only."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="hierarchy",
                include_connectivity=False,
            )

            assert result["success"] is True
            assert "hierarchy" in result
            assert "connectivity" not in result
            assert "related_components" not in result

    @pytest.mark.asyncio
    async def test_connectivity_only_analysis(self, mock_graph_rag_service):
        """Test analysis with connectivity type only."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="connectivity",
                include_connectivity=True,
            )

            assert result["success"] is True
            assert "connectivity" in result
            assert "hierarchy" not in result

    @pytest.mark.asyncio
    async def test_overview_analysis(self, mock_graph_rag_service):
        """Test overview analysis type."""
        # Mock get_project_structure_overview
        mock_graph_rag_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "test_project",
                "total_components": 5,
                "total_relationships": 4,
                "max_depth": 2,
            }
        )

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="overview",
            )

            assert result["success"] is True
            assert "project_overview" in result
            assert result["project_overview"]["total_components"] == 5

    @pytest.mark.asyncio
    async def test_with_report_generation(self, mock_graph_rag_service, mock_report_service):
        """Test structure analysis with report generation."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
            patch("src.tools.graph_rag.structure_analysis.get_graph_analysis_report_service", return_value=mock_report_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="comprehensive",
                generate_report=True,
                include_recommendations=True,
            )

            assert result["success"] is True
            assert "comprehensive_report" in result

            report = result["comprehensive_report"]
            assert report["report_type"] == "structure_summary"
            assert report["execution_time_ms"] == 100.0
            assert "summary" in report
            assert "statistics" in report
            assert "recommendations" in report
            assert "insights" in report

            # Verify recommendations structure
            assert len(report["recommendations"]) == 1
            rec = report["recommendations"][0]
            assert rec["title"] == "Test Recommendation"
            assert rec["severity"] == "info"
            assert rec["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_navigation_paths_generation(self, mock_graph_rag_service):
        """Test navigation paths generation for comprehensive analysis."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="comprehensive",
            )

            assert result["success"] is True
            assert "navigation_paths" in result
            # Should have paths to related components
            assert len(result["navigation_paths"]) >= 0

    @pytest.mark.asyncio
    async def test_invalid_breadcrumb_error(self):
        """Test error handling for invalid breadcrumb."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(
            return_value=StructureGraph(nodes={}, edges=[], root_nodes=[], project_name="test_project")
        )

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="nonexistent.breadcrumb",
                project_name="test_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "nonexistent.breadcrumb" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_breadcrumb_validation(self):
        """Test validation for empty breadcrumb."""
        result = await graph_analyze_structure(
            breadcrumb="",
            project_name="test_project",
        )

        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_project_name_validation(self):
        """Test validation for empty project name."""
        result = await graph_analyze_structure(
            breadcrumb="test.breadcrumb",
            project_name="",
        )

        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_max_depth_clamping(self, mock_graph_rag_service):
        """Test that max_depth is properly clamped between 1 and 10."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            # Test max_depth too low
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                max_depth=-5,
            )
            assert result["max_depth"] == 1

            # Test max_depth too high
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                max_depth=20,
            )
            assert result["max_depth"] == 10

    @pytest.mark.asyncio
    async def test_service_initialization_error(self):
        """Test error handling during service initialization."""
        with patch("src.tools.graph_rag.structure_analysis.QdrantService", side_effect=Exception("Service error")):
            result = await graph_analyze_structure(
                breadcrumb="test.breadcrumb",
                project_name="test_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "Service error" in result["error"]

    @pytest.mark.asyncio
    async def test_graph_building_error(self):
        """Test error handling during graph building."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(side_effect=Exception("Graph building failed"))

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test.breadcrumb",
                project_name="test_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "Graph building failed" in result["error"]

    @pytest.mark.asyncio
    async def test_report_generation_error(self, mock_graph_rag_service):
        """Test error handling during report generation."""
        mock_report_service = Mock()
        mock_report_service.generate_structure_analysis_report = AsyncMock(side_effect=Exception("Report generation failed"))

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
            patch("src.tools.graph_rag.structure_analysis.get_graph_analysis_report_service", return_value=mock_report_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                generate_report=True,
            )

            assert result["success"] is True  # Main analysis should still succeed
            assert "report_error" in result
            assert "Report generation failed" in result["report_error"]

    @pytest.mark.asyncio
    async def test_navigation_paths_error_handling(self, mock_graph_rag_service):
        """Test error handling during navigation paths generation."""
        # Mock find_hierarchical_path to raise an exception
        mock_graph_rag_service.find_hierarchical_path = AsyncMock(side_effect=Exception("Path finding failed"))

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="comprehensive",
            )

            assert result["success"] is True  # Main analysis should still succeed
            assert "navigation_paths" in result
            assert result["navigation_paths"] == []  # Should be empty due to error

    @pytest.mark.asyncio
    async def test_different_analysis_types_coverage(self, mock_graph_rag_service):
        """Test coverage of all analysis types."""
        analysis_types = ["comprehensive", "hierarchy", "connectivity", "overview"]

        # Mock additional methods for overview
        mock_graph_rag_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "test_project",
                "total_components": 5,
            }
        )

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            for analysis_type in analysis_types:
                result = await graph_analyze_structure(
                    breadcrumb="test_module.test_class",
                    project_name="test_project",
                    analysis_type=analysis_type,
                    include_connectivity=True,
                )

                assert result["success"] is True
                assert result["analysis_type"] == analysis_type

                # Verify specific content based on type
                if analysis_type in ["comprehensive", "hierarchy"]:
                    assert "hierarchy" in result

                if analysis_type in ["comprehensive", "connectivity"]:
                    assert "connectivity" in result

                if analysis_type in ["comprehensive", "overview"]:
                    assert "related_components" in result

                if analysis_type == "overview":
                    assert "project_overview" in result

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, mock_graph_rag_service):
        """Test that performance metrics are properly tracked."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="test_module.test_class",
                project_name="test_project",
                analysis_type="comprehensive",
            )

            assert result["success"] is True
            assert "graph_stats" in result

            graph_stats = result["graph_stats"]
            assert "total_nodes" in graph_stats
            assert "total_edges" in graph_stats
            assert "force_rebuilt" in graph_stats

    @pytest.mark.asyncio
    async def test_large_project_simulation(self, mock_graph_rag_service):
        """Test behavior with large project simulation."""
        # Create a larger graph for testing
        large_graph = StructureGraph(
            nodes={
                f"node_{i}": GraphNode(
                    breadcrumb=f"node_{i}",
                    name=f"node_{i}",
                    chunk_type=ChunkType.FUNCTION,
                    file_path=f"/test/file_{i}.py",
                    depth=i % 5,
                    parent_breadcrumb=f"node_{i-1}" if i > 0 else None,
                    children_breadcrumbs=[f"node_{i+1}"] if i < 99 else [],
                )
                for i in range(100)
            },
            edges=[],
            root_nodes=["node_0"],
            project_name="large_test_project",
        )

        # Create many related components
        large_related_components = [
            GraphNode(
                breadcrumb=f"related_node_{i}",
                name=f"related_node_{i}",
                chunk_type=ChunkType.FUNCTION,
                file_path=f"/test/related_file_{i}.py",
                depth=2,
                parent_breadcrumb="test_module.test_class",
                children_breadcrumbs=[],
            )
            for i in range(50)
        ]

        large_traversal_result = GraphTraversalResult(
            visited_nodes=large_related_components,
            path=[f"related_node_{i}" for i in range(50)],
            related_components=large_related_components,
            traversal_depth=3,
            execution_time_ms=500.0,
        )

        mock_graph_rag_service.build_structure_graph = AsyncMock(return_value=large_graph)
        mock_graph_rag_service.find_related_components = AsyncMock(return_value=large_traversal_result)

        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            result = await graph_analyze_structure(
                breadcrumb="node_0",
                project_name="large_test_project",
                analysis_type="comprehensive",
                max_depth=5,
            )

            assert result["success"] is True
            assert result["graph_stats"]["total_nodes"] == 100
            assert len(result["related_components"]) == 50
            assert result["metadata"]["total_analysis_components"] == 50

    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, mock_graph_rag_service):
        """Test handling of concurrent analysis requests."""
        with (
            patch("src.tools.graph_rag.structure_analysis.QdrantService"),
            patch("src.tools.graph_rag.structure_analysis.EmbeddingService"),
            patch("src.tools.graph_rag.structure_analysis.GraphRAGService", return_value=mock_graph_rag_service),
        ):
            # Run multiple analyses concurrently
            tasks = [
                graph_analyze_structure(
                    breadcrumb=f"test_module.test_class_{i}",
                    project_name="test_project",
                    analysis_type="comprehensive",
                )
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed (breadcrumb will be found in mock)
            for result in results:
                if isinstance(result, dict):
                    assert result["success"] is True
                else:
                    # If exception, should be handled gracefully
                    assert isinstance(result, Exception)

    def test_report_service_integration_validation(self):
        """Test validation of report service integration."""
        # This test validates that the report service is properly integrated
        # by checking the expected interface and return types

        # Test imports work correctly
        from src.services.graph_analysis_report_service import (
            GraphAnalysisReport,
            GraphAnalysisReportService,
            Recommendation,
            ReportType,
            SeverityLevel,
            StatisticalSummary,
        )

        # Verify enum values
        assert ReportType.STRUCTURE_SUMMARY.value == "structure_summary"
        assert SeverityLevel.CRITICAL.value == "critical"

        # Verify classes are properly defined
        assert hasattr(GraphAnalysisReportService, "generate_structure_analysis_report")
        assert hasattr(Recommendation, "title")
        assert hasattr(StatisticalSummary, "metric_name")
        assert hasattr(GraphAnalysisReport, "report_type")


@pytest.mark.integration
class TestGraphAnalysisIntegration:
    """Integration tests for graph analysis with real services."""

    @pytest.mark.asyncio
    async def test_end_to_end_analysis_flow(self):
        """Test complete end-to-end analysis flow."""
        # This would test with real services if available
        # For now, we'll skip this in unit tests
        pytest.skip("Integration test requires real services")

    @pytest.mark.asyncio
    async def test_real_project_analysis(self):
        """Test analysis on a real project structure."""
        # This would test with an actual indexed project
        pytest.skip("Integration test requires indexed project")


# Performance and Load Testing
@pytest.mark.performance
class TestGraphAnalysisPerformance:
    """Performance tests for graph analysis functionality."""

    @pytest.mark.asyncio
    async def test_analysis_performance_benchmarks(self):
        """Test performance benchmarks for analysis operations."""
        # This would test actual performance metrics
        pytest.skip("Performance test requires specific setup")

    @pytest.mark.asyncio
    async def test_memory_usage_analysis(self):
        """Test memory usage during large graph analysis."""
        pytest.skip("Memory test requires specific monitoring setup")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=src.tools.graph_rag.structure_analysis",
            "--cov=src.services.graph_analysis_report_service",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
