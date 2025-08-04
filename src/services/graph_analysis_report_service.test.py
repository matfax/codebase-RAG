"""
Comprehensive Unit Tests for Graph Analysis Report Service

This module provides thorough testing for the report generation functionality,
covering all report types, statistics, and recommendation generation.
"""

import statistics as stats
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.code_chunk import ChunkType
from src.services.graph_analysis_report_service import (
    GraphAnalysisReport,
    GraphAnalysisReportService,
    Recommendation,
    ReportType,
    SeverityLevel,
    StatisticalSummary,
    get_graph_analysis_report_service,
)
from src.services.graph_rag_service import GraphRAGService, GraphTraversalResult
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


class TestGraphAnalysisReportService:
    """Test suite for Graph Analysis Report Service."""

    @pytest.fixture
    def sample_structure_graph(self):
        """Sample StructureGraph for testing."""
        nodes = {
            "app.module": GraphNode(
                breadcrumb="app.module",
                name="module",
                chunk_type=ChunkType.MODULE,
                file_path="/app/module.py",
                depth=0,
                parent_breadcrumb=None,
                children_breadcrumbs=["app.module.MyClass", "app.module.my_function"],
            ),
            "app.module.MyClass": GraphNode(
                breadcrumb="app.module.MyClass",
                name="MyClass",
                chunk_type=ChunkType.CLASS,
                file_path="/app/module.py",
                depth=1,
                parent_breadcrumb="app.module",
                children_breadcrumbs=["app.module.MyClass.method1", "app.module.MyClass.method2"],
            ),
            "app.module.my_function": GraphNode(
                breadcrumb="app.module.my_function",
                name="my_function",
                chunk_type=ChunkType.FUNCTION,
                file_path="/app/module.py",
                depth=1,
                parent_breadcrumb="app.module",
                children_breadcrumbs=[],
            ),
        }

        edges = [
            GraphEdge("app.module", "app.module.MyClass", "contains", 1.0),
            GraphEdge("app.module", "app.module.my_function", "contains", 1.0),
        ]

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["app.module"],
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
                    "breadcrumb": "app.module.MyClass",
                    "name": "MyClass",
                    "type": "class",
                    "depth": 1,
                },
                "ancestors": [{"breadcrumb": "app.module", "name": "module", "type": "module"}],
                "descendants": [
                    {"breadcrumb": "app.module.MyClass.method1", "name": "method1", "type": "method"},
                    {"breadcrumb": "app.module.MyClass.method2", "name": "method2", "type": "method"},
                ],
                "siblings": [{"breadcrumb": "app.module.my_function", "name": "my_function", "type": "function"}],
            }
        )

        # Mock analyze_component_connectivity
        mock_service.analyze_component_connectivity = AsyncMock(
            return_value={
                "influence_score": 0.8,
                "connection_count": 3,
                "in_degree": 1,
                "out_degree": 2,
            }
        )

        # Mock find_related_components
        mock_related_components = [
            GraphNode(
                breadcrumb="app.module.MyClass.method1",
                name="method1",
                chunk_type=ChunkType.METHOD,
                file_path="/app/module.py",
                depth=2,
                parent_breadcrumb="app.module.MyClass",
                children_breadcrumbs=[],
            ),
        ]

        mock_traversal_result = GraphTraversalResult(
            visited_nodes=mock_related_components,
            path=["app.module.MyClass", "app.module.MyClass.method1"],
            related_components=mock_related_components,
            traversal_depth=2,
            execution_time_ms=100.0,
        )

        mock_service.find_related_components = AsyncMock(return_value=mock_traversal_result)

        # Mock get_project_structure_overview
        mock_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "test_project",
                "total_components": 3,
                "total_relationships": 2,
                "root_components": 1,
                "max_depth": 1,
                "breakdown": {
                    "by_type": {"module": 1, "class": 1, "function": 1},
                    "by_depth": {0: 1, 1: 2},
                    "by_language": {"py": 3},
                    "by_relationship": {"contains": 2},
                },
                "largest_components": [
                    {"breadcrumb": "app.module.MyClass", "children_count": 2},
                ],
                "structure_health": {
                    "orphaned_nodes": 0,
                    "average_children_per_node": 0.67,
                    "relationship_density": 0.33,
                },
            }
        )

        # Mock get_performance_stats
        mock_service.get_performance_stats = Mock(
            return_value={
                "graphs_built": 5,
                "cache_hits": 10,
                "cache_misses": 2,
                "traversals_executed": 15,
                "avg_build_time_ms": 150.0,
                "avg_traversal_time_ms": 75.0,
            }
        )

        return mock_service

    @pytest.fixture
    def report_service(self, mock_graph_rag_service):
        """GraphAnalysisReportService instance for testing."""
        return GraphAnalysisReportService(mock_graph_rag_service)

    @pytest.mark.asyncio
    async def test_generate_structure_analysis_report_basic(self, report_service):
        """Test basic structure analysis report generation."""
        report = await report_service.generate_structure_analysis_report(
            breadcrumb="app.module.MyClass",
            project_name="test_project",
            analysis_type="comprehensive",
        )

        assert isinstance(report, GraphAnalysisReport)
        assert report.report_type == ReportType.STRUCTURE_SUMMARY
        assert report.project_name == "test_project"
        assert report.execution_time_ms > 0

        # Check summary structure
        summary = report.summary
        assert "target_component" in summary
        assert summary["target_component"]["breadcrumb"] == "app.module.MyClass"
        assert "hierarchy_depth" in summary
        assert "children_count" in summary
        assert "connectivity_score" in summary

        # Check that we have statistics
        assert isinstance(report.statistics, list)

        # Check that we have insights
        assert isinstance(report.insights, list)
        assert len(report.insights) > 0

        # Check confidence and quality scores
        assert 0.0 <= report.confidence_score <= 1.0
        assert 0.0 <= report.data_quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_generate_structure_analysis_report_with_recommendations(self, report_service):
        """Test structure analysis report with recommendations."""
        report = await report_service.generate_structure_analysis_report(
            breadcrumb="app.module.MyClass",
            project_name="test_project",
            include_optimization_suggestions=True,
        )

        # Should have recommendations
        assert isinstance(report.recommendations, list)

        # Verify recommendation structure
        for rec in report.recommendations:
            assert isinstance(rec, Recommendation)
            assert hasattr(rec, "title")
            assert hasattr(rec, "description")
            assert hasattr(rec, "severity")
            assert hasattr(rec, "category")
            assert hasattr(rec, "impact")
            assert hasattr(rec, "suggested_actions")
            assert isinstance(rec.suggested_actions, list)
            assert 0.0 <= rec.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_generate_structure_analysis_report_performance_insights(self, report_service):
        """Test structure analysis report with performance insights."""
        report = await report_service.generate_structure_analysis_report(
            breadcrumb="app.module.MyClass",
            project_name="test_project",
            include_performance_insights=True,
        )

        # Should have performance metrics
        assert "performance_metrics" in report.__dict__
        assert isinstance(report.performance_metrics, dict)

        if report.performance_metrics:
            assert "graph_size" in report.performance_metrics
            assert "traversal_depth" in report.performance_metrics
            assert "execution_time_ms" in report.performance_metrics

    @pytest.mark.asyncio
    async def test_generate_pattern_analysis_report(self, report_service):
        """Test pattern analysis report generation."""
        report = await report_service.generate_pattern_analysis_report(
            project_name="test_project",
            pattern_types=["structural", "behavioral"],
            min_confidence=0.6,
        )

        assert isinstance(report, GraphAnalysisReport)
        assert report.report_type == ReportType.PATTERN_ANALYSIS
        assert report.project_name == "test_project"

        # Check summary structure
        summary = report.summary
        assert "total_patterns_identified" in summary
        assert "confidence_threshold" in summary
        assert summary["confidence_threshold"] == 0.6
        assert "pattern_quality_score" in summary

        # Should have pattern metrics
        assert hasattr(report, "pattern_metrics")
        if report.pattern_metrics:
            assert "architectural_patterns" in report.pattern_metrics
            assert "design_patterns" in report.pattern_metrics
            assert "naming_patterns" in report.pattern_metrics

    @pytest.mark.asyncio
    async def test_generate_project_health_report(self, report_service):
        """Test project health report generation."""
        report = await report_service.generate_project_health_report(
            project_name="test_project",
            include_performance_analysis=True,
            include_recommendations=True,
        )

        assert isinstance(report, GraphAnalysisReport)
        assert report.report_type == ReportType.PROJECT_HEALTH
        assert report.project_name == "test_project"

        # Check summary structure
        summary = report.summary
        assert "overall_health_score" in summary
        assert "structure_quality_score" in summary
        assert "maintainability_score" in summary
        assert "total_components" in summary
        assert "critical_issues" in summary
        assert "warnings" in summary
        assert "optimization_opportunities" in summary

        # Health scores should be between 0 and 1
        assert 0.0 <= summary["overall_health_score"] <= 1.0
        assert 0.0 <= summary["structure_quality_score"] <= 1.0
        assert 0.0 <= summary["maintainability_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_generate_comprehensive_report(self, report_service):
        """Test comprehensive report generation."""
        report = await report_service.generate_comprehensive_report(
            project_name="test_project",
            target_breadcrumb="app.module.MyClass",
            include_all_metrics=True,
        )

        assert isinstance(report, GraphAnalysisReport)
        assert report.report_type == ReportType.COMPREHENSIVE
        assert report.project_name == "test_project"

        # Check summary structure
        summary = report.summary
        assert "analysis_scope" in summary
        assert summary["analysis_scope"] == "comprehensive"
        assert "target_component" in summary
        assert summary["target_component"] == "app.module.MyClass"
        assert "total_insights" in summary
        assert "total_recommendations" in summary
        assert "reports_combined" in summary

        # Should have combined metrics
        assert hasattr(report, "structural_metrics")
        assert isinstance(report.structural_metrics, dict)

    @pytest.mark.asyncio
    async def test_error_handling_invalid_breadcrumb(self, report_service):
        """Test error handling for invalid breadcrumb."""
        # Mock the service to return empty graph
        report_service.graph_rag_service.build_structure_graph = AsyncMock(
            return_value=StructureGraph(nodes={}, edges=[], root_nodes=[], project_name="test_project")
        )

        report = await report_service.generate_structure_analysis_report(
            breadcrumb="nonexistent.breadcrumb",
            project_name="test_project",
        )

        # Should return error report
        assert report.summary.get("error") is not None
        assert "not found" in report.summary["error"]
        assert len(report.recommendations) > 0  # Should have error recommendation
        assert report.recommendations[0].severity == SeverityLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_error_handling_service_exception(self, report_service):
        """Test error handling when service throws exception."""
        # Mock the service to throw exception
        report_service.graph_rag_service.build_structure_graph = AsyncMock(side_effect=Exception("Service unavailable"))

        report = await report_service.generate_structure_analysis_report(
            breadcrumb="app.module.MyClass",
            project_name="test_project",
        )

        # Should return error report
        assert report.summary.get("error") is not None
        assert "Service unavailable" in report.summary["error"]
        assert report.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_statistical_summary_generation(self, report_service):
        """Test statistical summary generation."""
        report = await report_service.generate_structure_analysis_report(
            breadcrumb="app.module.MyClass",
            project_name="test_project",
        )

        # Should have statistical summaries
        assert isinstance(report.statistics, list)

        for stat in report.statistics:
            assert isinstance(stat, StatisticalSummary)
            assert hasattr(stat, "metric_name")
            assert hasattr(stat, "count")
            assert hasattr(stat, "mean")
            assert hasattr(stat, "median")
            assert stat.count >= 0
            if stat.count > 0:
                assert stat.min_value <= stat.mean <= stat.max_value
                assert stat.min_value <= stat.median <= stat.max_value

    def test_recommendation_severity_levels(self):
        """Test all severity levels are properly defined."""
        # Test all severity levels
        severities = [
            SeverityLevel.INFO,
            SeverityLevel.WARNING,
            SeverityLevel.CRITICAL,
            SeverityLevel.OPTIMIZATION,
        ]

        for severity in severities:
            assert hasattr(severity, "value")
            assert isinstance(severity.value, str)

    def test_report_type_enumeration(self):
        """Test all report types are properly defined."""
        report_types = [
            ReportType.STRUCTURE_SUMMARY,
            ReportType.PATTERN_ANALYSIS,
            ReportType.CONNECTIVITY_INSIGHTS,
            ReportType.PERFORMANCE_ANALYSIS,
            ReportType.COMPREHENSIVE,
            ReportType.PROJECT_HEALTH,
        ]

        for report_type in report_types:
            assert hasattr(report_type, "value")
            assert isinstance(report_type.value, str)

    @pytest.mark.asyncio
    async def test_recommendation_prioritization(self, report_service):
        """Test recommendation prioritization."""
        # Create a mock with multiple recommendations of different severities
        mock_recommendations = [
            Recommendation(
                title="Info Recommendation",
                description="Info level",
                severity=SeverityLevel.INFO,
                category="test",
                impact="Low",
                suggested_actions=["action"],
                confidence=0.5,
            ),
            Recommendation(
                title="Critical Recommendation",
                description="Critical level",
                severity=SeverityLevel.CRITICAL,
                category="test",
                impact="High",
                suggested_actions=["urgent action"],
                confidence=0.9,
            ),
            Recommendation(
                title="Warning Recommendation",
                description="Warning level",
                severity=SeverityLevel.WARNING,
                category="test",
                impact="Medium",
                suggested_actions=["action needed"],
                confidence=0.7,
            ),
        ]

        # Test prioritization method
        prioritized = report_service._prioritize_recommendations(mock_recommendations)

        # Should be sorted by severity (critical first)
        assert prioritized[0].severity == SeverityLevel.CRITICAL
        assert prioritized[-1].severity == SeverityLevel.INFO

    @pytest.mark.asyncio
    async def test_insights_generation_variety(self, report_service):
        """Test variety of insights generation."""
        # Test with different scenarios
        scenarios = [
            {"hierarchy_depth": 0, "children_count": 0, "connectivity_score": 0.1},
            {"hierarchy_depth": 5, "children_count": 20, "connectivity_score": 0.9},
            {"hierarchy_depth": 2, "children_count": 5, "connectivity_score": 0.5},
        ]

        for scenario in scenarios:
            # Mock different hierarchy data
            report_service.graph_rag_service.get_component_hierarchy = AsyncMock(
                return_value={
                    "ancestors": [{"breadcrumb": f"ancestor_{i}"} for i in range(scenario["hierarchy_depth"])],
                    "descendants": [{"breadcrumb": f"descendant_{i}"} for i in range(scenario["children_count"])],
                    "siblings": [],
                }
            )

            report_service.graph_rag_service.analyze_component_connectivity = AsyncMock(
                return_value={
                    "influence_score": scenario["connectivity_score"],
                    "connection_count": scenario["children_count"],
                }
            )

            report = await report_service.generate_structure_analysis_report(
                breadcrumb="app.module.MyClass",
                project_name="test_project",
            )

            # Should generate insights appropriate to the scenario
            assert len(report.insights) > 0
            insights_text = " ".join(report.insights).lower()

            if scenario["hierarchy_depth"] > 3:
                assert "deeply nested" in insights_text or "depth" in insights_text

            if scenario["children_count"] > 10:
                assert "high complexity" in insights_text or "many" in insights_text

            if scenario["connectivity_score"] > 0.8:
                assert "high" in insights_text and "connectivity" in insights_text

    def test_data_quality_score_calculation(self, report_service):
        """Test data quality score calculation."""
        from src.services.structure_relationship_builder import GraphNode, StructureGraph

        # Test with complete node data
        complete_node = GraphNode(
            breadcrumb="app.module.MyClass",
            name="MyClass",
            chunk_type=ChunkType.CLASS,
            file_path="/app/module.py",
            depth=1,
            parent_breadcrumb="app.module",
            children_breadcrumbs=[],
        )
        complete_node.docstring = "Complete documentation"

        graph = StructureGraph(
            nodes={"app.module.MyClass": complete_node},
            edges=[],
            root_nodes=[],
            project_name="test_project",
        )

        quality_score = report_service._calculate_data_quality_score(graph, "app.module.MyClass")
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # Should be high for complete data

        # Test with incomplete node data
        incomplete_node = GraphNode(
            breadcrumb="app.module.IncompleteClass",
            name="",  # Missing name
            chunk_type=ChunkType.CLASS,
            file_path="",  # Missing file path
            depth=1,
            parent_breadcrumb=None,
            children_breadcrumbs=[],
        )

        graph_incomplete = StructureGraph(
            nodes={"app.module.IncompleteClass": incomplete_node},
            edges=[],
            root_nodes=[],
            project_name="test_project",
        )

        quality_score_incomplete = report_service._calculate_data_quality_score(graph_incomplete, "app.module.IncompleteClass")
        assert quality_score_incomplete < quality_score  # Should be lower for incomplete data

    def test_confidence_score_calculation(self, report_service):
        """Test confidence score calculation."""
        # Test with good data
        good_hierarchy = {"current": {"breadcrumb": "test"}, "ancestors": [], "descendants": []}
        good_connectivity = {"influence_score": 0.8, "connection_count": 5}

        confidence = report_service._calculate_confidence_score(good_hierarchy, good_connectivity)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high for good data

        # Test with error data
        error_hierarchy = {"error": "Not found"}
        error_connectivity = {"error": "Analysis failed"}

        confidence_error = report_service._calculate_confidence_score(error_hierarchy, error_connectivity)
        assert confidence_error < confidence  # Should be lower for error data

    def test_singleton_service_access(self, mock_graph_rag_service):
        """Test singleton service access pattern."""
        # Test first initialization
        service1 = get_graph_analysis_report_service(mock_graph_rag_service)
        assert isinstance(service1, GraphAnalysisReportService)

        # Test subsequent access without parameter
        service2 = get_graph_analysis_report_service()
        assert service1 is service2  # Should be same instance

        # Reset global instance for clean test
        import src.services.graph_analysis_report_service

        src.services.graph_analysis_report_service._report_service_instance = None

    def test_singleton_service_error_handling(self):
        """Test singleton service error handling."""
        # Reset global instance
        import src.services.graph_analysis_report_service

        src.services.graph_analysis_report_service._report_service_instance = None

        # Test error when no service provided on first call
        with pytest.raises(ValueError, match="graph_rag_service is required"):
            get_graph_analysis_report_service()

    def test_report_generation_statistics_tracking(self, report_service):
        """Test report generation statistics tracking."""
        initial_stats = report_service.get_report_statistics()
        initial_count = initial_stats["generation_stats"]["reports_generated"]

        # Simulate report generation by calling _update_generation_stats
        report_service._update_generation_stats(100.0)
        report_service._update_generation_stats(200.0)

        updated_stats = report_service.get_report_statistics()

        # Should have incremented count
        assert updated_stats["generation_stats"]["reports_generated"] == initial_count + 2

        # Should have updated average time
        expected_avg = (100.0 + 200.0) / 2
        actual_avg = updated_stats["generation_stats"]["avg_generation_time_ms"]
        assert abs(actual_avg - expected_avg) < 0.01  # Allow for floating point precision

    @pytest.mark.asyncio
    async def test_large_project_report_generation(self, report_service):
        """Test report generation for large projects."""
        # Mock large project data
        large_graph = StructureGraph(
            nodes={
                f"node_{i}": GraphNode(
                    breadcrumb=f"node_{i}",
                    name=f"node_{i}",
                    chunk_type=ChunkType.FUNCTION,
                    file_path=f"/app/file_{i}.py",
                    depth=i % 5,
                    parent_breadcrumb=f"node_{i-1}" if i > 0 else None,
                    children_breadcrumbs=[],
                )
                for i in range(1000)
            },
            edges=[],
            root_nodes=["node_0"],
            project_name="large_test_project",
        )

        report_service.graph_rag_service.build_structure_graph = AsyncMock(return_value=large_graph)

        # Mock large project overview
        report_service.graph_rag_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "large_test_project",
                "total_components": 1000,
                "total_relationships": 999,
                "breakdown": {
                    "by_type": {"function": 1000},
                    "by_depth": dict.fromkeys(range(5), 200),
                },
                "structure_health": {
                    "orphaned_nodes": 0,
                    "average_children_per_node": 0.999,
                    "relationship_density": 0.001,
                },
            }
        )

        report = await report_service.generate_project_health_report(
            project_name="large_test_project",
        )

        assert isinstance(report, GraphAnalysisReport)
        assert report.summary["total_components"] == 1000
        assert len(report.insights) > 0
        assert len(report.recommendations) >= 0

    @pytest.mark.asyncio
    async def test_pattern_quality_score_calculation(self, report_service):
        """Test pattern quality score calculation."""
        # Test with high-quality patterns
        high_quality_patterns = [
            {"confidence": 0.9, "pattern_type": "structural"},
            {"confidence": 0.8, "pattern_type": "behavioral"},
            {"confidence": 0.85, "pattern_type": "creational"},
        ]

        quality_score = report_service._calculate_pattern_quality_score(high_quality_patterns, [], [])
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.7  # Should be high for high-quality patterns

        # Test with low-quality patterns
        low_quality_patterns = [
            {"confidence": 0.3, "pattern_type": "structural"},
            {"confidence": 0.2, "pattern_type": "structural"},
        ]

        low_quality_score = report_service._calculate_pattern_quality_score(low_quality_patterns, [], [])
        assert low_quality_score < quality_score  # Should be lower

        # Test with no patterns
        no_patterns_score = report_service._calculate_pattern_quality_score([], [], [])
        assert no_patterns_score == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, report_service):
        """Test concurrent report generation."""
        import asyncio

        # Generate multiple reports concurrently
        tasks = [
            report_service.generate_structure_analysis_report(
                breadcrumb="app.module.MyClass",
                project_name=f"test_project_{i}",
            )
            for i in range(5)
        ]

        reports = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for report in reports:
            assert isinstance(report, GraphAnalysisReport)
            assert report.execution_time_ms > 0

    def test_structural_metrics_extraction(self, report_service):
        """Test structural metrics extraction."""
        hierarchy_data = {
            "ancestors": [{"breadcrumb": "parent"}],
            "descendants": [{"breadcrumb": "child1"}, {"breadcrumb": "child2"}],
            "siblings": [{"breadcrumb": "sibling"}],
        }

        connectivity_data = {
            "influence_score": 0.75,
            "connection_count": 5,
        }

        metrics = report_service._extract_structural_metrics(hierarchy_data, connectivity_data)

        assert "hierarchy_metrics" in metrics
        assert "connectivity_metrics" in metrics

        hierarchy_metrics = metrics["hierarchy_metrics"]
        assert hierarchy_metrics["ancestor_count"] == 1
        assert hierarchy_metrics["descendant_count"] == 2
        assert hierarchy_metrics["sibling_count"] == 1

        connectivity_metrics = metrics["connectivity_metrics"]
        assert connectivity_metrics["influence_score"] == 0.75
        assert connectivity_metrics["connection_count"] == 5


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=src.services.graph_analysis_report_service",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
