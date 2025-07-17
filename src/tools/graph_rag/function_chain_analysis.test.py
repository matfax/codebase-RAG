"""Unit tests for Function Chain Analysis Tool

This module contains comprehensive tests for the trace_function_chain tool,
covering various tracing directions, boundary conditions, and error scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.code_chunk import ChunkType
from src.services.breadcrumb_resolver_service import BreadcrumbCandidate, BreadcrumbResolutionResult
from src.services.implementation_chain_service import (
    ChainDirection,
    ChainLink,
    ChainType,
    ImplementationChain,
)
from src.services.structure_relationship_builder import GraphNode
from src.tools.graph_rag.function_chain_analysis import (
    _format_arrow_output,
    _format_chain_links,
    _format_mermaid_output,
    _identify_branch_points,
    _identify_terminal_points,
    _validate_input_parameters,
    trace_function_chain,
)


class TestTraceFunction:
    """Test suite for the main trace_function_chain function."""

    @pytest.fixture
    def mock_breadcrumb_resolver(self):
        """Create a mock breadcrumb resolver."""
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock()
        return mock_resolver

    @pytest.fixture
    def mock_implementation_chain_service(self):
        """Create a mock implementation chain service."""
        mock_service = MagicMock()
        mock_service.trace_implementation_chain = AsyncMock()
        return mock_service

    @pytest.fixture
    def sample_graph_node(self):
        """Create a sample graph node for testing."""
        return GraphNode(
            chunk_id="test_chunk_1",
            breadcrumb="test.module.function",
            name="test_function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/module.py",
        )

    @pytest.fixture
    def sample_chain_link(self, sample_graph_node):
        """Create a sample chain link for testing."""
        target_node = GraphNode(
            chunk_id="test_chunk_2",
            breadcrumb="test.module.helper",
            name="helper_function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/module.py",
        )

        return ChainLink(
            source_component=sample_graph_node,
            target_component=target_node,
            relationship_type="calls",
            link_strength=0.8,
            interaction_type="method_invocation",
            evidence_source="graph_traversal",
            confidence=0.9,
        )

    @pytest.fixture
    def sample_implementation_chain(self, sample_graph_node, sample_chain_link):
        """Create a sample implementation chain for testing."""
        return ImplementationChain(
            chain_id="test_chain_1",
            chain_type=ChainType.EXECUTION_FLOW,
            entry_point=sample_graph_node,
            terminal_points=[sample_chain_link.target_component],
            links=[sample_chain_link],
            depth=2,
            branch_count=0,
            complexity_score=0.3,
            completeness_score=0.8,
            reliability_score=0.85,
            project_name="test_project",
            functional_purpose="Test execution flow",
        )

    @pytest.fixture
    def successful_breadcrumb_result(self, sample_graph_node):
        """Create a successful breadcrumb resolution result."""
        candidate = BreadcrumbCandidate(
            breadcrumb="test.module.function",
            confidence_score=0.9,
            source_chunk=None,
            reasoning="Exact match found",
            match_type="exact",
        )

        return BreadcrumbResolutionResult(
            query="test function",
            success=True,
            primary_candidate=candidate,
            resolution_time_ms=150.0,
            search_results_count=1,
        )

    @pytest.mark.asyncio
    async def test_trace_function_chain_success_forward(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test successful forward function chain tracing."""
        # Setup mocks
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                direction="forward",
                max_depth=5,
                output_format="arrow",
            )

        # Verify results
        assert result["success"] is True
        assert result["entry_point"] == "test function"
        assert result["project_name"] == "test_project"
        assert result["direction"] == "forward"
        assert result["resolved_breadcrumb"] == "test.module.function"
        assert result["breadcrumb_confidence"] == 0.9

        # Verify chain info
        chain_info = result["chain_info"]
        assert chain_info["depth"] == 2
        assert chain_info["total_links"] == 1
        assert chain_info["complexity_score"] == 0.3

        # Verify arrow format output
        assert "arrow_format" in result
        assert "test_function => helper_function" in result["arrow_format"]

        # Verify performance monitoring
        assert "performance" in result
        assert result["performance"]["total_time"] > 0

        # Verify service calls
        mock_breadcrumb_resolver.resolve.assert_called_once()
        mock_implementation_chain_service.trace_implementation_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_trace_function_chain_success_backward(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test successful backward function chain tracing."""
        # Setup mocks
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                direction="backward",
                max_depth=8,
                output_format="mermaid",
            )

        # Verify results
        assert result["success"] is True
        assert result["direction"] == "backward"

        # Verify Mermaid format output
        assert "mermaid_format" in result
        assert "graph TD" in result["mermaid_format"]

        # Verify service was called with correct direction
        call_args = mock_implementation_chain_service.trace_implementation_chain.call_args
        assert call_args[1]["direction"] == ChainDirection.BACKWARD

    @pytest.mark.asyncio
    async def test_trace_function_chain_success_bidirectional(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test successful bidirectional function chain tracing."""
        # Setup mocks
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                direction="bidirectional",
                max_depth=10,
                output_format="both",
            )

        # Verify results
        assert result["success"] is True
        assert result["direction"] == "bidirectional"

        # Verify both output formats
        assert "arrow_format" in result
        assert "mermaid_format" in result

        # Verify service was called with correct direction
        call_args = mock_implementation_chain_service.trace_implementation_chain.call_args
        assert call_args[1]["direction"] == ChainDirection.BIDIRECTIONAL

    @pytest.mark.asyncio
    async def test_trace_function_chain_breadcrumb_resolution_failure(self, mock_breadcrumb_resolver, mock_implementation_chain_service):
        """Test failure in breadcrumb resolution."""
        # Setup failed breadcrumb resolution
        failed_result = BreadcrumbResolutionResult(
            query="nonexistent function",
            success=False,
            error_message="Function not found in codebase",
            resolution_time_ms=200.0,
            search_results_count=0,
        )
        mock_breadcrumb_resolver.resolve.return_value = failed_result

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="nonexistent function",
                project_name="test_project",
                direction="forward",
            )

        # Verify failure handling
        assert result["success"] is False
        assert "Failed to resolve entry point" in result["error"]
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

        # Verify chain service was not called
        mock_implementation_chain_service.trace_implementation_chain.assert_not_called()

    @pytest.mark.asyncio
    async def test_trace_function_chain_invalid_parameters(self):
        """Test various invalid parameter combinations."""
        # Test empty entry point
        result = await trace_function_chain(
            entry_point="",
            project_name="test_project",
        )
        assert result["success"] is False
        assert "Entry point is required" in result["error"]

        # Test empty project name
        result = await trace_function_chain(
            entry_point="test function",
            project_name="",
        )
        assert result["success"] is False
        assert "Project name is required" in result["error"]

        # Test invalid direction
        result = await trace_function_chain(
            entry_point="test function",
            project_name="test_project",
            direction="invalid_direction",
        )
        assert result["success"] is False
        assert "Invalid direction" in result["error"]

        # Test invalid max_depth
        result = await trace_function_chain(
            entry_point="test function",
            project_name="test_project",
            max_depth=0,
        )
        assert result["success"] is False
        assert "Invalid max_depth" in result["error"]

    @pytest.mark.asyncio
    async def test_trace_function_chain_exception_handling(self, mock_breadcrumb_resolver, mock_implementation_chain_service):
        """Test exception handling in the main function."""
        # Setup exception in breadcrumb resolver
        mock_breadcrumb_resolver.resolve.side_effect = Exception("Resolver error")

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
            )

        # Verify error handling
        assert result["success"] is False
        assert "Error tracing function chain" in result["error"]
        assert "suggestions" in result
        assert result["performance"]["error_occurred"] is True


class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_input_parameters_success(self):
        """Test successful parameter validation."""
        result = _validate_input_parameters(
            entry_point="test function",
            project_name="test_project",
            direction="forward",
            max_depth=10,
            output_format="arrow",
            chain_type="execution_flow",
            min_link_strength=0.5,
        )

        assert result["valid"] is True

    def test_validate_input_parameters_failures(self):
        """Test various parameter validation failures."""
        # Test empty entry point
        result = _validate_input_parameters("", "test_project", "forward", 10, "arrow", "execution_flow", 0.5)
        assert result["valid"] is False
        assert "Entry point is required" in result["error"]

        # Test invalid direction
        result = _validate_input_parameters("test", "test_project", "invalid", 10, "arrow", "execution_flow", 0.5)
        assert result["valid"] is False
        assert "Invalid direction" in result["error"]

        # Test invalid max_depth
        result = _validate_input_parameters("test", "test_project", "forward", 0, "arrow", "execution_flow", 0.5)
        assert result["valid"] is False
        assert "Invalid max_depth" in result["error"]

        # Test invalid output_format
        result = _validate_input_parameters("test", "test_project", "forward", 10, "invalid", "execution_flow", 0.5)
        assert result["valid"] is False
        assert "Invalid output_format" in result["error"]

        # Test invalid chain_type
        result = _validate_input_parameters("test", "test_project", "forward", 10, "arrow", "invalid_type", 0.5)
        assert result["valid"] is False
        assert "Invalid chain_type" in result["error"]

        # Test invalid min_link_strength
        result = _validate_input_parameters("test", "test_project", "forward", 10, "arrow", "execution_flow", 1.5)
        assert result["valid"] is False
        assert "Invalid min_link_strength" in result["error"]


class TestFormattingFunctions:
    """Test suite for formatting functions."""

    @pytest.fixture
    def sample_nodes(self):
        """Create sample graph nodes for testing."""
        node1 = GraphNode(
            chunk_id="chunk1",
            breadcrumb="module.ClassA.method1",
            name="method1",
            chunk_type=ChunkType.METHOD,
            file_path="/src/module.py",
        )

        node2 = GraphNode(
            chunk_id="chunk2",
            breadcrumb="module.ClassA.method2",
            name="method2",
            chunk_type=ChunkType.METHOD,
            file_path="/src/module.py",
        )

        node3 = GraphNode(
            chunk_id="chunk3",
            breadcrumb="module.ClassB.helper",
            name="helper",
            chunk_type=ChunkType.METHOD,
            file_path="/src/module.py",
        )

        return node1, node2, node3

    @pytest.fixture
    def sample_chain_with_branches(self, sample_nodes):
        """Create a sample chain with branches for testing."""
        node1, node2, node3 = sample_nodes

        # Create links: node1 -> node2 and node1 -> node3 (branch)
        link1 = ChainLink(
            source_component=node1,
            target_component=node2,
            relationship_type="calls",
            link_strength=0.8,
            interaction_type="method_invocation",
            evidence_source="graph_traversal",
            confidence=0.9,
        )

        link2 = ChainLink(
            source_component=node1,
            target_component=node3,
            relationship_type="calls",
            link_strength=0.7,
            interaction_type="method_invocation",
            evidence_source="graph_traversal",
            confidence=0.8,
        )

        return ImplementationChain(
            chain_id="test_chain_branches",
            chain_type=ChainType.EXECUTION_FLOW,
            entry_point=node1,
            terminal_points=[node2, node3],
            links=[link1, link2],
            depth=2,
            branch_count=1,
            complexity_score=0.4,
            completeness_score=0.8,
            reliability_score=0.85,
            project_name="test_project",
            functional_purpose="Test execution flow with branches",
        )

    def test_identify_branch_points(self, sample_chain_with_branches):
        """Test branch point identification."""
        branch_points = _identify_branch_points(sample_chain_with_branches)

        assert len(branch_points) == 1
        branch_point = branch_points[0]
        assert branch_point["name"] == "method1"
        assert branch_point["branch_count"] == 2
        assert len(branch_point["target_components"]) == 2
        assert "method2" in branch_point["target_components"]
        assert "helper" in branch_point["target_components"]

    def test_identify_terminal_points(self, sample_chain_with_branches):
        """Test terminal point identification."""
        terminal_points = _identify_terminal_points(sample_chain_with_branches)

        assert len(terminal_points) == 2
        terminal_names = [tp["name"] for tp in terminal_points]
        assert "method2" in terminal_names
        assert "helper" in terminal_names

    def test_format_arrow_output(self, sample_chain_with_branches):
        """Test arrow format output."""
        arrow_output = _format_arrow_output(sample_chain_with_branches)

        assert "method1 =>" in arrow_output
        assert "with 1 branches" in arrow_output

    def test_format_arrow_output_empty_chain(self, sample_nodes):
        """Test arrow format with empty chain."""
        node1, _, _ = sample_nodes
        empty_chain = ImplementationChain(
            chain_id="empty_chain",
            chain_type=ChainType.EXECUTION_FLOW,
            entry_point=node1,
            terminal_points=[],
            links=[],
            depth=0,
            branch_count=0,
            complexity_score=0.0,
            completeness_score=0.0,
            reliability_score=0.0,
            project_name="test_project",
        )

        arrow_output = _format_arrow_output(empty_chain)
        assert "method1" in arrow_output
        assert "no connections found" in arrow_output

    def test_format_mermaid_output(self, sample_chain_with_branches):
        """Test Mermaid format output."""
        mermaid_output = _format_mermaid_output(sample_chain_with_branches)

        assert "graph TD" in mermaid_output
        assert "method1" in mermaid_output
        assert "method2" in mermaid_output
        assert "helper" in mermaid_output
        assert "-->" in mermaid_output
        assert "classDef entryPoint" in mermaid_output

    def test_format_mermaid_output_empty_chain(self, sample_nodes):
        """Test Mermaid format with empty chain."""
        node1, _, _ = sample_nodes
        empty_chain = ImplementationChain(
            chain_id="empty_chain",
            chain_type=ChainType.EXECUTION_FLOW,
            entry_point=node1,
            terminal_points=[],
            links=[],
            depth=0,
            branch_count=0,
            complexity_score=0.0,
            completeness_score=0.0,
            reliability_score=0.0,
            project_name="test_project",
        )

        mermaid_output = _format_mermaid_output(empty_chain)
        assert "graph TD" in mermaid_output
        assert "method1" in mermaid_output
        assert "No connections found" in mermaid_output

    def test_format_chain_links(self, sample_chain_with_branches):
        """Test chain link formatting."""
        formatted_links = _format_chain_links(sample_chain_with_branches)

        assert len(formatted_links) == 2

        # Check first link
        link1 = formatted_links[0]
        assert link1["index"] == 1
        assert link1["source"]["name"] == "method1"
        assert link1["target"]["name"] == "method2"
        assert link1["relationship"]["type"] == "calls"
        assert link1["relationship"]["strength"] == 0.8

        # Check second link
        link2 = formatted_links[1]
        assert link2["index"] == 2
        assert link2["source"]["name"] == "method1"
        assert link2["target"]["name"] == "helper"


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_trace_function_chain_max_depth_1(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test with minimum depth value."""
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                max_depth=1,
            )

        assert result["success"] is True
        assert result["max_depth"] == 1

    @pytest.mark.asyncio
    async def test_trace_function_chain_max_depth_50(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test with maximum depth value."""
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                max_depth=50,
            )

        assert result["success"] is True
        assert result["max_depth"] == 50

    @pytest.mark.asyncio
    async def test_trace_function_chain_various_chain_types(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test with various chain types."""
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        chain_types = ["execution_flow", "data_flow", "dependency_chain", "inheritance_chain"]

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            for chain_type in chain_types:
                result = await trace_function_chain(
                    entry_point="test function",
                    project_name="test_project",
                    chain_type=chain_type,
                )

                assert result["success"] is True
                assert result["chain_type"] == chain_type

    @pytest.mark.asyncio
    async def test_trace_function_chain_min_link_strength_bounds(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test with minimum and maximum link strength values."""
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            # Test minimum value
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                min_link_strength=0.0,
            )
            assert result["success"] is True

            # Test maximum value
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                min_link_strength=1.0,
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trace_function_chain_performance_monitoring_disabled(
        self, mock_breadcrumb_resolver, mock_implementation_chain_service, successful_breadcrumb_result, sample_implementation_chain
    ):
        """Test with performance monitoring disabled."""
        mock_breadcrumb_resolver.resolve.return_value = successful_breadcrumb_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = sample_implementation_chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await trace_function_chain(
                entry_point="test function",
                project_name="test_project",
                performance_monitoring=False,
            )

        assert result["success"] is True
        assert "performance" not in result


if __name__ == "__main__":
    pytest.main([__file__])
