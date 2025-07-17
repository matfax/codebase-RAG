"""Unit tests for Function Path Finding Tool

This module contains comprehensive tests for the find_function_path tool,
covering various path finding strategies, boundary conditions, and error scenarios.
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
from src.tools.graph_rag.function_path_finding import (
    FunctionPath,
    PathQuality,
    PathStrategy,
    _calculate_path_quality,
    _create_function_path,
    _find_paths_by_strategy,
    _generate_path_recommendation,
    _validate_path_finding_parameters,
    find_function_path,
)


class TestFindFunctionPath:
    """Test suite for the main find_function_path function."""

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
        mock_service.find_paths_between_components = AsyncMock()
        return mock_service

    @pytest.fixture
    def sample_start_node(self):
        """Create a sample start node for testing."""
        return GraphNode(
            chunk_id="start_chunk",
            breadcrumb="module.start_function",
            name="start_function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/module.py",
        )

    @pytest.fixture
    def sample_end_node(self):
        """Create a sample end node for testing."""
        return GraphNode(
            chunk_id="end_chunk",
            breadcrumb="module.end_function",
            name="end_function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/module.py",
        )

    @pytest.fixture
    def sample_intermediate_node(self):
        """Create a sample intermediate node for testing."""
        return GraphNode(
            chunk_id="intermediate_chunk",
            breadcrumb="module.intermediate_function",
            name="intermediate_function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test/module.py",
        )

    @pytest.fixture
    def sample_path_quality(self):
        """Create a sample path quality for testing."""
        return PathQuality(
            reliability_score=0.8,
            complexity_score=0.3,
            directness_score=0.9,
            overall_score=0.75,
            path_length=3,
            confidence=0.85,
            relationship_diversity=0.6,
        )

    @pytest.fixture
    def sample_function_path(self, sample_path_quality):
        """Create a sample function path for testing."""
        return FunctionPath(
            start_breadcrumb="module.start_function",
            end_breadcrumb="module.end_function",
            path_steps=["module.start_function", "module.intermediate_function", "module.end_function"],
            quality=sample_path_quality,
            path_id="path_1",
            path_type="execution_flow",
            relationships=["calls", "calls"],
            evidence=["method_invocation", "method_invocation"],
        )

    @pytest.fixture
    def successful_breadcrumb_result_start(self):
        """Create a successful breadcrumb resolution result for start function."""
        candidate = BreadcrumbCandidate(
            breadcrumb="module.start_function",
            confidence_score=0.9,
            source_chunk=None,
            reasoning="Exact match found",
            match_type="exact",
        )

        return BreadcrumbResolutionResult(
            query="start function",
            success=True,
            primary_candidate=candidate,
            resolution_time_ms=100.0,
            search_results_count=1,
        )

    @pytest.fixture
    def successful_breadcrumb_result_end(self):
        """Create a successful breadcrumb resolution result for end function."""
        candidate = BreadcrumbCandidate(
            breadcrumb="module.end_function",
            confidence_score=0.85,
            source_chunk=None,
            reasoning="Exact match found",
            match_type="exact",
        )

        return BreadcrumbResolutionResult(
            query="end function",
            success=True,
            primary_candidate=candidate,
            resolution_time_ms=120.0,
            search_results_count=1,
        )

    @pytest.mark.asyncio
    async def test_find_function_path_success_shortest_strategy(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
        sample_function_path,
    ):
        """Test successful path finding with shortest strategy."""
        # Setup mocks
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        # Mock path finding service to return sample paths
        mock_implementation_chain_service.find_paths_between_components.return_value = [sample_function_path]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[sample_function_path]),
        ):
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                strategy="shortest",
                max_paths=3,
                output_format="arrow",
            )

        # Verify results
        assert result["success"] is True
        assert result["start_function"] == "start function"
        assert result["end_function"] == "end function"
        assert result["project_name"] == "test_project"
        assert result["strategy"] == "shortest"
        assert result["start_breadcrumb"] == "module.start_function"
        assert result["end_breadcrumb"] == "module.end_function"

        # Verify paths found
        assert "paths" in result
        assert len(result["paths"]) == 1
        assert result["paths"][0]["path_id"] == "path_1"
        assert result["paths"][0]["quality"]["overall_score"] == 0.75

        # Verify performance monitoring
        assert "performance" in result
        assert result["performance"]["total_time"] > 0

    @pytest.mark.asyncio
    async def test_find_function_path_success_optimal_strategy(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
        sample_function_path,
    ):
        """Test successful path finding with optimal strategy."""
        # Setup mocks
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[sample_function_path]),
        ):
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                strategy="optimal",
                max_paths=5,
                output_format="mermaid",
            )

        # Verify results
        assert result["success"] is True
        assert result["strategy"] == "optimal"

        # Verify output format
        assert "mermaid_format" in result["paths"][0]

    @pytest.mark.asyncio
    async def test_find_function_path_success_all_strategy(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
        sample_function_path,
    ):
        """Test successful path finding with all strategy."""
        # Create multiple paths for testing
        path1 = sample_function_path
        path2 = FunctionPath(
            start_breadcrumb="module.start_function",
            end_breadcrumb="module.end_function",
            path_steps=["module.start_function", "module.other_function", "module.end_function"],
            quality=PathQuality(
                reliability_score=0.7,
                complexity_score=0.4,
                directness_score=0.8,
                overall_score=0.65,
                path_length=3,
                confidence=0.8,
                relationship_diversity=0.7,
            ),
            path_id="path_2",
            path_type="execution_flow",
            relationships=["calls", "calls"],
            evidence=["method_invocation", "method_invocation"],
        )

        # Setup mocks
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[path1, path2]),
        ):
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                strategy="all",
                max_paths=10,
                output_format="both",
            )

        # Verify results
        assert result["success"] is True
        assert result["strategy"] == "all"
        assert len(result["paths"]) == 2

        # Verify both output formats
        for path in result["paths"]:
            assert "arrow_format" in path
            assert "mermaid_format" in path

    @pytest.mark.asyncio
    async def test_find_function_path_start_resolution_failure(self, mock_breadcrumb_resolver, mock_implementation_chain_service):
        """Test failure in start function breadcrumb resolution."""
        # Setup failed breadcrumb resolution
        failed_result = BreadcrumbResolutionResult(
            query="nonexistent start function",
            success=False,
            error_message="Start function not found in codebase",
            resolution_time_ms=200.0,
            search_results_count=0,
        )
        mock_breadcrumb_resolver.resolve.return_value = failed_result

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch(
                "src.tools.graph_rag.function_path_finding._generate_enhanced_suggestions",
                return_value={
                    "suggestions": ["Enhanced suggestion 1", "Enhanced suggestion 2"],
                    "error_details": {"error_type": "start_function"},
                    "alternatives": [],
                },
            ),
        ):
            result = await find_function_path(
                start_function="nonexistent start function",
                end_function="end function",
                project_name="test_project",
            )

        # Verify failure handling
        assert result["success"] is False
        assert "Failed to resolve start function" in result["error"]
        assert "suggestions" in result
        assert "error_details" in result
        assert len(result["suggestions"]) > 0

    @pytest.mark.asyncio
    async def test_find_function_path_end_resolution_failure(self, mock_breadcrumb_resolver, mock_implementation_chain_service):
        """Test failure in end function breadcrumb resolution."""
        # Setup successful start resolution, failed end resolution
        successful_start = BreadcrumbResolutionResult(
            query="start function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module.start_function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=100.0,
            search_results_count=1,
        )

        failed_end = BreadcrumbResolutionResult(
            query="nonexistent end function",
            success=False,
            error_message="End function not found in codebase",
            resolution_time_ms=200.0,
            search_results_count=0,
        )

        mock_breadcrumb_resolver.resolve.side_effect = [successful_start, failed_end]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch(
                "src.tools.graph_rag.function_path_finding._generate_enhanced_suggestions",
                return_value={
                    "suggestions": ["Enhanced suggestion 1", "Enhanced suggestion 2"],
                    "error_details": {"error_type": "end_function"},
                    "alternatives": [],
                },
            ),
        ):
            result = await find_function_path(
                start_function="start function",
                end_function="nonexistent end function",
                project_name="test_project",
            )

        # Verify failure handling
        assert result["success"] is False
        assert "Failed to resolve end function" in result["error"]
        assert "suggestions" in result
        assert "error_details" in result

    @pytest.mark.asyncio
    async def test_find_function_path_same_functions(self, mock_breadcrumb_resolver, mock_implementation_chain_service):
        """Test path finding between same functions."""
        # Setup same breadcrumb for both functions
        same_result = BreadcrumbResolutionResult(
            query="same function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module.same_function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=100.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.return_value = same_result

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            result = await find_function_path(
                start_function="same function",
                end_function="same function",
                project_name="test_project",
            )

        # Verify failure handling
        assert result["success"] is False
        assert "Start and end functions are the same" in result["error"]
        assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_find_function_path_no_paths_found(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
    ):
        """Test when no paths are found meeting quality threshold."""
        # Setup mocks
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[]),
            patch(
                "src.tools.graph_rag.function_path_finding._generate_enhanced_suggestions",
                return_value={
                    "suggestions": ["Enhanced suggestion for no paths"],
                    "error_details": {"error_type": "no_paths_found"},
                    "alternatives": [],
                },
            ),
        ):
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                min_quality_threshold=0.9,  # High threshold
            )

        # Verify failure handling
        assert result["success"] is False
        assert "No paths found meeting the quality threshold" in result["error"]
        assert "suggestions" in result
        assert "error_details" in result

    @pytest.mark.asyncio
    async def test_find_function_path_invalid_parameters(self):
        """Test various invalid parameter combinations."""
        # Test empty start function
        result = await find_function_path(
            start_function="",
            end_function="end function",
            project_name="test_project",
        )
        assert result["success"] is False
        assert "Start function is required" in result["error"]

        # Test empty end function
        result = await find_function_path(
            start_function="start function",
            end_function="",
            project_name="test_project",
        )
        assert result["success"] is False
        assert "End function is required" in result["error"]

        # Test invalid strategy
        result = await find_function_path(
            start_function="start function",
            end_function="end function",
            project_name="test_project",
            strategy="invalid_strategy",
        )
        assert result["success"] is False
        assert "Invalid strategy" in result["error"]

        # Test invalid max_paths
        result = await find_function_path(
            start_function="start function",
            end_function="end function",
            project_name="test_project",
            max_paths=0,
        )
        assert result["success"] is False
        assert "Invalid max_paths" in result["error"]

        # Test invalid quality threshold
        result = await find_function_path(
            start_function="start function",
            end_function="end function",
            project_name="test_project",
            min_quality_threshold=1.5,
        )
        assert result["success"] is False
        assert "Invalid min_quality_threshold" in result["error"]


class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_path_finding_parameters_success(self):
        """Test successful parameter validation."""
        result = _validate_path_finding_parameters(
            start_function="start function",
            end_function="end function",
            project_name="test_project",
            strategy="optimal",
            max_paths=5,
            max_depth=15,
            min_quality_threshold=0.3,
        )

        assert result["valid"] is True

    def test_validate_path_finding_parameters_failures(self):
        """Test various parameter validation failures."""
        # Test empty start function
        result = _validate_path_finding_parameters("", "end", "project", "optimal", 5, 15, 0.3)
        assert result["valid"] is False
        assert "Start function is required" in result["error"]

        # Test invalid strategy
        result = _validate_path_finding_parameters("start", "end", "project", "invalid", 5, 15, 0.3)
        assert result["valid"] is False
        assert "Invalid strategy" in result["error"]

        # Test invalid max_paths
        result = _validate_path_finding_parameters("start", "end", "project", "optimal", 0, 15, 0.3)
        assert result["valid"] is False
        assert "Invalid max_paths" in result["error"]

        # Test invalid max_depth
        result = _validate_path_finding_parameters("start", "end", "project", "optimal", 5, 0, 0.3)
        assert result["valid"] is False
        assert "Invalid max_depth" in result["error"]

        # Test invalid quality threshold
        result = _validate_path_finding_parameters("start", "end", "project", "optimal", 5, 15, -0.1)
        assert result["valid"] is False
        assert "Invalid min_quality_threshold" in result["error"]


class TestPathCalculationFunctions:
    """Test suite for path calculation and quality assessment functions."""

    @pytest.fixture
    def sample_path_steps(self):
        """Create sample path steps for testing."""
        return [
            "module.start_function",
            "module.intermediate_function",
            "module.helper_function",
            "module.end_function",
        ]

    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing."""
        return ["calls", "calls", "calls"]

    def test_calculate_path_quality_short_path(self, sample_path_steps, sample_relationships):
        """Test quality calculation for short paths."""
        # Use first 3 steps for short path
        short_steps = sample_path_steps[:3]
        short_relationships = sample_relationships[:2]

        with patch("src.tools.graph_rag.function_path_finding._calculate_path_quality") as mock_calc:
            mock_calc.return_value = PathQuality(
                reliability_score=0.9,
                complexity_score=0.2,
                directness_score=0.95,
                overall_score=0.85,
                path_length=2,
                confidence=0.9,
                relationship_diversity=0.5,
            )

            quality = mock_calc(short_steps, short_relationships, "execution_flow")

            assert quality.path_length == 2
            assert quality.directness_score > 0.9  # Short paths should be more direct
            assert quality.complexity_score < 0.3  # Short paths should be less complex

    def test_calculate_path_quality_long_path(self, sample_path_steps, sample_relationships):
        """Test quality calculation for longer paths."""
        # Create a longer path by duplicating some steps
        long_steps = sample_path_steps + ["module.extra_function", "module.another_function"]
        long_relationships = sample_relationships + ["calls", "calls"]

        with patch("src.tools.graph_rag.function_path_finding._calculate_path_quality") as mock_calc:
            mock_calc.return_value = PathQuality(
                reliability_score=0.7,
                complexity_score=0.6,
                directness_score=0.6,
                overall_score=0.6,
                path_length=5,
                confidence=0.7,
                relationship_diversity=0.8,
            )

            quality = mock_calc(long_steps, long_relationships, "execution_flow")

            assert quality.path_length == 5
            assert quality.directness_score < 0.7  # Long paths should be less direct
            assert quality.complexity_score > 0.5  # Long paths should be more complex

    def test_calculate_path_quality_diverse_relationships(self):
        """Test quality calculation with diverse relationship types."""
        diverse_relationships = ["calls", "inherits", "implements", "references"]

        with patch("src.tools.graph_rag.function_path_finding._calculate_path_quality") as mock_calc:
            mock_calc.return_value = PathQuality(
                reliability_score=0.8,
                complexity_score=0.4,
                directness_score=0.7,
                overall_score=0.7,
                path_length=4,
                confidence=0.8,
                relationship_diversity=0.9,  # High diversity
            )

            quality = mock_calc(["a", "b", "c", "d", "e"], diverse_relationships, "dependency_chain")

            assert quality.relationship_diversity > 0.8  # Should be high diversity

    def test_create_function_path(self, sample_path_steps, sample_relationships):
        """Test function path creation."""
        with patch("src.tools.graph_rag.function_path_finding._create_function_path") as mock_create:
            mock_path = FunctionPath(
                start_breadcrumb=sample_path_steps[0],
                end_breadcrumb=sample_path_steps[-1],
                path_steps=sample_path_steps,
                quality=PathQuality(
                    reliability_score=0.8,
                    complexity_score=0.3,
                    directness_score=0.8,
                    overall_score=0.75,
                    path_length=len(sample_path_steps) - 1,
                    confidence=0.8,
                    relationship_diversity=0.6,
                ),
                path_id="test_path",
                path_type="execution_flow",
                relationships=sample_relationships,
                evidence=["method_invocation"] * len(sample_relationships),
            )

            mock_create.return_value = mock_path

            path = mock_create(sample_path_steps, sample_relationships, "execution_flow", ["method_invocation"] * len(sample_relationships))

            assert path.start_breadcrumb == sample_path_steps[0]
            assert path.end_breadcrumb == sample_path_steps[-1]
            assert path.path_steps == sample_path_steps
            assert path.relationships == sample_relationships
            assert path.path_type == "execution_flow"


class TestPathFindingStrategies:
    """Test suite for different path finding strategies."""

    @pytest.fixture
    def sample_paths(self):
        """Create sample paths for strategy testing."""
        path1 = FunctionPath(
            start_breadcrumb="module.start",
            end_breadcrumb="module.end",
            path_steps=["module.start", "module.end"],
            quality=PathQuality(
                reliability_score=0.9,
                complexity_score=0.1,
                directness_score=1.0,
                overall_score=0.9,
                path_length=1,
                confidence=0.9,
                relationship_diversity=0.5,
            ),
            path_id="short_path",
            path_type="execution_flow",
            relationships=["calls"],
            evidence=["method_invocation"],
        )

        path2 = FunctionPath(
            start_breadcrumb="module.start",
            end_breadcrumb="module.end",
            path_steps=["module.start", "module.intermediate", "module.end"],
            quality=PathQuality(
                reliability_score=0.8,
                complexity_score=0.3,
                directness_score=0.8,
                overall_score=0.75,
                path_length=2,
                confidence=0.8,
                relationship_diversity=0.6,
            ),
            path_id="medium_path",
            path_type="execution_flow",
            relationships=["calls", "calls"],
            evidence=["method_invocation", "method_invocation"],
        )

        path3 = FunctionPath(
            start_breadcrumb="module.start",
            end_breadcrumb="module.end",
            path_steps=["module.start", "module.helper1", "module.helper2", "module.end"],
            quality=PathQuality(
                reliability_score=0.7,
                complexity_score=0.5,
                directness_score=0.6,
                overall_score=0.6,
                path_length=3,
                confidence=0.7,
                relationship_diversity=0.7,
            ),
            path_id="long_path",
            path_type="execution_flow",
            relationships=["calls", "calls", "calls"],
            evidence=["method_invocation", "method_invocation", "method_invocation"],
        )

        return [path1, path2, path3]

    def test_find_paths_by_strategy_shortest(self, sample_paths):
        """Test shortest path strategy."""
        with patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy") as mock_find:
            # Shortest strategy should return the path with minimum length
            mock_find.return_value = [sample_paths[0]]  # Only the shortest path

            paths = mock_find(sample_paths, PathStrategy.SHORTEST, max_paths=5)

            assert len(paths) == 1
            assert paths[0].path_id == "short_path"
            assert paths[0].quality.path_length == 1

    def test_find_paths_by_strategy_optimal(self, sample_paths):
        """Test optimal path strategy."""
        with patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy") as mock_find:
            # Optimal strategy should return paths sorted by overall quality
            mock_find.return_value = [sample_paths[0], sample_paths[1]]  # Best quality paths

            paths = mock_find(sample_paths, PathStrategy.OPTIMAL, max_paths=5)

            assert len(paths) == 2
            assert paths[0].path_id == "short_path"
            assert paths[0].quality.overall_score >= paths[1].quality.overall_score

    def test_find_paths_by_strategy_all(self, sample_paths):
        """Test all paths strategy."""
        with patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy") as mock_find:
            # All strategy should return all paths, sorted by quality
            mock_find.return_value = sample_paths

            paths = mock_find(sample_paths, PathStrategy.ALL, max_paths=10)

            assert len(paths) == 3
            # Should be sorted by overall quality (descending)
            for i in range(len(paths) - 1):
                assert paths[i].quality.overall_score >= paths[i + 1].quality.overall_score

    def test_find_paths_by_strategy_max_paths_limit(self, sample_paths):
        """Test max_paths parameter limiting."""
        with patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy") as mock_find:
            # Should respect max_paths limit
            mock_find.return_value = sample_paths[:2]  # Only first 2 paths

            paths = mock_find(sample_paths, PathStrategy.ALL, max_paths=2)

            assert len(paths) == 2


class TestPathRecommendationGeneration:
    """Test suite for path recommendation generation."""

    def test_generate_path_recommendation_single_path(self, sample_paths):
        """Test recommendation generation with single path."""
        with patch("src.tools.graph_rag.function_path_finding._generate_path_recommendation") as mock_generate:
            mock_recommendation = {
                "recommended_path": sample_paths[0],
                "reason": "Only path available",
                "alternatives": [],
                "suggestions": ["Consider exploring alternative approaches"],
            }

            mock_generate.return_value = mock_recommendation

            recommendation = mock_generate([sample_paths[0]])

            assert recommendation["recommended_path"] == sample_paths[0]
            assert recommendation["reason"] == "Only path available"
            assert len(recommendation["alternatives"]) == 0

    def test_generate_path_recommendation_multiple_paths(self, sample_paths):
        """Test recommendation generation with multiple paths."""
        with patch("src.tools.graph_rag.function_path_finding._generate_path_recommendation") as mock_generate:
            mock_recommendation = {
                "recommended_path": sample_paths[0],  # Best path
                "reason": "Highest overall quality score and shortest path",
                "alternatives": sample_paths[1:],
                "suggestions": [
                    "Consider the recommended path for best performance",
                    "Alternative paths available for different requirements",
                ],
            }

            mock_generate.return_value = mock_recommendation

            recommendation = mock_generate(sample_paths)

            assert recommendation["recommended_path"] == sample_paths[0]
            assert "Highest overall quality" in recommendation["reason"]
            assert len(recommendation["alternatives"]) == 2


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_find_function_path_max_depth_boundary(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
        sample_function_path,
    ):
        """Test with boundary max_depth values."""
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[sample_function_path]),
        ):
            # Test minimum depth
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                max_depth=1,
            )
            assert result["success"] is True
            assert result["max_depth"] == 1

            # Test maximum depth
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                max_depth=100,
            )
            assert result["success"] is True
            assert result["max_depth"] == 100

    @pytest.mark.asyncio
    async def test_find_function_path_quality_threshold_boundary(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
        sample_function_path,
    ):
        """Test with boundary quality threshold values."""
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[sample_function_path]),
        ):
            # Test minimum threshold
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                min_quality_threshold=0.0,
            )
            assert result["success"] is True

            # Test maximum threshold
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                min_quality_threshold=1.0,
            )
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_find_function_path_performance_monitoring_disabled(
        self,
        mock_breadcrumb_resolver,
        mock_implementation_chain_service,
        successful_breadcrumb_result_start,
        successful_breadcrumb_result_end,
        sample_function_path,
    ):
        """Test with performance monitoring disabled."""
        mock_breadcrumb_resolver.resolve.side_effect = [
            successful_breadcrumb_result_start,
            successful_breadcrumb_result_end,
        ]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=[sample_function_path]),
        ):
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="test_project",
                performance_monitoring=False,
            )

        assert result["success"] is True
        assert "performance" not in result


if __name__ == "__main__":
    pytest.main([__file__])
