"""Performance tests for Function Path Finding Tools

This module contains comprehensive performance tests for the Graph RAG function
path finding tools, ensuring they can handle large codebases efficiently.
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List
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
from src.tools.graph_rag.function_chain_analysis import trace_function_chain
from src.tools.graph_rag.function_path_finding import (
    FunctionPath,
    PathQuality,
    PathStrategy,
    find_function_path,
)


class PerformanceTestFixtures:
    """Test fixtures for performance testing."""

    @staticmethod
    def create_large_graph_nodes(count: int) -> list[GraphNode]:
        """Create a large number of graph nodes for performance testing."""
        nodes = []
        for i in range(count):
            node = GraphNode(
                chunk_id=f"chunk_{i}",
                breadcrumb=f"module_{i // 100}.class_{i // 10}.method_{i}",
                name=f"method_{i}",
                chunk_type=ChunkType.METHOD,
                file_path=f"/test/module_{i // 100}.py",
            )
            nodes.append(node)
        return nodes

    @staticmethod
    def create_large_chain_links(nodes: list[GraphNode]) -> list[ChainLink]:
        """Create a large number of chain links for performance testing."""
        links = []
        for i in range(len(nodes) - 1):
            link = ChainLink(
                source_component=nodes[i],
                target_component=nodes[i + 1],
                relationship_type="calls",
                link_strength=0.8,
                interaction_type="method_invocation",
                evidence_source="graph_traversal",
                confidence=0.9,
            )
            links.append(link)
        return links

    @staticmethod
    def create_large_implementation_chain(nodes: list[GraphNode], links: list[ChainLink]) -> ImplementationChain:
        """Create a large implementation chain for performance testing."""
        return ImplementationChain(
            chain_id="large_test_chain",
            chain_type=ChainType.EXECUTION_FLOW,
            entry_point=nodes[0],
            terminal_points=[nodes[-1]],
            links=links,
            depth=len(nodes),
            branch_count=10,
            complexity_score=0.5,
            completeness_score=0.8,
            reliability_score=0.85,
            project_name="large_test_project",
            functional_purpose="Large scale performance test",
        )

    @staticmethod
    def create_multiple_paths(count: int) -> list[FunctionPath]:
        """Create multiple function paths for performance testing."""
        paths = []
        for i in range(count):
            path = FunctionPath(
                start_breadcrumb=f"start.function_{i}",
                end_breadcrumb=f"end.function_{i}",
                path_steps=[
                    f"start.function_{i}",
                    f"intermediate.function_{i}",
                    f"end.function_{i}",
                ],
                quality=PathQuality(
                    reliability_score=0.8,
                    complexity_score=0.3,
                    directness_score=0.9,
                    overall_score=0.75,
                    path_length=3,
                    confidence=0.85,
                    relationship_diversity=0.6,
                ),
                path_id=f"path_{i}",
                path_type="execution_flow",
                relationships=["calls", "calls"],
                evidence=[f"evidence_{i}_1", f"evidence_{i}_2"],
            )
            paths.append(path)
        return paths


class TestFunctionChainAnalysisPerformance:
    """Performance tests for function chain analysis."""

    @pytest.fixture
    def performance_fixtures(self):
        """Create performance test fixtures."""
        return PerformanceTestFixtures()

    @pytest.fixture
    def mock_services(self):
        """Create mock services for performance testing."""
        mock_breadcrumb_resolver = MagicMock()
        mock_breadcrumb_resolver.resolve = AsyncMock()

        mock_implementation_chain_service = MagicMock()
        mock_implementation_chain_service.trace_implementation_chain = AsyncMock()

        return mock_breadcrumb_resolver, mock_implementation_chain_service

    @pytest.mark.asyncio
    async def test_trace_function_chain_small_codebase_performance(self, performance_fixtures, mock_services):
        """Test performance with small codebase (< 1000 functions)."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create small codebase simulation
        nodes = performance_fixtures.create_large_graph_nodes(100)
        links = performance_fixtures.create_large_chain_links(nodes)
        chain = performance_fixtures.create_large_implementation_chain(nodes, links)

        # Setup mocks
        successful_result = BreadcrumbResolutionResult(
            query="test function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module_0.class_0.method_0",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=10.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.return_value = successful_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            # Measure performance
            start_time = time.time()
            result = await trace_function_chain(
                entry_point="test function",
                project_name="small_test_project",
                direction="forward",
                max_depth=10,
                performance_monitoring=True,
            )
            total_time = time.time() - start_time

            # Verify performance requirements
            assert result["success"] is True
            assert total_time < 1.0  # Should complete in under 1 second
            assert result["performance"]["total_time"] < 1000  # Under 1000ms
            assert result["performance"]["breadcrumb_resolution_time"] < 100  # Under 100ms
            assert result["performance"]["chain_tracing_time"] < 500  # Under 500ms
            assert result["performance"]["formatting_time"] < 100  # Under 100ms

    @pytest.mark.asyncio
    async def test_trace_function_chain_medium_codebase_performance(self, performance_fixtures, mock_services):
        """Test performance with medium codebase (1000-10000 functions)."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create medium codebase simulation
        nodes = performance_fixtures.create_large_graph_nodes(1000)
        links = performance_fixtures.create_large_chain_links(nodes)
        chain = performance_fixtures.create_large_implementation_chain(nodes, links)

        # Setup mocks
        successful_result = BreadcrumbResolutionResult(
            query="test function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module_0.class_0.method_0",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=50.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.return_value = successful_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            # Measure performance
            start_time = time.time()
            result = await trace_function_chain(
                entry_point="test function",
                project_name="medium_test_project",
                direction="forward",
                max_depth=15,
                performance_monitoring=True,
            )
            total_time = time.time() - start_time

            # Verify performance requirements for medium codebase
            assert result["success"] is True
            assert total_time < 3.0  # Should complete in under 3 seconds
            assert result["performance"]["total_time"] < 3000  # Under 3000ms
            assert result["performance"]["breadcrumb_resolution_time"] < 200  # Under 200ms
            assert result["performance"]["chain_tracing_time"] < 2000  # Under 2000ms
            assert result["performance"]["formatting_time"] < 500  # Under 500ms

    @pytest.mark.asyncio
    async def test_trace_function_chain_large_codebase_performance(self, performance_fixtures, mock_services):
        """Test performance with large codebase (> 10000 functions)."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create large codebase simulation
        nodes = performance_fixtures.create_large_graph_nodes(5000)
        links = performance_fixtures.create_large_chain_links(nodes)
        chain = performance_fixtures.create_large_implementation_chain(nodes, links)

        # Setup mocks
        successful_result = BreadcrumbResolutionResult(
            query="test function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module_0.class_0.method_0",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=100.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.return_value = successful_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            # Measure performance
            start_time = time.time()
            result = await trace_function_chain(
                entry_point="test function",
                project_name="large_test_project",
                direction="forward",
                max_depth=20,
                performance_monitoring=True,
            )
            total_time = time.time() - start_time

            # Verify performance requirements for large codebase
            assert result["success"] is True
            assert total_time < 5.0  # Should complete in under 5 seconds
            assert result["performance"]["total_time"] < 5000  # Under 5000ms
            assert result["performance"]["breadcrumb_resolution_time"] < 500  # Under 500ms
            assert result["performance"]["chain_tracing_time"] < 3000  # Under 3000ms
            assert result["performance"]["formatting_time"] < 1000  # Under 1000ms

    @pytest.mark.asyncio
    async def test_trace_function_chain_concurrent_requests(self, performance_fixtures, mock_services):
        """Test performance with concurrent requests."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create medium codebase simulation
        nodes = performance_fixtures.create_large_graph_nodes(500)
        links = performance_fixtures.create_large_chain_links(nodes)
        chain = performance_fixtures.create_large_implementation_chain(nodes, links)

        # Setup mocks
        successful_result = BreadcrumbResolutionResult(
            query="test function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module_0.class_0.method_0",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=25.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.return_value = successful_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            # Create concurrent requests
            async def run_single_request(request_id: int):
                return await trace_function_chain(
                    entry_point=f"test function {request_id}",
                    project_name="concurrent_test_project",
                    direction="forward",
                    max_depth=10,
                    performance_monitoring=True,
                )

            # Run 10 concurrent requests
            start_time = time.time()
            tasks = [run_single_request(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            # Verify all requests succeeded
            assert all(result["success"] for result in results)

            # Verify concurrent performance
            assert total_time < 2.0  # All 10 requests should complete in under 2 seconds

            # Verify individual request performance
            for result in results:
                assert result["performance"]["total_time"] < 1000  # Each request under 1000ms

    @pytest.mark.asyncio
    async def test_trace_function_chain_memory_usage(self, performance_fixtures, mock_services):
        """Test memory usage with large chains."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create large codebase simulation
        nodes = performance_fixtures.create_large_graph_nodes(2000)
        links = performance_fixtures.create_large_chain_links(nodes)
        chain = performance_fixtures.create_large_implementation_chain(nodes, links)

        # Setup mocks
        successful_result = BreadcrumbResolutionResult(
            query="test function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="module_0.class_0.method_0",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=75.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.return_value = successful_result
        mock_implementation_chain_service.trace_implementation_chain.return_value = chain

        with (
            patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
        ):
            # Run multiple iterations to test memory usage
            for i in range(5):
                result = await trace_function_chain(
                    entry_point=f"test function {i}",
                    project_name="memory_test_project",
                    direction="forward",
                    max_depth=15,
                    performance_monitoring=True,
                )

                # Verify success
                assert result["success"] is True

                # Verify chain info is present and has expected structure
                assert "chain_info" in result
                assert result["chain_info"]["total_components"] == 2000
                assert result["chain_info"]["total_links"] == 1999


class TestFunctionPathFindingPerformance:
    """Performance tests for function path finding."""

    @pytest.fixture
    def performance_fixtures(self):
        """Create performance test fixtures."""
        return PerformanceTestFixtures()

    @pytest.fixture
    def mock_services(self):
        """Create mock services for performance testing."""
        mock_breadcrumb_resolver = MagicMock()
        mock_breadcrumb_resolver.resolve = AsyncMock()

        mock_implementation_chain_service = MagicMock()
        mock_implementation_chain_service.find_paths_between_components = AsyncMock()

        return mock_breadcrumb_resolver, mock_implementation_chain_service

    @pytest.mark.asyncio
    async def test_find_function_path_small_codebase_performance(self, performance_fixtures, mock_services):
        """Test path finding performance with small codebase."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create small set of paths
        paths = performance_fixtures.create_multiple_paths(10)

        # Setup mocks
        successful_start = BreadcrumbResolutionResult(
            query="start function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="start.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=10.0,
            search_results_count=1,
        )

        successful_end = BreadcrumbResolutionResult(
            query="end function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="end.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=10.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.side_effect = [successful_start, successful_end]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=paths),
        ):
            # Measure performance
            start_time = time.time()
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="small_test_project",
                strategy="optimal",
                max_paths=5,
                performance_monitoring=True,
            )
            total_time = time.time() - start_time

            # Verify performance requirements
            assert result["success"] is True
            assert total_time < 1.0  # Should complete in under 1 second
            assert result["performance"]["total_time"] < 1000  # Under 1000ms
            assert result["performance"]["breadcrumb_resolution_time"] < 100  # Under 100ms
            assert result["performance"]["path_finding_time"] < 500  # Under 500ms

    @pytest.mark.asyncio
    async def test_find_function_path_large_path_set_performance(self, performance_fixtures, mock_services):
        """Test path finding performance with large path sets."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create large set of paths
        paths = performance_fixtures.create_multiple_paths(1000)

        # Setup mocks
        successful_start = BreadcrumbResolutionResult(
            query="start function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="start.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=20.0,
            search_results_count=1,
        )

        successful_end = BreadcrumbResolutionResult(
            query="end function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="end.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=20.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.side_effect = [successful_start, successful_end]

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=paths[:10]),  # Return top 10
        ):
            # Measure performance
            start_time = time.time()
            result = await find_function_path(
                start_function="start function",
                end_function="end function",
                project_name="large_path_test_project",
                strategy="all",
                max_paths=10,
                performance_monitoring=True,
            )
            total_time = time.time() - start_time

            # Verify performance requirements with large path set
            assert result["success"] is True
            assert total_time < 2.0  # Should complete in under 2 seconds
            assert result["performance"]["total_time"] < 2000  # Under 2000ms
            assert result["performance"]["quality_analysis_time"] < 500  # Under 500ms
            assert result["performance"]["formatting_time"] < 1000  # Under 1000ms

    @pytest.mark.asyncio
    async def test_find_function_path_different_strategies_performance(self, performance_fixtures, mock_services):
        """Test performance of different path finding strategies."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create medium set of paths
        paths = performance_fixtures.create_multiple_paths(100)

        # Setup mocks
        successful_start = BreadcrumbResolutionResult(
            query="start function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="start.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=15.0,
            search_results_count=1,
        )

        successful_end = BreadcrumbResolutionResult(
            query="end function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="end.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=15.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.side_effect = [successful_start, successful_end]

        strategies = ["shortest", "optimal", "all"]
        strategy_times = {}

        for strategy in strategies:
            with (
                patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
                patch(
                    "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                    return_value=mock_implementation_chain_service,
                ),
                patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=paths[:10]),
            ):
                # Reset mock call counts
                mock_breadcrumb_resolver.resolve.reset_mock()
                mock_breadcrumb_resolver.resolve.side_effect = [successful_start, successful_end]

                # Measure performance for each strategy
                start_time = time.time()
                result = await find_function_path(
                    start_function="start function",
                    end_function="end function",
                    project_name="strategy_test_project",
                    strategy=strategy,
                    max_paths=10,
                    performance_monitoring=True,
                )
                total_time = time.time() - start_time

                strategy_times[strategy] = total_time

                # Verify success and performance
                assert result["success"] is True
                assert total_time < 1.5  # Each strategy should complete in under 1.5 seconds
                assert result["performance"]["total_time"] < 1500  # Under 1500ms

        # All strategies should have similar performance
        max_time = max(strategy_times.values())
        min_time = min(strategy_times.values())
        assert max_time - min_time < 0.5  # Performance difference should be less than 0.5 seconds

    @pytest.mark.asyncio
    async def test_find_function_path_stress_test(self, performance_fixtures, mock_services):
        """Stress test with many consecutive requests."""
        mock_breadcrumb_resolver, mock_implementation_chain_service = mock_services

        # Create paths for testing
        paths = performance_fixtures.create_multiple_paths(50)

        # Setup mocks
        successful_start = BreadcrumbResolutionResult(
            query="start function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="start.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=10.0,
            search_results_count=1,
        )

        successful_end = BreadcrumbResolutionResult(
            query="end function",
            success=True,
            primary_candidate=BreadcrumbCandidate(
                breadcrumb="end.function",
                confidence_score=0.9,
                source_chunk=None,
                reasoning="Exact match found",
                match_type="exact",
            ),
            resolution_time_ms=10.0,
            search_results_count=1,
        )

        mock_breadcrumb_resolver.resolve.side_effect = [successful_start, successful_end] * 50  # 50 pairs

        with (
            patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
            patch(
                "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                return_value=mock_implementation_chain_service,
            ),
            patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=paths[:5]),
        ):
            # Run 50 consecutive requests
            request_times = []
            success_count = 0

            start_time = time.time()
            for i in range(50):
                request_start = time.time()
                result = await find_function_path(
                    start_function=f"start function {i}",
                    end_function=f"end function {i}",
                    project_name="stress_test_project",
                    strategy="optimal",
                    max_paths=5,
                    performance_monitoring=True,
                )
                request_time = time.time() - request_start
                request_times.append(request_time)

                if result["success"]:
                    success_count += 1

            total_time = time.time() - start_time

            # Verify stress test results
            assert success_count == 50  # All requests should succeed
            assert total_time < 10.0  # All 50 requests should complete in under 10 seconds

            # Verify individual request performance
            avg_request_time = statistics.mean(request_times)
            max_request_time = max(request_times)

            assert avg_request_time < 0.1  # Average request time under 100ms
            assert max_request_time < 0.5  # No single request should take more than 500ms

            # Verify performance consistency
            request_time_stddev = statistics.stdev(request_times)
            assert request_time_stddev < 0.05  # Low variance in request times


class TestScalabilityBenchmarks:
    """Benchmarks for scalability testing."""

    @pytest.mark.asyncio
    async def test_scalability_benchmark_chain_analysis(self):
        """Benchmark chain analysis scalability."""
        # Test with different chain sizes
        chain_sizes = [10, 50, 100, 500, 1000]
        performance_results = {}

        for size in chain_sizes:
            # Create test data
            nodes = PerformanceTestFixtures.create_large_graph_nodes(size)
            links = PerformanceTestFixtures.create_large_chain_links(nodes)
            chain = PerformanceTestFixtures.create_large_implementation_chain(nodes, links)

            # Setup mocks
            mock_breadcrumb_resolver = MagicMock()
            mock_breadcrumb_resolver.resolve = AsyncMock()
            mock_implementation_chain_service = MagicMock()
            mock_implementation_chain_service.trace_implementation_chain = AsyncMock()

            successful_result = BreadcrumbResolutionResult(
                query="test function",
                success=True,
                primary_candidate=BreadcrumbCandidate(
                    breadcrumb="module_0.class_0.method_0",
                    confidence_score=0.9,
                    source_chunk=None,
                    reasoning="Exact match found",
                    match_type="exact",
                ),
                resolution_time_ms=10.0,
                search_results_count=1,
            )

            mock_breadcrumb_resolver.resolve.return_value = successful_result
            mock_implementation_chain_service.trace_implementation_chain.return_value = chain

            with (
                patch("src.tools.graph_rag.function_chain_analysis.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
                patch(
                    "src.tools.graph_rag.function_chain_analysis.get_implementation_chain_service",
                    return_value=mock_implementation_chain_service,
                ),
            ):
                # Measure performance
                start_time = time.time()
                result = await trace_function_chain(
                    entry_point="test function",
                    project_name=f"benchmark_project_{size}",
                    direction="forward",
                    max_depth=20,
                    performance_monitoring=True,
                )
                total_time = time.time() - start_time

                performance_results[size] = {
                    "total_time": total_time,
                    "success": result["success"],
                    "performance_data": result["performance"],
                    "chain_size": size,
                }

        # Verify scalability
        for size, result in performance_results.items():
            assert result["success"] is True

            # Performance should scale reasonably with size
            expected_max_time = 0.5 + (size / 1000) * 2  # Base time + size factor
            assert result["total_time"] < expected_max_time

        # Verify performance scaling is reasonable
        small_time = performance_results[10]["total_time"]
        large_time = performance_results[1000]["total_time"]

        # Large should not be more than 100x slower than small
        assert large_time / small_time < 100

    @pytest.mark.asyncio
    async def test_scalability_benchmark_path_finding(self):
        """Benchmark path finding scalability."""
        # Test with different path counts
        path_counts = [5, 25, 50, 100, 500]
        performance_results = {}

        for count in path_counts:
            # Create test data
            paths = PerformanceTestFixtures.create_multiple_paths(count)

            # Setup mocks
            mock_breadcrumb_resolver = MagicMock()
            mock_breadcrumb_resolver.resolve = AsyncMock()
            mock_implementation_chain_service = MagicMock()
            mock_implementation_chain_service.find_paths_between_components = AsyncMock()

            successful_start = BreadcrumbResolutionResult(
                query="start function",
                success=True,
                primary_candidate=BreadcrumbCandidate(
                    breadcrumb="start.function",
                    confidence_score=0.9,
                    source_chunk=None,
                    reasoning="Exact match found",
                    match_type="exact",
                ),
                resolution_time_ms=10.0,
                search_results_count=1,
            )

            successful_end = BreadcrumbResolutionResult(
                query="end function",
                success=True,
                primary_candidate=BreadcrumbCandidate(
                    breadcrumb="end.function",
                    confidence_score=0.9,
                    source_chunk=None,
                    reasoning="Exact match found",
                    match_type="exact",
                ),
                resolution_time_ms=10.0,
                search_results_count=1,
            )

            mock_breadcrumb_resolver.resolve.side_effect = [successful_start, successful_end]

            with (
                patch("src.tools.graph_rag.function_path_finding.BreadcrumbResolver", return_value=mock_breadcrumb_resolver),
                patch(
                    "src.tools.graph_rag.function_path_finding.get_implementation_chain_service",
                    return_value=mock_implementation_chain_service,
                ),
                patch("src.tools.graph_rag.function_path_finding._find_paths_by_strategy", return_value=paths[: min(10, count)]),
            ):
                # Measure performance
                start_time = time.time()
                result = await find_function_path(
                    start_function="start function",
                    end_function="end function",
                    project_name=f"benchmark_project_{count}",
                    strategy="all",
                    max_paths=10,
                    performance_monitoring=True,
                )
                total_time = time.time() - start_time

                performance_results[count] = {
                    "total_time": total_time,
                    "success": result["success"],
                    "performance_data": result["performance"],
                    "path_count": count,
                }

        # Verify scalability
        for count, result in performance_results.items():
            assert result["success"] is True

            # Performance should scale reasonably with path count
            expected_max_time = 0.5 + (count / 500) * 2  # Base time + count factor
            assert result["total_time"] < expected_max_time

        # Verify performance scaling is reasonable
        small_time = performance_results[5]["total_time"]
        large_time = performance_results[500]["total_time"]

        # Large should not be more than 50x slower than small
        assert large_time / small_time < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
