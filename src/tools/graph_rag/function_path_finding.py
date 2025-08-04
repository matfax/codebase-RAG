"""Function Path Finding Tool

This module provides MCP tools for finding paths between functions in codebases
using Graph RAG capabilities. It enables users to discover how functions are
connected and identify the most efficient ways to navigate between them.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.services.implementation_chain_service import (
    ChainDirection,
    ChainType,
    ImplementationChainService,
    get_implementation_chain_service,
)
from src.utils.output_formatters import (
    MermaidStyle,
    OutputFormat,
    format_arrow_path,
    format_comprehensive_output,
    format_mermaid_path,
)

logger = logging.getLogger(__name__)


class PathStrategy(Enum):
    """Path finding strategies."""

    SHORTEST = "shortest"  # Find the shortest path
    OPTIMAL = "optimal"  # Find the most reliable/efficient path
    ALL = "all"  # Find all possible paths


@dataclass
class PathQuality:
    """Quality metrics for a function path."""

    reliability_score: float  # How reliable the path is (0.0-1.0)
    complexity_score: float  # How complex the path is (0.0-1.0, lower is better)
    directness_score: float  # How direct the path is (0.0-1.0, higher is better)
    overall_score: float  # Overall quality score (0.0-1.0)

    # Additional metrics
    path_length: int  # Number of steps in the path
    confidence: float  # Confidence in the path existence
    relationship_diversity: float  # Diversity of relationship types (0.0-1.0)


@dataclass
class FunctionPath:
    """Represents a path between two functions."""

    start_breadcrumb: str
    end_breadcrumb: str
    path_steps: list[str]  # List of breadcrumbs in the path
    quality: PathQuality

    # Path metadata
    path_id: str
    path_type: str  # Type of path (execution, data_flow, etc.)
    relationships: list[str]  # Types of relationships in the path
    evidence: list[str]  # Evidence for each step

    # Formatting
    arrow_format: str = ""
    mermaid_format: str = ""


@dataclass
class PathRecommendation:
    """Recommendation for the best path."""

    recommended_path: FunctionPath
    reason: str  # Why this path is recommended
    alternatives: list[FunctionPath]  # Alternative paths
    suggestions: list[str]  # Suggestions for improvement


async def find_function_path(
    start_function: str,
    end_function: str,
    project_name: str,
    strategy: str = "optimal",
    max_paths: int = 10,  # Increased default for better results
    max_depth: int = 25,  # Increased default for deeper analysis
    min_quality_threshold: float = 0.3,
    include_path_diversity: bool = True,
    output_format: str = "arrow",
    include_mermaid: bool = True,
    performance_monitoring: bool = True,
) -> dict[str, Any]:
    """
    Find paths between two functions in a codebase.

    This tool discovers how functions are connected and identifies the most
    efficient ways to navigate between them. It supports multiple path-finding
    strategies and provides detailed quality analysis.

    Args:
        start_function: Starting function (breadcrumb or natural language)
        end_function: Target function (breadcrumb or natural language)
        project_name: Name of the project to analyze
        strategy: Path finding strategy ("shortest", "optimal", "all")
        max_paths: Maximum number of paths to return (default: 5)
        max_depth: Maximum depth for path traversal (default: 15)
        min_quality_threshold: Minimum quality threshold for paths (0.0-1.0)
        include_path_diversity: Whether to include path diversity analysis
        output_format: Output format ("arrow", "mermaid", "both")
        include_mermaid: Whether to include Mermaid diagram output
        performance_monitoring: Whether to include performance monitoring

    Returns:
        Dictionary containing path analysis results with formatted output
    """
    start_time = time.time()

    try:
        logger.info(f"Starting path finding from '{start_function}' to '{end_function}' in project '{project_name}'")

        # Initialize results structure
        results = {
            "success": True,
            "start_function": start_function,
            "end_function": end_function,
            "project_name": project_name,
            "strategy": strategy,
            "max_paths": max_paths,
            "max_depth": max_depth,
            "timestamp": time.time(),
        }

        # Initialize performance monitoring
        if performance_monitoring:
            results["performance"] = {
                "start_time": start_time,
                "breadcrumb_resolution_time": 0.0,
                "path_finding_time": 0.0,
                "quality_analysis_time": 0.0,
                "formatting_time": 0.0,
                "total_time": 0.0,
            }

        # Validate input parameters
        validation_result = _validate_path_finding_parameters(
            start_function, end_function, project_name, strategy, max_paths, max_depth, min_quality_threshold
        )

        if not validation_result["valid"]:
            results["success"] = False
            results["error"] = validation_result["error"]
            results["suggestions"] = validation_result.get("suggestions", [])
            return results

        # Convert strategy to enum
        try:
            path_strategy = PathStrategy(strategy.lower())
        except ValueError:
            results["success"] = False
            results["error"] = f"Invalid strategy: {strategy}"
            results["suggestions"] = ["Valid strategies: shortest, optimal, all"]
            return results

        # Initialize services
        from src.services.embedding_service import EmbeddingService
        from src.services.graph_rag_service import get_graph_rag_service
        from src.services.hybrid_search_service import get_hybrid_search_service
        from src.services.implementation_chain_service import get_implementation_chain_service
        from src.services.qdrant_service import QdrantService

        breadcrumb_resolver = BreadcrumbResolver()
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()

        # Initialize Graph RAG and Hybrid Search services with required dependencies
        graph_rag_service = get_graph_rag_service(qdrant_service, embedding_service)
        hybrid_search_service = get_hybrid_search_service()

        implementation_chain_service = get_implementation_chain_service(
            graph_rag_service=graph_rag_service, hybrid_search_service=hybrid_search_service
        )

        # Step 1: Resolve breadcrumbs for both functions
        breadcrumb_start_time = time.time()

        # Resolve start function breadcrumb
        start_result = await breadcrumb_resolver.resolve(query=start_function, target_projects=[project_name])

        if not start_result.success:
            results["success"] = False
            results["error"] = f"Failed to resolve start function: {start_result.error_message}"

            # Enhanced error handling with intelligent suggestions
            from src.tools.graph_rag.function_chain_analysis import _generate_enhanced_suggestions

            enhanced_suggestions = await _generate_enhanced_suggestions(start_function, project_name, start_result, "start_function")
            results["suggestions"] = enhanced_suggestions["suggestions"]
            results["error_details"] = enhanced_suggestions["error_details"]
            results["alternatives"] = enhanced_suggestions["alternatives"]
            return results

        # Resolve end function breadcrumb
        end_result = await breadcrumb_resolver.resolve(query=end_function, target_projects=[project_name])

        if not end_result.success:
            results["success"] = False
            results["error"] = f"Failed to resolve end function: {end_result.error_message}"

            # Enhanced error handling with intelligent suggestions
            from src.tools.graph_rag.function_chain_analysis import _generate_enhanced_suggestions

            enhanced_suggestions = await _generate_enhanced_suggestions(end_function, project_name, end_result, "end_function")
            results["suggestions"] = enhanced_suggestions["suggestions"]
            results["error_details"] = enhanced_suggestions["error_details"]
            results["alternatives"] = enhanced_suggestions["alternatives"]
            return results

        # Check if primary_candidate exists and has breadcrumb attribute
        if not start_result.primary_candidate:
            results["success"] = False
            results["error"] = f"No valid breadcrumb found for start function: {start_function}"
            return results

        if not end_result.primary_candidate:
            results["success"] = False
            results["error"] = f"No valid breadcrumb found for end function: {end_function}"
            return results

        # Additional safety checks for breadcrumb access
        if hasattr(start_result.primary_candidate, "breadcrumb"):
            start_breadcrumb = start_result.primary_candidate.breadcrumb
        else:
            start_breadcrumb = str(start_result.primary_candidate)

        if hasattr(end_result.primary_candidate, "breadcrumb"):
            end_breadcrumb = end_result.primary_candidate.breadcrumb
        else:
            end_breadcrumb = str(end_result.primary_candidate)

        results["resolved_start_breadcrumb"] = start_breadcrumb
        results["resolved_end_breadcrumb"] = end_breadcrumb
        results["start_breadcrumb_confidence"] = getattr(start_result.primary_candidate, "confidence_score", 0.0)
        results["end_breadcrumb_confidence"] = getattr(end_result.primary_candidate, "confidence_score", 0.0)

        if performance_monitoring:
            results["performance"]["breadcrumb_resolution_time"] = (time.time() - breadcrumb_start_time) * 1000

        # Step 2: Find paths between the functions
        path_finding_start_time = time.time()

        # Check if start and end are the same
        if start_breadcrumb == end_breadcrumb:
            results["success"] = False
            results["error"] = "Start and end functions are the same"
            results["suggestions"] = [
                "Specify different start and end functions",
                "Use the trace_function_chain tool to analyze a single function",
            ]
            return results

        # Find multiple paths using bidirectional search
        try:
            paths = await _find_multiple_paths(
                start_breadcrumb=start_breadcrumb,
                end_breadcrumb=end_breadcrumb,
                project_name=project_name,
                strategy=path_strategy,
                max_paths=max_paths,
                max_depth=max_depth,
                implementation_chain_service=implementation_chain_service,
            )
        except Exception as e:
            import traceback

            logger.error(f"Detailed error in _find_multiple_paths: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            results["success"] = False
            results["error"] = f"Error finding paths from '{start_function}' to '{end_function}': {str(e)}"
            results["detailed_error"] = traceback.format_exc()
            return results

        if performance_monitoring:
            results["performance"]["path_finding_time"] = (time.time() - path_finding_start_time) * 1000

        # Step 3: Analyze path quality and filter results
        quality_start_time = time.time()

        # Filter paths by quality threshold
        quality_paths = []
        for path in paths:
            if path.quality.overall_score >= min_quality_threshold:
                quality_paths.append(path)

        if not quality_paths:
            results["success"] = False
            results["error"] = "No paths found meeting the quality threshold"

            # Enhanced error handling for no paths found
            from src.tools.graph_rag.function_chain_analysis import _generate_enhanced_suggestions

            enhanced_suggestions = await _generate_enhanced_suggestions(
                f"{start_function} -> {end_function}", project_name, None, "no_paths_found"
            )

            # Add specific suggestions for no paths found
            specific_suggestions = [
                "No paths found between the specified functions. Try these approaches:",
                f"• Lower the quality threshold (current: {min_quality_threshold})",
                f"• Increase the maximum depth (current: {max_depth})",
                "• Check if the functions are in the same codebase",
                "• Use the search tool to find intermediate functions",
                "• Try different path finding strategies",
            ]

            results["suggestions"] = specific_suggestions + enhanced_suggestions["suggestions"]
            results["error_details"] = enhanced_suggestions["error_details"]
            results["alternatives"] = enhanced_suggestions["alternatives"]
            results["error_details"]["paths_analyzed"] = len(paths)
            results["error_details"]["quality_threshold"] = min_quality_threshold
            results["error_details"]["max_depth"] = max_depth
            return results

        # Sort paths by quality score (descending)
        quality_paths.sort(key=lambda p: p.quality.overall_score, reverse=True)

        # Limit results to max_paths
        final_paths = quality_paths[:max_paths]

        if performance_monitoring:
            results["performance"]["quality_analysis_time"] = (time.time() - quality_start_time) * 1000

        # Step 4: Format output using comprehensive formatting
        format_start_time = time.time()

        # Convert FunctionPath objects to dictionaries for comprehensive formatting
        path_dicts = []
        for path in final_paths:
            path_dict = {
                "path_id": path.path_id,
                "start_breadcrumb": path.start_breadcrumb,
                "end_breadcrumb": path.end_breadcrumb,
                "path_steps": path.path_steps,
                "quality": {
                    "reliability_score": path.quality.reliability_score,
                    "complexity_score": path.quality.complexity_score,
                    "directness_score": path.quality.directness_score,
                    "overall_score": path.quality.overall_score,
                    "path_length": path.quality.path_length,
                    "confidence": path.quality.confidence,
                    "relationship_diversity": path.quality.relationship_diversity,
                },
                "path_type": path.path_type,
                "relationships": path.relationships,
                "evidence": path.evidence,
            }
            path_dicts.append(path_dict)

        # Determine output format enum
        output_format_enum = OutputFormat.BOTH
        if output_format == "arrow":
            output_format_enum = OutputFormat.ARROW
        elif output_format == "mermaid":
            output_format_enum = OutputFormat.MERMAID

        # Use comprehensive formatting with enhanced features
        comprehensive_output = format_comprehensive_output(
            path_dicts,
            output_format=output_format_enum,
            include_comparison=True,
            include_recommendations=True,
            mermaid_style=MermaidStyle.FLOWCHART,
            custom_styling=None,
        )

        # Extract formatted results
        results["paths"] = comprehensive_output["paths"]
        results["total_paths_found"] = len(paths)
        results["paths_meeting_threshold"] = len(quality_paths)
        results["paths_returned"] = len(final_paths)
        results["summary"] = comprehensive_output["summary"]

        # Add path diversity analysis if requested
        if include_path_diversity and final_paths:
            results["path_diversity"] = _analyze_path_diversity(final_paths)

        # Add comprehensive recommendations and comparison
        if "recommendations" in comprehensive_output:
            results["recommendation"] = comprehensive_output["recommendations"]
        if "comparison" in comprehensive_output:
            results["path_comparison"] = comprehensive_output["comparison"]

        if performance_monitoring:
            results["performance"]["formatting_time"] = (time.time() - format_start_time) * 1000
            results["performance"]["total_time"] = (time.time() - start_time) * 1000

        logger.info(f"Path finding completed successfully - found {len(final_paths)} quality paths")

        return results

    except Exception as e:
        error_msg = f"Error finding paths from '{start_function}' to '{end_function}': {str(e)}"
        logger.error(error_msg, exc_info=True)

        results = {
            "success": False,
            "error": error_msg,
            "start_function": start_function,
            "end_function": end_function,
            "project_name": project_name,
            "strategy": strategy,
            "suggestions": [
                "Check if the project has been indexed properly",
                "Try using the search tool to find the exact function names",
                "Verify the project name is correct",
                "Consider using simpler function descriptions",
            ],
            "timestamp": time.time(),
        }

        if performance_monitoring:
            results["performance"] = {
                "total_time": (time.time() - start_time) * 1000,
                "error_occurred": True,
            }

        return results


def _validate_path_finding_parameters(
    start_function: str, end_function: str, project_name: str, strategy: str, max_paths: int, max_depth: int, min_quality_threshold: float
) -> dict[str, Any]:
    """
    Validate input parameters for path finding.

    Args:
        start_function: Starting function identifier
        end_function: Target function identifier
        project_name: Name of the project
        strategy: Path finding strategy
        max_paths: Maximum number of paths to return
        max_depth: Maximum depth for traversal
        min_quality_threshold: Minimum quality threshold

    Returns:
        Dictionary with validation results
    """
    if not start_function or not start_function.strip():
        return {
            "valid": False,
            "error": "Start function is required and cannot be empty",
            "suggestions": ["Provide a function name or natural language description"],
        }

    if not end_function or not end_function.strip():
        return {
            "valid": False,
            "error": "End function is required and cannot be empty",
            "suggestions": ["Provide a function name or natural language description"],
        }

    if not project_name or not project_name.strip():
        return {
            "valid": False,
            "error": "Project name is required and cannot be empty",
            "suggestions": ["Provide a valid project name"],
        }

    if strategy not in ["shortest", "optimal", "all"]:
        return {
            "valid": False,
            "error": f"Invalid strategy: {strategy}",
            "suggestions": ["Valid strategies: shortest, optimal, all"],
        }

    if not isinstance(max_paths, int) or max_paths < 1 or max_paths > 20:
        return {
            "valid": False,
            "error": f"Invalid max_paths: {max_paths}. Must be between 1 and 20",
            "suggestions": ["Use a reasonable number of paths between 1 and 20"],
        }

    if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 50:
        return {
            "valid": False,
            "error": f"Invalid max_depth: {max_depth}. Must be between 1 and 50",
            "suggestions": ["Use a reasonable depth value between 1 and 50"],
        }

    if not isinstance(min_quality_threshold, Union[int, float]) or min_quality_threshold < 0.0 or min_quality_threshold > 1.0:
        return {
            "valid": False,
            "error": f"Invalid min_quality_threshold: {min_quality_threshold}. Must be between 0.0 and 1.0",
            "suggestions": ["Use a quality threshold between 0.0 and 1.0"],
        }

    return {"valid": True}


async def _find_multiple_paths(
    start_breadcrumb: str,
    end_breadcrumb: str,
    project_name: str,
    strategy: PathStrategy,
    max_paths: int,
    max_depth: int,
    implementation_chain_service: ImplementationChainService,
) -> list[FunctionPath]:
    """
    Find multiple paths between two functions using different strategies.

    This implementation uses bidirectional search and multiple chain types
    to discover various paths between the start and end functions.

    Args:
        start_breadcrumb: Starting function breadcrumb
        end_breadcrumb: Target function breadcrumb
        project_name: Name of the project
        strategy: Path finding strategy
        max_paths: Maximum number of paths to return
        max_depth: Maximum depth for traversal
        implementation_chain_service: Service for chain analysis

    Returns:
        List of FunctionPath objects
    """
    logger.info(f"Finding paths from {start_breadcrumb} to {end_breadcrumb} using strategy: {strategy}")

    paths = []

    # Get the project structure graph
    project_graph = await implementation_chain_service.graph_rag_service.build_structure_graph(project_name, force_rebuild=True)

    if not project_graph:
        logger.warning(f"No structure graph found for project: {project_name}")
        return paths

    # Find start and end nodes in the graph
    start_node = implementation_chain_service._find_component_by_breadcrumb(project_graph, start_breadcrumb)
    end_node = implementation_chain_service._find_component_by_breadcrumb(project_graph, end_breadcrumb)

    if not start_node or not end_node:
        logger.warning(f"Start or end node not found: {start_breadcrumb} -> {end_breadcrumb}")
        return paths

    # Use different chain types based on strategy
    chain_types = _get_chain_types_for_strategy(strategy)

    # Try different approaches to find paths
    for chain_type in chain_types:
        if len(paths) >= max_paths:
            break

        # Method 1: Bidirectional search
        bidirectional_paths = await _find_bidirectional_paths(
            start_node, end_node, project_graph, chain_type, max_depth, implementation_chain_service
        )

        # Method 2: Forward chain analysis
        forward_paths = await _find_forward_paths(start_node, end_node, project_graph, chain_type, max_depth, implementation_chain_service)

        # Method 3: Backward chain analysis
        backward_paths = await _find_backward_paths(
            start_node, end_node, project_graph, chain_type, max_depth, implementation_chain_service
        )

        # Combine all found paths
        all_found_paths = bidirectional_paths + forward_paths + backward_paths

        # Convert to FunctionPath objects
        for path_info in all_found_paths:
            if len(paths) >= max_paths:
                break

            function_path = _create_function_path(path_info, start_breadcrumb, end_breadcrumb, chain_type, strategy)

            if function_path and _is_valid_path(function_path):
                paths.append(function_path)

    # Apply strategy-specific filtering and sorting
    paths = _apply_strategy_filtering(paths, strategy, max_paths)

    logger.info(f"Found {len(paths)} paths between {start_breadcrumb} and {end_breadcrumb}")
    return paths


def _get_chain_types_for_strategy(strategy: PathStrategy) -> list[ChainType]:
    """
    Get chain types to use based on the path finding strategy.

    Args:
        strategy: Path finding strategy

    Returns:
        List of chain types to try
    """
    if strategy == PathStrategy.SHORTEST:
        # For shortest paths, prioritize direct execution flows
        return [
            ChainType.EXECUTION_FLOW,
            ChainType.DEPENDENCY_CHAIN,
            ChainType.SERVICE_LAYER_CHAIN,
        ]
    elif strategy == PathStrategy.OPTIMAL:
        # For optimal paths, include all reliable chain types
        return [
            ChainType.EXECUTION_FLOW,
            ChainType.DEPENDENCY_CHAIN,
            ChainType.SERVICE_LAYER_CHAIN,
            ChainType.INHERITANCE_CHAIN,
            ChainType.INTERFACE_IMPLEMENTATION,
        ]
    elif strategy == PathStrategy.ALL:
        # For all paths, try every chain type
        return [
            ChainType.EXECUTION_FLOW,
            ChainType.DEPENDENCY_CHAIN,
            ChainType.SERVICE_LAYER_CHAIN,
            ChainType.INHERITANCE_CHAIN,
            ChainType.INTERFACE_IMPLEMENTATION,
            ChainType.DATA_FLOW,
            ChainType.API_ENDPOINT_CHAIN,
        ]
    else:
        return [ChainType.EXECUTION_FLOW]


async def _find_bidirectional_paths(start_node, end_node, project_graph, chain_type, max_depth, implementation_chain_service) -> list[dict]:
    """
    Find paths using bidirectional search.

    Args:
        start_node: Starting graph node
        end_node: Target graph node
        project_graph: Project structure graph
        chain_type: Type of chain to analyze
        max_depth: Maximum depth for traversal
        implementation_chain_service: Service for chain analysis

    Returns:
        List of path information dictionaries
    """
    paths = []

    try:
        # Trace forward from start
        start_breadcrumb = getattr(start_node, "breadcrumb", str(start_node))
        forward_chain = await implementation_chain_service.trace_implementation_chain(
            start_breadcrumb,
            project_graph.project_name,
            chain_type,
            ChainDirection.FORWARD,
            max_depth // 2,
            0.2,  # Lower threshold for more coverage
        )

        # Trace backward from end
        end_breadcrumb = getattr(end_node, "breadcrumb", str(end_node))
        backward_chain = await implementation_chain_service.trace_implementation_chain(
            end_breadcrumb,
            project_graph.project_name,
            chain_type,
            ChainDirection.BACKWARD,
            max_depth // 2,
            0.2,  # Lower threshold for more coverage
        )

        # Find intersections between forward and backward chains
        intersections = _find_chain_intersections(forward_chain, backward_chain)

        # Create paths through intersections
        for intersection in intersections:
            path_info = _build_path_through_intersection(start_node, end_node, intersection, forward_chain, backward_chain)
            if path_info:
                paths.append(path_info)

    except Exception as e:
        logger.error(f"Error in bidirectional path finding: {e}")

    return paths


async def _find_forward_paths(start_node, end_node, project_graph, chain_type, max_depth, implementation_chain_service) -> list[dict]:
    """
    Find paths using forward chain analysis.

    Args:
        start_node: Starting graph node
        end_node: Target graph node
        project_graph: Project structure graph
        chain_type: Type of chain to analyze
        max_depth: Maximum depth for traversal
        implementation_chain_service: Service for chain analysis

    Returns:
        List of path information dictionaries
    """
    paths = []

    try:
        # Trace forward from start
        start_breadcrumb = getattr(start_node, "breadcrumb", str(start_node))
        forward_chain = await implementation_chain_service.trace_implementation_chain(
            start_breadcrumb,
            project_graph.project_name,
            chain_type,
            ChainDirection.FORWARD,
            max_depth,
            0.2,  # Lower threshold for more coverage
        )

        # Check if end node is reachable in forward chain
        if _is_node_in_chain(end_node, forward_chain):
            path_info = _extract_path_from_chain(start_node, end_node, forward_chain)
            if path_info:
                paths.append(path_info)

    except Exception as e:
        logger.error(f"Error in forward path finding: {e}")

    return paths


async def _find_backward_paths(start_node, end_node, project_graph, chain_type, max_depth, implementation_chain_service) -> list[dict]:
    """
    Find paths using backward chain analysis.

    Args:
        start_node: Starting graph node
        end_node: Target graph node
        project_graph: Project structure graph
        chain_type: Type of chain to analyze
        max_depth: Maximum depth for traversal
        implementation_chain_service: Service for chain analysis

    Returns:
        List of path information dictionaries
    """
    paths = []

    try:
        # Trace backward from end
        end_breadcrumb = getattr(end_node, "breadcrumb", str(end_node))
        backward_chain = await implementation_chain_service.trace_implementation_chain(
            end_breadcrumb,
            project_graph.project_name,
            chain_type,
            ChainDirection.BACKWARD,
            max_depth,
            0.2,  # Lower threshold for more coverage
        )

        # Check if start node is reachable in backward chain
        if _is_node_in_chain(start_node, backward_chain):
            path_info = _extract_path_from_chain(start_node, end_node, backward_chain)
            if path_info:
                paths.append(path_info)

    except Exception as e:
        logger.error(f"Error in backward path finding: {e}")

    return paths


def _find_chain_intersections(forward_chain, backward_chain) -> list[dict]:
    """
    Find intersection points between forward and backward chains.

    Args:
        forward_chain: Forward implementation chain
        backward_chain: Backward implementation chain

    Returns:
        List of intersection information
    """
    intersections = []

    # Get all nodes from both chains
    forward_nodes = set()
    backward_nodes = set()

    for link in forward_chain.links:
        forward_nodes.add(link.source_component.chunk_id)
        forward_nodes.add(link.target_component.chunk_id)

    for link in backward_chain.links:
        backward_nodes.add(link.source_component.chunk_id)
        backward_nodes.add(link.target_component.chunk_id)

    # Find common nodes
    common_nodes = forward_nodes & backward_nodes

    # Create intersection info for each common node
    for node_id in common_nodes:
        # Find the actual node object
        node = None
        for link in forward_chain.links:
            if link.source_component.chunk_id == node_id:
                node = link.source_component
                break
            elif link.target_component.chunk_id == node_id:
                node = link.target_component
                break

        if node:
            intersections.append(
                {
                    "node": node,
                    "forward_distance": _calculate_distance_in_chain(forward_chain, node),
                    "backward_distance": _calculate_distance_in_chain(backward_chain, node),
                }
            )

    return intersections


def _build_path_through_intersection(start_node, end_node, intersection, forward_chain, backward_chain) -> dict:
    """
    Build a path through an intersection point.

    Args:
        start_node: Starting graph node
        end_node: Target graph node
        intersection: Intersection point information
        forward_chain: Forward implementation chain
        backward_chain: Backward implementation chain

    Returns:
        Path information dictionary
    """
    try:
        # Get path from start to intersection
        forward_path = _get_path_to_node(forward_chain, start_node, intersection["node"])

        # Get path from intersection to end
        backward_path = _get_path_to_node(backward_chain, intersection["node"], end_node)

        if forward_path and backward_path:
            # Combine paths
            combined_path = forward_path + backward_path[1:]  # Remove duplicate intersection node

            return {
                "path_nodes": combined_path,
                "path_length": len(combined_path),
                "intersection_node": intersection["node"],
                "forward_distance": intersection["forward_distance"],
                "backward_distance": intersection["backward_distance"],
                "chain_type": "bidirectional",
            }
    except Exception as e:
        logger.error(f"Error building path through intersection: {e}")

    return None


def _is_node_in_chain(node, chain) -> bool:
    """
    Check if a node is present in a chain.

    Args:
        node: Graph node to check
        chain: Implementation chain

    Returns:
        True if node is in chain, False otherwise
    """
    node_id = getattr(node, "chunk_id", str(node))
    for link in chain.links:
        source_id = getattr(link.source_component, "chunk_id", str(link.source_component))
        target_id = getattr(link.target_component, "chunk_id", str(link.target_component))
        if source_id == node_id or target_id == node_id:
            return True
    return False


def _extract_path_from_chain(start_node, end_node, chain) -> dict:
    """
    Extract a path from start to end node within a chain.

    Args:
        start_node: Starting graph node
        end_node: Target graph node
        chain: Implementation chain

    Returns:
        Path information dictionary or None if no path found
    """
    try:
        # Build a graph from the chain links
        graph = {}
        for link in chain.links:
            source_id = getattr(link.source_component, "chunk_id", str(link.source_component))

            if source_id not in graph:
                graph[source_id] = []
            graph[source_id].append(link.target_component)

        # Find path using BFS
        from collections import deque

        start_node_id = getattr(start_node, "chunk_id", str(start_node))
        end_node_id = getattr(end_node, "chunk_id", str(end_node))

        queue = deque([(start_node, [start_node])])
        visited = {start_node_id}

        while queue:
            current_node, path = queue.popleft()
            current_node_id = getattr(current_node, "chunk_id", str(current_node))

            if current_node_id == end_node_id:
                return {
                    "path_nodes": path,
                    "path_length": len(path),
                    "chain_type": "direct",
                }

            if current_node_id in graph:
                for neighbor in graph[current_node_id]:
                    neighbor_id = getattr(neighbor, "chunk_id", str(neighbor))
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor, path + [neighbor]))

        return None

    except Exception as e:
        logger.error(f"Error extracting path from chain: {e}")
        return None


def _calculate_distance_in_chain(chain, node) -> int:
    """
    Calculate distance of a node from the entry point in a chain.

    Args:
        chain: Implementation chain
        node: Graph node

    Returns:
        Distance as integer
    """
    # Simple implementation - can be enhanced
    for i, link in enumerate(chain.links):
        if link.source_component.chunk_id == node.chunk_id or link.target_component.chunk_id == node.chunk_id:
            return i
    return 0


def _get_path_to_node(chain, start_node, target_node) -> list:
    """
    Get the path from start to target node within a chain.

    Args:
        chain: Implementation chain
        start_node: Starting graph node
        target_node: Target graph node

    Returns:
        List of nodes in the path
    """
    # Simple implementation - can be enhanced
    path = [start_node]
    current_node = start_node

    # Get node IDs with safety checks
    current_node_id = getattr(current_node, "chunk_id", str(current_node))
    target_node_id = getattr(target_node, "chunk_id", str(target_node))

    for link in chain.links:
        source_id = getattr(link.source_component, "chunk_id", str(link.source_component))
        if source_id == current_node_id:
            path.append(link.target_component)
            current_node = link.target_component
            current_node_id = getattr(current_node, "chunk_id", str(current_node))

            if current_node_id == target_node_id:
                break

    return path


def _create_function_path(path_info, start_breadcrumb, end_breadcrumb, chain_type, strategy) -> FunctionPath:
    """
    Create a FunctionPath object from path information.

    Args:
        path_info: Path information dictionary
        start_breadcrumb: Starting function breadcrumb
        end_breadcrumb: Target function breadcrumb
        chain_type: Type of chain
        strategy: Path finding strategy

    Returns:
        FunctionPath object
    """
    if not path_info or not path_info.get("path_nodes"):
        return None

    path_nodes = path_info["path_nodes"]
    # Add safety check for node types
    path_steps = []
    for node in path_nodes:
        if hasattr(node, "breadcrumb"):
            path_steps.append(node.breadcrumb)
        else:
            # Fallback for string nodes
            path_steps.append(str(node))

    # Calculate quality metrics
    quality = _calculate_path_quality(path_info, path_nodes, strategy)

    # Generate unique path ID
    path_id = f"{strategy.value}_{chain_type.value}_{len(path_steps)}_{hash(str(path_steps)) % 10000}"

    # Determine relationships
    relationships = _extract_relationships(path_info, path_nodes)

    # Generate evidence
    evidence = _generate_path_evidence(path_info, path_nodes)

    return FunctionPath(
        start_breadcrumb=start_breadcrumb,
        end_breadcrumb=end_breadcrumb,
        path_steps=path_steps,
        quality=quality,
        path_id=path_id,
        path_type=chain_type.value,
        relationships=relationships,
        evidence=evidence,
    )


def _calculate_path_quality(path_info, path_nodes, strategy) -> PathQuality:
    """
    Calculate quality metrics for a path.

    Args:
        path_info: Path information dictionary
        path_nodes: List of nodes in the path
        strategy: Path finding strategy

    Returns:
        PathQuality object
    """
    path_length = len(path_nodes)

    # Calculate directness score (shorter paths are more direct)
    directness_score = max(0.1, 1.0 - (path_length - 2) * 0.1)

    # Calculate complexity score (lower is better)
    complexity_score = min(1.0, path_length * 0.15)

    # Calculate reliability score based on path type
    reliability_score = 0.8  # Base reliability
    if path_info.get("chain_type") == "bidirectional":
        reliability_score *= 0.9  # Slightly lower for bidirectional

    # Calculate relationship diversity
    relationship_diversity = 0.7  # Placeholder

    # Calculate confidence
    confidence = reliability_score * directness_score

    # Calculate overall score
    if strategy == PathStrategy.SHORTEST:
        overall_score = directness_score * 0.6 + reliability_score * 0.4
    elif strategy == PathStrategy.OPTIMAL:
        overall_score = reliability_score * 0.5 + directness_score * 0.3 + (1.0 - complexity_score) * 0.2
    else:  # ALL
        overall_score = (reliability_score + directness_score + (1.0 - complexity_score)) / 3.0

    return PathQuality(
        reliability_score=reliability_score,
        complexity_score=complexity_score,
        directness_score=directness_score,
        overall_score=overall_score,
        path_length=path_length,
        confidence=confidence,
        relationship_diversity=relationship_diversity,
    )


def _extract_relationships(path_info, path_nodes) -> list[str]:
    """
    Extract relationship types from path information.

    Args:
        path_info: Path information dictionary
        path_nodes: List of nodes in the path

    Returns:
        List of relationship type strings
    """
    # Placeholder implementation
    relationships = []

    for i in range(len(path_nodes) - 1):
        # This would be enhanced to extract actual relationship types
        relationships.append("function_call")

    return relationships


def _generate_path_evidence(path_info, path_nodes) -> list[str]:
    """
    Generate evidence for each step in the path.

    Args:
        path_info: Path information dictionary
        path_nodes: List of nodes in the path

    Returns:
        List of evidence strings
    """
    evidence = []

    for i, node in enumerate(path_nodes):
        if hasattr(node, "breadcrumb"):
            evidence.append(f"Step {i + 1}: {node.breadcrumb} ({getattr(node, 'chunk_type', 'unknown')})")
        else:
            evidence.append(f"Step {i + 1}: {str(node)} (unknown type)")

    return evidence


def _is_valid_path(path: FunctionPath) -> bool:
    """
    Validate that a path is valid and meaningful.

    Args:
        path: FunctionPath to validate

    Returns:
        True if path is valid, False otherwise
    """
    # Basic validation
    if not path.path_steps or len(path.path_steps) < 2:
        return False

    # Check that start and end match
    if path.path_steps[0] != path.start_breadcrumb or path.path_steps[-1] != path.end_breadcrumb:
        return False

    # Check quality thresholds
    if path.quality.overall_score < 0.1:
        return False

    return True


def _apply_strategy_filtering(paths: list[FunctionPath], strategy: PathStrategy, max_paths: int) -> list[FunctionPath]:
    """
    Apply strategy-specific filtering and sorting to paths.

    Args:
        paths: List of paths to filter
        strategy: Path finding strategy
        max_paths: Maximum number of paths to return

    Returns:
        Filtered and sorted list of paths
    """
    if not paths:
        return paths

    # Remove duplicates based on path steps
    unique_paths = []
    seen_paths = set()

    for path in paths:
        path_key = tuple(path.path_steps)
        if path_key not in seen_paths:
            seen_paths.add(path_key)
            unique_paths.append(path)

    # Apply strategy-specific sorting
    if strategy == PathStrategy.SHORTEST:
        unique_paths.sort(key=lambda p: (p.quality.path_length, -p.quality.overall_score))
    elif strategy == PathStrategy.OPTIMAL:
        unique_paths.sort(key=lambda p: -p.quality.overall_score)
    elif strategy == PathStrategy.ALL:
        unique_paths.sort(key=lambda p: (-p.quality.overall_score, p.quality.path_length))

    return unique_paths[:max_paths]


def _format_path_output(path: FunctionPath, output_format: str, include_mermaid: bool) -> dict[str, Any]:
    """
    Format path output for display using comprehensive formatting utilities.

    Args:
        path: The path to format
        output_format: Output format preference
        include_mermaid: Whether to include Mermaid output

    Returns:
        Formatted path information
    """
    # Base path information
    formatted = {
        "path_id": path.path_id,
        "start_breadcrumb": path.start_breadcrumb,
        "end_breadcrumb": path.end_breadcrumb,
        "path_steps": path.path_steps,
        "quality": {
            "reliability_score": path.quality.reliability_score,
            "complexity_score": path.quality.complexity_score,
            "directness_score": path.quality.directness_score,
            "overall_score": path.quality.overall_score,
            "path_length": path.quality.path_length,
            "confidence": path.quality.confidence,
            "relationship_diversity": path.quality.relationship_diversity,
        },
        "path_type": path.path_type,
        "relationships": path.relationships,
        "evidence": path.evidence,
    }

    # Add comprehensive arrow formatting
    if output_format in ["arrow", "both"]:
        formatted["arrow_format"] = format_arrow_path(
            path.path_steps, path.relationships, include_relationships=True, custom_separator=" => ", max_line_length=80
        )

    # Add comprehensive Mermaid formatting
    if output_format in ["mermaid", "both"] or include_mermaid:
        formatted["mermaid_format"] = format_mermaid_path(
            path.path_steps,
            path.relationships,
            path.path_id,
            MermaidStyle.FLOWCHART,
            include_quality_info=True,
            quality_scores=formatted["quality"],
            custom_styling=None,
        )

    return formatted


def _analyze_path_diversity(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Analyze diversity of paths including relationship types and structural variety.

    This function calculates multiple diversity metrics to help users understand
    the variety of paths available between two functions.

    Args:
        paths: List of paths to analyze

    Returns:
        Diversity analysis results with comprehensive metrics
    """
    if not paths:
        return {
            "total_paths": 0,
            "diversity_score": 0.0,
            "relationship_diversity": 0.0,
            "length_diversity": 0.0,
            "type_diversity": 0.0,
            "analysis": "No paths available for diversity analysis",
        }

    # Basic path statistics
    total_paths = len(paths)
    path_lengths = [path.quality.path_length for path in paths]
    average_path_length = sum(path_lengths) / total_paths

    # 1. Relationship Type Diversity
    all_relationships = set()
    relationship_frequency = {}

    for path in paths:
        for rel in path.relationships:
            all_relationships.add(rel)
            relationship_frequency[rel] = relationship_frequency.get(rel, 0) + 1

    unique_relationship_types = len(all_relationships)

    # Calculate relationship diversity using Shannon entropy
    relationship_diversity = _calculate_shannon_entropy(relationship_frequency)

    # 2. Path Length Diversity
    length_variance = _calculate_variance(path_lengths)
    length_diversity = min(1.0, length_variance / 10.0)  # Normalize to 0-1

    # 3. Path Type Diversity
    path_types = [path.path_type for path in paths]
    path_type_frequency = {}
    for path_type in path_types:
        path_type_frequency[path_type] = path_type_frequency.get(path_type, 0) + 1

    unique_path_types = len(set(path_types))
    type_diversity = _calculate_shannon_entropy(path_type_frequency)

    # 4. Quality Score Diversity
    quality_scores = [path.quality.overall_score for path in paths]
    quality_variance = _calculate_variance(quality_scores)
    quality_diversity = min(1.0, quality_variance * 4.0)  # Normalize to 0-1

    # 5. Structural Complexity Diversity
    complexity_scores = [path.quality.complexity_score for path in paths]
    complexity_variance = _calculate_variance(complexity_scores)
    complexity_diversity = min(1.0, complexity_variance * 4.0)  # Normalize to 0-1

    # 6. Overall Diversity Score
    # Weighted combination of different diversity metrics
    diversity_score = (
        relationship_diversity * 0.3
        + length_diversity * 0.2
        + type_diversity * 0.2
        + quality_diversity * 0.15
        + complexity_diversity * 0.15
    )

    # 7. Path Uniqueness Analysis
    unique_path_signatures = set()
    for path in paths:
        # Create a signature based on key characteristics
        signature = (
            tuple(path.path_steps),
            path.path_type,
            len(path.relationships),
        )
        unique_path_signatures.add(signature)

    uniqueness_ratio = len(unique_path_signatures) / total_paths

    # 8. Relationship Pattern Analysis
    relationship_patterns = _analyze_relationship_patterns(paths)

    # 9. Coverage Analysis
    coverage_analysis = _analyze_path_coverage(paths)

    return {
        "total_paths": total_paths,
        "diversity_score": diversity_score,
        # Core diversity metrics
        "relationship_diversity": relationship_diversity,
        "length_diversity": length_diversity,
        "type_diversity": type_diversity,
        "quality_diversity": quality_diversity,
        "complexity_diversity": complexity_diversity,
        # Basic statistics
        "unique_relationship_types": unique_relationship_types,
        "unique_path_types": unique_path_types,
        "average_path_length": average_path_length,
        "path_length_range": {
            "min": min(path_lengths),
            "max": max(path_lengths),
            "variance": length_variance,
        },
        # Advanced analysis
        "uniqueness_ratio": uniqueness_ratio,
        "relationship_frequency": relationship_frequency,
        "path_type_distribution": path_type_frequency,
        "relationship_patterns": relationship_patterns,
        "coverage_analysis": coverage_analysis,
        # Quality analysis
        "quality_score_range": {
            "min": min(quality_scores),
            "max": max(quality_scores),
            "average": sum(quality_scores) / len(quality_scores),
            "variance": quality_variance,
        },
        # Recommendations
        "diversity_recommendations": _generate_diversity_recommendations(
            diversity_score, relationship_diversity, length_diversity, type_diversity
        ),
    }


def _calculate_shannon_entropy(frequency_dict: dict[str, int]) -> float:
    """
    Calculate Shannon entropy for diversity measurement.

    Args:
        frequency_dict: Dictionary of item frequencies

    Returns:
        Shannon entropy value (0.0-1.0, normalized)
    """
    if not frequency_dict:
        return 0.0

    total = sum(frequency_dict.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in frequency_dict.values():
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)

    # Normalize to 0-1 range (max entropy is log2(n) where n is number of unique items)
    max_entropy = math.log2(len(frequency_dict)) if len(frequency_dict) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _calculate_variance(values: list[float]) -> float:
    """
    Calculate variance of a list of values.

    Args:
        values: List of numeric values

    Returns:
        Variance value
    """
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance


def _analyze_relationship_patterns(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Analyze patterns in relationship types across paths.

    Args:
        paths: List of paths to analyze

    Returns:
        Relationship pattern analysis
    """

    # Pattern 1: Common relationship sequences
    relationship_sequences = []
    for path in paths:
        if len(path.relationships) > 1:
            for i in range(len(path.relationships) - 1):
                sequence = (path.relationships[i], path.relationships[i + 1])
                relationship_sequences.append(sequence)

    sequence_frequency = {}
    for seq in relationship_sequences:
        sequence_frequency[seq] = sequence_frequency.get(seq, 0) + 1

    # Pattern 2: Relationship dominance
    all_relationships = []
    for path in paths:
        all_relationships.extend(path.relationships)

    relationship_counts = {}
    for rel in all_relationships:
        relationship_counts[rel] = relationship_counts.get(rel, 0) + 1

    # Find dominant relationship type
    dominant_relationship = max(relationship_counts.items(), key=lambda x: x[1]) if relationship_counts else None

    # Pattern 3: Path relationship consistency
    path_consistency = []
    for path in paths:
        unique_rels = set(path.relationships)
        consistency = len(unique_rels) / len(path.relationships) if path.relationships else 0.0
        path_consistency.append(consistency)

    avg_consistency = sum(path_consistency) / len(path_consistency) if path_consistency else 0.0

    return {
        "common_sequences": dict(sorted(sequence_frequency.items(), key=lambda x: x[1], reverse=True)[:5]),
        "dominant_relationship": dominant_relationship,
        "relationship_distribution": relationship_counts,
        "average_path_consistency": avg_consistency,
        "total_relationship_instances": len(all_relationships),
    }


def _analyze_path_coverage(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Analyze coverage provided by the set of paths.

    Args:
        paths: List of paths to analyze

    Returns:
        Coverage analysis results
    """
    if not paths:
        return {"coverage_score": 0.0, "analysis": "No paths for coverage analysis"}

    # 1. Node coverage - how many unique nodes are covered
    all_nodes = set()
    for path in paths:
        all_nodes.update(path.path_steps)

    unique_nodes_covered = len(all_nodes)

    # 2. Relationship coverage - how many unique relationships are covered
    all_relationships = set()
    for path in paths:
        all_relationships.update(path.relationships)

    unique_relationships_covered = len(all_relationships)

    # 3. Path type coverage - how many different path types are covered
    path_types = {path.path_type for path in paths}
    unique_path_types_covered = len(path_types)

    # 4. Quality spectrum coverage - how well different quality levels are covered
    quality_scores = [path.quality.overall_score for path in paths]
    quality_range = max(quality_scores) - min(quality_scores) if quality_scores else 0.0

    # 5. Length spectrum coverage - how well different path lengths are covered
    path_lengths = [path.quality.path_length for path in paths]
    length_range = max(path_lengths) - min(path_lengths) if path_lengths else 0.0

    # Overall coverage score
    coverage_score = min(
        1.0,
        (
            unique_nodes_covered * 0.3
            + unique_relationships_covered * 0.3
            + unique_path_types_covered * 0.2
            + quality_range * 0.1
            + length_range * 0.1
        )
        / 10.0,
    )  # Normalize

    return {
        "coverage_score": coverage_score,
        "unique_nodes_covered": unique_nodes_covered,
        "unique_relationships_covered": unique_relationships_covered,
        "unique_path_types_covered": unique_path_types_covered,
        "quality_range": quality_range,
        "length_range": length_range,
        "coverage_breakdown": {
            "node_coverage": unique_nodes_covered,
            "relationship_coverage": unique_relationships_covered,
            "type_coverage": unique_path_types_covered,
            "quality_spectrum": quality_range,
            "length_spectrum": length_range,
        },
    }


def _generate_diversity_recommendations(
    diversity_score: float,
    relationship_diversity: float,
    length_diversity: float,
    type_diversity: float,
) -> list[str]:
    """
    Generate recommendations based on diversity analysis.

    Args:
        diversity_score: Overall diversity score
        relationship_diversity: Relationship type diversity
        length_diversity: Path length diversity
        type_diversity: Path type diversity

    Returns:
        List of recommendations
    """
    recommendations = []

    if diversity_score < 0.3:
        recommendations.append("Low overall diversity - consider expanding search parameters")
    elif diversity_score > 0.8:
        recommendations.append("High path diversity - good coverage of different approaches")

    if relationship_diversity < 0.3:
        recommendations.append("Limited relationship type diversity - paths use similar connection types")

    if length_diversity < 0.2:
        recommendations.append("Similar path lengths - consider different search strategies for varied complexity")

    if type_diversity < 0.3:
        recommendations.append("Limited path type diversity - try different chain types for more variety")

    # Positive recommendations
    if diversity_score > 0.6:
        recommendations.append("Good diversity provides multiple viable approaches")

    if relationship_diversity > 0.7:
        recommendations.append("Rich relationship diversity shows various connection patterns")

    if not recommendations:
        recommendations.append("Balanced diversity across all metrics")

    return recommendations


def _generate_path_recommendation(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Generate intelligent recommendations for the best paths based on multiple criteria.

    This function analyzes all paths to identify the most direct, most reliable,
    and most balanced options, providing detailed reasoning for each recommendation.

    Args:
        paths: List of paths to analyze

    Returns:
        Comprehensive path recommendation with alternatives and reasoning
    """
    if not paths:
        return {
            "recommended_path": None,
            "reason": "No paths available",
            "alternatives": [],
            "suggestions": ["Try using different search parameters", "Increase max_depth parameter", "Lower quality threshold"],
        }

    # Sort paths by overall quality score (descending)
    sorted_paths = sorted(paths, key=lambda p: p.quality.overall_score, reverse=True)

    # Find specialized paths
    most_direct = min(paths, key=lambda p: p.quality.path_length)
    most_reliable = max(paths, key=lambda p: p.quality.reliability_score)
    most_balanced = max(paths, key=lambda p: p.quality.overall_score)
    least_complex = min(paths, key=lambda p: p.quality.complexity_score)

    # Primary recommendation (highest overall score)
    primary_recommendation = sorted_paths[0]

    # Generate detailed reasoning for the primary recommendation
    primary_reasoning = _generate_recommendation_reasoning(primary_recommendation, paths)

    # Generate alternative recommendations
    alternatives = []

    # Add most direct path if different from primary
    if most_direct.path_id != primary_recommendation.path_id:
        alternatives.append(
            {
                "path_id": most_direct.path_id,
                "type": "most_direct",
                "overall_score": most_direct.quality.overall_score,
                "path_length": most_direct.quality.path_length,
                "reason": f"Shortest path with only {most_direct.quality.path_length} steps",
                "trade_offs": "May sacrifice some reliability for directness",
            }
        )

    # Add most reliable path if different from primary
    if most_reliable.path_id != primary_recommendation.path_id:
        alternatives.append(
            {
                "path_id": most_reliable.path_id,
                "type": "most_reliable",
                "overall_score": most_reliable.quality.overall_score,
                "reliability_score": most_reliable.quality.reliability_score,
                "reason": f"Highest reliability score ({most_reliable.quality.reliability_score:.2f})",
                "trade_offs": "May be longer but more dependable",
            }
        )

    # Add least complex path if different from primary
    if least_complex.path_id != primary_recommendation.path_id:
        alternatives.append(
            {
                "path_id": least_complex.path_id,
                "type": "least_complex",
                "overall_score": least_complex.quality.overall_score,
                "complexity_score": least_complex.quality.complexity_score,
                "reason": f"Lowest complexity score ({least_complex.quality.complexity_score:.2f})",
                "trade_offs": "Simplest path, easier to understand and maintain",
            }
        )

    # Add up to 2 more high-quality alternatives
    remaining_paths = [p for p in sorted_paths[1:] if p.path_id not in [alt.get("path_id") for alt in alternatives]]
    for i, path in enumerate(remaining_paths[:2]):
        alternatives.append(
            {
                "path_id": path.path_id,
                "type": "high_quality_alternative",
                "overall_score": path.quality.overall_score,
                "reason": f"Alternative #{i+1} with good overall quality ({path.quality.overall_score:.2f})",
                "trade_offs": "Balanced approach with good all-around performance",
            }
        )

    # Generate usage suggestions
    suggestions = _generate_usage_suggestions(primary_recommendation, paths, alternatives)

    # Calculate recommendation confidence
    confidence = _calculate_recommendation_confidence(primary_recommendation, paths)

    return {
        "recommended_path": {
            "path_id": primary_recommendation.path_id,
            "start_breadcrumb": primary_recommendation.start_breadcrumb,
            "end_breadcrumb": primary_recommendation.end_breadcrumb,
            "overall_score": primary_recommendation.quality.overall_score,
            "path_length": primary_recommendation.quality.path_length,
            "reliability_score": primary_recommendation.quality.reliability_score,
            "complexity_score": primary_recommendation.quality.complexity_score,
            "path_type": primary_recommendation.path_type,
            "confidence": confidence,
        },
        "reason": primary_reasoning,
        "alternatives": alternatives[:5],  # Limit to 5 alternatives
        "suggestions": suggestions,
        "analysis": {
            "total_paths_analyzed": len(paths),
            "recommendation_confidence": confidence,
            "quality_spread": {
                "highest": max(p.quality.overall_score for p in paths),
                "lowest": min(p.quality.overall_score for p in paths),
                "average": sum(p.quality.overall_score for p in paths) / len(paths),
            },
            "specialized_paths": {
                "most_direct_id": most_direct.path_id,
                "most_reliable_id": most_reliable.path_id,
                "most_balanced_id": most_balanced.path_id,
                "least_complex_id": least_complex.path_id,
            },
        },
    }


def _generate_recommendation_reasoning(path: FunctionPath, all_paths: list[FunctionPath]) -> str:
    """
    Generate detailed reasoning for why a path is recommended.

    Args:
        path: The recommended path
        all_paths: All available paths for comparison

    Returns:
        Detailed reasoning string
    """
    reasons = []

    # Overall quality reasoning
    if path.quality.overall_score >= 0.8:
        reasons.append(f"exceptional overall quality score ({path.quality.overall_score:.2f})")
    elif path.quality.overall_score >= 0.6:
        reasons.append(f"good overall quality score ({path.quality.overall_score:.2f})")
    else:
        reasons.append(f"acceptable quality score ({path.quality.overall_score:.2f})")

    # Reliability reasoning
    if path.quality.reliability_score >= 0.8:
        reasons.append("high reliability")
    elif path.quality.reliability_score >= 0.6:
        reasons.append("good reliability")

    # Directness reasoning
    if path.quality.path_length <= 3:
        reasons.append("very direct connection")
    elif path.quality.path_length <= 5:
        reasons.append("reasonably direct path")

    # Complexity reasoning
    if path.quality.complexity_score <= 0.3:
        reasons.append("low complexity")
    elif path.quality.complexity_score <= 0.5:
        reasons.append("moderate complexity")

    # Comparison with alternatives
    if len(all_paths) > 1:
        better_quality_count = sum(1 for p in all_paths if p.quality.overall_score > path.quality.overall_score)
        if better_quality_count == 0:
            reasons.append("highest quality among all options")
        elif better_quality_count <= len(all_paths) * 0.2:
            reasons.append("among the top-quality options")

    # Path type reasoning
    if path.path_type == "execution_flow":
        reasons.append("follows execution flow patterns")
    elif path.path_type == "dependency_chain":
        reasons.append("based on dependency relationships")

    return f"Recommended because it has {', '.join(reasons)}."


def _generate_usage_suggestions(primary_path: FunctionPath, all_paths: list[FunctionPath], alternatives: list[dict]) -> list[str]:
    """
    Generate usage suggestions based on path analysis.

    Args:
        primary_path: The primary recommended path
        all_paths: All available paths
        alternatives: Alternative path recommendations

    Returns:
        List of usage suggestions
    """
    suggestions = []

    # Primary path suggestions
    if primary_path.quality.overall_score >= 0.8:
        suggestions.append("The recommended path is high-quality and suitable for most use cases")
    elif primary_path.quality.overall_score >= 0.6:
        suggestions.append("The recommended path provides good balance of quality factors")
    else:
        suggestions.append("Consider alternatives if higher quality is needed")

    # Alternative suggestions
    if alternatives:
        if any(alt.get("type") == "most_direct" for alt in alternatives):
            suggestions.append("Choose the most direct alternative for simple, quick connections")

        if any(alt.get("type") == "most_reliable" for alt in alternatives):
            suggestions.append("Choose the most reliable alternative for production-critical paths")

        if any(alt.get("type") == "least_complex" for alt in alternatives):
            suggestions.append("Choose the least complex alternative for easier maintenance")

    # Path length suggestions
    if primary_path.quality.path_length > 5:
        suggestions.append("Consider if a shorter path might be more maintainable")

    # Reliability suggestions
    if primary_path.quality.reliability_score < 0.6:
        suggestions.append("Verify the path reliability before using in production")

    # Complexity suggestions
    if primary_path.quality.complexity_score > 0.7:
        suggestions.append("This path is complex - consider documentation or simplification")

    # General suggestions
    if len(all_paths) > 3:
        suggestions.append("Multiple paths available - choose based on your specific requirements")

    if not suggestions:
        suggestions.append("The recommended path is well-suited for typical use cases")

    return suggestions


def _calculate_recommendation_confidence(path: FunctionPath, all_paths: list[FunctionPath]) -> float:
    """
    Calculate confidence in the recommendation.

    Args:
        path: The recommended path
        all_paths: All available paths

    Returns:
        Confidence score (0.0-1.0)
    """
    if len(all_paths) == 1:
        return path.quality.overall_score

    # Base confidence from path quality
    base_confidence = path.quality.overall_score

    # Confidence boost from being significantly better than alternatives
    other_scores = [p.quality.overall_score for p in all_paths if p.path_id != path.path_id]
    if other_scores:
        score_advantage = path.quality.overall_score - max(other_scores)
        advantage_bonus = min(0.2, score_advantage * 0.5)  # Up to 0.2 bonus
        base_confidence += advantage_bonus

    # Confidence adjustment based on path characteristics
    if path.quality.reliability_score >= 0.8:
        base_confidence *= 1.1
    elif path.quality.reliability_score < 0.5:
        base_confidence *= 0.9

    # Confidence adjustment based on path length
    if path.quality.path_length <= 3:
        base_confidence *= 1.05
    elif path.quality.path_length > 7:
        base_confidence *= 0.95

    return min(1.0, base_confidence)


def _compare_paths(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Perform comprehensive comparison of multiple paths across various dimensions.

    This function analyzes paths to identify trade-offs, strengths, and weaknesses,
    providing detailed comparative analysis to help users make informed decisions.

    Args:
        paths: List of paths to compare

    Returns:
        Comprehensive path comparison results
    """
    if len(paths) < 2:
        return {
            "comparison": "Need at least 2 paths for comparison",
            "total_paths": len(paths),
            "analysis": "Cannot perform comparative analysis with fewer than 2 paths",
        }

    # Basic statistics
    quality_scores = [path.quality.overall_score for path in paths]
    path_lengths = [path.quality.path_length for path in paths]
    reliability_scores = [path.quality.reliability_score for path in paths]

    # Find specialized paths
    most_direct = min(paths, key=lambda p: p.quality.path_length)
    most_reliable = max(paths, key=lambda p: p.quality.reliability_score)
    highest_quality = max(paths, key=lambda p: p.quality.overall_score)
    least_complex = min(paths, key=lambda p: p.quality.complexity_score)
    most_confident = max(paths, key=lambda p: p.quality.confidence)

    # Calculate comparison metrics
    quality_variance = _calculate_variance(quality_scores)
    length_variance = _calculate_variance(path_lengths)
    reliability_variance = _calculate_variance(reliability_scores)

    # Path type analysis
    path_types = [path.path_type for path in paths]
    path_type_distribution = {}
    for path_type in path_types:
        path_type_distribution[path_type] = path_type_distribution.get(path_type, 0) + 1

    # Relationship analysis
    all_relationships = set()
    for path in paths:
        all_relationships.update(path.relationships)

    # Trade-off analysis
    trade_offs = _analyze_path_trade_offs(paths)

    # Clustering analysis
    path_clusters = _cluster_similar_paths(paths)

    # Ranking analysis
    rankings = _generate_path_rankings(paths)

    # Comparative advantages
    comparative_advantages = _identify_comparative_advantages(paths)

    return {
        "total_paths": len(paths),
        "comparison_summary": {
            "quality_spread": max(quality_scores) - min(quality_scores),
            "length_spread": max(path_lengths) - min(path_lengths),
            "reliability_spread": max(reliability_scores) - min(reliability_scores),
            "avg_quality": sum(quality_scores) / len(quality_scores),
            "avg_length": sum(path_lengths) / len(path_lengths),
            "avg_reliability": sum(reliability_scores) / len(reliability_scores),
        },
        # Detailed statistics
        "quality_statistics": {
            "min": min(quality_scores),
            "max": max(quality_scores),
            "avg": sum(quality_scores) / len(quality_scores),
            "variance": quality_variance,
            "distribution": _create_score_distribution(quality_scores),
        },
        "length_statistics": {
            "min": min(path_lengths),
            "max": max(path_lengths),
            "avg": sum(path_lengths) / len(path_lengths),
            "variance": length_variance,
            "distribution": _create_length_distribution(path_lengths),
        },
        "reliability_statistics": {
            "min": min(reliability_scores),
            "max": max(reliability_scores),
            "avg": sum(reliability_scores) / len(reliability_scores),
            "variance": reliability_variance,
            "distribution": _create_score_distribution(reliability_scores),
        },
        # Specialized paths
        "specialized_paths": {
            "most_direct": {
                "path_id": most_direct.path_id,
                "length": most_direct.quality.path_length,
                "quality": most_direct.quality.overall_score,
                "advantage": "Shortest connection",
            },
            "most_reliable": {
                "path_id": most_reliable.path_id,
                "reliability": most_reliable.quality.reliability_score,
                "quality": most_reliable.quality.overall_score,
                "advantage": "Highest reliability",
            },
            "highest_quality": {
                "path_id": highest_quality.path_id,
                "quality": highest_quality.quality.overall_score,
                "advantage": "Best overall quality",
            },
            "least_complex": {
                "path_id": least_complex.path_id,
                "complexity": least_complex.quality.complexity_score,
                "quality": least_complex.quality.overall_score,
                "advantage": "Simplest implementation",
            },
            "most_confident": {
                "path_id": most_confident.path_id,
                "confidence": most_confident.quality.confidence,
                "quality": most_confident.quality.overall_score,
                "advantage": "Highest confidence",
            },
        },
        # Path type analysis
        "path_type_analysis": {
            "distribution": path_type_distribution,
            "unique_types": len(set(path_types)),
            "dominant_type": max(path_type_distribution.items(), key=lambda x: x[1])[0],
        },
        # Relationship analysis
        "relationship_analysis": {
            "total_unique_relationships": len(all_relationships),
            "relationship_types": list(all_relationships),
            "paths_with_most_relationships": max(paths, key=lambda p: len(p.relationships)).path_id,
        },
        # Advanced analysis
        "trade_off_analysis": trade_offs,
        "path_clusters": path_clusters,
        "rankings": rankings,
        "comparative_advantages": comparative_advantages,
        # Recommendations
        "comparison_insights": _generate_comparison_insights(paths, trade_offs),
        "selection_guidance": _generate_selection_guidance(paths, rankings),
    }


def _analyze_path_trade_offs(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Analyze trade-offs between different paths.

    Args:
        paths: List of paths to analyze

    Returns:
        Trade-off analysis results
    """
    trade_offs = {}

    # Quality vs Length trade-off
    quality_length_correlation = _calculate_correlation([p.quality.overall_score for p in paths], [p.quality.path_length for p in paths])

    # Reliability vs Complexity trade-off
    reliability_complexity_correlation = _calculate_correlation(
        [p.quality.reliability_score for p in paths], [p.quality.complexity_score for p in paths]
    )

    # Directness vs Reliability trade-off
    directness_reliability_correlation = _calculate_correlation(
        [1.0 / p.quality.path_length for p in paths],
        [p.quality.reliability_score for p in paths],  # Inverse length as directness
    )

    trade_offs["quality_vs_length"] = {
        "correlation": quality_length_correlation,
        "interpretation": _interpret_correlation(quality_length_correlation, "quality", "length"),
    }

    trade_offs["reliability_vs_complexity"] = {
        "correlation": reliability_complexity_correlation,
        "interpretation": _interpret_correlation(reliability_complexity_correlation, "reliability", "complexity"),
    }

    trade_offs["directness_vs_reliability"] = {
        "correlation": directness_reliability_correlation,
        "interpretation": _interpret_correlation(directness_reliability_correlation, "directness", "reliability"),
    }

    return trade_offs


def _calculate_correlation(x_values: list[float], y_values: list[float]) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        x_values: First set of values
        y_values: Second set of values

    Returns:
        Correlation coefficient (-1.0 to 1.0)
    """
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0

    n = len(x_values)
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=False))

    x_variance = sum((x - x_mean) ** 2 for x in x_values)
    y_variance = sum((y - y_mean) ** 2 for y in y_values)

    if x_variance == 0 or y_variance == 0:
        return 0.0

    denominator = math.sqrt(x_variance * y_variance)
    return numerator / denominator


def _interpret_correlation(correlation: float, factor1: str, factor2: str) -> str:
    """
    Interpret correlation coefficient.

    Args:
        correlation: Correlation coefficient
        factor1: Name of first factor
        factor2: Name of second factor

    Returns:
        Human-readable interpretation
    """
    abs_corr = abs(correlation)

    if abs_corr < 0.1:
        strength = "no"
    elif abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if correlation > 0 else "negative"

    return f"{strength} {direction} correlation between {factor1} and {factor2}"


def _cluster_similar_paths(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Cluster paths based on similarity.

    Args:
        paths: List of paths to cluster

    Returns:
        Clustering results
    """
    if len(paths) < 2:
        return {"clusters": [], "analysis": "Not enough paths for clustering"}

    # Simple clustering based on path length and quality
    clusters = {}

    for path in paths:
        # Create cluster key based on path characteristics
        length_bucket = "short" if path.quality.path_length <= 3 else "medium" if path.quality.path_length <= 6 else "long"
        quality_bucket = "high" if path.quality.overall_score >= 0.7 else "medium" if path.quality.overall_score >= 0.4 else "low"

        cluster_key = f"{length_bucket}_{quality_bucket}"

        if cluster_key not in clusters:
            clusters[cluster_key] = []
        clusters[cluster_key].append(
            {
                "path_id": path.path_id,
                "quality": path.quality.overall_score,
                "length": path.quality.path_length,
                "reliability": path.quality.reliability_score,
            }
        )

    return {
        "clusters": clusters,
        "cluster_count": len(clusters),
        "largest_cluster": max(clusters.keys(), key=lambda k: len(clusters[k])) if clusters else None,
    }


def _generate_path_rankings(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Generate various rankings of paths.

    Args:
        paths: List of paths to rank

    Returns:
        Various path rankings
    """
    return {
        "by_overall_quality": [
            {"path_id": p.path_id, "score": p.quality.overall_score}
            for p in sorted(paths, key=lambda p: p.quality.overall_score, reverse=True)
        ],
        "by_directness": [
            {"path_id": p.path_id, "length": p.quality.path_length} for p in sorted(paths, key=lambda p: p.quality.path_length)
        ],
        "by_reliability": [
            {"path_id": p.path_id, "reliability": p.quality.reliability_score}
            for p in sorted(paths, key=lambda p: p.quality.reliability_score, reverse=True)
        ],
        "by_simplicity": [
            {"path_id": p.path_id, "complexity": p.quality.complexity_score}
            for p in sorted(paths, key=lambda p: p.quality.complexity_score)
        ],
    }


def _identify_comparative_advantages(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Identify comparative advantages of each path.

    Args:
        paths: List of paths to analyze

    Returns:
        Comparative advantages analysis
    """
    advantages = {}

    for path in paths:
        path_advantages = []

        # Compare against all other paths
        others = [p for p in paths if p.path_id != path.path_id]

        if not others:
            advantages[path.path_id] = ["Only available path"]
            continue

        # Check if this path is best in any category
        if path.quality.overall_score == max(p.quality.overall_score for p in paths):
            path_advantages.append("Highest overall quality")

        if path.quality.path_length == min(p.quality.path_length for p in paths):
            path_advantages.append("Shortest path")

        if path.quality.reliability_score == max(p.quality.reliability_score for p in paths):
            path_advantages.append("Most reliable")

        if path.quality.complexity_score == min(p.quality.complexity_score for p in paths):
            path_advantages.append("Least complex")

        if path.quality.confidence == max(p.quality.confidence for p in paths):
            path_advantages.append("Highest confidence")

        # Check for above-average performance
        avg_quality = sum(p.quality.overall_score for p in paths) / len(paths)
        if path.quality.overall_score > avg_quality:
            path_advantages.append("Above average quality")

        if not path_advantages:
            path_advantages.append("Balanced performance")

        advantages[path.path_id] = path_advantages

    return advantages


def _create_score_distribution(scores: list[float]) -> dict[str, int]:
    """
    Create distribution of scores.

    Args:
        scores: List of scores

    Returns:
        Score distribution
    """
    distribution = {"high": 0, "medium": 0, "low": 0}

    for score in scores:
        if score >= 0.7:
            distribution["high"] += 1
        elif score >= 0.4:
            distribution["medium"] += 1
        else:
            distribution["low"] += 1

    return distribution


def _create_length_distribution(lengths: list[int]) -> dict[str, int]:
    """
    Create distribution of path lengths.

    Args:
        lengths: List of path lengths

    Returns:
        Length distribution
    """
    distribution = {"short": 0, "medium": 0, "long": 0}

    for length in lengths:
        if length <= 3:
            distribution["short"] += 1
        elif length <= 6:
            distribution["medium"] += 1
        else:
            distribution["long"] += 1

    return distribution


def _generate_comparison_insights(paths: list[FunctionPath], trade_offs: dict[str, Any]) -> list[str]:
    """
    Generate insights from path comparison.

    Args:
        paths: List of paths
        trade_offs: Trade-off analysis results

    Returns:
        List of insights
    """
    insights = []

    # Quality insights
    quality_scores = [p.quality.overall_score for p in paths]
    quality_range = max(quality_scores) - min(quality_scores)

    if quality_range > 0.5:
        insights.append("Significant quality differences between paths - choose carefully based on requirements")
    elif quality_range < 0.2:
        insights.append("Similar quality across paths - other factors may be more important for selection")

    # Length insights
    path_lengths = [p.quality.path_length for p in paths]
    length_range = max(path_lengths) - min(path_lengths)

    if length_range > 3:
        insights.append("Wide range of path lengths - consider complexity vs directness trade-offs")

    # Trade-off insights
    quality_length_corr = trade_offs.get("quality_vs_length", {}).get("correlation", 0)
    if quality_length_corr < -0.5:
        insights.append("Longer paths tend to have higher quality - consider if extra steps are worth it")
    elif quality_length_corr > 0.5:
        insights.append("Shorter paths also have higher quality - directness is beneficial")

    if not insights:
        insights.append("Paths show balanced characteristics across different metrics")

    return insights


def _generate_selection_guidance(paths: list[FunctionPath], rankings: dict[str, Any]) -> list[str]:
    """
    Generate guidance for path selection.

    Args:
        paths: List of paths
        rankings: Path rankings

    Returns:
        List of selection guidance
    """
    guidance = []

    # Check if same path dominates multiple categories
    top_quality = rankings["by_overall_quality"][0]["path_id"]
    top_direct = rankings["by_directness"][0]["path_id"]
    top_reliable = rankings["by_reliability"][0]["path_id"]

    if top_quality == top_direct == top_reliable:
        guidance.append(f"Path {top_quality} is optimal across all key metrics")
    elif top_quality == top_direct:
        guidance.append(f"Path {top_quality} offers best combination of quality and directness")
    elif top_quality == top_reliable:
        guidance.append(f"Path {top_quality} provides best quality and reliability")
    else:
        guidance.append("No single path dominates - consider your priorities")

    # General guidance
    guidance.append("Choose based on your specific requirements:")
    guidance.append("- For speed: prioritize directness")
    guidance.append("- For production: prioritize reliability")
    guidance.append("- For maintenance: prioritize simplicity")
    guidance.append("- For overall use: prioritize quality")

    return guidance


def _find_paths_by_strategy(all_paths: list[FunctionPath], strategy: PathStrategy, max_paths: int) -> list[FunctionPath]:
    """
    Find paths using the specified strategy.

    Args:
        all_paths: All available paths
        strategy: Path finding strategy
        max_paths: Maximum number of paths to return

    Returns:
        Filtered and sorted list of paths
    """
    if not all_paths:
        return []

    # Apply strategy-specific filtering and sorting
    filtered_paths = _apply_strategy_filtering(all_paths, strategy, max_paths)

    return filtered_paths


def _validate_path_finding_parameters(
    start_function: str, end_function: str, project_name: str, strategy: str, max_paths: int, max_depth: int, min_quality_threshold: float
) -> dict[str, Any]:
    """
    Validate input parameters for path finding.

    Args:
        start_function: Starting function name
        end_function: End function name
        project_name: Project name
        strategy: Path finding strategy
        max_paths: Maximum number of paths
        max_depth: Maximum search depth
        min_quality_threshold: Minimum quality threshold

    Returns:
        Validation result dictionary
    """
    if not start_function or not start_function.strip():
        return {
            "valid": False,
            "error": "Start function is required and cannot be empty",
            "suggestions": ["Provide a valid start function name or description"],
        }

    if not end_function or not end_function.strip():
        return {
            "valid": False,
            "error": "End function is required and cannot be empty",
            "suggestions": ["Provide a valid end function name or description"],
        }

    if not project_name or not project_name.strip():
        return {"valid": False, "error": "Project name is required and cannot be empty", "suggestions": ["Provide a valid project name"]}

    if strategy not in ["shortest", "optimal", "all"]:
        return {"valid": False, "error": f"Invalid strategy: {strategy}", "suggestions": ["Valid strategies: shortest, optimal, all"]}

    if not isinstance(max_paths, int) or max_paths < 1 or max_paths > 100:
        return {
            "valid": False,
            "error": f"Invalid max_paths: {max_paths}. Must be between 1 and 100",
            "suggestions": ["Use a reasonable max_paths value between 1 and 100"],
        }

    if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 100:
        return {
            "valid": False,
            "error": f"Invalid max_depth: {max_depth}. Must be between 1 and 100",
            "suggestions": ["Use a reasonable max_depth value between 1 and 100"],
        }

    if not isinstance(min_quality_threshold, int | float) or min_quality_threshold < 0.0 or min_quality_threshold > 1.0:
        return {
            "valid": False,
            "error": f"Invalid min_quality_threshold: {min_quality_threshold}. Must be between 0.0 and 1.0",
            "suggestions": ["Use a quality threshold value between 0.0 and 1.0"],
        }

    return {"valid": True}
