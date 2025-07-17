"""Function Path Finding Tool

This module provides MCP tools for finding paths between functions in codebases
using Graph RAG capabilities. It enables users to discover how functions are
connected and identify the most efficient ways to navigate between them.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.services.implementation_chain_service import (
    ChainDirection,
    ChainType,
    ImplementationChainService,
    get_implementation_chain_service,
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
    max_paths: int = 5,
    max_depth: int = 15,
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
        breadcrumb_resolver = BreadcrumbResolver()
        implementation_chain_service = get_implementation_chain_service()

        # Step 1: Resolve breadcrumbs for both functions
        breadcrumb_start_time = time.time()

        # Resolve start function breadcrumb
        start_result = await breadcrumb_resolver.resolve(query=start_function, target_projects=[project_name])

        if not start_result.success:
            results["success"] = False
            results["error"] = f"Failed to resolve start function: {start_result.error_message}"
            results["suggestions"] = [
                "Try using a more specific function name or description",
                "Use the search tool to find the exact function name first",
                "Check if the project has been indexed properly",
            ]
            return results

        # Resolve end function breadcrumb
        end_result = await breadcrumb_resolver.resolve(query=end_function, target_projects=[project_name])

        if not end_result.success:
            results["success"] = False
            results["error"] = f"Failed to resolve end function: {end_result.error_message}"
            results["suggestions"] = [
                "Try using a more specific function name or description",
                "Use the search tool to find the exact function name first",
                "Check if the project has been indexed properly",
            ]
            return results

        start_breadcrumb = start_result.primary_candidate.breadcrumb
        end_breadcrumb = end_result.primary_candidate.breadcrumb

        results["resolved_start_breadcrumb"] = start_breadcrumb
        results["resolved_end_breadcrumb"] = end_breadcrumb
        results["start_breadcrumb_confidence"] = start_result.primary_candidate.confidence_score
        results["end_breadcrumb_confidence"] = end_result.primary_candidate.confidence_score

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
        paths = await _find_multiple_paths(
            start_breadcrumb=start_breadcrumb,
            end_breadcrumb=end_breadcrumb,
            project_name=project_name,
            strategy=path_strategy,
            max_paths=max_paths,
            max_depth=max_depth,
            implementation_chain_service=implementation_chain_service,
        )

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
            results["suggestions"] = [
                "Try lowering the quality threshold",
                "Increase the maximum depth parameter",
                "Check if the functions are actually connected",
                "Use the search tool to find intermediate functions",
            ]
            return results

        # Sort paths by quality score (descending)
        quality_paths.sort(key=lambda p: p.quality.overall_score, reverse=True)

        # Limit results to max_paths
        final_paths = quality_paths[:max_paths]

        if performance_monitoring:
            results["performance"]["quality_analysis_time"] = (time.time() - quality_start_time) * 1000

        # Step 4: Format output
        format_start_time = time.time()

        # Format all paths
        formatted_paths = []
        for path in final_paths:
            formatted_path = _format_path_output(path, output_format, include_mermaid)
            formatted_paths.append(formatted_path)

        results["paths"] = formatted_paths
        results["total_paths_found"] = len(paths)
        results["paths_meeting_threshold"] = len(quality_paths)
        results["paths_returned"] = len(final_paths)

        # Add path diversity analysis if requested
        if include_path_diversity and final_paths:
            results["path_diversity"] = _analyze_path_diversity(final_paths)

        # Add path recommendation
        recommendation = _generate_path_recommendation(final_paths)
        results["recommendation"] = recommendation

        # Add path comparison
        if len(final_paths) > 1:
            results["path_comparison"] = _compare_paths(final_paths)

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

    if not isinstance(min_quality_threshold, int | float) or min_quality_threshold < 0.0 or min_quality_threshold > 1.0:
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
    project_graph = await implementation_chain_service.graph_rag_service.get_project_structure_graph(project_name)

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
        forward_chain = await implementation_chain_service.trace_implementation_chain(
            start_node.breadcrumb,
            project_graph.project_name,
            chain_type,
            ChainDirection.FORWARD,
            max_depth // 2,
            0.2,  # Lower threshold for more coverage
        )

        # Trace backward from end
        backward_chain = await implementation_chain_service.trace_implementation_chain(
            end_node.breadcrumb,
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
        forward_chain = await implementation_chain_service.trace_implementation_chain(
            start_node.breadcrumb,
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
        backward_chain = await implementation_chain_service.trace_implementation_chain(
            end_node.breadcrumb,
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
    for link in chain.links:
        if link.source_component.chunk_id == node.chunk_id or link.target_component.chunk_id == node.chunk_id:
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
            source_id = link.source_component.chunk_id

            if source_id not in graph:
                graph[source_id] = []
            graph[source_id].append(link.target_component)

        # Find path using BFS
        from collections import deque

        queue = deque([(start_node, [start_node])])
        visited = {start_node.chunk_id}

        while queue:
            current_node, path = queue.popleft()

            if current_node.chunk_id == end_node.chunk_id:
                return {
                    "path_nodes": path,
                    "path_length": len(path),
                    "chain_type": "direct",
                }

            if current_node.chunk_id in graph:
                for neighbor in graph[current_node.chunk_id]:
                    if neighbor.chunk_id not in visited:
                        visited.add(neighbor.chunk_id)
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

    for link in chain.links:
        if link.source_component.chunk_id == current_node.chunk_id:
            path.append(link.target_component)
            current_node = link.target_component

            if current_node.chunk_id == target_node.chunk_id:
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
    path_steps = [node.breadcrumb for node in path_nodes]

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
        evidence.append(f"Step {i + 1}: {node.breadcrumb} ({node.chunk_type})")

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
    Format path output for display.

    Args:
        path: The path to format
        output_format: Output format preference
        include_mermaid: Whether to include Mermaid output

    Returns:
        Formatted path information
    """
    # This will be implemented in subtask 3.8
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

    if output_format in ["arrow", "both"]:
        formatted["arrow_format"] = " => ".join(path.path_steps)

    if output_format in ["mermaid", "both"] or include_mermaid:
        formatted["mermaid_format"] = _generate_mermaid_format(path)

    return formatted


def _generate_mermaid_format(path: FunctionPath) -> str:
    """
    Generate Mermaid diagram format for a path.

    Args:
        path: The path to format

    Returns:
        Mermaid diagram string
    """
    # This will be enhanced in subtask 3.8
    lines = ["graph TD"]

    for i, step in enumerate(path.path_steps):
        node_id = f"N{i}"
        lines.append(f"    {node_id}[{step}]")

        if i < len(path.path_steps) - 1:
            next_node_id = f"N{i + 1}"
            lines.append(f"    {node_id} --> {next_node_id}")

    return "\n".join(lines)


def _analyze_path_diversity(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Analyze diversity of paths.

    Args:
        paths: List of paths to analyze

    Returns:
        Diversity analysis results
    """
    # This will be implemented in subtask 3.5
    return {
        "total_paths": len(paths),
        "unique_relationship_types": len({rel for path in paths for rel in path.relationships}),
        "average_path_length": sum(path.quality.path_length for path in paths) / len(paths),
        "diversity_score": 0.7,  # Placeholder
    }


def _generate_path_recommendation(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Generate recommendation for the best path.

    Args:
        paths: List of paths to analyze

    Returns:
        Path recommendation
    """
    # This will be implemented in subtask 3.7
    if not paths:
        return {
            "recommended_path": None,
            "reason": "No paths available",
            "alternatives": [],
            "suggestions": ["Try using different search parameters"],
        }

    # For now, recommend the first path (highest quality score)
    recommended = paths[0]
    alternatives = paths[1:3]  # Up to 2 alternatives

    return {
        "recommended_path": {
            "path_id": recommended.path_id,
            "start_breadcrumb": recommended.start_breadcrumb,
            "end_breadcrumb": recommended.end_breadcrumb,
            "overall_score": recommended.quality.overall_score,
            "reason": "Highest overall quality score",
        },
        "reason": f"This path has the highest overall quality score ({recommended.quality.overall_score:.2f})",
        "alternatives": [
            {
                "path_id": alt.path_id,
                "overall_score": alt.quality.overall_score,
                "reason": "Alternative with good quality",
            }
            for alt in alternatives
        ],
        "suggestions": [
            "Consider the recommended path for most reliable results",
            "Check alternatives if you need different characteristics",
        ],
    }


def _compare_paths(paths: list[FunctionPath]) -> dict[str, Any]:
    """
    Compare multiple paths.

    Args:
        paths: List of paths to compare

    Returns:
        Path comparison results
    """
    # This will be implemented in subtask 3.7
    if len(paths) < 2:
        return {"comparison": "Need at least 2 paths for comparison"}

    # Calculate statistics
    quality_scores = [path.quality.overall_score for path in paths]
    path_lengths = [path.quality.path_length for path in paths]

    return {
        "total_paths": len(paths),
        "quality_range": {
            "min": min(quality_scores),
            "max": max(quality_scores),
            "avg": sum(quality_scores) / len(quality_scores),
        },
        "length_range": {
            "min": min(path_lengths),
            "max": max(path_lengths),
            "avg": sum(path_lengths) / len(path_lengths),
        },
        "most_direct": min(paths, key=lambda p: p.quality.path_length).path_id,
        "most_reliable": max(paths, key=lambda p: p.quality.reliability_score).path_id,
    }
