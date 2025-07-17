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
    Find multiple paths between two functions.

    This is a placeholder implementation that will be enhanced in subsequent subtasks.

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
    # This will be implemented in subtask 3.3
    # For now, create a placeholder path
    placeholder_path = FunctionPath(
        start_breadcrumb=start_breadcrumb,
        end_breadcrumb=end_breadcrumb,
        path_steps=[start_breadcrumb, end_breadcrumb],
        quality=PathQuality(
            reliability_score=0.5,
            complexity_score=0.3,
            directness_score=0.8,
            overall_score=0.6,
            path_length=2,
            confidence=0.7,
            relationship_diversity=0.5,
        ),
        path_id="placeholder_path_1",
        path_type="execution_flow",
        relationships=["direct_call"],
        evidence=["placeholder_evidence"],
    )

    return [placeholder_path]


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
