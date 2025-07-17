"""
Output formatting utilities for function chain tools.

This module provides comprehensive formatting functions for converting function paths
and chains into various output formats including arrow format and Mermaid diagrams.
"""

import logging
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats for function paths."""

    ARROW = "arrow"
    MERMAID = "mermaid"
    BOTH = "both"


class MermaidStyle(Enum):
    """Mermaid diagram styles."""

    FLOWCHART = "flowchart"
    GRAPH = "graph"
    SEQUENCE = "sequence"


def format_arrow_path(
    path_steps: list[str],
    relationship_types: list[str] | None = None,
    include_relationships: bool = True,
    custom_separator: str = " => ",
    max_line_length: int = 80,
) -> str:
    """
    Format a function path as an arrow diagram.

    Args:
        path_steps: List of function breadcrumbs in the path
        relationship_types: Optional list of relationship types between steps
        include_relationships: Whether to include relationship type annotations
        custom_separator: Custom separator between path steps
        max_line_length: Maximum line length before wrapping

    Returns:
        Formatted arrow path string
    """
    if not path_steps:
        return ""

    if len(path_steps) == 1:
        return path_steps[0]

    # Build the path with relationship annotations
    if include_relationships and relationship_types:
        formatted_parts = []
        for i in range(len(path_steps) - 1):
            formatted_parts.append(path_steps[i])

            # Add relationship type annotation if available
            if i < len(relationship_types):
                rel_type = relationship_types[i]
                formatted_parts.append(f"--[{rel_type}]-->")
            else:
                formatted_parts.append(custom_separator.strip())

        formatted_parts.append(path_steps[-1])
        result = " ".join(formatted_parts)
    else:
        result = custom_separator.join(path_steps)

    # Handle line wrapping for long paths
    if len(result) > max_line_length:
        return _wrap_arrow_path(result, max_line_length, custom_separator)

    return result


def _wrap_arrow_path(path: str, max_length: int, separator: str) -> str:
    """
    Wrap a long arrow path across multiple lines.

    Args:
        path: The path string to wrap
        max_length: Maximum line length
        separator: Separator used in the path

    Returns:
        Multi-line wrapped path
    """
    parts = path.split(separator)
    lines = []
    current_line = ""

    for i, part in enumerate(parts):
        if i == 0:
            current_line = part
        else:
            test_line = current_line + separator + part
            if len(test_line) <= max_length:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = "  " + part  # Indent continuation lines

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def format_mermaid_path(
    path_steps: list[str],
    relationship_types: list[str] | None = None,
    path_id: str | None = None,
    style: MermaidStyle = MermaidStyle.FLOWCHART,
    include_quality_info: bool = False,
    quality_scores: dict[str, float] | None = None,
    custom_styling: dict[str, str] | None = None,
) -> str:
    """
    Format a function path as a Mermaid diagram.

    Args:
        path_steps: List of function breadcrumbs in the path
        relationship_types: Optional list of relationship types between steps
        path_id: Optional identifier for the path
        style: Mermaid diagram style
        include_quality_info: Whether to include quality information
        quality_scores: Optional quality scores for styling
        custom_styling: Optional custom CSS styling

    Returns:
        Mermaid diagram string
    """
    if not path_steps:
        return ""

    if len(path_steps) == 1:
        return f"graph TD\n    A[{path_steps[0]}]"

    # Generate diagram based on style
    if style == MermaidStyle.FLOWCHART:
        return _generate_flowchart_mermaid(path_steps, relationship_types, path_id, include_quality_info, quality_scores, custom_styling)
    elif style == MermaidStyle.GRAPH:
        return _generate_graph_mermaid(path_steps, relationship_types, path_id, include_quality_info, quality_scores, custom_styling)
    elif style == MermaidStyle.SEQUENCE:
        return _generate_sequence_mermaid(path_steps, relationship_types, path_id, include_quality_info, quality_scores)
    else:
        return _generate_flowchart_mermaid(path_steps, relationship_types, path_id, include_quality_info, quality_scores, custom_styling)


def _generate_flowchart_mermaid(
    path_steps: list[str],
    relationship_types: list[str] | None,
    path_id: str | None,
    include_quality_info: bool,
    quality_scores: dict[str, float] | None,
    custom_styling: dict[str, str] | None,
) -> str:
    """Generate flowchart-style Mermaid diagram."""
    lines = ["flowchart TD"]

    # Add title if path_id provided
    if path_id:
        lines.append(f"    %% Path: {path_id}")

    # Generate nodes with cleaned names
    node_mappings = {}
    for i, step in enumerate(path_steps):
        node_id = f"N{i}"
        clean_name = _clean_mermaid_text(step)
        node_mappings[step] = node_id

        # Add quality styling if available
        if include_quality_info and quality_scores:
            node_style = _get_quality_style(quality_scores, i)
            lines.append(f"    {node_id}[{clean_name}]{node_style}")
        else:
            lines.append(f"    {node_id}[{clean_name}]")

    # Generate connections
    for i in range(len(path_steps) - 1):
        current_node = f"N{i}"
        next_node = f"N{i + 1}"

        # Add relationship type if available
        if relationship_types and i < len(relationship_types):
            rel_type = _clean_mermaid_text(relationship_types[i])
            lines.append(f"    {current_node} -->|{rel_type}| {next_node}")
        else:
            lines.append(f"    {current_node} --> {next_node}")

    # Add custom styling if provided
    if custom_styling:
        lines.extend(_generate_mermaid_styling(custom_styling))

    return "\n".join(lines)


def _generate_graph_mermaid(
    path_steps: list[str],
    relationship_types: list[str] | None,
    path_id: str | None,
    include_quality_info: bool,
    quality_scores: dict[str, float] | None,
    custom_styling: dict[str, str] | None,
) -> str:
    """Generate graph-style Mermaid diagram."""
    lines = ["graph TD"]

    # Add title if path_id provided
    if path_id:
        lines.append(f"    %% Path: {path_id}")

    # Generate connections with embedded labels
    for i in range(len(path_steps) - 1):
        current_step = _clean_mermaid_text(path_steps[i])
        next_step = _clean_mermaid_text(path_steps[i + 1])

        if relationship_types and i < len(relationship_types):
            rel_type = _clean_mermaid_text(relationship_types[i])
            lines.append(f"    {current_step} -->|{rel_type}| {next_step}")
        else:
            lines.append(f"    {current_step} --> {next_step}")

    # Add custom styling if provided
    if custom_styling:
        lines.extend(_generate_mermaid_styling(custom_styling))

    return "\n".join(lines)


def _generate_sequence_mermaid(
    path_steps: list[str],
    relationship_types: list[str] | None,
    path_id: str | None,
    include_quality_info: bool,
    quality_scores: dict[str, float] | None,
) -> str:
    """Generate sequence-style Mermaid diagram."""
    lines = ["sequenceDiagram"]

    # Add title if path_id provided
    if path_id:
        lines.append(f"    title {path_id}")

    # Generate sequence interactions
    for i in range(len(path_steps) - 1):
        current_step = _clean_mermaid_text(path_steps[i])
        next_step = _clean_mermaid_text(path_steps[i + 1])

        if relationship_types and i < len(relationship_types):
            rel_type = _clean_mermaid_text(relationship_types[i])
            lines.append(f"    {current_step}->>+{next_step}: {rel_type}")
        else:
            lines.append(f"    {current_step}->>+{next_step}: calls")

    return "\n".join(lines)


def _clean_mermaid_text(text: str) -> str:
    """
    Clean text for safe use in Mermaid diagrams.

    Args:
        text: Text to clean

    Returns:
        Cleaned text safe for Mermaid
    """
    # Remove or replace problematic characters
    cleaned = text.replace('"', "'")
    cleaned = cleaned.replace("[", "(")
    cleaned = cleaned.replace("]", ")")
    cleaned = cleaned.replace("{", "(")
    cleaned = cleaned.replace("}", ")")
    cleaned = cleaned.replace("|", ":")
    cleaned = cleaned.replace("\\", "/")

    # Truncate very long names
    if len(cleaned) > 40:
        cleaned = cleaned[:37] + "..."

    return cleaned


def _get_quality_style(quality_scores: dict[str, float], index: int) -> str:
    """
    Get quality-based styling for a node.

    Args:
        quality_scores: Quality scores dictionary
        index: Node index

    Returns:
        Mermaid styling string
    """
    overall_score = quality_scores.get("overall_score", 0.5)

    if overall_score >= 0.8:
        return ":::highQuality"
    elif overall_score >= 0.6:
        return ":::mediumQuality"
    else:
        return ":::lowQuality"


def _generate_mermaid_styling(custom_styling: dict[str, str]) -> list[str]:
    """
    Generate Mermaid CSS styling lines.

    Args:
        custom_styling: Custom styling dictionary

    Returns:
        List of styling lines
    """
    lines = []

    # Default quality styles
    lines.extend(
        [
            "    classDef highQuality fill:#d4edda,stroke:#155724,stroke-width:2px",
            "    classDef mediumQuality fill:#fff3cd,stroke:#856404,stroke-width:2px",
            "    classDef lowQuality fill:#f8d7da,stroke:#721c24,stroke-width:2px",
        ]
    )

    # Add custom styles
    for class_name, style in custom_styling.items():
        lines.append(f"    classDef {class_name} {style}")

    return lines


def format_path_comparison(
    paths: list[dict[str, Any]], comparison_metrics: list[str] = None, include_recommendations: bool = True
) -> dict[str, Any]:
    """
    Format multiple paths for comparison.

    Args:
        paths: List of path dictionaries
        comparison_metrics: Metrics to compare
        include_recommendations: Whether to include recommendations

    Returns:
        Formatted comparison results
    """
    if not paths:
        return {"paths": [], "comparison": {}, "recommendations": {}}

    if comparison_metrics is None:
        comparison_metrics = ["overall_score", "path_length", "reliability_score", "complexity_score"]

    # Format each path
    formatted_paths = []
    for i, path in enumerate(paths):
        formatted_path = {
            "index": i,
            "arrow_format": path.get("arrow_format", ""),
            "mermaid_format": path.get("mermaid_format", ""),
            "quality": path.get("quality", {}),
            "path_steps": path.get("path_steps", []),
            "path_length": len(path.get("path_steps", [])),
            "relationships": path.get("relationships", []),
        }
        formatted_paths.append(formatted_path)

    # Generate comparison analysis
    comparison_analysis = _analyze_path_comparison(formatted_paths, comparison_metrics)

    # Generate recommendations if requested
    recommendations = {}
    if include_recommendations:
        recommendations = _generate_path_recommendations(formatted_paths)

    return {"paths": formatted_paths, "comparison": comparison_analysis, "recommendations": recommendations, "total_paths": len(paths)}


def _analyze_path_comparison(paths: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    """Analyze paths for comparison."""
    analysis = {}

    for metric in metrics:
        values = []
        for path in paths:
            if metric in path.get("quality", {}):
                values.append(path["quality"][metric])
            elif metric == "path_length":
                values.append(path["path_length"])

        if values:
            analysis[metric] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "range": max(values) - min(values),
            }

    return analysis


def _generate_path_recommendations(paths: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate recommendations based on path analysis."""
    if not paths:
        return {}

    recommendations = {}

    # Find best paths for different criteria
    best_overall = max(paths, key=lambda p: p.get("quality", {}).get("overall_score", 0))
    shortest = min(paths, key=lambda p: p["path_length"])
    most_reliable = max(paths, key=lambda p: p.get("quality", {}).get("reliability_score", 0))
    most_direct = max(paths, key=lambda p: p.get("quality", {}).get("directness_score", 0))

    recommendations["best_overall"] = {
        "index": best_overall["index"],
        "reason": "Highest overall quality score",
        "score": best_overall.get("quality", {}).get("overall_score", 0),
    }

    recommendations["shortest"] = {"index": shortest["index"], "reason": "Shortest path length", "length": shortest["path_length"]}

    recommendations["most_reliable"] = {
        "index": most_reliable["index"],
        "reason": "Highest reliability score",
        "score": most_reliable.get("quality", {}).get("reliability_score", 0),
    }

    recommendations["most_direct"] = {
        "index": most_direct["index"],
        "reason": "Highest directness score",
        "score": most_direct.get("quality", {}).get("directness_score", 0),
    }

    return recommendations


def format_comprehensive_output(
    paths: list[dict[str, Any]],
    output_format: OutputFormat = OutputFormat.BOTH,
    include_comparison: bool = True,
    include_recommendations: bool = True,
    mermaid_style: MermaidStyle = MermaidStyle.FLOWCHART,
    custom_styling: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Format comprehensive output for multiple paths.

    Args:
        paths: List of path dictionaries
        output_format: Desired output format
        include_comparison: Whether to include comparison analysis
        include_recommendations: Whether to include recommendations
        mermaid_style: Mermaid diagram style
        custom_styling: Custom styling options

    Returns:
        Comprehensive formatted output
    """
    if not paths:
        return {"paths": [], "summary": {"total_paths": 0, "message": "No paths found"}, "comparison": {}, "recommendations": {}}

    # Format each path
    formatted_paths = []
    for i, path in enumerate(paths):
        formatted_path = {
            "index": i,
            "path_id": path.get("path_id", f"path_{i}"),
            "start_breadcrumb": path.get("start_breadcrumb", ""),
            "end_breadcrumb": path.get("end_breadcrumb", ""),
            "path_steps": path.get("path_steps", []),
            "quality": path.get("quality", {}),
            "path_type": path.get("path_type", ""),
            "relationships": path.get("relationships", []),
            "evidence": path.get("evidence", []),
        }

        # Add format-specific outputs
        if output_format in [OutputFormat.ARROW, OutputFormat.BOTH]:
            formatted_path["arrow_format"] = format_arrow_path(
                formatted_path["path_steps"], formatted_path["relationships"], include_relationships=True
            )

        if output_format in [OutputFormat.MERMAID, OutputFormat.BOTH]:
            formatted_path["mermaid_format"] = format_mermaid_path(
                formatted_path["path_steps"],
                formatted_path["relationships"],
                formatted_path["path_id"],
                mermaid_style,
                include_quality_info=True,
                quality_scores=formatted_path["quality"],
                custom_styling=custom_styling,
            )

        formatted_paths.append(formatted_path)

    # Generate summary
    summary = {
        "total_paths": len(paths),
        "output_format": output_format.value,
        "average_path_length": sum(len(p["path_steps"]) for p in formatted_paths) / len(formatted_paths),
        "mermaid_style": mermaid_style.value if output_format in [OutputFormat.MERMAID, OutputFormat.BOTH] else None,
    }

    result = {"paths": formatted_paths, "summary": summary}

    # Add comparison if requested
    if include_comparison:
        comparison_result = format_path_comparison(
            formatted_paths, ["overall_score", "path_length", "reliability_score", "complexity_score"], include_recommendations
        )
        result["comparison"] = comparison_result["comparison"]

        if include_recommendations:
            result["recommendations"] = comparison_result["recommendations"]

    return result
