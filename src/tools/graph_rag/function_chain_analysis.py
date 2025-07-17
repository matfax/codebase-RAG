"""Function Chain Analysis Tool

This module provides MCP tools for tracing function call chains and implementation
flows in codebases using Graph RAG capabilities. It enables users to understand
execution paths, data flows, and dependencies between functions.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.services.implementation_chain_service import (
    ChainDirection,
    ChainType,
    ImplementationChainService,
    get_implementation_chain_service,
)

logger = logging.getLogger(__name__)


async def trace_function_chain(
    entry_point: str,
    project_name: str,
    direction: str = "forward",
    max_depth: int = 10,
    output_format: str = "arrow",
    include_mermaid: bool = False,
    chain_type: str = "execution_flow",
    min_link_strength: float = 0.3,
    identify_branch_points: bool = True,
    identify_terminal_points: bool = True,
    performance_monitoring: bool = True,
) -> dict[str, Any]:
    """
    Trace a complete function chain from an entry point with various analysis options.

    This tool provides comprehensive function chain tracing capabilities, supporting
    multiple directions (forward/backward/bidirectional), output formats, and
    detailed analysis of execution flows, data flows, and dependencies.

    Args:
        entry_point: Function/class identifier (breadcrumb or natural language)
        project_name: Name of the project to analyze
        direction: Tracing direction ("forward", "backward", "bidirectional")
        max_depth: Maximum depth for chain traversal (default: 10)
        output_format: Output format ("arrow", "mermaid", "both")
        include_mermaid: Whether to include Mermaid diagram output
        chain_type: Type of chain to trace ("execution_flow", "data_flow", "dependency_chain")
        min_link_strength: Minimum link strength threshold (0.0-1.0)
        identify_branch_points: Whether to identify branch points in the chain
        identify_terminal_points: Whether to identify terminal points in the chain
        performance_monitoring: Whether to include performance monitoring

    Returns:
        Dictionary containing chain analysis results with formatted output
    """
    start_time = time.time()

    try:
        logger.info(f"Starting function chain trace for '{entry_point}' in project '{project_name}'")

        # Initialize results structure
        results = {
            "success": True,
            "entry_point": entry_point,
            "project_name": project_name,
            "direction": direction,
            "max_depth": max_depth,
            "chain_type": chain_type,
            "output_format": output_format,
            "timestamp": time.time(),
        }

        # Initialize performance monitoring
        if performance_monitoring:
            results["performance"] = {
                "start_time": start_time,
                "breadcrumb_resolution_time": 0.0,
                "chain_tracing_time": 0.0,
                "formatting_time": 0.0,
                "total_time": 0.0,
            }

        # Initialize services
        breadcrumb_resolver = BreadcrumbResolver()
        implementation_chain_service = get_implementation_chain_service()

        # Validate input parameters
        validation_result = _validate_input_parameters(
            entry_point, project_name, direction, max_depth, output_format, chain_type, min_link_strength
        )

        if not validation_result["valid"]:
            results["success"] = False
            results["error"] = validation_result["error"]
            results["suggestions"] = validation_result.get("suggestions", [])
            return results

        # Convert direction and chain_type to enums
        try:
            chain_direction = ChainDirection(direction.lower())
            chain_type_enum = ChainType(chain_type.lower())
        except ValueError as e:
            results["success"] = False
            results["error"] = f"Invalid parameter: {str(e)}"
            results["suggestions"] = [
                "Valid directions: forward, backward, bidirectional",
                "Valid chain types: execution_flow, data_flow, dependency_chain",
            ]
            return results

        # Step 1: Resolve breadcrumb from entry point (natural language or direct breadcrumb)
        breadcrumb_start_time = time.time()

        breadcrumb_result = await breadcrumb_resolver.resolve(query=entry_point, target_projects=[project_name])

        if performance_monitoring:
            results["performance"]["breadcrumb_resolution_time"] = (time.time() - breadcrumb_start_time) * 1000

        if not breadcrumb_result.success:
            results["success"] = False
            results["error"] = f"Failed to resolve entry point: {breadcrumb_result.error_message}"
            results["suggestions"] = [
                "Try using a more specific function name or description",
                "Use the search tool to find the exact function name first",
                "Check if the project has been indexed properly",
            ]
            return results

        resolved_breadcrumb = breadcrumb_result.primary_candidate.breadcrumb
        results["resolved_breadcrumb"] = resolved_breadcrumb
        results["breadcrumb_confidence"] = breadcrumb_result.primary_candidate.confidence_score

        # Step 2: Trace the implementation chain
        chain_start_time = time.time()

        implementation_chain = await implementation_chain_service.trace_implementation_chain(
            entry_point_breadcrumb=resolved_breadcrumb,
            project_name=project_name,
            chain_type=chain_type_enum,
            direction=chain_direction,
            max_depth=max_depth,
            min_link_strength=min_link_strength,
        )

        if performance_monitoring:
            results["performance"]["chain_tracing_time"] = (time.time() - chain_start_time) * 1000

        # Step 3: Process and format results
        format_start_time = time.time()

        # Basic chain information
        results["chain_info"] = {
            "chain_id": implementation_chain.chain_id,
            "depth": implementation_chain.depth,
            "branch_count": implementation_chain.branch_count,
            "total_components": implementation_chain.total_components,
            "complexity_score": implementation_chain.complexity_score,
            "completeness_score": implementation_chain.completeness_score,
            "reliability_score": implementation_chain.reliability_score,
            "functional_purpose": implementation_chain.functional_purpose,
            "total_links": len(implementation_chain.links),
        }

        # Identify branch points and terminal points
        if identify_branch_points:
            results["branch_points"] = _identify_branch_points(implementation_chain)

        if identify_terminal_points:
            results["terminal_points"] = _identify_terminal_points(implementation_chain)

        # Format output based on requested format
        if output_format in ["arrow", "both"]:
            results["arrow_format"] = _format_arrow_output(implementation_chain)

        if output_format in ["mermaid", "both"] or include_mermaid:
            results["mermaid_format"] = _format_mermaid_output(implementation_chain)

        # Add detailed link information
        results["chain_links"] = _format_chain_links(implementation_chain)

        # Add component statistics
        results["component_statistics"] = {
            "components_by_type": dict(implementation_chain.components_by_type),
            "average_link_strength": implementation_chain.avg_link_strength,
            "scope_breadcrumb": implementation_chain.scope_breadcrumb,
        }

        if performance_monitoring:
            results["performance"]["formatting_time"] = (time.time() - format_start_time) * 1000
            results["performance"]["total_time"] = (time.time() - start_time) * 1000

        logger.info(f"Function chain trace completed successfully for '{entry_point}' - found {len(implementation_chain.links)} links")

        return results

    except Exception as e:
        error_msg = f"Error tracing function chain for '{entry_point}': {str(e)}"
        logger.error(error_msg, exc_info=True)

        results = {
            "success": False,
            "error": error_msg,
            "entry_point": entry_point,
            "project_name": project_name,
            "direction": direction,
            "suggestions": [
                "Check if the project has been indexed properly",
                "Try using the search tool to find the exact function name",
                "Verify the project name is correct",
                "Consider using a simpler entry point description",
            ],
            "timestamp": time.time(),
        }

        if performance_monitoring:
            results["performance"] = {
                "total_time": (time.time() - start_time) * 1000,
                "error_occurred": True,
            }

        return results


def _validate_input_parameters(
    entry_point: str, project_name: str, direction: str, max_depth: int, output_format: str, chain_type: str, min_link_strength: float
) -> dict[str, Any]:
    """
    Validate input parameters for the trace_function_chain function.

    Args:
        entry_point: Function/class identifier
        project_name: Name of the project
        direction: Tracing direction
        max_depth: Maximum depth for traversal
        output_format: Output format
        chain_type: Type of chain to trace
        min_link_strength: Minimum link strength threshold

    Returns:
        Dictionary with validation results
    """
    if not entry_point or not entry_point.strip():
        return {
            "valid": False,
            "error": "Entry point is required and cannot be empty",
            "suggestions": ["Provide a function name or natural language description"],
        }

    if not project_name or not project_name.strip():
        return {"valid": False, "error": "Project name is required and cannot be empty", "suggestions": ["Provide a valid project name"]}

    if direction not in ["forward", "backward", "bidirectional"]:
        return {
            "valid": False,
            "error": f"Invalid direction: {direction}",
            "suggestions": ["Valid directions: forward, backward, bidirectional"],
        }

    if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 50:
        return {
            "valid": False,
            "error": f"Invalid max_depth: {max_depth}. Must be between 1 and 50",
            "suggestions": ["Use a reasonable depth value between 1 and 50"],
        }

    if output_format not in ["arrow", "mermaid", "both"]:
        return {"valid": False, "error": f"Invalid output_format: {output_format}", "suggestions": ["Valid formats: arrow, mermaid, both"]}

    valid_chain_types = [
        "execution_flow",
        "data_flow",
        "dependency_chain",
        "inheritance_chain",
        "interface_implementation",
        "service_layer_chain",
        "api_endpoint_chain",
    ]

    if chain_type not in valid_chain_types:
        return {
            "valid": False,
            "error": f"Invalid chain_type: {chain_type}",
            "suggestions": [f"Valid chain types: {', '.join(valid_chain_types)}"],
        }

    if not isinstance(min_link_strength, int | float) or min_link_strength < 0.0 or min_link_strength > 1.0:
        return {
            "valid": False,
            "error": f"Invalid min_link_strength: {min_link_strength}. Must be between 0.0 and 1.0",
            "suggestions": ["Use a link strength value between 0.0 and 1.0"],
        }

    return {"valid": True}


def _identify_branch_points(implementation_chain) -> list[dict[str, Any]]:
    """
    Identify branch points in the implementation chain.

    Args:
        implementation_chain: The implementation chain to analyze

    Returns:
        List of branch point information
    """
    branch_points = []

    try:
        # Count outgoing links for each component
        outgoing_counts = {}
        outgoing_links = {}

        for link in implementation_chain.links:
            source_id = link.source_component.chunk_id
            outgoing_counts[source_id] = outgoing_counts.get(source_id, 0) + 1

            if source_id not in outgoing_links:
                outgoing_links[source_id] = []
            outgoing_links[source_id].append(link)

        # Identify components with multiple outgoing links (branch points)
        for component_id, count in outgoing_counts.items():
            if count > 1:
                # Find the component
                component = None
                for link in implementation_chain.links:
                    if link.source_component.chunk_id == component_id:
                        component = link.source_component
                        break

                if component:
                    targets = [link.target_component.name for link in outgoing_links[component_id]]
                    branch_points.append(
                        {
                            "component_id": component_id,
                            "breadcrumb": component.breadcrumb,
                            "name": component.name,
                            "chunk_type": component.chunk_type,
                            "branch_count": count,
                            "target_components": targets,
                        }
                    )

    except Exception as e:
        logger.error(f"Error identifying branch points: {e}")

    return branch_points


def _identify_terminal_points(implementation_chain) -> list[dict[str, Any]]:
    """
    Identify terminal points in the implementation chain.

    Args:
        implementation_chain: The implementation chain to analyze

    Returns:
        List of terminal point information
    """
    terminal_points = []

    try:
        # Get all components that appear as targets in links
        all_targets = set()
        all_sources = set()

        for link in implementation_chain.links:
            all_targets.add(link.target_component.chunk_id)
            all_sources.add(link.source_component.chunk_id)

        # Terminal points are targets that are not sources (no outgoing links)
        terminal_ids = all_targets - all_sources

        # Also include explicitly marked terminal points
        for terminal_component in implementation_chain.terminal_points:
            terminal_ids.add(terminal_component.chunk_id)

        # Collect terminal point information
        for terminal_id in terminal_ids:
            # Find the component
            component = None
            for link in implementation_chain.links:
                if link.target_component.chunk_id == terminal_id:
                    component = link.target_component
                    break

            # Also check terminal_points list
            if not component:
                for terminal_component in implementation_chain.terminal_points:
                    if terminal_component.chunk_id == terminal_id:
                        component = terminal_component
                        break

            if component:
                terminal_points.append(
                    {
                        "component_id": terminal_id,
                        "breadcrumb": component.breadcrumb,
                        "name": component.name,
                        "chunk_type": component.chunk_type,
                        "file_path": component.file_path,
                    }
                )

    except Exception as e:
        logger.error(f"Error identifying terminal points: {e}")

    return terminal_points


def _format_arrow_output(implementation_chain) -> str:
    """
    Format the chain as arrow-style output (A => B => C).

    Args:
        implementation_chain: The implementation chain to format

    Returns:
        Arrow-formatted string representation
    """
    if not implementation_chain.links:
        return f"[{implementation_chain.entry_point.name}] (no connections found)"

    try:
        # Build the chain path
        chain_elements = []
        visited = set()

        # Start with entry point
        current = implementation_chain.entry_point
        chain_elements.append(current.name or current.breadcrumb)
        visited.add(current.chunk_id)

        # Follow the chain
        while True:
            next_link = None
            for link in implementation_chain.links:
                if link.source_component.chunk_id == current.chunk_id and link.target_component.chunk_id not in visited:
                    next_link = link
                    break

            if not next_link:
                break

            current = next_link.target_component
            visited.add(current.chunk_id)
            chain_elements.append(current.name or current.breadcrumb)

            # Prevent infinite loops
            if len(chain_elements) > 100:
                chain_elements.append("...")
                break

        # Create arrow format
        arrow_format = " => ".join(chain_elements)

        # Add branch information if there are multiple paths
        if implementation_chain.branch_count > 0:
            arrow_format += f" (with {implementation_chain.branch_count} branches)"

        return arrow_format

    except Exception as e:
        logger.error(f"Error formatting arrow output: {e}")
        return f"[{implementation_chain.entry_point.name}] (formatting error)"


def _format_mermaid_output(implementation_chain) -> str:
    """
    Format the chain as Mermaid diagram.

    Args:
        implementation_chain: The implementation chain to format

    Returns:
        Mermaid diagram string
    """
    if not implementation_chain.links:
        return f"graph TD\n    A[{implementation_chain.entry_point.name}]\n    A --> B[No connections found]"

    try:
        mermaid_lines = ["graph TD"]

        # Define nodes
        node_map = {}
        node_counter = 0

        def get_node_id(component):
            nonlocal node_counter
            if component.chunk_id not in node_map:
                node_id = f"N{node_counter}"
                node_counter += 1
                node_map[component.chunk_id] = node_id
                # Add node definition
                name = component.name or component.breadcrumb
                mermaid_lines.append(f"    {node_id}[{name}]")
            return node_map[component.chunk_id]

        # Add entry point first
        entry_node_id = get_node_id(implementation_chain.entry_point)

        # Add all links
        for link in implementation_chain.links:
            source_id = get_node_id(link.source_component)
            target_id = get_node_id(link.target_component)

            # Add edge with relationship type
            mermaid_lines.append(f"    {source_id} --> {target_id}")

        # Add styling for entry point
        mermaid_lines.append("    classDef entryPoint fill:#e1f5fe")
        mermaid_lines.append(f"    class {entry_node_id} entryPoint")

        return "\n".join(mermaid_lines)

    except Exception as e:
        logger.error(f"Error formatting Mermaid output: {e}")
        return f"graph TD\n    A[{implementation_chain.entry_point.name}]\n    A --> B[Formatting error]"


def _format_chain_links(implementation_chain) -> list[dict[str, Any]]:
    """
    Format chain links with detailed information.

    Args:
        implementation_chain: The implementation chain to format

    Returns:
        List of formatted link information
    """
    formatted_links = []

    try:
        for i, link in enumerate(implementation_chain.links):
            formatted_link = {
                "index": i + 1,
                "source": {
                    "name": link.source_component.name,
                    "breadcrumb": link.source_component.breadcrumb,
                    "chunk_type": link.source_component.chunk_type,
                    "file_path": link.source_component.file_path,
                },
                "target": {
                    "name": link.target_component.name,
                    "breadcrumb": link.target_component.breadcrumb,
                    "chunk_type": link.target_component.chunk_type,
                    "file_path": link.target_component.file_path,
                },
                "relationship": {
                    "type": link.relationship_type,
                    "interaction_type": link.interaction_type,
                    "strength": link.link_strength,
                    "confidence": link.confidence,
                },
                "evidence": {
                    "source": link.evidence_source,
                    "parameters_passed": link.parameters_passed,
                    "return_values": link.return_values,
                },
            }
            formatted_links.append(formatted_link)

    except Exception as e:
        logger.error(f"Error formatting chain links: {e}")

    return formatted_links
