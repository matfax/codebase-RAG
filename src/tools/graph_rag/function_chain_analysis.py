"""Function Chain Analysis Tool

This module provides MCP tools for tracing function call chains and implementation
flows in codebases using Graph RAG capabilities. It enables users to understand
execution paths, data flows, and dependencies between functions.
"""

import asyncio
import logging
import time
from typing import Any, Optional, Union

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

            # Enhanced error handling with intelligent suggestions
            enhanced_suggestions = await _generate_enhanced_suggestions(entry_point, project_name, breadcrumb_result, "entry_point")
            results["suggestions"] = enhanced_suggestions["suggestions"]
            results["error_details"] = enhanced_suggestions["error_details"]
            results["alternatives"] = enhanced_suggestions["alternatives"]
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

        # Generate enhanced suggestions for the error
        enhanced_suggestions = await _generate_enhanced_suggestions(entry_point, project_name, None, "general_error", error_msg)

        results = {
            "success": False,
            "error": error_msg,
            "entry_point": entry_point,
            "project_name": project_name,
            "direction": direction,
            "suggestions": enhanced_suggestions["suggestions"],
            "error_details": enhanced_suggestions["error_details"],
            "alternatives": enhanced_suggestions["alternatives"],
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

    if not isinstance(min_link_strength, Union[int, float]) or min_link_strength < 0.0 or min_link_strength > 1.0:
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


async def _generate_enhanced_suggestions(
    entry_point: str, project_name: str, breadcrumb_result: Any | None = None, error_type: str = "general", error_message: str = ""
) -> dict[str, Any]:
    """
    Generate enhanced error suggestions and alternatives when paths don't exist.

    Args:
        entry_point: The original entry point that failed
        project_name: The project name being searched
        breadcrumb_result: The breadcrumb resolution result (if available)
        error_type: Type of error ("entry_point", "general_error", etc.)
        error_message: The original error message

    Returns:
        Dictionary with enhanced suggestions, error details, and alternatives
    """
    suggestions = []
    error_details = {}
    alternatives = []

    try:
        # Import here to avoid circular imports
        from src.services.project_analysis_service import ProjectAnalysisService
        from src.tools.indexing.search_tools import search_async_cached

        # Analyze the entry point to understand what went wrong
        error_details["original_entry_point"] = entry_point
        error_details["project_name"] = project_name
        error_details["error_type"] = error_type
        error_details["error_message"] = error_message

        # Check if project exists and is indexed
        try:
            project_analysis_service = ProjectAnalysisService()
            project_info = await project_analysis_service.get_project_info(project_name)

            if not project_info:
                suggestions.extend(
                    [
                        f"Project '{project_name}' not found. Check if it has been indexed.",
                        "Use the list_indexed_projects tool to see available projects.",
                        "Run index_directory to index the project first.",
                    ]
                )
                error_details["project_status"] = "not_indexed"
                return {"suggestions": suggestions, "error_details": error_details, "alternatives": alternatives}

            error_details["project_status"] = "indexed"
            error_details["project_file_count"] = project_info.get("file_count", 0)
            error_details["project_functions_count"] = project_info.get("functions_count", 0)

        except Exception as e:
            logger.warning(f"Could not get project info: {e}")
            error_details["project_status"] = "unknown"

        # Try to find similar functions using fuzzy search
        try:
            # Search for partial matches
            search_results = await search_async_cached(
                query=entry_point, n_results=10, target_projects=[project_name], search_mode="hybrid", include_context=False
            )

            if search_results and search_results.get("results"):
                # Extract function names from search results
                found_functions = []
                for result in search_results["results"][:5]:  # Top 5 results
                    chunk = result.get("chunk", {})
                    if chunk.get("name"):
                        found_functions.append(
                            {
                                "name": chunk["name"],
                                "breadcrumb": chunk.get("breadcrumb", ""),
                                "chunk_type": chunk.get("chunk_type", ""),
                                "file_path": chunk.get("file_path", ""),
                                "confidence": result.get("score", 0.0),
                            }
                        )

                if found_functions:
                    alternatives = found_functions
                    suggestions.extend(
                        [
                            f"Found {len(found_functions)} similar functions in the project:",
                            "Consider using one of the alternatives provided.",
                            "Use the search tool with more specific keywords.",
                        ]
                    )
                    error_details["similar_functions_found"] = len(found_functions)
                else:
                    suggestions.extend(
                        [
                            "No similar functions found in the project.",
                            "Try searching with different keywords or descriptions.",
                            "Check if the function name is spelled correctly.",
                        ]
                    )
                    error_details["similar_functions_found"] = 0
            else:
                suggestions.extend(
                    [
                        "No search results found for the entry point.",
                        "Verify the function exists in the indexed project.",
                        "Try using broader search terms.",
                    ]
                )
                error_details["search_results_count"] = 0

        except Exception as e:
            logger.warning(f"Could not perform similarity search: {e}")
            suggestions.append("Could not perform similarity search. Check system status.")
            error_details["similarity_search_error"] = str(e)

        # Add error-type specific suggestions
        if error_type == "entry_point":
            suggestions.extend(
                [
                    "Entry point resolution failed. Try these approaches:",
                    "1. Use exact function names instead of descriptions",
                    "2. Include class names for methods (e.g., 'ClassName.method_name')",
                    "3. Check if the function is in a different module or namespace",
                    "4. Use the search tool to explore the codebase first",
                ]
            )

            # Add breadcrumb-specific suggestions if available
            if breadcrumb_result and hasattr(breadcrumb_result, "alternative_candidates"):
                if breadcrumb_result.alternative_candidates:
                    suggestions.append("Alternative candidates were found but didn't meet confidence threshold.")
                    for candidate in breadcrumb_result.alternative_candidates[:3]:  # Top 3
                        alternatives.append(
                            {
                                "breadcrumb": candidate.breadcrumb,
                                "confidence": candidate.confidence_score,
                                "reasoning": candidate.reasoning,
                                "match_type": candidate.match_type,
                            }
                        )

        elif error_type == "start_function":
            suggestions.extend(
                [
                    "Start function resolution failed. Try these approaches:",
                    "1. Use exact function names instead of descriptions",
                    "2. Include module/class paths for better specificity",
                    "3. Check if the function is in a different file or package",
                    "4. Search for similar function names in the project",
                ]
            )

        elif error_type == "end_function":
            suggestions.extend(
                [
                    "End function resolution failed. Try these approaches:",
                    "1. Use exact function names instead of descriptions",
                    "2. Include module/class paths for better specificity",
                    "3. Check if the function is in a different file or package",
                    "4. Search for similar function names in the project",
                ]
            )

        elif error_type == "no_paths_found":
            suggestions.extend(
                [
                    "No paths found between functions. This could mean:",
                    "1. Functions are not connected in the codebase",
                    "2. Connection requires deeper traversal than current depth",
                    "3. Path quality is below the threshold",
                    "4. Functions are in different modules/packages",
                    "5. Consider checking for indirect connections",
                ]
            )

        elif error_type == "general_error":
            suggestions.extend(
                [
                    "General error occurred. Try these steps:",
                    "1. Check if all required services are running",
                    "2. Verify the project is properly indexed",
                    "3. Try with a simpler entry point description",
                    "4. Check system logs for more details",
                ]
            )

            # Add specific suggestions based on error message
            if "timeout" in error_message.lower():
                suggestions.append("Request timed out. Try reducing search scope or depth.")
            elif "permission" in error_message.lower():
                suggestions.append("Permission error. Check file access permissions.")
            elif "connection" in error_message.lower():
                suggestions.append("Connection error. Check database connectivity.")

        # Add common troubleshooting suggestions
        suggestions.extend(
            [
                "Common troubleshooting steps:",
                "• Use 'search' tool to explore available functions",
                "• Check 'get_project_info' for project statistics",
                "• Try 'analyze_repository' to understand codebase structure",
                "• Use more specific or alternative keywords",
            ]
        )

        # Add performance suggestions if applicable
        if error_details.get("project_file_count", 0) > 10000:
            suggestions.append("Large project detected. Consider using more specific search terms.")

        return {"suggestions": suggestions, "error_details": error_details, "alternatives": alternatives}

    except Exception as e:
        logger.error(f"Error generating enhanced suggestions: {e}")
        return {
            "suggestions": [
                "Could not generate enhanced suggestions due to system error.",
                "Try basic troubleshooting steps:",
                "• Check if the project is indexed",
                "• Verify the function name is correct",
                "• Use the search tool to explore the codebase",
            ],
            "error_details": {"suggestion_generation_error": str(e)},
            "alternatives": [],
        }
