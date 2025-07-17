"""Structure Analysis Graph RAG Tool

This module provides MCP tools for analyzing structural relationships
of specific breadcrumbs in the codebase using Graph RAG capabilities.
"""

import logging
from typing import Any, Optional

from src.services.embedding_service import EmbeddingService
from src.services.graph_rag_service import GraphRAGService
from src.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


async def graph_analyze_structure(
    breadcrumb: str,
    project_name: str,
    analysis_type: str = "comprehensive",
    max_depth: int = 3,
    include_siblings: bool = False,
    include_connectivity: bool = True,
    force_rebuild_graph: bool = False,
) -> dict[str, Any]:
    """
    Analyze the structural relationships of a specific breadcrumb in the codebase.

    This tool leverages Graph RAG capabilities to provide deep structural analysis
    of code components, including hierarchical relationships, connectivity patterns,
    and related components within the codebase structure.

    Args:
        breadcrumb: The breadcrumb path to analyze (e.g., "MyClass.method_name")
        project_name: Name of the project to analyze within
        analysis_type: Type of analysis ("comprehensive", "hierarchy", "connectivity", "overview")
        max_depth: Maximum depth for relationship traversal (1-10, default: 3)
        include_siblings: Whether to include sibling components in the analysis
        include_connectivity: Whether to analyze component connectivity patterns
        force_rebuild_graph: Whether to force rebuild the structure graph

    Returns:
        Dictionary containing structural analysis results with hierarchical relationships,
        connectivity patterns, and related components
    """
    try:
        logger.info(f"Starting structure analysis for breadcrumb '{breadcrumb}' in project '{project_name}'")

        # Initialize core services
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()
        graph_rag_service = GraphRAGService(qdrant_service, embedding_service)

        # Validate parameters
        if not breadcrumb or not breadcrumb.strip():
            return {"success": False, "error": "Breadcrumb is required and cannot be empty", "analysis_type": analysis_type}

        if not project_name or not project_name.strip():
            return {"success": False, "error": "Project name is required and cannot be empty", "analysis_type": analysis_type}

        max_depth = max(1, min(max_depth, 10))  # Clamp between 1 and 10

        # Build/get structure graph
        structure_graph = await graph_rag_service.build_structure_graph(project_name=project_name, force_rebuild=force_rebuild_graph)

        if not structure_graph or not structure_graph.nodes:
            return {
                "success": False,
                "error": f"No structure graph available for project '{project_name}'. Try indexing the project first.",
                "project_name": project_name,
                "analysis_type": analysis_type,
            }

        # Initialize results structure
        results = {
            "success": True,
            "breadcrumb": breadcrumb,
            "project_name": project_name,
            "analysis_type": analysis_type,
            "max_depth": max_depth,
            "graph_stats": {
                "total_nodes": len(structure_graph.nodes),
                "total_edges": len(structure_graph.edges),
                "force_rebuilt": force_rebuild_graph,
            },
        }

        # Core analysis: Component hierarchy
        if analysis_type in ["comprehensive", "hierarchy"]:
            hierarchy_result = await graph_rag_service.get_component_hierarchy(
                breadcrumb=breadcrumb, project_name=project_name, include_siblings=include_siblings
            )
            results["hierarchy"] = hierarchy_result

        # Connectivity analysis
        if analysis_type in ["comprehensive", "connectivity"] and include_connectivity:
            connectivity_result = await graph_rag_service.analyze_component_connectivity(breadcrumb=breadcrumb, project_name=project_name)
            results["connectivity"] = connectivity_result

        # Find related components
        if analysis_type in ["comprehensive", "overview"]:
            related_components_result = await graph_rag_service.find_related_components(
                breadcrumb=breadcrumb, project_name=project_name, max_depth=max_depth
            )
            related_components = related_components_result.related_components if related_components_result else []
            results["related_components"] = [
                {
                    "breadcrumb": comp.breadcrumb,
                    "chunk_type": comp.chunk_type,
                    "file_path": comp.file_path,
                    "similarity_score": getattr(comp, "similarity_score", 0.0),
                }
                for comp in related_components
            ]

        # Project structure overview for context
        if analysis_type == "overview":
            overview = await graph_rag_service.get_project_structure_overview(project_name)
            results["project_overview"] = overview

        # Find optimal navigation paths (if multiple related components found)
        if analysis_type == "comprehensive" and results.get("related_components") and len(results["related_components"]) > 1:
            try:
                # Find paths to top related components
                target_breadcrumbs = [comp["breadcrumb"] for comp in results["related_components"][:3] if comp["breadcrumb"] != breadcrumb]

                navigation_paths = []
                for target in target_breadcrumbs:
                    path = await graph_rag_service.find_hierarchical_path(
                        from_breadcrumb=breadcrumb, to_breadcrumb=target, project_name=project_name
                    )
                    if path:
                        navigation_paths.append({"target": target, "path": path, "path_length": len(path)})

                results["navigation_paths"] = navigation_paths

            except Exception as e:
                logger.warning(f"Could not generate navigation paths: {e}")
                results["navigation_paths"] = []

        # Add metadata
        results["metadata"] = {
            "analysis_completed_at": "now",
            "include_siblings": include_siblings,
            "include_connectivity": include_connectivity,
            "total_analysis_components": len(results.get("related_components", [])),
            "hierarchy_depth": len(results.get("hierarchy", {}).get("ancestors", [])),
        }

        logger.info(f"Structure analysis completed successfully for '{breadcrumb}'")
        return results

    except Exception as e:
        error_msg = f"Error analyzing structure for breadcrumb '{breadcrumb}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "breadcrumb": breadcrumb,
            "project_name": project_name,
            "analysis_type": analysis_type,
        }
