"""Structure Analysis Graph RAG Tool

This module provides MCP tools for analyzing structural relationships
of specific breadcrumbs in the codebase using Graph RAG capabilities.
"""

import logging
from typing import Any, Optional

from src.services.embedding_service import EmbeddingService
from src.services.graph_analysis_report_service import (
    ReportType,
    get_graph_analysis_report_service,
)
from src.services.graph_performance_optimizer import (
    OptimizationStrategy,
    ProcessingPhase,
    ProgressUpdate,
    get_graph_performance_optimizer,
)
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
    generate_report: bool = False,
    include_recommendations: bool = True,
    enable_performance_optimization: bool = True,
    progress_callback: callable | None = None,
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
        generate_report: Whether to generate a comprehensive analysis report
        include_recommendations: Whether to include optimization recommendations in the report
        enable_performance_optimization: Whether to enable performance optimizations for large projects
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary containing structural analysis results with hierarchical relationships,
        connectivity patterns, and related components. If generate_report=True, includes
        a comprehensive report with statistics and recommendations.
    """
    try:
        logger.info(f"Starting structure analysis for breadcrumb '{breadcrumb}' in project '{project_name}'")

        # Initialize core services
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()
        graph_rag_service = GraphRAGService(qdrant_service, embedding_service)

        # Initialize performance optimizer if enabled
        performance_optimizer = None
        optimization_strategy = None
        if enable_performance_optimization:
            performance_optimizer = get_graph_performance_optimizer()
            if progress_callback:
                performance_optimizer.add_progress_callback(progress_callback)

        # Validate parameters
        if not breadcrumb or not breadcrumb.strip():
            return {"success": False, "error": "Breadcrumb is required and cannot be empty", "analysis_type": analysis_type}

        if not project_name or not project_name.strip():
            return {"success": False, "error": "Project name is required and cannot be empty", "analysis_type": analysis_type}

        max_depth = max(1, min(max_depth, 10))  # Clamp between 1 and 10

        # Build/get structure graph with optimization
        if performance_optimizer:
            await performance_optimizer._update_progress(
                phase=ProcessingPhase.GRAPH_BUILDING,
                current_step=0,
                total_steps=1,
                message="Building structure graph",
                elapsed_time_ms=0,
                items_processed=0,
            )

        structure_graph = await graph_rag_service.build_structure_graph(project_name=project_name, force_rebuild=force_rebuild_graph)

        # Determine optimization strategy based on graph size
        if performance_optimizer:
            optimization_strategy = performance_optimizer.determine_optimization_strategy(
                project_name=project_name,
                estimated_nodes=len(structure_graph.nodes),
                estimated_edges=len(structure_graph.edges),
            )
            logger.info(f"Using optimization strategy: {optimization_strategy.value}")

            # Optimize graph structure for large projects
            if optimization_strategy in [OptimizationStrategy.LARGE_PROJECT, OptimizationStrategy.ENTERPRISE_PROJECT]:
                structure_graph = await performance_optimizer._optimize_graph_structure(structure_graph, optimization_strategy)

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
            "performance_optimization_enabled": enable_performance_optimization,
            "optimization_strategy": optimization_strategy.value if optimization_strategy else None,
        }

        # Add performance metrics if optimizer was used
        if performance_optimizer:
            current_progress = performance_optimizer.get_current_progress()
            if current_progress:
                results["performance_metrics"] = {
                    "total_execution_time_ms": current_progress.elapsed_time_ms,
                    "memory_usage_mb": current_progress.memory_usage_mb,
                    "items_processed": current_progress.items_processed,
                    "processing_rate": current_progress.items_per_second,
                }

        # Generate comprehensive report if requested
        if generate_report:
            try:
                report_service = get_graph_analysis_report_service(graph_rag_service)
                analysis_report = await report_service.generate_structure_analysis_report(
                    breadcrumb=breadcrumb,
                    project_name=project_name,
                    analysis_type=analysis_type,
                    include_performance_insights=True,
                    include_optimization_suggestions=include_recommendations,
                )

                # Convert report to dictionary format
                results["comprehensive_report"] = {
                    "report_type": analysis_report.report_type.value,
                    "generated_at": analysis_report.generated_at.isoformat(),
                    "execution_time_ms": analysis_report.execution_time_ms,
                    "summary": analysis_report.summary,
                    "statistics": [
                        {
                            "metric_name": stat.metric_name,
                            "count": stat.count,
                            "mean": stat.mean,
                            "median": stat.median,
                            "std_dev": stat.std_dev,
                            "min_value": stat.min_value,
                            "max_value": stat.max_value,
                            "distribution": stat.distribution,
                        }
                        for stat in analysis_report.statistics
                    ],
                    "recommendations": [
                        {
                            "title": rec.title,
                            "description": rec.description,
                            "severity": rec.severity.value,
                            "category": rec.category,
                            "impact": rec.impact,
                            "suggested_actions": rec.suggested_actions,
                            "affected_components": rec.affected_components,
                            "confidence": rec.confidence,
                            "estimated_effort": rec.estimated_effort,
                        }
                        for rec in analysis_report.recommendations
                    ],
                    "insights": analysis_report.insights,
                    "structural_metrics": analysis_report.structural_metrics,
                    "performance_metrics": analysis_report.performance_metrics,
                    "confidence_score": analysis_report.confidence_score,
                    "data_quality_score": analysis_report.data_quality_score,
                }

                logger.info(f"Generated comprehensive report with {len(analysis_report.recommendations)} recommendations")

            except Exception as e:
                logger.warning(f"Could not generate comprehensive report: {e}")
                results["report_error"] = f"Report generation failed: {str(e)}"

        logger.info(f"Structure analysis completed successfully for '{breadcrumb}'")

        # Clean up performance optimizer callback
        if performance_optimizer and progress_callback:
            performance_optimizer.remove_progress_callback(progress_callback)

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
