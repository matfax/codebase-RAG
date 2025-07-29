"""
Multi-Modal Search Tools for MCP Server

This module provides MCP tools for the LightRAG-inspired multi-modal retrieval system
with four distinct retrieval modes: Local, Global, Hybrid, and Mix.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ...models.query_features import QueryFeatures
from ...services.embedding_service import EmbeddingService
from ...services.multi_modal_retrieval_strategy import get_multi_modal_retrieval_strategy
from ...services.qdrant_service import QdrantService
from ...services.query_analyzer import get_query_analyzer
from ...services.retrieval_mode_performance_monitor import get_performance_monitor
from ..core.errors import SearchError, ValidationError
from ..core.performance_monitor import with_performance_monitoring
from ..project.project_utils import get_available_project_names

logger = logging.getLogger(__name__)


@with_performance_monitoring(timeout_seconds=20, tool_name="multi_modal_search")
async def multi_modal_search(
    query: str,
    n_results: int = 10,
    mode: str | None = None,
    target_projects: list[str] | None = None,
    cross_project: bool = False,
    enable_manual_mode_selection: bool = False,
    include_analysis: bool = True,
    include_performance_metrics: bool = False,
) -> dict[str, Any]:
    """
    Perform multi-modal search using LightRAG-inspired retrieval modes.

    This tool implements four distinct retrieval modes:
    - Local: Deep entity-focused retrieval using low-level keywords
    - Global: Broad relationship-focused retrieval using high-level keywords
    - Hybrid: Combined local+global with balanced context
    - Mix: Intelligent automatic mode selection based on query analysis

    Args:
        query: Natural language search query
        n_results: Number of results to return (1-50, default: 10)
        mode: Optional manual mode selection ('local', 'global', 'hybrid', 'mix')
        target_projects: List of specific project names to search in
        cross_project: Whether to search across all projects (default: False)
        enable_manual_mode_selection: Whether to allow manual mode override
        include_analysis: Whether to include query analysis in response
        include_performance_metrics: Whether to include performance metrics

    Returns:
        Dictionary containing search results with multi-modal metadata and analysis
    """
    try:
        # Input validation
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string", field_name="query", value=str(query))

        if not isinstance(n_results, int) or n_results < 1 or n_results > 50:
            raise ValidationError("n_results must be between 1 and 50", field_name="n_results", value=str(n_results))

        if mode and mode not in ["local", "global", "hybrid", "mix"]:
            raise ValidationError("mode must be one of: 'local', 'global', 'hybrid', 'mix'", field_name="mode", value=mode)

        logger.info(f"[MULTI_MODAL_START] Starting multi-modal search: '{query[:50]}...' with mode={mode or 'auto'}")

        # If mode is explicitly specified, enable manual mode selection
        effective_enable_manual = enable_manual_mode_selection or (mode is not None)

        # [DEBUG] Log multi-modal configuration
        multi_modal_config = {
            "requested_mode": mode,
            "enable_manual_mode_selection": enable_manual_mode_selection,
            "effective_enable_manual": effective_enable_manual,
            "include_analysis": include_analysis,
            "include_performance_metrics": include_performance_metrics,
            "target_projects": target_projects,
            "cross_project": cross_project,
            "n_results": n_results,
        }
        logger.debug(f"[MULTI_MODAL_CONFIG] {multi_modal_config}")

        # Determine target projects
        project_names = []
        if target_projects:
            # Validate project names
            all_collections = []  # This would come from Qdrant in real implementation
            try:
                from ...tools.indexing.search_tools import get_qdrant_client

                qdrant_client = get_qdrant_client()
                all_collections = [c.name for c in qdrant_client.get_collections().collections]
            except Exception as e:
                logger.warning(f"Could not get collections for project validation: {e}")

            available_projects = get_available_project_names(all_collections) if all_collections else target_projects

            # Filter to existing projects
            project_names = [p for p in target_projects if p in available_projects]
            if not project_names:
                return {
                    "error": f"No indexed collections found for projects: {target_projects}",
                    "available_projects": available_projects,
                    "query": query,
                    "results": [],
                    "total": 0,
                }
        elif cross_project:
            # Get all available projects
            try:
                from ...tools.indexing.search_tools import get_qdrant_client

                qdrant_client = get_qdrant_client()
                all_collections = [c.name for c in qdrant_client.get_collections().collections]
                project_names = get_available_project_names(all_collections)
            except Exception as e:
                logger.error(f"Error getting cross-project collections: {e}")
                return {
                    "error": f"Failed to get cross-project collections: {str(e)}",
                    "query": query,
                    "results": [],
                    "total": 0,
                }
        else:
            # Use current project
            from ...tools.indexing.search_tools import get_current_project

            current_project = get_current_project()
            if current_project:
                project_names = [current_project["name"]]
            else:
                return {
                    "error": "No current project detected and no target projects specified",
                    "query": query,
                    "results": [],
                    "total": 0,
                    "suggestion": "Use cross_project=True or specify target_projects",
                }

        if not project_names:
            return {
                "error": "No projects available for search",
                "query": query,
                "results": [],
                "total": 0,
            }

        # Initialize required services
        logger.debug("[SERVICE_INIT] Initializing services for multi-modal retrieval")

        try:
            # Initialize Qdrant service
            qdrant_service = QdrantService()
            logger.debug("[SERVICE_INIT] Qdrant service initialized")

            # Initialize embedding service
            embedding_service = EmbeddingService()
            logger.debug("[SERVICE_INIT] Embedding service initialized")

            # Get multi-modal retrieval service with proper dependencies
            retrieval_service = get_multi_modal_retrieval_strategy(qdrant_service=qdrant_service, embedding_service=embedding_service)
            logger.debug("[SERVICE_INIT] Multi-modal retrieval strategy initialized")

        except Exception as e:
            logger.error(f"[MULTI_MODAL_ERROR] Failed to initialize services: {e}")
            return {
                "error": f"Multi-modal service initialization failed: {str(e)}",
                "query": query,
                "results": [],
                "total": 0,
            }

        if retrieval_service is None:
            logger.error("[MULTI_MODAL_ERROR] Failed to initialize retrieval service")
            return {
                "error": "Multi-modal retrieval service initialization failed",
                "query": query,
                "results": [],
                "total": 0,
            }

        logger.info(
            f"[MULTI_MODAL_SEARCH] Executing multi-modal search - "
            f"projects: {project_names}, "
            f"mode: {mode or 'auto'}, "
            f"manual_selection: {effective_enable_manual}"
        )

        # Perform multi-modal search
        retrieval_result = await retrieval_service.search(
            query=query,
            project_names=project_names,
            mode=mode,
            n_results=n_results,
            enable_manual_mode_selection=effective_enable_manual,
        )

        logger.debug(
            f"[MULTI_MODAL_RESULT] Search completed - "
            f"mode_used: {getattr(retrieval_result, 'mode_used', 'unknown')}, "
            f"results: {getattr(retrieval_result, 'total_results', 0)}"
        )

        # Record performance metrics
        if include_performance_metrics:
            performance_monitor = get_performance_monitor()
            performance_monitor.record_query_result(retrieval_result)

        # Build response
        response = {
            "query": query,
            "mode_used": retrieval_result.mode_used,
            "results": retrieval_result.results,
            "total": retrieval_result.total_results,
            "projects_searched": project_names,
            "performance": {
                "total_execution_time_ms": retrieval_result.total_execution_time_ms,
                "query_analysis_time_ms": retrieval_result.query_analysis_time_ms,
                "retrieval_time_ms": retrieval_result.retrieval_time_ms,
                "post_processing_time_ms": retrieval_result.post_processing_time_ms,
                "average_confidence": retrieval_result.average_confidence,
                "result_diversity_score": retrieval_result.result_diversity_score,
            },
            "multi_modal_metadata": {
                "mode_selection": "manual" if (enable_manual_mode_selection and mode) else "automatic",
                "fallback_used": retrieval_result.fallback_used,
                "cache_hit": retrieval_result.cache_hit,
            },
        }

        # Add query analysis if requested
        if include_analysis:
            try:
                query_analyzer = await get_query_analyzer()
                query_features = await query_analyzer.analyze_query(query)

                response["query_analysis"] = {
                    "query_type": query_features.query_type.value,
                    "complexity": query_features.complexity.value,
                    "confidence_score": query_features.confidence_score,
                    "recommended_mode": query_features.recommended_mode,
                    "mode_confidence": query_features.mode_confidence,
                    "has_specific_entities": query_features.has_specific_entities,
                    "has_relationships": query_features.has_relationships,
                    "has_patterns": query_features.has_patterns,
                    "keywords": {
                        "entity_names": query_features.keywords.entity_names,
                        "concept_terms": query_features.keywords.concept_terms,
                        "relationship_indicators": query_features.keywords.relationship_indicators,
                        "high_level_keywords": query_features.keywords.high_level_keywords,
                        "low_level_keywords": query_features.keywords.low_level_keywords,
                    },
                    "context_hints": {
                        "language_hints": query_features.language_hints,
                        "framework_hints": query_features.framework_hints,
                        "domain_hints": query_features.domain_hints,
                    },
                }
            except Exception as e:
                logger.warning(f"Failed to add query analysis: {e}")
                response["query_analysis"] = {"error": str(e)}

        # Add performance metrics if requested
        if include_performance_metrics:
            try:
                performance_monitor = get_performance_monitor()
                mode_stats = performance_monitor.get_mode_statistics(retrieval_result.mode_used.split("(")[0])
                response["performance_context"] = mode_stats
            except Exception as e:
                logger.warning(f"Failed to add performance context: {e}")

        # Add error information if present
        if retrieval_result.error_message:
            response["error"] = retrieval_result.error_message

        logger.info(
            f"Multi-modal search completed: {retrieval_result.total_results} results "
            f"in {retrieval_result.total_execution_time_ms:.2f}ms using {retrieval_result.mode_used} mode"
        )

        return response

    except (ValidationError, SearchError) as e:
        logger.error(f"Multi-modal search failed with known error: {e}")
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "query": query,
            "results": [],
            "total": 0,
        }
    except Exception as e:
        logger.error(f"Multi-modal search failed with unexpected error: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "error_type": "UnexpectedError",
            "query": query,
            "results": [],
            "total": 0,
        }


async def analyze_query_features(query: str) -> dict[str, Any]:
    """
    Analyze query features and recommend optimal retrieval mode.

    This tool provides detailed analysis of a search query to understand
    its characteristics and recommend the best retrieval strategy.

    Args:
        query: The search query to analyze

    Returns:
        Dictionary containing comprehensive query analysis
    """
    try:
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string", field_name="query", value=str(query))

        logger.info(f"Analyzing query features: '{query[:50]}...'")

        # Get query analyzer
        query_analyzer = await get_query_analyzer()

        # Perform analysis
        features = await query_analyzer.analyze_query(query)

        # Build comprehensive response
        response = {
            "query": features.original_query,
            "normalized_query": features.normalized_query,
            "analysis": {
                "query_type": features.query_type.value,
                "complexity": features.complexity.value,
                "confidence_score": features.confidence_score,
                "query_length": features.query_length,
                "word_count": features.word_count,
            },
            "semantic_analysis": {
                "has_specific_entities": features.has_specific_entities,
                "has_relationships": features.has_relationships,
                "has_patterns": features.has_patterns,
                "has_implementation_focus": features.has_implementation_focus,
                "has_conceptual_focus": features.has_conceptual_focus,
            },
            "structural_indicators": {
                "mentions_functions": features.mentions_functions,
                "mentions_classes": features.mentions_classes,
                "mentions_files": features.mentions_files,
                "mentions_modules": features.mentions_modules,
                "mentions_patterns": features.mentions_patterns,
            },
            "keywords": {
                "entity_names": features.keywords.entity_names,
                "concept_terms": features.keywords.concept_terms,
                "technical_terms": features.keywords.technical_terms,
                "relationship_indicators": features.keywords.relationship_indicators,
                "high_level_keywords": features.keywords.high_level_keywords,
                "low_level_keywords": features.keywords.low_level_keywords,
                "all_keywords": features.keywords.get_all_keywords(),
            },
            "context_hints": {
                "language_hints": features.language_hints,
                "framework_hints": features.framework_hints,
                "domain_hints": features.domain_hints,
            },
            "recommendation": {
                "recommended_mode": features.recommended_mode,
                "mode_confidence": features.mode_confidence,
                "reasoning": _generate_mode_reasoning(features),
            },
            "metadata": {
                "processing_time_ms": features.processing_time_ms,
                "analysis_timestamp": features.analysis_timestamp,
            },
        }

        logger.info(f"Query analysis completed: type={features.query_type.value}, recommended_mode={features.recommended_mode}")

        return response

    except ValidationError as e:
        logger.error(f"Query analysis failed with validation error: {e}")
        return {
            "error": str(e),
            "error_type": "ValidationError",
            "query": query,
        }
    except Exception as e:
        logger.error(f"Query analysis failed with unexpected error: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "error_type": "UnexpectedError",
            "query": query,
        }


async def get_retrieval_mode_performance(
    mode: str | None = None,
    include_comparison: bool = True,
    include_alerts: bool = True,
    include_history: bool = False,
    history_limit: int = 50,
) -> dict[str, Any]:
    """
    Get performance metrics and analytics for retrieval modes.

    This tool provides comprehensive performance monitoring data
    for the multi-modal retrieval system.

    Args:
        mode: Specific mode to get metrics for ('local', 'global', 'hybrid', 'mix')
        include_comparison: Whether to include mode comparison
        include_alerts: Whether to include active alerts
        include_history: Whether to include query history
        history_limit: Limit for query history (default: 50)

    Returns:
        Dictionary containing performance metrics and analytics
    """
    try:
        logger.info(f"Getting retrieval mode performance for mode: {mode or 'all'}")

        # Get performance monitor
        performance_monitor = get_performance_monitor()

        # Build response
        response = {
            "timestamp": "2024-07-24T12:00:00",
            "metrics": {},
        }

        # Get mode statistics
        if mode:
            if mode not in ["local", "global", "hybrid", "mix"]:
                raise ValidationError("mode must be one of: 'local', 'global', 'hybrid', 'mix'", field_name="mode", value=mode)
            response["metrics"] = performance_monitor.get_mode_statistics(mode)
        else:
            response["metrics"] = performance_monitor.get_mode_statistics()

        # Add mode comparison
        if include_comparison:
            try:
                comparison = performance_monitor.compare_modes()
                response["comparison"] = {
                    "timestamp": comparison.timestamp,
                    "modes_compared": comparison.modes_compared,
                    "metrics": comparison.metrics_compared,
                    "best_performing_mode": comparison.best_performing_mode,
                    "recommendations": comparison.recommendations,
                }
            except Exception as e:
                logger.warning(f"Failed to include comparison: {e}")
                response["comparison"] = {"error": str(e)}

        # Add alerts
        if include_alerts:
            try:
                active_alerts = performance_monitor.get_active_alerts()
                response["alerts"] = {
                    "active_count": len(active_alerts),
                    "alerts": [
                        {
                            "id": alert.alert_id,
                            "mode": alert.mode_name,
                            "type": alert.alert_type,
                            "severity": alert.severity,
                            "message": alert.message,
                            "timestamp": alert.timestamp,
                            "metrics": alert.metrics,
                            "recommendations": alert.recommendations,
                        }
                        for alert in active_alerts
                    ],
                }
            except Exception as e:
                logger.warning(f"Failed to include alerts: {e}")
                response["alerts"] = {"error": str(e)}

        # Add query history
        if include_history:
            try:
                history = performance_monitor.get_query_history(mode=mode, limit=history_limit)
                response["history"] = {
                    "total_entries": len(history),
                    "entries": history,
                }
            except Exception as e:
                logger.warning(f"Failed to include history: {e}")
                response["history"] = {"error": str(e)}

        # Add summary
        all_stats = performance_monitor.get_mode_statistics()
        total_queries = sum(stats.get("total_queries", 0) for stats in all_stats.values())
        successful_queries = sum(stats.get("successful_queries", 0) for stats in all_stats.values())

        response["summary"] = {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "overall_success_rate": successful_queries / total_queries if total_queries > 0 else 0.0,
            "active_alerts": len(performance_monitor.get_active_alerts()),
            "monitoring_status": "active",
        }

        logger.info(
            f"Performance metrics retrieved: {total_queries} total queries, {len(performance_monitor.get_active_alerts())} active alerts"
        )

        return response

    except ValidationError as e:
        logger.error(f"Performance metrics request failed with validation error: {e}")
        return {
            "error": str(e),
            "error_type": "ValidationError",
        }
    except Exception as e:
        logger.error(f"Performance metrics request failed with unexpected error: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "error_type": "UnexpectedError",
        }


def _generate_mode_reasoning(features: QueryFeatures) -> str:
    """Generate human-readable reasoning for mode recommendation."""
    reasoning_parts = []

    # Analyze query type
    if features.query_type.value == "entity_focused":
        reasoning_parts.append("Query focuses on specific entities")
    elif features.query_type.value == "relationship_focused":
        reasoning_parts.append("Query emphasizes relationships and connections")
    elif features.query_type.value == "conceptual":
        reasoning_parts.append("Query is conceptual and high-level")

    # Analyze complexity
    if features.complexity.value == "simple":
        reasoning_parts.append("Simple query structure")
    elif features.complexity.value == "complex":
        reasoning_parts.append("Complex query with multiple concepts")
    elif features.complexity.value == "multi_faceted":
        reasoning_parts.append("Multi-faceted query requiring comprehensive approach")

    # Analyze specific characteristics
    if features.has_specific_entities:
        reasoning_parts.append("Contains specific entity references")

    if features.has_relationships:
        reasoning_parts.append("Includes relationship indicators")

    if features.has_patterns:
        reasoning_parts.append("Mentions architectural patterns")

    # Mode-specific reasoning
    if features.recommended_mode == "local":
        reasoning_parts.append("→ Local mode recommended for focused entity search")
    elif features.recommended_mode == "global":
        reasoning_parts.append("→ Global mode recommended for broad relationship exploration")
    elif features.recommended_mode == "hybrid":
        reasoning_parts.append("→ Hybrid mode recommended for balanced approach")
    elif features.recommended_mode == "mix":
        reasoning_parts.append("→ Mix mode recommended for adaptive strategy selection")

    return ". ".join(reasoning_parts) + "."


# Register tools for MCP
MULTI_MODAL_SEARCH_TOOLS = {
    "multi_modal_search": multi_modal_search,
    "analyze_query_features": analyze_query_features,
    "get_retrieval_mode_performance": get_retrieval_mode_performance,
}
