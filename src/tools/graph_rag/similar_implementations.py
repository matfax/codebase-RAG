"""Similar Implementations Graph RAG Tool

This module provides MCP tools for finding similar implementations
across projects using Graph RAG capabilities and cross-project search.
"""

import logging
from typing import Any, Optional

from src.models.code_chunk import ChunkType, CodeChunk
from src.services.cross_project_search_service import (
    CrossProjectSearchFilter,
    CrossProjectSearchService,
)
from src.services.embedding_service import EmbeddingService
from src.services.graph_rag_service import GraphRAGService
from src.services.implementation_chain_service import ImplementationChainService
from src.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


async def graph_find_similar_implementations(
    query: str,
    source_breadcrumb: str = None,
    source_project: str = None,
    target_projects: list[str] = None,
    exclude_projects: list[str] = None,
    chunk_types: list[str] = None,
    languages: list[str] = None,
    similarity_threshold: float = 0.7,
    structural_weight: float = 0.5,
    max_results: int = 10,
    include_implementation_chains: bool = False,
    include_architectural_context: bool = True,
) -> dict[str, Any]:
    """
    Find similar implementations across projects using Graph RAG capabilities.

    This tool leverages cross-project search and implementation chain analysis
    to find similar code implementations, patterns, and architectural solutions
    across multiple indexed projects.

    Args:
        query: Natural language description of what to search for
        source_breadcrumb: Optional specific breadcrumb to find similar implementations for
        source_project: Optional source project name (used with source_breadcrumb)
        target_projects: List of specific projects to search in (default: all projects)
        exclude_projects: List of projects to exclude from search
        chunk_types: List of chunk types to include ("function", "class", "method", etc.)
        languages: List of programming languages to include
        similarity_threshold: Minimum semantic similarity score (0.0-1.0, default: 0.7)
        structural_weight: Weight for structural vs semantic similarity (0.0-1.0, default: 0.5)
        max_results: Maximum number of similar implementations to return (1-50, default: 10)
        include_implementation_chains: Whether to include implementation chain analysis
        include_architectural_context: Whether to include architectural context analysis

    Returns:
        Dictionary containing similar implementations with similarity scores,
        architectural context, and optional implementation chains
    """
    try:
        logger.info(f"Starting similar implementations search for query: '{query}'")

        # Initialize core services
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()
        graph_rag_service = GraphRAGService(qdrant_service, embedding_service)
        cross_project_service = CrossProjectSearchService(qdrant_service, embedding_service, graph_rag_service)

        # Validate parameters
        if not query or not query.strip():
            return {"success": False, "error": "Query is required and cannot be empty"}

        max_results = max(1, min(max_results, 50))  # Clamp between 1 and 50
        similarity_threshold = max(0.0, min(similarity_threshold, 1.0))  # Clamp between 0 and 1
        structural_weight = max(0.0, min(structural_weight, 1.0))  # Clamp between 0 and 1

        # Convert string chunk types to ChunkType enum if provided
        chunk_type_enums = []
        if chunk_types:
            for chunk_type_str in chunk_types:
                try:
                    chunk_type_enums.append(ChunkType(chunk_type_str.lower()))
                except ValueError:
                    logger.warning(f"Invalid chunk type: {chunk_type_str}")

        # Build search filters
        search_filters = CrossProjectSearchFilter(
            target_projects=target_projects or [],
            exclude_projects=exclude_projects or [],
            chunk_types=chunk_type_enums,
            similarity_threshold=similarity_threshold,
            structural_weight=structural_weight,
            languages=languages or [],
        )

        # Initialize results structure
        results = {
            "success": True,
            "query": query,
            "search_filters": {
                "target_projects": target_projects,
                "exclude_projects": exclude_projects,
                "chunk_types": chunk_types,
                "languages": languages,
                "similarity_threshold": similarity_threshold,
                "structural_weight": structural_weight,
            },
            "max_results": max_results,
        }

        # Check for source-specific search
        if source_breadcrumb and source_project:
            # Find the source chunk first
            query_embedding = await embedding_service.generate_embeddings([source_breadcrumb]).__anext__()
            source_chunks = await qdrant_service.search_vectors(
                collection_name=f"project_{source_project}_code",
                query_vector=query_embedding,
                limit=5,
                score_threshold=0.8,
            )

            if not source_chunks:
                return {
                    "success": False,
                    "error": f"Could not find source breadcrumb '{source_breadcrumb}' in project '{source_project}'",
                    "source_breadcrumb": source_breadcrumb,
                    "source_project": source_project,
                }

            # Use the best matching source chunk
            source_chunk = source_chunks[0]
            search_result = await cross_project_service.find_similar_implementations(
                source_chunk=source_chunk, search_filters=search_filters, max_results=max_results
            )

            results["search_type"] = "source_based"
            results["source_breadcrumb"] = source_breadcrumb
            results["source_project"] = source_project
            results["source_chunk"] = {
                "breadcrumb": source_chunk.breadcrumb,
                "chunk_type": source_chunk.chunk_type.value if source_chunk.chunk_type else None,
                "file_path": source_chunk.file_path,
                "name": source_chunk.name,
            }
        else:
            # Perform general cross-project search
            search_result = await cross_project_service.search_across_projects(
                query=query, search_filters=search_filters, max_results=max_results
            )

            results["search_type"] = "query_based"

        # Process search results
        similar_implementations = []

        for match in search_result.matches:
            implementation = {
                "breadcrumb": match.chunk.breadcrumb,
                "project_name": match.project_name,
                "chunk_type": match.chunk.chunk_type.value if match.chunk.chunk_type else None,
                "file_path": match.chunk.file_path,
                "name": match.chunk.name,
                "similarity_score": match.similarity_score,
                "structural_score": match.structural_score,
                "combined_score": match.combined_score,
                "start_line": match.chunk.start_line,
                "end_line": match.chunk.end_line,
            }

            # Add content excerpt if available
            if hasattr(match.chunk, "content") and match.chunk.content:
                # Include first few lines as preview
                content_lines = match.chunk.content.strip().split("\n")
                preview_lines = content_lines[:5]  # First 5 lines
                if len(content_lines) > 5:
                    preview_lines.append("...")
                implementation["content_preview"] = "\n".join(preview_lines)

            # Add architectural context if requested
            if include_architectural_context:
                implementation["architectural_context"] = match.architectural_context
                implementation["related_components"] = [
                    {
                        "breadcrumb": comp.breadcrumb,
                        "chunk_type": comp.chunk_type,
                        "file_path": comp.file_path,
                    }
                    for comp in match.related_components
                ]
                implementation["usage_patterns"] = match.usage_patterns

            similar_implementations.append(implementation)

        results["similar_implementations"] = similar_implementations

        # Add implementation chain analysis if requested
        if include_implementation_chains and similar_implementations:
            try:
                impl_chain_service = ImplementationChainService(qdrant_service, embedding_service)

                # Analyze implementation chains for top results
                chain_analyses = []
                for impl in similar_implementations[:3]:  # Top 3 results
                    chain_result = await impl_chain_service.find_similar_implementation_patterns(
                        project_name=impl["project_name"], component_breadcrumb=impl["breadcrumb"], similarity_threshold=0.6, max_patterns=5
                    )

                    if chain_result and chain_result.get("similar_patterns"):
                        chain_analyses.append(
                            {
                                "implementation": impl["breadcrumb"],
                                "project": impl["project_name"],
                                "similar_patterns": chain_result["similar_patterns"][:3],  # Top 3 patterns
                            }
                        )

                results["implementation_chains"] = chain_analyses

            except Exception as e:
                logger.warning(f"Could not analyze implementation chains: {e}")
                results["implementation_chains"] = []

        # Add search statistics
        results["search_statistics"] = {
            "projects_searched": search_result.projects_searched,
            "total_chunks_examined": search_result.total_chunks_examined,
            "execution_time_ms": search_result.execution_time_ms,
            "semantic_weight_used": search_result.semantic_weight_used,
            "structural_weight_used": search_result.structural_weight_used,
            "results_count": len(similar_implementations),
        }

        logger.info(f"Found {len(similar_implementations)} similar implementations for query '{query}'")
        return results

    except Exception as e:
        error_msg = f"Error finding similar implementations for query '{query}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg, "query": query}
