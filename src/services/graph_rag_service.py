"""
Graph RAG Service - Core controller for Graph RAG functionality.

This service acts as the main orchestrator for Graph RAG operations, providing:
- Structure relationship graph construction and analysis
- Hierarchical traversal algorithms for code navigation
- Caching mechanisms for performance optimization
- Deep integration with existing Qdrant vector database

Built on top of Wave 1's enhanced CodeChunk model and StructureAnalyzerService.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from src.models.code_chunk import ChunkType, CodeChunk
from src.models.file_metadata import FileMetadata

from .embedding_service import EmbeddingService
from .graph_rag_cache_service import get_graph_rag_cache_service
from .graph_traversal_algorithms import (
    ComponentCluster,
    GraphTraversalAlgorithms,
    RelationshipFilter,
    TraversalOptions,
    TraversalPath,
    TraversalStrategy,
)
from .qdrant_service import QdrantService
from .structure_analyzer_service import FileStructureContext, StructureAnalyzerService, get_structure_analyzer
from .structure_relationship_builder import (
    GraphEdge,
    GraphNode,
    StructureGraph,
    StructureRelationshipBuilder,
)


@dataclass
class GraphTraversalResult:
    """Result of graph traversal operations."""

    visited_nodes: list[GraphNode]
    path: list[str]  # breadcrumb path
    related_components: list[GraphNode]
    traversal_depth: int
    execution_time_ms: float
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GraphRAGService:
    """
    Core service for Graph RAG functionality.

    This service orchestrates the construction and analysis of code structure
    relationship graphs, providing hierarchical navigation and component discovery
    capabilities built on top of Wave 1's enhanced CodeChunk infrastructure.
    """

    def __init__(self, qdrant_service: QdrantService, embedding_service: EmbeddingService):
        """
        Initialize the Graph RAG service.

        Args:
            qdrant_service: Qdrant vector database service for data storage/retrieval
            embedding_service: Embedding service for semantic similarity
        """
        self.logger = logging.getLogger(__name__)

        # Core services
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.structure_analyzer = get_structure_analyzer()

        # Initialize relationship builder (implemented in task 2.2)
        self.relationship_builder = None

        # Initialize traversal algorithms (task 2.3)
        self.traversal_algorithms = GraphTraversalAlgorithms()

        # Initialize Graph RAG cache service (task 2.4)
        self.graph_cache_service = get_graph_rag_cache_service()

        # Cache for structure graphs (basic local cache)
        self._graph_cache: dict[str, StructureGraph] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl = 3600  # 1 hour default TTL

        # Performance metrics
        self._performance_stats = {
            "graphs_built": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "traversals_executed": 0,
            "avg_build_time_ms": 0.0,
            "avg_traversal_time_ms": 0.0,
        }

        self.logger.info("GraphRAGService initialized")

    async def build_structure_graph(self, project_name: str, force_rebuild: bool = False) -> StructureGraph:
        """
        Build or retrieve a code structure relationship graph for a project.

        Args:
            project_name: Name of the project to analyze
            force_rebuild: Force rebuilding even if cached version exists

        Returns:
            StructureGraph representing the project's code relationships
        """
        start_time = time.time()

        try:
            # Check cache first
            if not force_rebuild and project_name in self._graph_cache:
                cache_timestamp = self._cache_timestamps.get(project_name, 0)
                if time.time() - cache_timestamp < self._cache_ttl:
                    self._performance_stats["cache_hits"] += 1
                    self.logger.debug(f"Cache hit for project graph: {project_name}")
                    return self._graph_cache[project_name]

            # Cache miss - build new graph
            self._performance_stats["cache_misses"] += 1
            self.logger.info(f"Building structure graph for project: {project_name}")

            # Initialize relationship builder if needed
            if self.relationship_builder is None:
                self.relationship_builder = StructureRelationshipBuilder(self.qdrant_service, self.structure_analyzer)

            # Retrieve all code chunks for the project
            chunks = await self._get_project_chunks(project_name)
            if not chunks:
                self.logger.warning(f"No chunks found for project: {project_name}")
                return StructureGraph(nodes={}, edges=[], project_name=project_name)

            # Log project size for monitoring - now supporting full project processing
            self.logger.info(f"Processing {len(chunks)} chunks for project: {project_name}")

            # Build the graph structure
            graph = await self.relationship_builder.build_relationship_graph(chunks, project_name)

            # Cache the result
            self._graph_cache[project_name] = graph
            self._cache_timestamps[project_name] = time.time()

            # Update performance stats
            build_time = (time.time() - start_time) * 1000
            self._performance_stats["graphs_built"] += 1
            self._update_avg_build_time(build_time)

            self.logger.info(
                f"Built structure graph for {project_name}: "
                f"{len(graph.nodes)} nodes, {len(graph.edges)} edges "
                f"in {build_time:.2f}ms"
            )

            return graph

        except Exception as e:
            self.logger.error(f"Error building structure graph for {project_name}: {e}")
            raise

    async def find_related_components(
        self, breadcrumb: str, project_name: str, relationship_types: list[str] | None = None, max_depth: int = 3
    ) -> GraphTraversalResult:
        """
        Find components related to a given breadcrumb path.

        Args:
            breadcrumb: Starting breadcrumb path (e.g., "module.class.method")
            project_name: Project to search within
            relationship_types: Types of relationships to follow
            max_depth: Maximum traversal depth

        Returns:
            GraphTraversalResult with related components and traversal metadata
        """
        start_time = time.time()

        try:
            # Get or build the structure graph
            graph = await self.build_structure_graph(project_name)

            # Check if the starting breadcrumb exists
            if breadcrumb not in graph.nodes:
                self.logger.warning(f"Breadcrumb not found in graph: {breadcrumb}")
                return GraphTraversalResult(
                    visited_nodes=[],
                    path=[],
                    related_components=[],
                    traversal_depth=0,
                    execution_time_ms=0.0,
                    metadata={"error": "breadcrumb_not_found"},
                )

            # Perform graph traversal
            visited_nodes, path = await self._traverse_graph(graph, breadcrumb, relationship_types, max_depth)

            # Collect related components (excluding the starting node)
            related_components = [node for node in visited_nodes if node.breadcrumb != breadcrumb]

            # Calculate execution metrics
            execution_time = (time.time() - start_time) * 1000
            self._performance_stats["traversals_executed"] += 1
            self._update_avg_traversal_time(execution_time)

            result = GraphTraversalResult(
                visited_nodes=visited_nodes,
                path=path,
                related_components=related_components,
                traversal_depth=len(path) - 1 if path else 0,
                execution_time_ms=execution_time,
                metadata={
                    "starting_breadcrumb": breadcrumb,
                    "relationship_types_used": relationship_types or ["all"],
                    "max_depth_reached": len(path) >= max_depth,
                },
            )

            self.logger.debug(f"Found {len(related_components)} related components for {breadcrumb} " f"in {execution_time:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Error finding related components for {breadcrumb}: {e}")
            raise

    async def find_hierarchical_path(self, from_breadcrumb: str, to_breadcrumb: str, project_name: str) -> list[str] | None:
        """
        Find the hierarchical path between two breadcrumbs.

        Args:
            from_breadcrumb: Starting breadcrumb
            to_breadcrumb: Target breadcrumb
            project_name: Project to search within

        Returns:
            List of breadcrumbs representing the path, or None if no path exists
        """
        try:
            graph = await self.build_structure_graph(project_name)

            # Check if both breadcrumbs exist
            if from_breadcrumb not in graph.nodes or to_breadcrumb not in graph.nodes:
                return None

            # Use breadth-first search to find shortest path
            return await self._find_shortest_path(graph, from_breadcrumb, to_breadcrumb)

        except Exception as e:
            self.logger.error(f"Error finding hierarchical path: {e}")
            return None

    async def get_component_hierarchy(self, breadcrumb: str, project_name: str, include_siblings: bool = False) -> dict[str, Any]:
        """
        Get the complete hierarchy context for a component.

        Args:
            breadcrumb: Component breadcrumb to analyze
            project_name: Project name
            include_siblings: Include sibling components at each level

        Returns:
            Dictionary with hierarchy information including parents, children, and optionally siblings
        """
        try:
            graph = await self.build_structure_graph(project_name)

            if breadcrumb not in graph.nodes:
                return {"error": "breadcrumb_not_found"}

            node = graph.nodes[breadcrumb]

            # Build hierarchy context
            hierarchy = {
                "current": {"breadcrumb": breadcrumb, "name": node.name, "type": node.chunk_type.value, "depth": node.depth},
                "ancestors": await self._get_ancestors(graph, breadcrumb),
                "descendants": await self._get_descendants(graph, breadcrumb),
            }

            if include_siblings:
                hierarchy["siblings"] = await self._get_siblings(graph, breadcrumb)

            return hierarchy

        except Exception as e:
            self.logger.error(f"Error getting component hierarchy: {e}")
            return {"error": str(e)}

    async def invalidate_cache(self, project_name: str | None = None):
        """
        Invalidate cached structure graphs and related data.

        Args:
            project_name: Specific project to invalidate, or None for all projects
        """
        if project_name:
            # Invalidate local graph cache
            self._graph_cache.pop(project_name, None)
            self._cache_timestamps.pop(project_name, None)

            # Invalidate Graph RAG cache service
            await self.graph_cache_service.invalidate_project_cache(project_name)

            self.logger.info(f"Invalidated cache for project: {project_name}")
        else:
            # Clear local cache
            self._graph_cache.clear()
            self._cache_timestamps.clear()

            # Note: Graph RAG cache service doesn't have global clear method for safety
            # Individual project caches should be cleared as needed

            self.logger.info("Invalidated all graph caches")

    def configure_function_call_detection(self, enable: bool = True, confidence_threshold: float = 0.5, invalidate_cache: bool = True):
        """
        Configure function call detection settings for the Graph RAG service.

        Args:
            enable: Whether to enable function call detection in graph building
            confidence_threshold: Minimum confidence threshold for function call edges
            invalidate_cache: Whether to invalidate existing graph caches after configuration change
        """
        # Initialize relationship builder if not already done
        if self.relationship_builder is None:
            self.relationship_builder = StructureRelationshipBuilder(self.qdrant_service, self.structure_analyzer)

        # Configure the relationship builder
        self.relationship_builder.configure_function_call_detection(enable, confidence_threshold)

        # Invalidate caches if requested (since changing configuration affects graph building)
        if invalidate_cache:
            self._graph_cache.clear()
            self._cache_timestamps.clear()
            self.logger.info("Graph caches invalidated due to function call detection configuration change")

        self.logger.info(f"Function call detection configured: enabled={enable}, " f"confidence_threshold={confidence_threshold}")

    def get_function_call_detection_config(self) -> dict[str, any]:
        """
        Get current function call detection configuration.

        Returns:
            Dictionary with current configuration settings
        """
        if self.relationship_builder is None:
            # Return default configuration
            return {"enabled": True, "confidence_threshold": 0.5, "status": "not_initialized"}

        return {
            "enabled": self.relationship_builder.enable_function_call_detection,
            "confidence_threshold": self.relationship_builder.function_call_confidence_threshold,
            "status": "configured",
        }

    async def advanced_component_search(
        self,
        breadcrumb: str,
        project_name: str,
        strategy: TraversalStrategy = TraversalStrategy.DEPTH_FIRST,
        relationship_filter: RelationshipFilter = RelationshipFilter.ALL,
        max_depth: int = 3,
        max_nodes: int = 50,
    ) -> GraphTraversalResult:
        """
        Perform advanced component search using sophisticated traversal algorithms.

        Args:
            breadcrumb: Starting breadcrumb for search
            project_name: Project to search within
            strategy: Traversal strategy to use
            relationship_filter: Filter for relationship types
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to visit

        Returns:
            GraphTraversalResult with detailed search results
        """
        try:
            # Configure traversal options
            options = TraversalOptions(
                strategy=strategy,
                relationship_filter=relationship_filter,
                max_depth=max_depth,
                max_nodes=max_nodes,
                confidence_threshold=0.5,
            )

            # Check cache first
            cached_result = await self.graph_cache_service.get_traversal_result(breadcrumb, project_name, options)

            if cached_result is not None:
                visited_nodes, path, metadata = cached_result
                metadata["cache_hit"] = True
            else:
                # Get or build the structure graph
                graph = await self.build_structure_graph(project_name)

                # Perform advanced traversal
                visited_nodes, path, metadata = await self.traversal_algorithms.advanced_traversal(graph, breadcrumb, options)

                metadata["cache_hit"] = False

                # Cache the result
                await self.graph_cache_service.cache_traversal_result(breadcrumb, project_name, options, (visited_nodes, path, metadata))

            # Create result object
            related_components = [node for node in visited_nodes if node.breadcrumb != breadcrumb]

            result = GraphTraversalResult(
                visited_nodes=visited_nodes,
                path=path,
                related_components=related_components,
                traversal_depth=metadata.get("path_length", 0),
                execution_time_ms=metadata.get("execution_time_ms", 0.0),
                metadata={
                    **metadata,
                    "starting_breadcrumb": breadcrumb,
                    "strategy_used": strategy.value,
                    "relationship_filter": relationship_filter.value,
                },
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in advanced component search: {e}")
            raise

    async def discover_component_clusters(
        self, project_name: str, cluster_threshold: float = 0.7, min_cluster_size: int = 3, max_clusters: int = 10
    ) -> list[ComponentCluster]:
        """
        Discover clusters of highly related components within a project.

        Args:
            project_name: Project to analyze
            cluster_threshold: Minimum relationship strength for clustering
            min_cluster_size: Minimum number of nodes in a cluster
            max_clusters: Maximum number of clusters to return

        Returns:
            List of identified component clusters
        """
        try:
            # Check cache first
            cached_clusters = await self.graph_cache_service.get_component_clusters(
                project_name, cluster_threshold, min_cluster_size, max_clusters
            )

            if cached_clusters is not None:
                self.logger.debug(f"Cache hit for component clusters in {project_name}")
                return cached_clusters

            # Cache miss - compute clusters
            graph = await self.build_structure_graph(project_name)

            clusters = await self.traversal_algorithms.find_component_clusters(graph, cluster_threshold, min_cluster_size, max_clusters)

            # Cache the result
            await self.graph_cache_service.cache_component_clusters(
                project_name, cluster_threshold, min_cluster_size, max_clusters, clusters
            )

            self.logger.info(f"Discovered {len(clusters)} component clusters in {project_name}")
            return clusters

        except Exception as e:
            self.logger.error(f"Error discovering component clusters: {e}")
            return []

    async def find_optimal_navigation_paths(
        self, from_breadcrumb: str, to_breadcrumb: str, project_name: str, max_paths: int = 5
    ) -> list[TraversalPath]:
        """
        Find multiple optimal paths for navigating between components.

        Args:
            from_breadcrumb: Starting component
            to_breadcrumb: Target component
            project_name: Project to search within
            max_paths: Maximum number of paths to return

        Returns:
            List of optimal navigation paths
        """
        try:
            graph = await self.build_structure_graph(project_name)

            paths = await self.traversal_algorithms.find_optimal_paths(graph, from_breadcrumb, to_breadcrumb, max_paths)

            self.logger.debug(f"Found {len(paths)} optimal paths from {from_breadcrumb} to {to_breadcrumb}")
            return paths

        except Exception as e:
            self.logger.error(f"Error finding optimal navigation paths: {e}")
            return []

    async def analyze_component_connectivity(self, breadcrumb: str, project_name: str) -> dict[str, Any]:
        """
        Analyze the connectivity patterns and influence of a specific component.

        Args:
            breadcrumb: Component to analyze
            project_name: Project name

        Returns:
            Dictionary with detailed connectivity analysis
        """
        try:
            # Check cache first
            cached_analysis = await self.graph_cache_service.get_connectivity_analysis(breadcrumb, project_name)

            if cached_analysis is not None:
                self.logger.debug(f"Cache hit for connectivity analysis: {breadcrumb}")
                return cached_analysis

            # Cache miss - compute analysis
            graph = await self.build_structure_graph(project_name)

            analysis = await self.traversal_algorithms.analyze_connectivity(graph, breadcrumb)

            # Cache the result
            await self.graph_cache_service.cache_connectivity_analysis(breadcrumb, project_name, analysis)

            self.logger.debug(f"Analyzed connectivity for {breadcrumb}")
            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing component connectivity: {e}")
            return {"error": str(e)}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the Graph RAG service."""
        stats = self._performance_stats.copy()

        # Include traversal algorithm statistics
        if self.traversal_algorithms:
            stats["traversal_algorithms"] = self.traversal_algorithms.get_traversal_statistics()

        # Include Graph RAG cache statistics
        if self.graph_cache_service:
            stats["graph_rag_cache"] = self.graph_cache_service.get_cache_statistics()

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_stats = {
            "graphs_built": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "traversals_executed": 0,
            "avg_build_time_ms": 0.0,
            "avg_traversal_time_ms": 0.0,
        }

        # Reset traversal algorithm statistics
        if self.traversal_algorithms:
            self.traversal_algorithms.reset_statistics()

        # Reset Graph RAG cache statistics
        if self.graph_cache_service:
            self.graph_cache_service.reset_statistics()

        self.logger.info("Performance statistics reset")

    # =================== Private Helper Methods ===================

    async def _get_project_chunks(self, project_name: str) -> list[CodeChunk]:
        """
        Retrieve all code chunks for a project from Qdrant.

        Args:
            project_name: Project name to query

        Returns:
            List of CodeChunk objects for the project
        """
        try:
            # Initialize Qdrant service connection
            await self.qdrant_service._initialize_cache()

            chunks = []

            # Query all code collections for the project
            collection_names = [f"project_{project_name}_code", f"project_{project_name}_config", f"project_{project_name}_documentation"]

            for collection_name in collection_names:
                try:
                    # Use scroll to get all points from the collection
                    scroll_result = self.qdrant_service.client.scroll(
                        collection_name=collection_name,
                        limit=10000,  # Large limit to get all chunks
                        with_payload=True,
                        with_vectors=False,  # We don't need vectors for graph building
                    )

                    points = scroll_result[0]  # First element is the list of points

                    for point in points:
                        payload = point.payload

                        # Skip chunks without breadcrumbs (not useful for graph building)
                        if not payload.get("breadcrumb"):
                            continue

                        # Convert Qdrant point to CodeChunk
                        chunk = self._point_payload_to_code_chunk(payload)
                        if chunk:
                            chunks.append(chunk)

                except Exception as e:
                    # Collection might not exist - this is OK
                    self.logger.debug(f"Could not query collection {collection_name}: {e}")
                    continue

            self.logger.info(f"Retrieved {len(chunks)} code chunks for project {project_name}")
            return chunks

        except Exception as e:
            self.logger.error(f"Error retrieving project chunks: {e}")
            return []

    def _point_payload_to_code_chunk(self, payload: dict[str, Any]) -> CodeChunk | None:
        """
        Convert a Qdrant point payload to a CodeChunk object.

        Args:
            payload: Qdrant point payload

        Returns:
            CodeChunk object or None if conversion fails
        """
        try:
            from datetime import datetime

            from ..models.code_chunk import ChunkType

            # Extract required fields
            chunk_id = payload.get("chunk_id")
            file_path = payload.get("file_path")
            content = payload.get("content")

            if not all([chunk_id, file_path, content]):
                return None

            # Parse chunk type
            chunk_type_str = payload.get("chunk_type", "raw_code")
            try:
                chunk_type = ChunkType(chunk_type_str)
            except ValueError:
                chunk_type = ChunkType.RAW_CODE

            # Parse datetime fields
            indexed_at = None
            indexed_at_str = payload.get("indexed_at")
            if indexed_at_str:
                try:
                    indexed_at = datetime.fromisoformat(indexed_at_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

            # Create CodeChunk object
            chunk = CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=content,
                chunk_type=chunk_type,
                language=payload.get("language", "unknown"),
                start_line=payload.get("line_start", 0),
                end_line=payload.get("line_end", 0),
                start_byte=payload.get("start_byte", 0),
                end_byte=payload.get("end_byte", 0),
                name=payload.get("name"),
                parent_name=payload.get("parent_name"),
                signature=payload.get("signature"),
                docstring=payload.get("docstring"),
                breadcrumb=payload.get("breadcrumb"),
                context_before=payload.get("context_before"),
                context_after=payload.get("context_after"),
                content_hash=payload.get("content_hash"),
                embedding_text=payload.get("embedding_text"),
                indexed_at=indexed_at,
                tags=payload.get("tags", []),
                complexity_score=payload.get("complexity_score"),
                dependencies=payload.get("dependencies", []),
                access_modifier=payload.get("access_modifier"),
                imports_used=payload.get("imports_used", []),
                has_syntax_errors=payload.get("has_syntax_errors", False),
                error_details=payload.get("error_details"),
            )

            return chunk

        except Exception as e:
            self.logger.error(f"Error converting point payload to CodeChunk: {e}")
            return None

    async def semantic_component_search(
        self,
        query_text: str,
        project_name: str,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        include_graph_context: bool = True,
    ) -> dict[str, Any]:
        """
        Perform semantic search enhanced with graph context.

        Args:
            query_text: Natural language query
            project_name: Project to search within
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity score
            include_graph_context: Include related components from graph

        Returns:
            Dictionary with search results and graph context
        """
        try:
            # Generate embedding for query
            query_embedding_tensor = await self.embedding_service.generate_embeddings("nomic-embed-text", query_text)
            if query_embedding_tensor is None:
                return {"error": "Failed to generate query embedding"}

            # Convert tensor to list - ensure proper conversion
            if hasattr(query_embedding_tensor, "tolist"):
                query_embedding = query_embedding_tensor.tolist()
            elif hasattr(query_embedding_tensor, "numpy"):
                query_embedding = query_embedding_tensor.numpy().tolist()
            else:
                return {"error": f"Unexpected embedding type: {type(query_embedding_tensor)}"}

            # Validate embedding format and dimensions
            if not isinstance(query_embedding, list) or len(query_embedding) == 0:
                return {
                    "error": f"Generated query embedding is invalid: type={type(query_embedding)}, "
                    f"len={len(query_embedding) if hasattr(query_embedding, '__len__') else 'N/A'}"
                }

            if len(query_embedding) != 768:
                return {"error": f"Query embedding dimension mismatch: expected 768, got {len(query_embedding)}"}

            self.logger.info(f"Successfully generated query embedding: {len(query_embedding)} dimensions")

            # Search in Qdrant collections
            search_results = []
            collection_names = [f"project_{project_name}_code"]

            for collection_name in collection_names:
                try:
                    # Perform vector search
                    results = self.qdrant_service.client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding[0],
                        limit=max_results,
                        score_threshold=similarity_threshold,
                        with_payload=True,
                        with_vectors=False,
                    )

                    for result in results:
                        payload = result.payload
                        search_results.append(
                            {
                                "score": result.score,
                                "breadcrumb": payload.get("breadcrumb"),
                                "name": payload.get("name"),
                                "chunk_type": payload.get("chunk_type"),
                                "file_path": payload.get("file_path"),
                                "content": payload.get("content", "")[:500],  # Truncate for display
                                "docstring": payload.get("docstring"),
                                "signature": payload.get("signature"),
                            }
                        )

                except Exception as e:
                    self.logger.debug(f"Could not search collection {collection_name}: {e}")
                    continue

            # Sort by similarity score
            search_results.sort(key=lambda x: x["score"], reverse=True)

            # Add graph context if requested
            graph_context = {}
            if include_graph_context and search_results:
                graph = await self.build_structure_graph(project_name)

                for result in search_results[:5]:  # Get context for top 5 results
                    breadcrumb = result.get("breadcrumb")
                    if breadcrumb and breadcrumb in graph.nodes:
                        # Get hierarchical context
                        context = await self.get_component_hierarchy(breadcrumb, project_name, include_siblings=True)
                        graph_context[breadcrumb] = context

            return {
                "query": query_text,
                "total_results": len(search_results),
                "results": search_results[:max_results],
                "graph_context": graph_context,
                "project_name": project_name,
            }

        except Exception as e:
            self.logger.error(f"Error in semantic component search: {e}")
            return {"error": str(e)}

    async def find_components_by_breadcrumb_pattern(
        self, breadcrumb_pattern: str, project_name: str, exact_match: bool = False
    ) -> list[dict[str, Any]]:
        """
        Find components matching a breadcrumb pattern.

        Args:
            breadcrumb_pattern: Pattern to match (supports wildcards if not exact)
            project_name: Project to search within
            exact_match: Whether to use exact matching

        Returns:
            List of matching components with their metadata
        """
        try:
            chunks = await self._get_project_chunks(project_name)
            matches = []

            for chunk in chunks:
                if not chunk.breadcrumb:
                    continue

                is_match = False
                if exact_match:
                    is_match = chunk.breadcrumb == breadcrumb_pattern
                else:
                    # Simple wildcard matching
                    if "*" in breadcrumb_pattern:
                        pattern_parts = breadcrumb_pattern.split("*")
                        breadcrumb_lower = chunk.breadcrumb.lower()
                        pattern_lower = breadcrumb_pattern.lower()

                        if len(pattern_parts) == 2:  # Simple prefix*suffix pattern
                            prefix, suffix = pattern_parts
                            is_match = breadcrumb_lower.startswith(prefix.lower()) and breadcrumb_lower.endswith(suffix.lower())
                        else:
                            # Contains pattern
                            is_match = pattern_lower.replace("*", "") in breadcrumb_lower
                    else:
                        # Substring match
                        is_match = breadcrumb_pattern.lower() in chunk.breadcrumb.lower()

                if is_match:
                    matches.append(
                        {
                            "breadcrumb": chunk.breadcrumb,
                            "name": chunk.name,
                            "chunk_type": chunk.chunk_type.value,
                            "file_path": chunk.file_path,
                            "parent_name": chunk.parent_name,
                            "signature": chunk.signature,
                            "docstring": chunk.docstring,
                            "line_range": f"{chunk.start_line}-{chunk.end_line}",
                        }
                    )

            # Sort by breadcrumb
            matches.sort(key=lambda x: x["breadcrumb"])

            self.logger.debug(f"Found {len(matches)} components matching pattern: {breadcrumb_pattern}")
            return matches

        except Exception as e:
            self.logger.error(f"Error finding components by breadcrumb pattern: {e}")
            return []

    async def get_project_structure_overview(self, project_name: str) -> dict[str, Any]:
        """
        Get high-level overview of project structure.

        Args:
            project_name: Project to analyze

        Returns:
            Dictionary with project structure overview
        """
        try:
            graph = await self.build_structure_graph(project_name)

            # Calculate structure metrics
            total_nodes = len(graph.nodes)
            total_edges = len(graph.edges)

            # Group by chunk type
            type_breakdown = {}
            depth_distribution = {}
            language_breakdown = {}

            for node in graph.nodes.values():
                # Type breakdown
                chunk_type = node.chunk_type.value
                if chunk_type not in type_breakdown:
                    type_breakdown[chunk_type] = 0
                type_breakdown[chunk_type] += 1

                # Depth distribution
                depth = node.depth
                if depth not in depth_distribution:
                    depth_distribution[depth] = 0
                depth_distribution[depth] += 1

                # Language breakdown (extract from file path)
                file_ext = node.file_path.split(".")[-1] if "." in node.file_path else "unknown"
                if file_ext not in language_breakdown:
                    language_breakdown[file_ext] = 0
                language_breakdown[file_ext] += 1

            # Relationship type breakdown
            relationship_breakdown = {}
            for edge in graph.edges:
                rel_type = edge.relationship_type
                if rel_type not in relationship_breakdown:
                    relationship_breakdown[rel_type] = 0
                relationship_breakdown[rel_type] += 1

            # Find largest components (by children count)
            largest_components = []
            for breadcrumb, node in graph.nodes.items():
                if len(node.children_breadcrumbs) > 2:  # Has multiple children
                    largest_components.append(
                        {
                            "breadcrumb": breadcrumb,
                            "name": node.name,
                            "type": node.chunk_type.value,
                            "children_count": len(node.children_breadcrumbs),
                            "depth": node.depth,
                        }
                    )

            largest_components.sort(key=lambda x: x["children_count"], reverse=True)

            overview = {
                "project_name": project_name,
                "total_components": total_nodes,
                "total_relationships": total_edges,
                "root_components": len(graph.root_nodes),
                "max_depth": max(depth_distribution.keys()) if depth_distribution else 0,
                "breakdown": {
                    "by_type": type_breakdown,
                    "by_depth": depth_distribution,
                    "by_language": language_breakdown,
                    "by_relationship": relationship_breakdown,
                },
                "largest_components": largest_components[:10],  # Top 10
                "structure_health": {
                    "orphaned_nodes": sum(
                        1 for node in graph.nodes.values() if not node.parent_breadcrumb and node.breadcrumb not in graph.root_nodes
                    ),
                    "average_children_per_node": (
                        sum(len(node.children_breadcrumbs) for node in graph.nodes.values()) / total_nodes if total_nodes > 0 else 0
                    ),
                    "relationship_density": total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0,
                },
            }

            return overview

        except Exception as e:
            self.logger.error(f"Error getting project structure overview: {e}")
            return {"error": str(e)}

    async def _traverse_graph(
        self, graph: StructureGraph, start_breadcrumb: str, relationship_types: list[str] | None, max_depth: int
    ) -> tuple[list[GraphNode], list[str]]:
        """
        Perform depth-first traversal of the structure graph.

        Args:
            graph: Structure graph to traverse
            start_breadcrumb: Starting point for traversal
            relationship_types: Types of relationships to follow
            max_depth: Maximum depth to traverse

        Returns:
            Tuple of (visited_nodes, breadcrumb_path)
        """
        visited = set()
        visited_nodes = []
        path = []

        async def dfs(breadcrumb: str, depth: int):
            if depth > max_depth or breadcrumb in visited:
                return

            visited.add(breadcrumb)
            if breadcrumb in graph.nodes:
                visited_nodes.append(graph.nodes[breadcrumb])
                path.append(breadcrumb)

            # Follow edges based on relationship types
            for edge in graph.edges:
                if edge.source_breadcrumb == breadcrumb:
                    if not relationship_types or edge.relationship_type in relationship_types:
                        await dfs(edge.target_breadcrumb, depth + 1)

        await dfs(start_breadcrumb, 0)
        return visited_nodes, path

    async def _find_shortest_path(self, graph: StructureGraph, from_breadcrumb: str, to_breadcrumb: str) -> list[str] | None:
        """Find shortest path between two breadcrumbs using BFS."""
        if from_breadcrumb == to_breadcrumb:
            return [from_breadcrumb]

        queue = [(from_breadcrumb, [from_breadcrumb])]
        visited = {from_breadcrumb}

        while queue:
            current, path = queue.pop(0)

            # Check all outgoing edges
            for edge in graph.edges:
                if edge.source_breadcrumb == current:
                    next_breadcrumb = edge.target_breadcrumb

                    if next_breadcrumb == to_breadcrumb:
                        return path + [next_breadcrumb]

                    if next_breadcrumb not in visited:
                        visited.add(next_breadcrumb)
                        queue.append((next_breadcrumb, path + [next_breadcrumb]))

        return None  # No path found

    async def _get_ancestors(self, graph: StructureGraph, breadcrumb: str) -> list[dict[str, Any]]:
        """Get all ancestor nodes in the hierarchy."""
        ancestors = []
        current_node = graph.nodes.get(breadcrumb)

        while current_node and current_node.parent_breadcrumb:
            parent = graph.nodes.get(current_node.parent_breadcrumb)
            if parent:
                ancestors.append({"breadcrumb": parent.breadcrumb, "name": parent.name, "type": parent.chunk_type.value})
                current_node = parent
            else:
                break

        return ancestors

    async def _get_descendants(self, graph: StructureGraph, breadcrumb: str) -> list[dict[str, Any]]:
        """Get all descendant nodes in the hierarchy."""
        descendants = []
        node = graph.nodes.get(breadcrumb)

        if node:
            for child_breadcrumb in node.children_breadcrumbs:
                child = graph.nodes.get(child_breadcrumb)
                if child:
                    descendants.append({"breadcrumb": child.breadcrumb, "name": child.name, "type": child.chunk_type.value})

        return descendants

    async def _get_siblings(self, graph: StructureGraph, breadcrumb: str) -> list[dict[str, Any]]:
        """Get sibling nodes at the same hierarchy level."""
        siblings = []
        node = graph.nodes.get(breadcrumb)

        if node and node.parent_breadcrumb:
            parent = graph.nodes.get(node.parent_breadcrumb)
            if parent:
                for sibling_breadcrumb in parent.children_breadcrumbs:
                    if sibling_breadcrumb != breadcrumb:
                        sibling = graph.nodes.get(sibling_breadcrumb)
                        if sibling:
                            siblings.append({"breadcrumb": sibling.breadcrumb, "name": sibling.name, "type": sibling.chunk_type.value})

        return siblings

    def _update_avg_build_time(self, build_time_ms: float):
        """Update average build time with new measurement."""
        current_avg = self._performance_stats["avg_build_time_ms"]
        graphs_built = self._performance_stats["graphs_built"]

        if graphs_built == 1:
            self._performance_stats["avg_build_time_ms"] = build_time_ms
        else:
            # Running average calculation
            self._performance_stats["avg_build_time_ms"] = (current_avg * (graphs_built - 1) + build_time_ms) / graphs_built

    def _update_avg_traversal_time(self, traversal_time_ms: float):
        """Update average traversal time with new measurement."""
        current_avg = self._performance_stats["avg_traversal_time_ms"]
        traversals = self._performance_stats["traversals_executed"]

        if traversals == 1:
            self._performance_stats["avg_traversal_time_ms"] = traversal_time_ms
        else:
            # Running average calculation
            self._performance_stats["avg_traversal_time_ms"] = (current_avg * (traversals - 1) + traversal_time_ms) / traversals


# Singleton instance for global access
_graph_rag_service_instance: GraphRAGService | None = None


def get_graph_rag_service(
    qdrant_service: QdrantService | None = None, embedding_service: EmbeddingService | None = None
) -> GraphRAGService:
    """
    Get the global Graph RAG service instance.

    Args:
        qdrant_service: Optional QdrantService instance (required for first call)
        embedding_service: Optional EmbeddingService instance (required for first call)

    Returns:
        GraphRAGService singleton instance
    """
    global _graph_rag_service_instance

    if _graph_rag_service_instance is None:
        # Create services if not provided
        if qdrant_service is None:
            qdrant_service = QdrantService()
        if embedding_service is None:
            embedding_service = EmbeddingService()
        _graph_rag_service_instance = GraphRAGService(qdrant_service, embedding_service)

    return _graph_rag_service_instance
