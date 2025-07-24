"""
Lightweight Graph Service Implementation for Agentic RAG Performance Enhancement.

This service implements memory indexing mechanism to store key node metadata in memory
for fast querying, supporting the full project processing by removing MCP limitations.
"""

import asyncio
import logging
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from ..models.code_chunk import ChunkType, CodeChunk
from .cache_service import BaseCacheService
from .graph_rag_service import GraphRAGService
from .hybrid_search_service import HybridSearchService
from .structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


@dataclass
class NodeMetadata:
    """Lightweight metadata for graph nodes stored in memory index."""

    node_id: str
    name: str
    chunk_type: ChunkType
    file_path: str
    breadcrumb: str | None = None
    parent_name: str | None = None
    signature: str | None = None
    language: str = ""
    line_start: int = 0
    line_end: int = 0
    importance_score: float = 0.0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    # Fast lookup indices
    children_ids: set[str] = field(default_factory=set)
    parent_ids: set[str] = field(default_factory=set)
    dependency_ids: set[str] = field(default_factory=set)


@dataclass
class MemoryIndex:
    """Memory-based index for fast node lookups."""

    # Primary index: node_id -> NodeMetadata
    nodes: dict[str, NodeMetadata] = field(default_factory=dict)

    # Secondary indices for fast lookups
    by_name: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_type: dict[ChunkType, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_file: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_breadcrumb: dict[str, str] = field(default_factory=dict)  # breadcrumb -> node_id
    by_language: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Relationship indices
    children_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    parent_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    dependency_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Performance metrics
    total_nodes: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    miss_count: int = 0


class LightweightGraphService:
    """
    Lightweight Graph Service implementing memory indexing and on-demand graph building.

    This service addresses the performance limitations of the existing GraphRAGService
    by implementing:
    1. Memory indexing mechanism for fast metadata lookups
    2. On-demand partial graph construction
    3. Removal of max_chunks_for_mcp limitation
    4. Multi-layer caching strategy (L1-L3)
    5. Timeout handling with partial results
    6. Progressive result return with confidence scoring
    """

    def __init__(
        self,
        graph_rag_service: GraphRAGService,
        hybrid_search_service: HybridSearchService,
        cache_service: BaseCacheService | None = None,
        enable_timeout: bool = True,
        default_timeout: int = 15,
        enable_progressive_results: bool = True,
        confidence_threshold: float = 0.7,
    ):
        self.logger = logging.getLogger(__name__)
        self.graph_rag_service = graph_rag_service
        self.hybrid_search_service = hybrid_search_service
        self.cache_service = cache_service

        # Configuration
        self.enable_timeout = enable_timeout
        self.default_timeout = default_timeout
        self.enable_progressive_results = enable_progressive_results
        self.confidence_threshold = confidence_threshold

        # Memory index
        self.memory_index = MemoryIndex()

        # Pre-computed query cache
        self.precomputed_queries = {"entry_points": {}, "main_functions": {}, "public_apis": {}, "common_patterns": {}}

        # Multi-layer cache configuration
        self.l1_cache = {}  # In-memory fast cache
        self.l2_cache = {}  # Path-based cache
        self.l3_cache = {}  # Query result cache

        # Performance tracking
        self.performance_metrics = {"total_queries": 0, "cache_hits": 0, "timeouts": 0, "partial_results": 0, "average_response_time": 0.0}

        self._lock = asyncio.Lock()

    async def initialize_memory_index(self, project_name: str, force_rebuild: bool = False) -> bool:
        """
        Initialize the memory index with key node metadata.

        Args:
            project_name: Name of the project to index
            force_rebuild: Whether to force rebuild the index

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            self.logger.info(f"Initializing memory index for project: {project_name}")

            # Check if index already exists and is recent
            if not force_rebuild and self._is_index_fresh(project_name):
                self.logger.info("Using existing fresh memory index")
                return True

            # Get all chunks from hybrid search service (NO LIMITATIONS)
            chunks = await self.hybrid_search_service.get_all_chunks(project_name)
            if not chunks:
                self.logger.warning(f"No chunks found for project: {project_name}")
                return False

            self.logger.info(f"Building memory index from {len(chunks)} chunks (NO MCP LIMITATIONS)")

            # Reset memory index
            async with self._lock:
                self.memory_index = MemoryIndex()

                # Process all chunks to build metadata
                for chunk in chunks:
                    await self._add_chunk_to_index(chunk)

                # Build relationship indices
                await self._build_relationship_indices(chunks)

                # Pre-compute common queries
                await self._precompute_common_queries(project_name)

                self.memory_index.total_nodes = len(self.memory_index.nodes)
                self.memory_index.last_updated = datetime.now()

            elapsed_time = time.time() - start_time
            self.logger.info(f"Memory index initialized successfully: {self.memory_index.total_nodes} nodes in {elapsed_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize memory index: {e}")
            return False

    async def _add_chunk_to_index(self, chunk: CodeChunk) -> None:
        """Add a code chunk to the memory index."""

        node_metadata = NodeMetadata(
            node_id=chunk.chunk_id,
            name=chunk.name or "unnamed",
            chunk_type=chunk.chunk_type,
            file_path=chunk.file_path,
            breadcrumb=chunk.breadcrumb,
            parent_name=chunk.parent_name,
            signature=chunk.signature,
            language=chunk.language,
            line_start=chunk.start_line,
            line_end=chunk.end_line,
            importance_score=self._calculate_importance_score(chunk),
        )

        # Add to primary index
        self.memory_index.nodes[chunk.chunk_id] = node_metadata

        # Add to secondary indices
        self.memory_index.by_name[chunk.name or "unnamed"].add(chunk.chunk_id)
        self.memory_index.by_type[chunk.chunk_type].add(chunk.chunk_id)
        self.memory_index.by_file[chunk.file_path].add(chunk.chunk_id)
        self.memory_index.by_language[chunk.language].add(chunk.chunk_id)

        if chunk.breadcrumb:
            self.memory_index.by_breadcrumb[chunk.breadcrumb] = chunk.chunk_id

    async def _build_relationship_indices(self, chunks: list[CodeChunk]) -> None:
        """Build relationship indices for fast graph traversal."""

        for chunk in chunks:
            node_id = chunk.chunk_id

            # Build parent-child relationships based on breadcrumb hierarchy
            if chunk.breadcrumb and "." in chunk.breadcrumb:
                parent_breadcrumb = ".".join(chunk.breadcrumb.split(".")[:-1])
                if parent_breadcrumb in self.memory_index.by_breadcrumb:
                    parent_id = self.memory_index.by_breadcrumb[parent_breadcrumb]

                    # Update indices
                    self.memory_index.children_index[parent_id].add(node_id)
                    self.memory_index.parent_index[node_id].add(parent_id)

                    # Update node metadata
                    if node_id in self.memory_index.nodes:
                        self.memory_index.nodes[node_id].parent_ids.add(parent_id)
                    if parent_id in self.memory_index.nodes:
                        self.memory_index.nodes[parent_id].children_ids.add(node_id)

            # Build dependency relationships based on imports
            if hasattr(chunk, "imports_used") and chunk.imports_used:
                for import_name in chunk.imports_used:
                    # Find matching nodes by name
                    matching_nodes = self.memory_index.by_name.get(import_name, set())
                    for dep_id in matching_nodes:
                        self.memory_index.dependency_index[node_id].add(dep_id)
                        if node_id in self.memory_index.nodes:
                            self.memory_index.nodes[node_id].dependency_ids.add(dep_id)

    async def _precompute_common_queries(self, project_name: str) -> None:
        """Pre-compute common queries for fast retrieval."""

        try:
            # Entry points: main functions, __main__, etc.
            entry_points = []
            for node_id, metadata in self.memory_index.nodes.items():
                if (
                    metadata.name in ["main", "__main__", "index", "app"]
                    or metadata.chunk_type == ChunkType.FUNCTION
                    and "main" in metadata.name.lower()
                ):
                    entry_points.append(node_id)

            self.precomputed_queries["entry_points"][project_name] = entry_points

            # Main functions: functions with high importance scores
            main_functions = sorted(
                [
                    (node_id, metadata)
                    for node_id, metadata in self.memory_index.nodes.items()
                    if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]
                ],
                key=lambda x: x[1].importance_score,
                reverse=True,
            )[:20]  # Top 20 functions

            self.precomputed_queries["main_functions"][project_name] = [nid for nid, _ in main_functions]

            # Public APIs: exported functions, public methods
            public_apis = []
            for node_id, metadata in self.memory_index.nodes.items():
                if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD] and not metadata.name.startswith("_"):  # Not private
                    public_apis.append(node_id)

            self.precomputed_queries["public_apis"][project_name] = public_apis

            self.logger.info(
                f"Pre-computed queries: {len(entry_points)} entry points, "
                f"{len(main_functions)} main functions, {len(public_apis)} public APIs"
            )

        except Exception as e:
            self.logger.error(f"Failed to pre-compute common queries: {e}")

    def _calculate_importance_score(self, chunk: CodeChunk) -> float:
        """Calculate importance score for a code chunk."""

        score = 0.0

        # Base score by chunk type
        type_scores = {
            ChunkType.FUNCTION: 1.0,
            ChunkType.METHOD: 0.9,
            ChunkType.CLASS: 1.2,
            ChunkType.INTERFACE: 1.1,
            ChunkType.CONSTRUCTOR: 0.8,
            ChunkType.ASYNC_FUNCTION: 1.1,
        }
        score += type_scores.get(chunk.chunk_type, 0.5)

        # Name-based scoring
        if chunk.name:
            if chunk.name in ["main", "__main__", "index", "app"]:
                score += 1.0
            elif chunk.name.startswith("_"):
                score -= 0.3  # Private functions less important
            elif chunk.name.isupper():
                score += 0.2  # Constants are moderately important

        # Size-based scoring (larger functions might be more important)
        lines = chunk.end_line - chunk.start_line + 1
        if lines > 50:
            score += 0.3
        elif lines > 20:
            score += 0.1

        # Docstring presence adds importance
        if chunk.docstring:
            score += 0.2

        return min(score, 3.0)  # Cap at 3.0

    def _is_index_fresh(self, project_name: str, max_age_minutes: int = 30) -> bool:
        """Check if the memory index is fresh enough to use."""

        if not self.memory_index.nodes:
            return False

        age = datetime.now() - self.memory_index.last_updated
        return age < timedelta(minutes=max_age_minutes)

    async def build_partial_graph(
        self, project_name: str, query_scope: str, max_nodes: int = 50, include_context: bool = True
    ) -> StructureGraph | None:
        """
        Build partial graph for specific query scope (Task 1.2).

        Args:
            project_name: Project to query
            query_scope: Specific scope (e.g., function name, class name)
            max_nodes: Maximum nodes to include
            include_context: Whether to include related context nodes

        Returns:
            StructureGraph: Partial graph or None if failed
        """

        try:
            start_time = time.time()
            self.logger.info(f"Building partial graph for scope: {query_scope}")

            # Find target nodes matching the query scope
            target_nodes = await self._find_target_nodes(query_scope)
            if not target_nodes:
                self.logger.warning(f"No nodes found for scope: {query_scope}")
                return None

            # Start with target nodes
            selected_nodes = set(target_nodes[: max_nodes // 2])  # Reserve half for target nodes

            # Add context if enabled
            if include_context:
                context_nodes = await self._get_context_nodes(selected_nodes, max_nodes - len(selected_nodes))
                selected_nodes.update(context_nodes)

            # Build the partial graph structure
            graph = await self._build_graph_structure(selected_nodes, project_name)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Partial graph built: {len(selected_nodes)} nodes in {elapsed_time:.2f}s")

            return graph

        except Exception as e:
            self.logger.error(f"Failed to build partial graph: {e}")
            return None

    async def _find_target_nodes(self, query_scope: str) -> list[str]:
        """Find nodes matching the query scope."""

        target_nodes = []

        # Search by name
        if query_scope in self.memory_index.by_name:
            target_nodes.extend(self.memory_index.by_name[query_scope])

        # Search by partial name match
        for name, node_ids in self.memory_index.by_name.items():
            if query_scope.lower() in name.lower():
                target_nodes.extend(node_ids)

        # Search by breadcrumb
        for breadcrumb, node_id in self.memory_index.by_breadcrumb.items():
            if query_scope.lower() in breadcrumb.lower():
                target_nodes.append(node_id)

        # Remove duplicates and sort by importance
        unique_nodes = list(set(target_nodes))
        return sorted(
            unique_nodes,
            key=lambda nid: self.memory_index.nodes.get(nid, NodeMetadata("", "", ChunkType.RAW_CODE, "")).importance_score,
            reverse=True,
        )

    async def _get_context_nodes(self, target_nodes: set[str], max_context: int) -> set[str]:
        """Get context nodes related to target nodes."""

        context_nodes = set()

        for node_id in target_nodes:
            if len(context_nodes) >= max_context:
                break

            metadata = self.memory_index.nodes.get(node_id)
            if not metadata:
                continue

            # Add parent nodes
            context_nodes.update(list(metadata.parent_ids)[:2])

            # Add child nodes (limited)
            context_nodes.update(list(metadata.children_ids)[:3])

            # Add dependency nodes (limited)
            context_nodes.update(list(metadata.dependency_ids)[:2])

        return context_nodes - target_nodes  # Exclude target nodes

    async def _build_graph_structure(self, node_ids: set[str], project_name: str) -> StructureGraph:
        """Build graph structure from selected node IDs."""

        nodes = {}
        edges = []

        # Create nodes
        for node_id in node_ids:
            metadata = self.memory_index.nodes.get(node_id)
            if metadata:
                node = GraphNode(
                    chunk_id=node_id,
                    breadcrumb=metadata.breadcrumb or metadata.name,
                    name=metadata.name,
                    chunk_type=metadata.chunk_type,
                    file_path=metadata.file_path,
                    depth=0,
                    semantic_weight=metadata.importance_score,
                )
                nodes[metadata.breadcrumb or node_id] = node

        # Create edges based on relationships
        for node_id in node_ids:
            metadata = self.memory_index.nodes.get(node_id)
            if not metadata:
                continue

            # Parent-child edges
            for child_id in metadata.children_ids:
                if child_id in node_ids:
                    child_metadata = self.memory_index.nodes.get(child_id)
                    if child_metadata:
                        edges.append(
                            GraphEdge(
                                source_breadcrumb=metadata.breadcrumb or metadata.name,
                                target_breadcrumb=child_metadata.breadcrumb or child_metadata.name,
                                relationship_type="parent_child",
                            )
                        )

            # Dependency edges
            for dep_id in metadata.dependency_ids:
                if dep_id in node_ids:
                    dep_metadata = self.memory_index.nodes.get(dep_id)
                    if dep_metadata:
                        edges.append(
                            GraphEdge(
                                source_breadcrumb=metadata.breadcrumb or metadata.name,
                                target_breadcrumb=dep_metadata.breadcrumb or dep_metadata.name,
                                relationship_type="dependency",
                            )
                        )

        return StructureGraph(nodes=nodes, edges=edges, project_name=project_name)

    async def get_precomputed_query(self, project_name: str, query_type: str) -> list[str]:
        """Get pre-computed query results (Task 1.3)."""

        if query_type in self.precomputed_queries:
            return self.precomputed_queries[query_type].get(project_name, [])
        return []

    async def find_intelligent_path(self, start_node: str, end_node: str, max_depth: int = 10) -> list[str] | None:
        """
        Find intelligent path between nodes using cache and index (Task 1.4).

        Args:
            start_node: Starting node ID or name
            end_node: Target node ID or name
            max_depth: Maximum search depth

        Returns:
            List[str]: Path as list of node IDs, or None if no path found
        """

        try:
            # Check L1 cache first
            cache_key = f"path_{start_node}_{end_node}"
            if cache_key in self.l1_cache:
                self.performance_metrics["cache_hits"] += 1
                return self.l1_cache[cache_key]

            # Resolve node names to IDs if needed
            start_id = await self._resolve_node_id(start_node)
            end_id = await self._resolve_node_id(end_node)

            if not start_id or not end_id:
                return None

            # BFS path finding using memory index
            path = await self._bfs_path_search(start_id, end_id, max_depth)

            # Cache result
            self.l1_cache[cache_key] = path

            return path

        except Exception as e:
            self.logger.error(f"Failed to find intelligent path: {e}")
            return None

    async def _resolve_node_id(self, identifier: str) -> str | None:
        """Resolve node name or breadcrumb to node ID."""

        # Check if it's already a node ID
        if identifier in self.memory_index.nodes:
            return identifier

        # Check breadcrumb index
        if identifier in self.memory_index.by_breadcrumb:
            return self.memory_index.by_breadcrumb[identifier]

        # Check name index (return first match)
        if identifier in self.memory_index.by_name:
            node_ids = self.memory_index.by_name[identifier]
            if node_ids:
                return next(iter(node_ids))

        return None

    async def _bfs_path_search(self, start_id: str, end_id: str, max_depth: int) -> list[str] | None:
        """BFS path search using memory index."""

        if start_id == end_id:
            return [start_id]

        queue = [(start_id, [start_id])]
        visited = {start_id}

        for _ in range(max_depth):
            if not queue:
                break

            current_id, path = queue.pop(0)

            if current_id == end_id:
                return path

            # Get neighbors from memory index
            neighbors = set()
            metadata = self.memory_index.nodes.get(current_id)
            if metadata:
                neighbors.update(metadata.children_ids)
                neighbors.update(metadata.parent_ids)
                neighbors.update(metadata.dependency_ids)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_memory_index_stats(self) -> dict[str, Any]:
        """Get memory index statistics."""

        return {
            "total_nodes": self.memory_index.total_nodes,
            "nodes_by_type": {chunk_type.value: len(node_ids) for chunk_type, node_ids in self.memory_index.by_type.items()},
            "nodes_by_language": {lang: len(node_ids) for lang, node_ids in self.memory_index.by_language.items()},
            "hit_rate": (self.memory_index.hit_count / max(1, self.memory_index.hit_count + self.memory_index.miss_count)),
            "last_updated": self.memory_index.last_updated.isoformat(),
            "performance_metrics": self.performance_metrics,
        }
