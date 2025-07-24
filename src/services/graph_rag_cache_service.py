"""
Graph RAG Cache Service for optimizing structure relationship queries.

This service provides intelligent caching mechanisms for Graph RAG operations,
including traversal results, component clusters, and relationship analyses.
Designed to work with the existing cache infrastructure while providing
specialized optimizations for graph operations.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

from .cache_service import get_cache_service
from .graph_traversal_algorithms import (
    ComponentCluster,
    RelationshipFilter,
    TraversalOptions,
    TraversalPath,
    TraversalStrategy,
)
from .structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


@dataclass
class CacheKey:
    """Represents a cache key for Graph RAG operations."""

    operation_type: str  # "traversal", "cluster", "connectivity", "path"
    project_name: str
    primary_key: str  # Main identifier (e.g., breadcrumb)
    parameters_hash: str  # Hash of operation parameters

    def to_string(self) -> str:
        """Convert cache key to string format."""
        return f"graph_rag:{self.operation_type}:{self.project_name}:{self.primary_key}:{self.parameters_hash}"


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""

    key: str
    data: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl_seconds: int
    size_bytes: int = 0
    dependencies: set[str] = None  # Other cache keys this entry depends on

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of the cache entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Statistics for Graph RAG cache performance."""

    total_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalidations: int = 0
    total_cached_entries: int = 0
    total_cache_size_bytes: int = 0
    average_lookup_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.cache_hits / self.total_operations) * 100

    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate as percentage."""
        return 100.0 - self.hit_rate


class GraphRAGCacheService:
    """
    Specialized caching service for Graph RAG operations.

    Provides intelligent caching with dependency tracking, adaptive TTL,
    and operation-specific optimizations for graph traversal and analysis.
    """

    def __init__(self):
        """Initialize the Graph RAG cache service."""
        self.logger = logging.getLogger(__name__)

        # Core cache service integration
        self.cache_service = None
        self._cache_initialized = False

        # Cache configuration
        self.default_ttl = {
            "traversal": 1800,  # 30 minutes - traversal results
            "cluster": 3600,  # 1 hour - component clusters
            "connectivity": 900,  # 15 minutes - connectivity analysis
            "path": 1200,  # 20 minutes - optimal paths
            "graph": 7200,  # 2 hours - full graphs
        }

        # Local cache for metadata tracking
        self._cache_metadata: dict[str, CacheEntry] = {}
        self._dependency_graph: dict[str, set[str]] = {}  # key -> dependent_keys

        # Performance statistics
        self._stats = CacheStats()

        # Cache optimization settings
        self._max_cache_size_mb = 500  # Maximum cache size
        self._cleanup_threshold = 0.8  # Cleanup when 80% full
        self._max_access_age_hours = 24  # Remove entries not accessed in 24 hours

        self.logger.info("GraphRAGCacheService initialized")

    async def _ensure_cache_service(self):
        """Ensure the underlying cache service is initialized."""
        if not self._cache_initialized:
            try:
                self.cache_service = await get_cache_service()
                self._cache_initialized = True
                self.logger.info("Connected to underlying cache service")
            except Exception as e:
                self.logger.error(f"Failed to initialize cache service: {e}")
                self._cache_initialized = False

    async def get_traversal_result(
        self, breadcrumb: str, project_name: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str], dict[str, Any]] | None:
        """
        Get cached traversal result.

        Args:
            breadcrumb: Starting breadcrumb
            project_name: Project name
            options: Traversal options

        Returns:
            Cached traversal result or None if not found
        """
        start_time = time.time()

        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return None

            # Generate cache key
            cache_key = self._generate_traversal_cache_key(breadcrumb, project_name, options)

            # Check local metadata first
            if cache_key.to_string() in self._cache_metadata:
                entry = self._cache_metadata[cache_key.to_string()]
                if entry.is_expired:
                    await self._invalidate_cache_entry(cache_key.to_string())
                    self._stats.cache_misses += 1
                    return None

            # Get from underlying cache
            cached_data = await self.cache_service.get(cache_key.to_string())

            if cached_data is not None:
                # Update access metadata
                await self._update_access_metadata(cache_key.to_string())

                # Deserialize the data
                result = self._deserialize_traversal_result(cached_data)

                # Update statistics
                lookup_time = (time.time() - start_time) * 1000
                self._update_lookup_stats(True, lookup_time)

                self.logger.debug(f"Cache hit for traversal: {breadcrumb}")
                return result

            # Cache miss
            self._stats.cache_misses += 1
            self._stats.total_operations += 1

            return None

        except Exception as e:
            self.logger.error(f"Error getting traversal result from cache: {e}")
            return None

    async def cache_traversal_result(
        self, breadcrumb: str, project_name: str, options: TraversalOptions, result: tuple[list[GraphNode], list[str], dict[str, Any]]
    ):
        """
        Cache a traversal result.

        Args:
            breadcrumb: Starting breadcrumb
            project_name: Project name
            options: Traversal options
            result: Traversal result to cache
        """
        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return

            # Generate cache key
            cache_key = self._generate_traversal_cache_key(breadcrumb, project_name, options)

            # Serialize the result
            serialized_data = self._serialize_traversal_result(result)
            size_bytes = len(json.dumps(serialized_data).encode("utf-8"))

            # Check cache size limits
            if await self._should_cleanup_cache(size_bytes):
                await self._cleanup_cache()

            # Cache the result
            ttl = self.default_ttl["traversal"]
            await self.cache_service.set(cache_key.to_string(), serialized_data, ttl)

            # Update metadata
            dependencies = {f"graph:{project_name}"}  # Depends on project graph
            await self._add_cache_metadata(cache_key.to_string(), ttl, size_bytes, dependencies)

            self.logger.debug(f"Cached traversal result for {breadcrumb}")

        except Exception as e:
            self.logger.error(f"Error caching traversal result: {e}")

    async def get_component_clusters(
        self, project_name: str, cluster_threshold: float, min_cluster_size: int, max_clusters: int
    ) -> list[ComponentCluster] | None:
        """
        Get cached component clusters.

        Args:
            project_name: Project name
            cluster_threshold: Clustering threshold
            min_cluster_size: Minimum cluster size
            max_clusters: Maximum clusters

        Returns:
            Cached clusters or None if not found
        """
        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return None

            # Generate cache key
            cache_key = self._generate_cluster_cache_key(project_name, cluster_threshold, min_cluster_size, max_clusters)

            # Check cache
            cached_data = await self.cache_service.get(cache_key.to_string())

            if cached_data is not None:
                await self._update_access_metadata(cache_key.to_string())
                clusters = self._deserialize_component_clusters(cached_data)

                self._stats.cache_hits += 1
                self._stats.total_operations += 1

                self.logger.debug(f"Cache hit for component clusters: {project_name}")
                return clusters

            # Cache miss
            self._stats.cache_misses += 1
            self._stats.total_operations += 1
            return None

        except Exception as e:
            self.logger.error(f"Error getting component clusters from cache: {e}")
            return None

    async def cache_component_clusters(
        self, project_name: str, cluster_threshold: float, min_cluster_size: int, max_clusters: int, clusters: list[ComponentCluster]
    ):
        """
        Cache component clusters.

        Args:
            project_name: Project name
            cluster_threshold: Clustering threshold
            min_cluster_size: Minimum cluster size
            max_clusters: Maximum clusters
            clusters: Clusters to cache
        """
        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return

            # Generate cache key
            cache_key = self._generate_cluster_cache_key(project_name, cluster_threshold, min_cluster_size, max_clusters)

            # Serialize and cache
            serialized_data = self._serialize_component_clusters(clusters)
            size_bytes = len(json.dumps(serialized_data).encode("utf-8"))

            ttl = self.default_ttl["cluster"]
            await self.cache_service.set(cache_key.to_string(), serialized_data, ttl)

            # Update metadata
            dependencies = {f"graph:{project_name}"}
            await self._add_cache_metadata(cache_key.to_string(), ttl, size_bytes, dependencies)

            self.logger.debug(f"Cached component clusters for {project_name}")

        except Exception as e:
            self.logger.error(f"Error caching component clusters: {e}")

    async def get_connectivity_analysis(self, breadcrumb: str, project_name: str) -> dict[str, Any] | None:
        """
        Get cached connectivity analysis.

        Args:
            breadcrumb: Component breadcrumb
            project_name: Project name

        Returns:
            Cached analysis or None if not found
        """
        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return None

            cache_key = self._generate_connectivity_cache_key(breadcrumb, project_name)
            cached_data = await self.cache_service.get(cache_key.to_string())

            if cached_data is not None:
                await self._update_access_metadata(cache_key.to_string())

                self._stats.cache_hits += 1
                self._stats.total_operations += 1

                return cached_data

            self._stats.cache_misses += 1
            self._stats.total_operations += 1
            return None

        except Exception as e:
            self.logger.error(f"Error getting connectivity analysis from cache: {e}")
            return None

    async def cache_connectivity_analysis(self, breadcrumb: str, project_name: str, analysis: dict[str, Any]):
        """
        Cache connectivity analysis.

        Args:
            breadcrumb: Component breadcrumb
            project_name: Project name
            analysis: Analysis to cache
        """
        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return

            cache_key = self._generate_connectivity_cache_key(breadcrumb, project_name)
            size_bytes = len(json.dumps(analysis).encode("utf-8"))

            ttl = self.default_ttl["connectivity"]
            await self.cache_service.set(cache_key.to_string(), analysis, ttl)

            # Update metadata
            dependencies = {f"graph:{project_name}"}
            await self._add_cache_metadata(cache_key.to_string(), ttl, size_bytes, dependencies)

        except Exception as e:
            self.logger.error(f"Error caching connectivity analysis: {e}")

    async def invalidate_project_cache(self, project_name: str):
        """
        Invalidate all cached data for a specific project.

        Args:
            project_name: Project to invalidate
        """
        try:
            await self._ensure_cache_service()
            if not self._cache_initialized:
                return

            # Find all cache entries for this project
            graph_key = f"graph:{project_name}"
            dependent_keys = self._dependency_graph.get(graph_key, set())

            # Invalidate dependent entries
            for cache_key in dependent_keys:
                await self._invalidate_cache_entry(cache_key)

            # Invalidate the graph itself
            await self._invalidate_cache_entry(graph_key)

            self.logger.info(f"Invalidated cache for project: {project_name}")

        except Exception as e:
            self.logger.error(f"Error invalidating project cache: {e}")

    async def cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        try:
            expired_keys = []
            current_time = time.time()

            for key, entry in self._cache_metadata.items():
                if entry.is_expired or (current_time - entry.accessed_at) > (self._max_access_age_hours * 3600):
                    expired_keys.append(key)

            for key in expired_keys:
                await self._invalidate_cache_entry(key)

            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        return {
            "stats": asdict(self._stats),
            "cache_size": {
                "total_entries": len(self._cache_metadata),
                "total_size_mb": self._stats.total_cache_size_bytes / (1024 * 1024),
                "size_limit_mb": self._max_cache_size_mb,
            },
            "cache_health": {
                "hit_rate_percent": self._stats.hit_rate,
                "miss_rate_percent": self._stats.miss_rate,
                "avg_lookup_time_ms": self._stats.average_lookup_time_ms,
            },
        }

    def reset_statistics(self):
        """Reset cache statistics."""
        self._stats = CacheStats()
        self.logger.info("Cache statistics reset")

    # =================== Private Helper Methods ===================

    def _generate_traversal_cache_key(self, breadcrumb: str, project_name: str, options: TraversalOptions) -> CacheKey:
        """Generate cache key for traversal operations."""
        # Create a hash of the traversal options
        options_dict = {
            "strategy": options.strategy.value,
            "relationship_filter": options.relationship_filter.value,
            "max_depth": options.max_depth,
            "max_nodes": options.max_nodes,
            "confidence_threshold": options.confidence_threshold,
        }

        params_hash = hashlib.md5(json.dumps(options_dict, sort_keys=True).encode()).hexdigest()[:8]

        return CacheKey(operation_type="traversal", project_name=project_name, primary_key=breadcrumb, parameters_hash=params_hash)

    def _generate_cluster_cache_key(
        self, project_name: str, cluster_threshold: float, min_cluster_size: int, max_clusters: int
    ) -> CacheKey:
        """Generate cache key for cluster operations."""
        params_dict = {"threshold": cluster_threshold, "min_size": min_cluster_size, "max_clusters": max_clusters}

        params_hash = hashlib.md5(json.dumps(params_dict, sort_keys=True).encode()).hexdigest()[:8]

        return CacheKey(operation_type="cluster", project_name=project_name, primary_key="all", parameters_hash=params_hash)

    def _generate_connectivity_cache_key(self, breadcrumb: str, project_name: str) -> CacheKey:
        """Generate cache key for connectivity analysis."""
        return CacheKey(operation_type="connectivity", project_name=project_name, primary_key=breadcrumb, parameters_hash="default")

    def _serialize_traversal_result(self, result: tuple[list[GraphNode], list[str], dict[str, Any]]) -> dict[str, Any]:
        """Serialize traversal result for caching."""
        visited_nodes, path, metadata = result

        serialized_nodes = []
        for node in visited_nodes:
            serialized_nodes.append(
                {
                    "chunk_id": node.chunk_id,
                    "breadcrumb": node.breadcrumb,
                    "name": node.name,
                    "chunk_type": node.chunk_type.value,
                    "file_path": node.file_path,
                    "parent_breadcrumb": node.parent_breadcrumb,
                    "children_breadcrumbs": node.children_breadcrumbs,
                    "depth": node.depth,
                    "semantic_weight": node.semantic_weight,
                }
            )

        return {"visited_nodes": serialized_nodes, "path": path, "metadata": metadata}

    def _deserialize_traversal_result(self, data: dict[str, Any]) -> tuple[list[GraphNode], list[str], dict[str, Any]]:
        """Deserialize traversal result from cache."""
        from ..models.code_chunk import ChunkType

        visited_nodes = []
        for node_data in data["visited_nodes"]:
            node = GraphNode(
                chunk_id=node_data["chunk_id"],
                breadcrumb=node_data["breadcrumb"],
                name=node_data["name"],
                chunk_type=ChunkType(node_data["chunk_type"]),
                file_path=node_data["file_path"],
                parent_breadcrumb=node_data.get("parent_breadcrumb"),
                children_breadcrumbs=node_data.get("children_breadcrumbs", []),
                depth=node_data.get("depth", 0),
                semantic_weight=node_data.get("semantic_weight", 1.0),
            )
            visited_nodes.append(node)

        return visited_nodes, data["path"], data["metadata"]

    def _serialize_component_clusters(self, clusters: list[ComponentCluster]) -> list[dict[str, Any]]:
        """Serialize component clusters for caching."""
        serialized_clusters = []

        for cluster in clusters:
            serialized_cluster = {
                "central_node": {
                    "chunk_id": cluster.central_node.chunk_id,
                    "breadcrumb": cluster.central_node.breadcrumb,
                    "name": cluster.central_node.name,
                    "chunk_type": cluster.central_node.chunk_type.value,
                    "file_path": cluster.central_node.file_path,
                    "depth": cluster.central_node.depth,
                    "semantic_weight": cluster.central_node.semantic_weight,
                },
                "related_nodes": [],
                "cluster_score": cluster.cluster_score,
                "relationship_types": list(cluster.relationship_types),
                "max_distance": cluster.max_distance,
                "total_nodes": cluster.total_nodes,
            }

            for node in cluster.related_nodes:
                serialized_cluster["related_nodes"].append(
                    {
                        "chunk_id": node.chunk_id,
                        "breadcrumb": node.breadcrumb,
                        "name": node.name,
                        "chunk_type": node.chunk_type.value,
                        "file_path": node.file_path,
                        "depth": node.depth,
                        "semantic_weight": node.semantic_weight,
                    }
                )

            serialized_clusters.append(serialized_cluster)

        return serialized_clusters

    def _deserialize_component_clusters(self, data: list[dict[str, Any]]) -> list[ComponentCluster]:
        """Deserialize component clusters from cache."""
        from ..models.code_chunk import ChunkType

        clusters = []

        for cluster_data in data:
            # Deserialize central node
            central_data = cluster_data["central_node"]
            central_node = GraphNode(
                chunk_id=central_data["chunk_id"],
                breadcrumb=central_data["breadcrumb"],
                name=central_data["name"],
                chunk_type=ChunkType(central_data["chunk_type"]),
                file_path=central_data["file_path"],
                depth=central_data["depth"],
                semantic_weight=central_data["semantic_weight"],
            )

            # Deserialize related nodes
            related_nodes = []
            for node_data in cluster_data["related_nodes"]:
                node = GraphNode(
                    chunk_id=node_data["chunk_id"],
                    breadcrumb=node_data["breadcrumb"],
                    name=node_data["name"],
                    chunk_type=ChunkType(node_data["chunk_type"]),
                    file_path=node_data["file_path"],
                    depth=node_data["depth"],
                    semantic_weight=node_data["semantic_weight"],
                )
                related_nodes.append(node)

            cluster = ComponentCluster(
                central_node=central_node,
                related_nodes=related_nodes,
                cluster_score=cluster_data["cluster_score"],
                relationship_types=set(cluster_data["relationship_types"]),
                max_distance=cluster_data["max_distance"],
                total_nodes=cluster_data["total_nodes"],
            )

            clusters.append(cluster)

        return clusters

    async def _add_cache_metadata(self, cache_key: str, ttl: int, size_bytes: int, dependencies: set[str]):
        """Add metadata for a cache entry."""
        current_time = time.time()

        entry = CacheEntry(
            key=cache_key,
            data=None,  # Data stored in underlying cache
            created_at=current_time,
            accessed_at=current_time,
            access_count=1,
            ttl_seconds=ttl,
            size_bytes=size_bytes,
            dependencies=dependencies,
        )

        self._cache_metadata[cache_key] = entry

        # Update dependency graph
        for dep in dependencies:
            if dep not in self._dependency_graph:
                self._dependency_graph[dep] = set()
            self._dependency_graph[dep].add(cache_key)

        # Update statistics
        self._stats.total_cached_entries = len(self._cache_metadata)
        self._stats.total_cache_size_bytes += size_bytes

    async def _update_access_metadata(self, cache_key: str):
        """Update access metadata for a cache entry."""
        if cache_key in self._cache_metadata:
            entry = self._cache_metadata[cache_key]
            entry.accessed_at = time.time()
            entry.access_count += 1

    async def _invalidate_cache_entry(self, cache_key: str):
        """Invalidate a specific cache entry."""
        try:
            if self._cache_initialized and self.cache_service:
                await self.cache_service.delete(cache_key)

            if cache_key in self._cache_metadata:
                entry = self._cache_metadata[cache_key]
                self._stats.total_cache_size_bytes -= entry.size_bytes
                del self._cache_metadata[cache_key]

            # Clean up dependency graph
            for dep_set in self._dependency_graph.values():
                dep_set.discard(cache_key)

            self._stats.cache_invalidations += 1
            self._stats.total_cached_entries = len(self._cache_metadata)

        except Exception as e:
            self.logger.error(f"Error invalidating cache entry {cache_key}: {e}")

    async def _should_cleanup_cache(self, additional_size_bytes: int) -> bool:
        """Check if cache cleanup is needed."""
        current_size_mb = (self._stats.total_cache_size_bytes + additional_size_bytes) / (1024 * 1024)
        return current_size_mb > (self._max_cache_size_mb * self._cleanup_threshold)

    async def _cleanup_cache(self):
        """Perform cache cleanup by removing least recently used entries."""
        # Sort entries by last access time (oldest first)
        sorted_entries = sorted(self._cache_metadata.items(), key=lambda x: x[1].accessed_at)

        # Remove oldest 25% of entries
        cleanup_count = max(1, len(sorted_entries) // 4)

        for i in range(cleanup_count):
            cache_key, _ = sorted_entries[i]
            await self._invalidate_cache_entry(cache_key)

        self.logger.info(f"Cleaned up {cleanup_count} cache entries to free space")

    def _update_lookup_stats(self, cache_hit: bool, lookup_time_ms: float):
        """Update cache lookup statistics."""
        if cache_hit:
            self._stats.cache_hits += 1
        else:
            self._stats.cache_misses += 1

        self._stats.total_operations += 1

        # Update average lookup time
        total_ops = self._stats.total_operations
        current_avg = self._stats.average_lookup_time_ms

        self._stats.average_lookup_time_ms = (current_avg * (total_ops - 1) + lookup_time_ms) / total_ops


# Singleton instance for global access
_graph_rag_cache_service_instance: GraphRAGCacheService | None = None


def get_graph_rag_cache_service() -> GraphRAGCacheService:
    """Get the global Graph RAG cache service instance."""
    global _graph_rag_cache_service_instance
    if _graph_rag_cache_service_instance is None:
        _graph_rag_cache_service_instance = GraphRAGCacheService()
    return _graph_rag_cache_service_instance
