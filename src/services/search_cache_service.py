"""
Search results cache service for the Codebase RAG MCP Server.

This module provides specialized caching for search results with features including:
- Search query result caching to avoid repeated vector database queries
- Composite cache keys that include search parameters, filters, and project context
- Search result ranking preservation and consistency
- Contextual search variations caching (different n_results, search modes, etc.)
- Cache invalidation when underlying content changes
- Performance optimization for repeated searches
"""

import asyncio
import gzip
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from src.config.cache_config import CacheConfig, get_global_cache_config
from src.models.cache_models import (
    CacheEntry,
    CacheEntryMetadata,
    CacheEntryType,
    CacheStatistics,
    create_cache_entry,
)
from src.services.cache_service import BaseCacheService, get_cache_service
from src.utils.cache_key_generator import CacheKeyGenerator, KeyType
from src.utils.cache_utils import CompressionFormat, SerializationFormat


class SearchMode(Enum):
    """Types of search modes that can be cached."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchScope(Enum):
    """Scope of search operations."""

    CURRENT_PROJECT = "current_project"
    CROSS_PROJECT = "cross_project"
    TARGET_PROJECTS = "target_projects"


@dataclass
class SearchParameters:
    """Parameters that define a search operation."""

    # Core search parameters
    query: str
    n_results: int = 5
    search_mode: SearchMode = SearchMode.HYBRID
    search_scope: SearchScope = SearchScope.CURRENT_PROJECT

    # Context parameters
    include_context: bool = True
    context_chunks: int = 1

    # Project filtering
    target_projects: list[str] = field(default_factory=list)
    current_project: str = ""

    # Collection filtering
    collections_searched: list[str] = field(default_factory=list)

    # Additional parameters
    filters: dict[str, Any] = field(default_factory=dict)

    def to_cache_dict(self) -> dict[str, Any]:
        """Convert to dictionary for cache key generation."""
        return {
            "query": self.query,
            "n_results": self.n_results,
            "search_mode": self.search_mode.value,
            "search_scope": self.search_scope.value,
            "include_context": self.include_context,
            "context_chunks": self.context_chunks,
            "target_projects": sorted(self.target_projects),  # Sort for consistent hashing
            "current_project": self.current_project,
            "collections_searched": sorted(self.collections_searched),  # Sort for consistent hashing
            "filters": self.filters,
        }


@dataclass
class SearchResultMetadata:
    """Metadata for cached search results."""

    # Search operation info
    search_parameters: SearchParameters
    search_timestamp: float = field(default_factory=time.time)
    search_duration_ms: float = 0.0

    # Result characteristics
    total_results: int = 0
    collections_count: int = 0
    result_types: dict[str, int] = field(default_factory=dict)  # Count by type
    score_range: tuple[float, float] = (0.0, 0.0)  # Min and max scores

    # Performance info
    cache_generation_time_ms: float = 0.0
    compression_ratio: float = 1.0

    # Validity tracking
    content_version: str = ""  # For invalidation when content changes
    indexed_at: float = 0.0  # When the underlying content was indexed

    # Access tracking
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class SearchCacheEntry:
    """Cache entry specifically for search results."""

    key: str
    results: list[dict[str, Any]]
    metadata: SearchResultMetadata
    compressed_data: bytes | None = None

    def get_result_count(self) -> int:
        """Get number of results in this entry."""
        return len(self.results)

    def get_score_range(self) -> tuple[float, float]:
        """Get score range of results."""
        if not self.results:
            return (0.0, 0.0)

        scores = [result.get("score", 0.0) for result in self.results]
        return (min(scores), max(scores))


@dataclass
class SearchCacheMetrics:
    """Metrics for search cache performance."""

    # Cache statistics
    total_searches_cached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalidations: int = 0

    # Performance metrics
    avg_cache_lookup_time_ms: float = 0.0
    avg_result_serialization_time_ms: float = 0.0
    avg_result_deserialization_time_ms: float = 0.0

    # Storage metrics
    total_storage_bytes: int = 0
    total_compressed_bytes: int = 0
    total_cached_results: int = 0

    # Search operation savings
    vector_db_queries_saved: int = 0
    estimated_time_saved_ms: float = 0.0

    # Search pattern tracking
    popular_queries: dict[str, int] = field(default_factory=dict)
    search_mode_distribution: dict[str, int] = field(default_factory=dict)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests

    def get_compression_ratio(self) -> float:
        """Calculate overall compression ratio."""
        if self.total_storage_bytes == 0:
            return 1.0
        return self.total_compressed_bytes / self.total_storage_bytes


class SearchCacheService:
    """
    Specialized cache service for search results with compression, ranking preservation, and metrics.

    This service provides high-performance caching for search results with the following features:
    - Composite cache keys based on search parameters and context
    - Search result ranking preservation to maintain relevance ordering
    - Contextual search variation caching for different parameter combinations
    - Cache invalidation strategies for content changes
    - Comprehensive metrics and performance monitoring
    - Storage optimization through compression and efficient serialization
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the search cache service."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)
        self.key_generator = CacheKeyGenerator()

        # Cache service will be initialized later
        self._cache_service: BaseCacheService | None = None

        # Metrics tracking
        self.metrics = SearchCacheMetrics()

        # Configuration
        self.compression_enabled = True  # Enable compression for search results
        self.max_result_size = 10 * 1024 * 1024  # 10MB limit per search result set
        self.max_results_to_cache = 100  # Limit number of results cached per query

        # Content version tracking for invalidation
        self._content_versions: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the search cache service."""
        try:
            self._cache_service = await get_cache_service()
            self.logger.info("Search cache service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize search cache service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the search cache service."""
        if self._cache_service:
            # Note: Cache service shutdown is handled globally
            pass
        self.logger.info("Search cache service shutdown")

    def _generate_cache_key(self, search_params: SearchParameters) -> str:
        """
        Generate composite cache key for search parameters.

        The key includes all parameters that affect search results to ensure
        proper cache isolation and avoid false cache hits.

        Args:
            search_params: Search parameters to generate key for

        Returns:
            str: Generated cache key
        """
        # Create content hash from all search parameters
        content_dict = search_params.to_cache_dict()

        # Add content version for invalidation support
        content_dict["content_version"] = self._get_content_version(search_params.current_project)

        # Generate consistent hash
        content_string = json.dumps(content_dict, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(content_string.encode()).hexdigest()[:32]

        # Generate hierarchical key
        return self.key_generator.generate_key(
            key_type=KeyType.SEARCH,
            namespace="search_results",
            project_id=search_params.current_project or "default",
            content=content_hash,
            additional_params={
                "mode": search_params.search_mode.value,
                "scope": search_params.search_scope.value,
                "n_results": str(search_params.n_results),
                "context": str(search_params.include_context),
            },
        )

    def _get_content_version(self, project_id: str) -> str:
        """Get or create content version identifier for cache invalidation."""
        if project_id not in self._content_versions:
            # For now, use timestamp-based versioning
            # In production, this should be tied to actual content changes
            self._content_versions[project_id] = f"v_{int(time.time())}"
        return self._content_versions[project_id]

    def _compress_results(self, results: list[dict[str, Any]]) -> bytes:
        """
        Compress search results for storage.

        Args:
            results: Search results to compress

        Returns:
            bytes: Compressed results data
        """
        try:
            start_time = time.time()

            # Serialize to JSON
            json_data = json.dumps(results, separators=(",", ":"))
            json_bytes = json_data.encode("utf-8")

            if self.compression_enabled:
                # Use gzip compression
                compressed_data = gzip.compress(json_bytes, compresslevel=6)
            else:
                compressed_data = json_bytes

            serialization_time = (time.time() - start_time) * 1000
            self.metrics.avg_result_serialization_time_ms = (
                self.metrics.avg_result_serialization_time_ms * self.metrics.total_searches_cached + serialization_time
            ) / (self.metrics.total_searches_cached + 1)

            return compressed_data

        except Exception as e:
            self.logger.error(f"Failed to compress search results: {e}")
            raise

    def _decompress_results(self, compressed_data: bytes) -> list[dict[str, Any]]:
        """
        Decompress search results from storage.

        Args:
            compressed_data: Compressed results data

        Returns:
            List[Dict[str, Any]]: Decompressed search results
        """
        try:
            start_time = time.time()

            if self.compression_enabled:
                # Decompress data
                json_bytes = gzip.decompress(compressed_data)
            else:
                json_bytes = compressed_data

            # Parse JSON
            json_data = json_bytes.decode("utf-8")
            results = json.loads(json_data)

            deserialization_time = (time.time() - start_time) * 1000
            self.metrics.avg_result_deserialization_time_ms = (
                self.metrics.avg_result_deserialization_time_ms * self.metrics.cache_hits + deserialization_time
            ) / (self.metrics.cache_hits + 1)

            return results

        except Exception as e:
            self.logger.error(f"Failed to decompress search results: {e}")
            raise

    async def get_cached_search_results(self, search_params: SearchParameters) -> list[dict[str, Any]] | None:
        """
        Get cached search results for given parameters.

        Args:
            search_params: Search parameters to look up

        Returns:
            Optional[List[Dict[str, Any]]]: Cached search results or None if not found
        """
        if not self._cache_service:
            return None

        try:
            lookup_start = time.time()

            # Generate cache key
            cache_key = self._generate_cache_key(search_params)

            # Look up in cache
            cached_data = await self._cache_service.get(cache_key)

            if cached_data is None:
                self.metrics.cache_misses += 1
                lookup_time = (time.time() - lookup_start) * 1000
                self.metrics.avg_cache_lookup_time_ms = (
                    self.metrics.avg_cache_lookup_time_ms * (self.metrics.cache_hits + self.metrics.cache_misses - 1) + lookup_time
                ) / (self.metrics.cache_hits + self.metrics.cache_misses)
                return None

            # Deserialize cached entry
            if isinstance(cached_data, dict):
                entry_data = cached_data
            else:
                entry_data = json.loads(cached_data)

            # Extract and decompress results
            compressed_data = bytes.fromhex(entry_data["compressed_data"])
            results = self._decompress_results(compressed_data)

            # Update metrics
            self.metrics.cache_hits += 1
            lookup_time = (time.time() - lookup_start) * 1000
            self.metrics.avg_cache_lookup_time_ms = (
                self.metrics.avg_cache_lookup_time_ms * (self.metrics.cache_hits + self.metrics.cache_misses - 1) + lookup_time
            ) / (self.metrics.cache_hits + self.metrics.cache_misses)

            # Track query popularity
            query_hash = hashlib.sha256(search_params.query.encode()).hexdigest()[:16]
            self.metrics.popular_queries[query_hash] = self.metrics.popular_queries.get(query_hash, 0) + 1

            self.logger.debug(f"Cache hit for search: {cache_key}")
            return results

        except Exception as e:
            self.logger.error(f"Failed to get cached search results: {e}")
            self.metrics.cache_misses += 1
            return None

    async def cache_search_results(
        self,
        search_params: SearchParameters,
        results: list[dict[str, Any]],
        search_duration_ms: float = 0.0,
    ) -> bool:
        """
        Cache search results with ranking preservation.

        Args:
            search_params: Search parameters used to generate results
            results: Search results to cache
            search_duration_ms: Time taken to generate the results

        Returns:
            bool: True if successfully cached
        """
        if not self._cache_service:
            return False

        try:
            # Validate results size
            if len(results) > self.max_results_to_cache:
                self.logger.warning(f"Too many results to cache: {len(results)}, limiting to {self.max_results_to_cache}")
                results = results[: self.max_results_to_cache]

            # Generate cache key
            cache_key = self._generate_cache_key(search_params)

            # Calculate result characteristics
            total_results = len(results)
            result_types = {}
            for result in results:
                result_type = result.get("type", "unknown")
                result_types[result_type] = result_types.get(result_type, 0) + 1

            score_range = (0.0, 0.0)
            if results:
                scores = [result.get("score", 0.0) for result in results]
                score_range = (min(scores), max(scores))

            # Compress results
            compressed_data = self._compress_results(results)

            # Validate compressed size
            if len(compressed_data) > self.max_result_size:
                self.logger.warning(f"Compressed results too large to cache: {len(compressed_data)} bytes")
                return False

            # Create cache entry data
            entry_data = {
                "compressed_data": compressed_data.hex(),
                "search_params": asdict(search_params),
                "total_results": total_results,
                "result_types": result_types,
                "score_range": score_range,
                "search_duration_ms": search_duration_ms,
                "compression_ratio": len(compressed_data) / len(json.dumps(results).encode()) if results else 1.0,
                "content_version": self._get_content_version(search_params.current_project),
                "cached_at": time.time(),
                "collections_count": len(search_params.collections_searched),
            }

            # Store in cache
            ttl = self.config.search_cache.ttl_seconds if hasattr(self.config, "search_cache") else 3600
            success = await self._cache_service.set(cache_key, entry_data, ttl)

            if success:
                # Update metrics
                self.metrics.total_searches_cached += 1
                original_size = len(json.dumps(results).encode("utf-8"))
                self.metrics.total_storage_bytes += original_size
                self.metrics.total_compressed_bytes += len(compressed_data)
                self.metrics.total_cached_results += total_results

                # Track search mode distribution
                mode = search_params.search_mode.value
                self.metrics.search_mode_distribution[mode] = self.metrics.search_mode_distribution.get(mode, 0) + 1

                self.logger.debug(f"Successfully cached search results: {cache_key} ({total_results} results)")
                return True
            else:
                self.logger.warning(f"Failed to cache search results: {cache_key}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to cache search results: {e}")
            return False

    async def invalidate_project_cache(self, project_id: str) -> int:
        """
        Invalidate all cached search results for a specific project.

        This should be called when project content changes to ensure cache consistency.

        Args:
            project_id: Project identifier to invalidate

        Returns:
            int: Number of invalidated entries (placeholder)
        """
        try:
            # Update content version to invalidate existing cache entries
            old_version = self._content_versions.get(project_id, "v_0")
            new_version = f"v_{int(time.time())}"
            self._content_versions[project_id] = new_version

            self.logger.info(f"Invalidated search cache for project {project_id}: {old_version} -> {new_version}")
            self.metrics.cache_invalidations += 1

            # Note: Actual cache cleanup could be implemented with a background task
            # For now, entries will naturally expire or be overwritten

            return 1  # Placeholder count

        except Exception as e:
            self.logger.error(f"Failed to invalidate project cache: {e}")
            return 0

    async def invalidate_all_cache(self) -> int:
        """
        Invalidate all cached search results.

        Returns:
            int: Number of invalidated entries (placeholder)
        """
        try:
            # Update all content versions
            current_time = int(time.time())
            invalidated_count = len(self._content_versions)

            for project_id in self._content_versions:
                self._content_versions[project_id] = f"v_{current_time}"

            self.logger.info(f"Invalidated all search cache ({invalidated_count} projects)")
            self.metrics.cache_invalidations += invalidated_count

            return invalidated_count

        except Exception as e:
            self.logger.error(f"Failed to invalidate all cache: {e}")
            return 0

    def check_result_consistency(self, results1: list[dict[str, Any]], results2: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Check consistency between two sets of search results.

        This can be used to validate that cached results maintain proper ranking
        and haven't become stale.

        Args:
            results1: First set of results
            results2: Second set of results

        Returns:
            Dict[str, Any]: Consistency check results
        """
        try:
            # Basic checks
            consistency = {
                "length_match": len(results1) == len(results2),
                "score_order_preserved": True,
                "content_match": True,
                "ranking_consistency": 1.0,
            }

            # Check if results are empty
            if not results1 and not results2:
                return consistency

            if not results1 or not results2:
                consistency["content_match"] = False
                consistency["ranking_consistency"] = 0.0
                return consistency

            # Check score ordering preservation
            scores1 = [r.get("score", 0.0) for r in results1]
            scores2 = [r.get("score", 0.0) for r in results2]

            if len(scores1) == len(scores2):
                # Check if relative ordering is preserved
                order_matches = 0
                total_pairs = 0

                for i in range(len(scores1)):
                    for j in range(i + 1, len(scores1)):
                        total_pairs += 1
                        if (scores1[i] > scores1[j]) == (scores2[i] > scores2[j]):
                            order_matches += 1

                if total_pairs > 0:
                    consistency["ranking_consistency"] = order_matches / total_pairs

                # Check if scores are non-increasing (proper ranking)
                consistency["score_order_preserved"] = all(scores1[i] >= scores1[i + 1] for i in range(len(scores1) - 1)) and all(
                    scores2[i] >= scores2[i + 1] for i in range(len(scores2) - 1)
                )

            # Check content match (simplified - compare file paths and basic metadata)
            if len(results1) == len(results2):
                content_matches = 0
                for r1, r2 in zip(results1, results2, strict=False):
                    if r1.get("file_path") == r2.get("file_path") and r1.get("chunk_type") == r2.get("chunk_type"):
                        content_matches += 1

                consistency["content_match"] = content_matches == len(results1)
            else:
                consistency["content_match"] = False

            return consistency

        except Exception as e:
            self.logger.error(f"Failed to check result consistency: {e}")
            return {"error": str(e)}

    def get_metrics(self) -> SearchCacheMetrics:
        """Get current cache metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.metrics = SearchCacheMetrics()
        self.logger.info("Search cache metrics reset")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hit_rate": self.metrics.get_hit_rate(),
            "compression_ratio": self.metrics.get_compression_ratio(),
            "total_cached": self.metrics.total_searches_cached,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "storage_bytes": self.metrics.total_storage_bytes,
            "compressed_bytes": self.metrics.total_compressed_bytes,
            "cached_results": self.metrics.total_cached_results,
            "vector_db_queries_saved": self.metrics.vector_db_queries_saved,
            "time_saved_ms": self.metrics.estimated_time_saved_ms,
            "popular_queries": dict(self.metrics.popular_queries),
            "search_mode_distribution": dict(self.metrics.search_mode_distribution),
            "avg_lookup_time_ms": self.metrics.avg_cache_lookup_time_ms,
            "avg_serialization_time_ms": self.metrics.avg_result_serialization_time_ms,
            "avg_deserialization_time_ms": self.metrics.avg_result_deserialization_time_ms,
        }


# Global search cache service instance
_search_cache_service: SearchCacheService | None = None


async def get_search_cache_service() -> SearchCacheService:
    """
    Get the global search cache service instance.

    Returns:
        SearchCacheService: The global search cache service instance
    """
    global _search_cache_service
    if _search_cache_service is None:
        _search_cache_service = SearchCacheService()
        await _search_cache_service.initialize()
    return _search_cache_service


async def shutdown_search_cache_service() -> None:
    """Shutdown the global search cache service."""
    global _search_cache_service
    if _search_cache_service:
        await _search_cache_service.shutdown()
        _search_cache_service = None
