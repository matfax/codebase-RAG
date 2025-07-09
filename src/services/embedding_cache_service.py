"""
Embedding cache service for the Codebase RAG MCP Server.

This module provides specialized caching for embeddings with features including:
- Query text embedding caching to avoid repeated Ollama API calls
- File content embedding caching for incremental indexing
- Batch embedding operation caching for performance optimization
- Embedding model versioning for cache invalidation on model changes
- Compression and optimization for embedding storage efficiency
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

import numpy as np
import torch

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..models.cache_models import (
    CacheEntry,
    CacheEntryMetadata,
    CacheEntryType,
    CacheStatistics,
    create_cache_entry,
)
from ..services.cache_service import BaseCacheService, get_cache_service
from ..utils.cache_key_generator import CacheKeyGenerator, KeyType
from ..utils.cache_utils import CompressionFormat, SerializationFormat


class EmbeddingType(Enum):
    """Types of embeddings that can be cached."""

    QUERY_TEXT = "query_text"
    FILE_CONTENT = "file_content"
    BATCH_OPERATION = "batch_operation"
    CHUNK_CONTENT = "chunk_content"


class EmbeddingCompressionLevel(Enum):
    """Compression levels for embedding storage."""

    NONE = 0
    LOW = 3
    MEDIUM = 6
    HIGH = 9


@dataclass
class EmbeddingMetadata:
    """Metadata for cached embeddings."""

    # Core embedding info
    model_name: str
    model_version: str = "latest"
    embedding_type: EmbeddingType = EmbeddingType.QUERY_TEXT
    dimension: int = 0

    # Content information
    original_text: str = ""
    text_hash: str = ""
    text_length: int = 0

    # Processing information
    generation_time_ms: float = 0.0
    compression_ratio: float = 1.0

    # Cache behavior
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class EmbeddingCacheEntry:
    """Cache entry specifically for embeddings."""

    key: str
    embedding: torch.Tensor
    metadata: EmbeddingMetadata
    compressed_data: bytes | None = None

    def get_embedding_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        if isinstance(self.embedding, torch.Tensor):
            return self.embedding.numpy()
        return np.array(self.embedding)

    def get_embedding_tensor(self) -> torch.Tensor:
        """Get embedding as torch tensor."""
        if isinstance(self.embedding, torch.Tensor):
            return self.embedding
        return torch.tensor(self.embedding, dtype=torch.float32)


@dataclass
class EmbeddingCacheMetrics:
    """Metrics for embedding cache performance."""

    # Cache statistics
    total_embeddings_cached: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_invalidations: int = 0

    # Performance metrics
    avg_cache_lookup_time_ms: float = 0.0
    avg_compression_time_ms: float = 0.0
    avg_decompression_time_ms: float = 0.0

    # Storage metrics
    total_storage_bytes: int = 0
    total_compressed_bytes: int = 0
    total_embeddings_count: int = 0

    # API call savings
    ollama_calls_saved: int = 0
    estimated_time_saved_ms: float = 0.0

    # Model version tracking
    model_versions: dict[str, int] = field(default_factory=dict)

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


class EmbeddingCacheService:
    """
    Specialized cache service for embeddings with compression, versioning, and metrics.

    This service provides high-performance caching for embeddings with the following features:
    - Content-based cache keys to avoid duplicate embedding generation
    - Model version tracking for cache invalidation
    - Configurable compression for storage efficiency
    - Comprehensive metrics and monitoring
    - Batch operation support for performance optimization
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the embedding cache service."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)
        self.key_generator = CacheKeyGenerator()

        # Cache service will be initialized later
        self._cache_service: BaseCacheService | None = None

        # Metrics tracking
        self.metrics = EmbeddingCacheMetrics()

        # Configuration
        self.compression_enabled = self.config.embedding_cache.compression_enabled
        self.compression_level = EmbeddingCompressionLevel.MEDIUM
        self.max_embedding_size = 50 * 1024 * 1024  # 50MB limit per embedding

        # Model version tracking
        self._model_versions: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the embedding cache service."""
        try:
            self._cache_service = await get_cache_service()
            self.logger.info("Embedding cache service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding cache service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the embedding cache service."""
        if self._cache_service:
            # Note: Cache service shutdown is handled globally
            pass
        self.logger.info("Embedding cache service shutdown")

    def _generate_cache_key(
        self,
        text: str,
        model_name: str,
        embedding_type: EmbeddingType = EmbeddingType.QUERY_TEXT,
        additional_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate content-based cache key for embedding.

        Args:
            text: Text content to embed
            model_name: Name of the embedding model
            embedding_type: Type of embedding
            additional_context: Additional context for key generation

        Returns:
            str: Generated cache key
        """
        # Create content hash
        content_parts = [text, model_name, embedding_type.value, self._get_model_version(model_name)]

        if additional_context:
            content_parts.append(json.dumps(additional_context, sort_keys=True))

        content_string = "|".join(str(part) for part in content_parts)
        content_hash = hashlib.sha256(content_string.encode()).hexdigest()[:32]

        # Generate hierarchical key
        return self.key_generator.generate_hierarchical_key(
            key_type=KeyType.EMBEDDING,
            namespace="embeddings",
            project_id="default",
            content_hash=content_hash,
            additional_params={"model": model_name, "type": embedding_type.value, "version": self._get_model_version(model_name)},
        )

    def _get_model_version(self, model_name: str) -> str:
        """Get or create model version identifier."""
        if model_name not in self._model_versions:
            # For now, use timestamp-based versioning
            # In production, this could be tied to actual model versions
            self._model_versions[model_name] = f"v_{int(time.time())}"
        return self._model_versions[model_name]

    def _compress_embedding(self, embedding: torch.Tensor) -> bytes:
        """
        Compress embedding for storage.

        Args:
            embedding: Embedding tensor to compress

        Returns:
            bytes: Compressed embedding data
        """
        try:
            start_time = time.time()

            # Convert to numpy array
            if isinstance(embedding, torch.Tensor):
                array = embedding.numpy()
            else:
                array = np.array(embedding)

            # Serialize to bytes
            array_bytes = array.tobytes()

            if self.compression_enabled:
                # Use gzip compression
                compressed_data = gzip.compress(array_bytes, compresslevel=self.compression_level.value)
            else:
                compressed_data = array_bytes

            compression_time = (time.time() - start_time) * 1000
            self.metrics.avg_compression_time_ms = (
                self.metrics.avg_compression_time_ms * self.metrics.total_embeddings_cached + compression_time
            ) / (self.metrics.total_embeddings_cached + 1)

            return compressed_data

        except Exception as e:
            self.logger.error(f"Failed to compress embedding: {e}")
            raise

    def _decompress_embedding(self, compressed_data: bytes, shape: tuple[int, ...], dtype: str = "float32") -> torch.Tensor:
        """
        Decompress embedding from storage.

        Args:
            compressed_data: Compressed embedding data
            shape: Original shape of the embedding
            dtype: Data type of the embedding

        Returns:
            torch.Tensor: Decompressed embedding tensor
        """
        try:
            start_time = time.time()

            if self.compression_enabled:
                # Decompress data
                array_bytes = gzip.decompress(compressed_data)
            else:
                array_bytes = compressed_data

            # Reconstruct numpy array
            array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)

            # Convert to torch tensor
            tensor = torch.tensor(array, dtype=torch.float32)

            decompression_time = (time.time() - start_time) * 1000
            self.metrics.avg_decompression_time_ms = (
                self.metrics.avg_decompression_time_ms * self.metrics.cache_hits + decompression_time
            ) / (self.metrics.cache_hits + 1)

            return tensor

        except Exception as e:
            self.logger.error(f"Failed to decompress embedding: {e}")
            raise

    async def get_cached_embedding(
        self, text: str, model_name: str, embedding_type: EmbeddingType = EmbeddingType.QUERY_TEXT
    ) -> torch.Tensor | None:
        """
        Get cached embedding for text.

        Args:
            text: Text content
            model_name: Name of the embedding model
            embedding_type: Type of embedding

        Returns:
            Optional[torch.Tensor]: Cached embedding or None if not found
        """
        if not self._cache_service:
            return None

        try:
            lookup_start = time.time()

            # Generate cache key
            cache_key = self._generate_cache_key(text, model_name, embedding_type)

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

            # Extract embedding information
            compressed_data = bytes.fromhex(entry_data["compressed_data"])
            shape = tuple(entry_data["shape"])
            dtype = entry_data.get("dtype", "float32")

            # Decompress embedding
            embedding = self._decompress_embedding(compressed_data, shape, dtype)

            # Update metrics
            self.metrics.cache_hits += 1
            lookup_time = (time.time() - lookup_start) * 1000
            self.metrics.avg_cache_lookup_time_ms = (
                self.metrics.avg_cache_lookup_time_ms * (self.metrics.cache_hits + self.metrics.cache_misses - 1) + lookup_time
            ) / (self.metrics.cache_hits + self.metrics.cache_misses)

            self.logger.debug(f"Cache hit for embedding: {cache_key}")
            return embedding

        except Exception as e:
            self.logger.error(f"Failed to get cached embedding: {e}")
            self.metrics.cache_misses += 1
            return None

    async def cache_embedding(
        self,
        text: str,
        embedding: torch.Tensor,
        model_name: str,
        embedding_type: EmbeddingType = EmbeddingType.QUERY_TEXT,
        generation_time_ms: float = 0.0,
    ) -> bool:
        """
        Cache an embedding.

        Args:
            text: Original text content
            embedding: Generated embedding tensor
            model_name: Name of the embedding model
            embedding_type: Type of embedding
            generation_time_ms: Time taken to generate the embedding

        Returns:
            bool: True if successfully cached
        """
        if not self._cache_service:
            return False

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(text, model_name, embedding_type)

            # Validate embedding size
            embedding_size = embedding.numel() * embedding.element_size()
            if embedding_size > self.max_embedding_size:
                self.logger.warning(f"Embedding too large to cache: {embedding_size} bytes")
                return False

            # Compress embedding
            compressed_data = self._compress_embedding(embedding)

            # Create cache entry data
            entry_data = {
                "compressed_data": compressed_data.hex(),
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype).replace("torch.", ""),
                "model_name": model_name,
                "model_version": self._get_model_version(model_name),
                "embedding_type": embedding_type.value,
                "text_hash": hashlib.sha256(text.encode()).hexdigest(),
                "text_length": len(text),
                "generation_time_ms": generation_time_ms,
                "compression_ratio": len(compressed_data) / embedding_size if embedding_size > 0 else 1.0,
                "cached_at": time.time(),
            }

            # Store in cache
            ttl = self.config.embedding_cache.ttl_seconds
            success = await self._cache_service.set(cache_key, entry_data, ttl)

            if success:
                # Update metrics
                self.metrics.total_embeddings_cached += 1
                self.metrics.total_storage_bytes += embedding_size
                self.metrics.total_compressed_bytes += len(compressed_data)
                self.metrics.total_embeddings_count += 1

                # Track model version
                if model_name not in self.metrics.model_versions:
                    self.metrics.model_versions[model_name] = 0
                self.metrics.model_versions[model_name] += 1

                self.logger.debug(f"Successfully cached embedding: {cache_key}")
                return True
            else:
                self.logger.warning(f"Failed to cache embedding: {cache_key}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to cache embedding: {e}")
            return False

    async def get_cached_batch_embeddings(
        self, texts: list[str], model_name: str, embedding_type: EmbeddingType = EmbeddingType.BATCH_OPERATION
    ) -> tuple[list[torch.Tensor | None], list[str]]:
        """
        Get cached embeddings for a batch of texts.

        Args:
            texts: List of text content
            model_name: Name of the embedding model
            embedding_type: Type of embedding

        Returns:
            Tuple of (embeddings, uncached_texts): Cached embeddings and texts that need generation
        """
        embeddings = []
        uncached_texts = []

        for text in texts:
            cached_embedding = await self.get_cached_embedding(text, model_name, embedding_type)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                uncached_texts.append(text)

        return embeddings, uncached_texts

    async def cache_batch_embeddings(
        self,
        texts: list[str],
        embeddings: list[torch.Tensor],
        model_name: str,
        embedding_type: EmbeddingType = EmbeddingType.BATCH_OPERATION,
        generation_times_ms: list[float] | None = None,
    ) -> list[bool]:
        """
        Cache a batch of embeddings.

        Args:
            texts: List of original text content
            embeddings: List of generated embedding tensors
            model_name: Name of the embedding model
            embedding_type: Type of embedding
            generation_times_ms: List of generation times for each embedding

        Returns:
            List[bool]: Success status for each cached embedding
        """
        if len(texts) != len(embeddings):
            raise ValueError("Texts and embeddings lists must have the same length")

        results = []
        generation_times = generation_times_ms or [0.0] * len(texts)

        # Process embeddings in parallel for better performance
        cache_tasks = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
            if embedding is not None:
                task = self.cache_embedding(text, embedding, model_name, embedding_type, generation_times[i])
                cache_tasks.append(task)
            else:
                cache_tasks.append(None)

        # Execute cache operations
        for i, task in enumerate(cache_tasks):
            if task is not None:
                try:
                    success = await task
                    results.append(success)
                except Exception as e:
                    self.logger.error(f"Failed to cache embedding {i}: {e}")
                    results.append(False)
            else:
                results.append(False)

        return results

    async def invalidate_model_cache(self, model_name: str) -> int:
        """
        Invalidate all cached embeddings for a specific model.

        Args:
            model_name: Name of the model to invalidate

        Returns:
            int: Number of invalidated entries
        """
        try:
            # Update model version to invalidate existing cache entries
            old_version = self._model_versions.get(model_name, "v_0")
            new_version = f"v_{int(time.time())}"
            self._model_versions[model_name] = new_version

            self.logger.info(f"Invalidated cache for model {model_name}: {old_version} -> {new_version}")
            self.metrics.cache_invalidations += 1

            # Note: Actual cache cleanup could be implemented with a background task
            # For now, entries will naturally expire or be overwritten

            return 1  # Placeholder count

        except Exception as e:
            self.logger.error(f"Failed to invalidate model cache: {e}")
            return 0

    def get_metrics(self) -> EmbeddingCacheMetrics:
        """Get current cache metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self.metrics = EmbeddingCacheMetrics()
        self.logger.info("Embedding cache metrics reset")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "hit_rate": self.metrics.get_hit_rate(),
            "compression_ratio": self.metrics.get_compression_ratio(),
            "total_cached": self.metrics.total_embeddings_cached,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "storage_bytes": self.metrics.total_storage_bytes,
            "compressed_bytes": self.metrics.total_compressed_bytes,
            "ollama_calls_saved": self.metrics.ollama_calls_saved,
            "time_saved_ms": self.metrics.estimated_time_saved_ms,
            "model_versions": dict(self.metrics.model_versions),
            "avg_lookup_time_ms": self.metrics.avg_cache_lookup_time_ms,
            "avg_compression_time_ms": self.metrics.avg_compression_time_ms,
            "avg_decompression_time_ms": self.metrics.avg_decompression_time_ms,
        }

    # ============================================================================
    # Cache Warmup Support
    # ============================================================================

    async def get_warmup_candidates(self, historical_data: dict[str, Any]) -> list[Any]:
        """
        Get warmup candidates for embedding cache.

        Args:
            historical_data: Historical usage data for generating candidates

        Returns:
            List of WarmupItem objects for cache preloading
        """
        from ..utils.cache_warmup_utils import get_embedding_cache_warmup_candidates

        try:
            candidates = await get_embedding_cache_warmup_candidates(self, historical_data)
            self.logger.info(f"Generated {len(candidates)} embedding cache warmup candidates")
            return candidates
        except Exception as e:
            self.logger.error(f"Failed to get embedding cache warmup candidates: {e}")
            return []

    async def warmup_item(self, item_key: str, item_data: Any) -> float:
        """
        Warm up a specific cache item.

        Args:
            item_key: Cache key for the item
            item_data: Data to warm up (query text or embedding data)

        Returns:
            Memory used in MB for the warmed up item
        """
        try:
            memory_used = 0.0

            if isinstance(item_data, str):
                # Direct query text - create a mock embedding for warmup
                query_text = item_data
                model_name = "default"  # Use default model for warmup

                # Generate mock embedding (zeros) for warmup
                embedding = torch.zeros(384)  # Common embedding dimension

                # Cache the mock embedding
                await self.cache_embedding(
                    text=query_text, model_name=model_name, embedding=embedding, embedding_type=EmbeddingType.QUERY_TEXT
                )

                # Estimate memory usage
                memory_used = self.estimate_item_size(item_key, item_data)

            elif isinstance(item_data, dict):
                # Structured data with query, model, etc.
                query_text = item_data.get("query", "")
                model_name = item_data.get("model", "default")
                embedding_type = EmbeddingType.QUERY_TEXT

                # Check if we have cached embedding data
                if "embedding" in item_data:
                    embedding_data = item_data["embedding"]
                    if isinstance(embedding_data, list):
                        embedding = torch.tensor(embedding_data, dtype=torch.float32)
                    else:
                        embedding = torch.zeros(384)
                else:
                    embedding = torch.zeros(384)

                # Cache the embedding
                await self.cache_embedding(text=query_text, model_name=model_name, embedding=embedding, embedding_type=embedding_type)

                memory_used = self.estimate_item_size(item_key, item_data)

            self.logger.debug(f"Warmed up embedding cache item: {item_key} ({memory_used:.2f}MB)")
            return memory_used

        except Exception as e:
            self.logger.error(f"Failed to warm up embedding cache item {item_key}: {e}")
            return 0.0

    def estimate_item_size(self, item_key: str, item_data: Any) -> float:
        """
        Estimate memory size for a cache item.

        Args:
            item_key: Cache key for the item
            item_data: Data to estimate size for

        Returns:
            Estimated size in MB
        """
        try:
            base_size = 0.0

            if isinstance(item_data, str):
                # String data - estimate based on text length
                text_size = len(item_data.encode("utf-8"))
                # Add embedding size (assume 384 dimensions * 4 bytes per float)
                embedding_size = 384 * 4
                base_size = (text_size + embedding_size) / (1024 * 1024)

            elif isinstance(item_data, dict):
                # Dictionary data - estimate based on serialized size
                text_size = len(str(item_data).encode("utf-8"))
                embedding_size = 384 * 4  # Default embedding size

                # If embedding data is present, calculate actual size
                if "embedding" in item_data:
                    embedding_data = item_data["embedding"]
                    if isinstance(embedding_data, list):
                        embedding_size = len(embedding_data) * 4  # 4 bytes per float

                base_size = (text_size + embedding_size) / (1024 * 1024)

            elif isinstance(item_data, torch.Tensor):
                # Tensor data - calculate actual size
                base_size = item_data.numel() * item_data.element_size() / (1024 * 1024)

            else:
                # Default estimation
                base_size = 0.01  # 10KB default

            # Add overhead for cache metadata
            overhead_factor = 1.2  # 20% overhead
            total_size = base_size * overhead_factor

            return max(total_size, 0.001)  # Minimum 1KB

        except Exception as e:
            self.logger.warning(f"Failed to estimate size for {item_key}: {e}")
            return 0.01  # Default 10KB

    async def get_warmup_statistics(self) -> dict[str, Any]:
        """
        Get statistics relevant for warmup planning.

        Returns:
            Dictionary with warmup-relevant statistics
        """
        try:
            stats = self.get_cache_stats()

            # Add warmup-specific information
            warmup_stats = {
                "cache_type": "embedding_cache",
                "current_items": stats.get("total_cached", 0),
                "hit_rate": stats.get("hit_rate", 0.0),
                "average_item_size_mb": 0.0,
                "most_accessed_models": [],
                "frequent_query_patterns": [],
                "storage_efficiency": stats.get("compression_ratio", 1.0),
                "warmup_priority_score": 0.0,
            }

            # Calculate average item size
            if stats.get("total_cached", 0) > 0:
                total_storage = stats.get("storage_bytes", 0)
                warmup_stats["average_item_size_mb"] = total_storage / (1024 * 1024) / stats["total_cached"]

            # Calculate warmup priority score (higher = more important to warm up)
            hit_rate = stats.get("hit_rate", 0.0)
            total_items = stats.get("total_cached", 0)

            if hit_rate > 0.7 and total_items > 10:
                warmup_stats["warmup_priority_score"] = 8.0  # High priority
            elif hit_rate > 0.5 and total_items > 5:
                warmup_stats["warmup_priority_score"] = 6.0  # Medium priority
            elif hit_rate > 0.3:
                warmup_stats["warmup_priority_score"] = 4.0  # Low priority
            else:
                warmup_stats["warmup_priority_score"] = 2.0  # Very low priority

            # Add model version information
            model_versions = stats.get("model_versions", {})
            warmup_stats["most_accessed_models"] = [{"model": model, "version": version} for model, version in model_versions.items()]

            return warmup_stats

        except Exception as e:
            self.logger.error(f"Failed to get warmup statistics: {e}")
            return {
                "cache_type": "embedding_cache",
                "current_items": 0,
                "hit_rate": 0.0,
                "average_item_size_mb": 0.0,
                "warmup_priority_score": 1.0,
                "error": str(e),
            }


# Global embedding cache service instance
_embedding_cache_service: EmbeddingCacheService | None = None


async def get_embedding_cache_service() -> EmbeddingCacheService:
    """
    Get the global embedding cache service instance.

    Returns:
        EmbeddingCacheService: The global embedding cache service instance
    """
    global _embedding_cache_service
    if _embedding_cache_service is None:
        _embedding_cache_service = EmbeddingCacheService()
        await _embedding_cache_service.initialize()
    return _embedding_cache_service


async def shutdown_embedding_cache_service() -> None:
    """Shutdown the global embedding cache service."""
    global _embedding_cache_service
    if _embedding_cache_service:
        await _embedding_cache_service.shutdown()
        _embedding_cache_service = None
